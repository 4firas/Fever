import Accelerate
import CoreML
import CoreVideo
import Foundation
import simd

/// Native CoreML pose backend — the real PinoFBT NLF model
/// (`pose.mlmodelc`, 24-joint SMPL, camera space, +Y down).
///
/// The compiled model is a single-input graph (`image`, 384×384 BGRA) that carries
/// its own detector/crop/tracker, and emits *localizer fields* for 1048 query points:
/// `joints3D [1,1048,3]` metres and `joints2D [1,1048,2]` pixels in the fed frame.
/// The SMPL-24 joints are the **last 24** of those 1048 (verified geometrically:
/// indices 1024…1047 form the pelvis→hips→spine→limbs→head skeleton, the other
/// clusters are ~2 cm dense point groups).
///
/// This replaces the onnxruntime sidecar: no Python, no child process, no ONNX
/// export — the vendor's own compiled CoreML graph, on the ANE/GPU.
public final class CoreMLPoseLandmarker: NLFPoseSource, @unchecked Sendable {

    /// Model input side (the graph's image constraint).
    public static let inputSide = 384
    /// Index of the first SMPL-24 joint inside the 1048 localizer queries.
    public static let firstJoint = 1024
    private static let jointCount = SMPLJoint.count   // 24

    private let model: MLModel
    /// Guards `scratch`. `MLModel.prediction` is itself thread-safe; the pipeline
    /// drives this from a single inference worker, the lock just makes that explicit.
    private let lock = NSLock()
    private var scratch: CVPixelBuffer?

    public init(modelURL: URL) throws {
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .all              // ANE + GPU + CPU; the ANE carries this graph
        self.model = try MLModel(contentsOf: modelURL, configuration: cfg)
    }

    /// Resolve the model: env override, else the app bundle's Resources/models, else
    /// the user's Application Support copy, else the dev rig.
    public convenience init?() {
        guard let p = CoreMLModelPaths.resolve() else { return nil }
        try? self.init(modelURL: p.model)
    }

    public func reset() { /* the graph self-tracks; nothing to reset */ }

    public func detect(_ pixelBuffer: CVPixelBuffer, at time: TimeInterval) async -> SMPLPose? {
        infer(pixelBuffer, at: time)
    }

    /// The synchronous body. CoreML inference is fast enough to stay on the inference
    /// worker (the model runs on the ANE/GPU); the lock keeps the scratch buffer safe.
    private func infer(_ pixelBuffer: CVPixelBuffer, at time: TimeInterval) -> SMPLPose? {
        lock.lock()
        defer { lock.unlock() }
        guard let input = scaledInput(from: pixelBuffer) else { return nil }
        guard let provider = try? MLDictionaryFeatureProvider(
            dictionary: ["image": MLFeatureValue(pixelBuffer: input)]) else { return nil }
        guard let out = try? model.prediction(from: provider) else { return nil }
        guard let j3 = out.featureValue(for: "joints3D")?.multiArrayValue,
              let j2 = out.featureValue(for: "joints2D")?.multiArrayValue else { return nil }
        return Self.pose(from: j3, j2, timestamp: time)
    }

    /// Frame → the model's 384×384 BGRA input. Plain scale (the graph was trained on
    /// frames handed to CoreML at its constraint size, which stretches to fill).
    private func scaledInput(from src: CVPixelBuffer) -> CVPixelBuffer? {
        let side = Self.inputSide
        if scratch == nil {
            var pb: CVPixelBuffer?
            let attrs: [CFString: Any] = [
                kCVPixelBufferCGImageCompatibilityKey: false,
                kCVPixelBufferCGBitmapContextCompatibilityKey: false,
                kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary,
            ]
            CVPixelBufferCreate(kCFAllocatorDefault, side, side, kCVPixelFormatType_32BGRA,
                                attrs as CFDictionary, &pb)
            scratch = pb
        }
        guard let dst = scratch else { return nil }
        guard CVPixelBufferGetPixelFormatType(src) == kCVPixelFormatType_32BGRA else { return nil }

        CVPixelBufferLockBaseAddress(src, .readOnly)
        CVPixelBufferLockBaseAddress(dst, [])
        defer {
            CVPixelBufferUnlockBaseAddress(dst, [])
            CVPixelBufferUnlockBaseAddress(src, .readOnly)
        }
        guard let sb = CVPixelBufferGetBaseAddress(src), let db = CVPixelBufferGetBaseAddress(dst) else { return nil }
        var s = vImage_Buffer(data: sb, height: vImagePixelCount(CVPixelBufferGetHeight(src)),
                              width: vImagePixelCount(CVPixelBufferGetWidth(src)),
                              rowBytes: CVPixelBufferGetBytesPerRow(src))
        var d = vImage_Buffer(data: db, height: vImagePixelCount(side), width: vImagePixelCount(side),
                              rowBytes: CVPixelBufferGetBytesPerRow(dst))
        // BGRA is 4-channel 8-bit; the scale is channel-agnostic.
        guard vImageScale_ARGB8888(&s, &d, nil, vImage_Flags(kvImageHighQualityResampling)) == kvImageNoError
        else { return nil }
        return dst
    }

    // MARK: - output decoding

    private static func pose(from j3: MLMultiArray, _ j2: MLMultiArray, timestamp: Double) -> SMPLPose {
        if ProcessInfo.processInfo.environment["FEVER_COREML_DEBUG"] != nil {
            print("[coreml] joints3D shape=\(j3.shape) strides=\(j3.strides) dtype=\(j3.dataType.rawValue)")
            print("[coreml] joints2D shape=\(j2.shape) strides=\(j2.strides) dtype=\(j2.dataType.rawValue)")
        }
        // Read the outputs straight out of their buffers as fp16 — the arrays are dense
        // (the last dim is the innermost), so the flat element index is i*dim + c, the
        // same addressing the ObjC probe used to dump all 1048 queries.
        var p3 = [SIMD3<Float>](repeating: .zero, count: jointCount)
        var p2 = [SIMD2<Float>](repeating: .zero, count: jointCount)
        // Address the outputs through their OWN strides: the ANE/GPU path pads each
        // query point (stride 32 element slots for joints3D, not 3), while the
        // CPU-only path is dense. Assuming either layout silently reads the wrong
        // memory — it still returns numbers, just not the body's.
        let s3 = j3.dataPointer.assumingMemoryBound(to: UInt16.self)
        let st3 = j3.strides[1].intValue, st3c = j3.strides[2].intValue
        for i in 0..<jointCount {
            let base = (firstJoint + i) * st3
            p3[i] = SIMD3<Float>(Float(Float16(bitPattern: s3[base + 0 * st3c])),
                                 Float(Float16(bitPattern: s3[base + 1 * st3c])),
                                 Float(Float16(bitPattern: s3[base + 2 * st3c])))
        }
        let s2 = j2.dataPointer.assumingMemoryBound(to: UInt16.self)
        let st2 = j2.strides[1].intValue, st2c = j2.strides[2].intValue
        for i in 0..<jointCount {
            let base = (firstJoint + i) * st2
            p2[i] = SIMD2<Float>(Float(Float16(bitPattern: s2[base + 0 * st2c])),
                                 Float(Float16(bitPattern: s2[base + 1 * st2c])))
        }
        return SMPLPose(joints3D: p3, joints2D: p2,
                        hasTracked: confidence(p3) ? 1 : 0,
                        timestamp: timestamp,
                        width: inputSide, height: inputSide)
    }

    /// Body present? The graph has no `has_tracked` output, so the tell is scale:
    /// a real body is ~0.3-0.5 m across the shoulders/hips, while a no-person frame
    /// collapses the skeleton to a few centimetres (measured: 0.024 m).
    private static func confidence(_ p: [SIMD3<Float>]) -> Bool {
        for v in p where !v.x.isFinite || !v.y.isFinite || !v.z.isFinite { return false }
        let shoulder = simd_distance(p[SMPLJoint.leftShoulder.rawValue], p[SMPLJoint.rightShoulder.rawValue])
        let hip = simd_distance(p[SMPLJoint.leftHip.rawValue], p[SMPLJoint.rightHip.rawValue])
        return max(shoulder, hip) >= 0.10
    }
}

/// Where the compiled pose model lives.
public struct CoreMLModelPaths: Sendable {
    public let model: URL

    public static func resolve(env: [String: String] = ProcessInfo.processInfo.environment) -> CoreMLModelPaths? {
        var candidates: [String] = []
        if let p = env["FEVER_COREML_MODEL"] { candidates.append((p as NSString).expandingTildeInPath) }
        if let res = Bundle.main.resourceURL {
            candidates.append(res.appendingPathComponent("models/pose.mlmodelc").path)
        }
        candidates.append("~/Library/Application Support/Fever/models/pose.mlmodelc")
        candidates.append("~/Dev/Fever/models/pose.mlmodelc")
        candidates.append("~/pino_rig/model/pose.mlmodelc")
        for c in candidates {
            var isDir: ObjCBool = false
            let path = (c as NSString).expandingTildeInPath
            if FileManager.default.fileExists(atPath: path, isDirectory: &isDir), isDir.boolValue {
                return CoreMLModelPaths(model: URL(fileURLWithPath: path))
            }
        }
        return nil
    }
}
