import Accelerate
import CoreVideo
import CoreImage
import Foundation
import simd

/// "camera frame in → SMPL-24 pose out". Both the live sidecar backend and the
/// headless stub conform, so the pipeline is agnostic.
public protocol NLFPoseSource: AnyObject {
    func detect(_ pixelBuffer: CVPixelBuffer, at time: TimeInterval) async -> SMPLPose?
    func reset()
    /// False for the synthetic stub (canned standing pose used when the NLF runtime
    /// isn't installed) so the UI can warn that it's a DEMO, not real tracking.
    var isLive: Bool { get }
}

public extension NLFPoseSource {
    var isLive: Bool { true }
}

/// Pose backend backed by the NLF onnxruntime sidecar: downscales each camera
/// frame to tightly-packed RGB, sends it, and returns SMPL-24 joints. The model
/// letterboxes internally so the downscale is only bandwidth tuning; the sidecar
/// also auto-corrects vertical orientation. Confined to the serial inference worker.
public final class NLFPoseLandmarker {
    private let service: NLFInferenceService
    private let targetHeight: Int
    private var rgbBuffer = Data()
    private var rgbaScratch = [UInt8]()

    public init(service: NLFInferenceService, targetHeight: Int = 480) {
        self.service = service
        self.targetHeight = targetHeight
    }

    /// Resolve the external NLF runtime (FEVER_NLF_ROOT) and build a live sidecar;
    /// nil if the runtime isn't present.
    public convenience init?() {
        guard let paths = NLFPaths.resolve() else { return nil }
        self.init(service: NLFSidecar(paths: paths))
    }

    public func reset() { service.reset() }

    public func detect(_ pixelBuffer: CVPixelBuffer, at time: TimeInterval) async -> SMPLPose? {
        let srcW = CVPixelBufferGetWidth(pixelBuffer), srcH = CVPixelBufferGetHeight(pixelBuffer)
        guard srcW > 0, srcH > 0 else { return nil }
        let h = targetHeight
        let w = max(1, Int((Float(srcW) / Float(srcH) * Float(h)).rounded()))
        guard let rgb = renderRGB(pixelBuffer, width: w, height: h) else { return nil }
        return await service.infer(rgb: rgb, width: w, height: h, timestamp: time)
    }

    /// Downscale a BGRA pixel buffer to tightly-packed, TOP-LEFT RGB888 via a CPU
    /// CGContext draw (no Core Image / GPU render — frees the GPU for the model and
    /// keeps orientation deterministic so joints2D land on the body).
    private func renderRGB(_ pb: CVPixelBuffer, width: Int, height: Int) -> Data? {
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pb) else { return nil }
        let srcW = CVPixelBufferGetWidth(pb), srcH = CVPixelBufferGetHeight(pb)
        let bpr = CVPixelBufferGetBytesPerRow(pb)
        let cs = CGColorSpaceCreateDeviceRGB()
        let srcBI = CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        guard let srcCtx = CGContext(data: base, width: srcW, height: srcH, bitsPerComponent: 8,
                                     bytesPerRow: bpr, space: cs, bitmapInfo: srcBI),
              let cg = srcCtx.makeImage() else { return nil }

        if rgbaScratch.count != width * height * 4 { rgbaScratch = [UInt8](repeating: 0, count: width * height * 4) }
        if rgbBuffer.count != width * height * 3 { rgbBuffer = Data(count: width * height * 3) }
        var ok = false
        rgbaScratch.withUnsafeMutableBytes { ptr in
            guard let dst = CGContext(data: ptr.baseAddress, width: width, height: height,
                                      bitsPerComponent: 8, bytesPerRow: width * 4, space: cs,
                                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return }
            dst.draw(cg, in: CGRect(x: 0, y: 0, width: width, height: height))   // row0 = top
            ok = true
        }
        guard ok else { return nil }
        // Pack RGBA8888 → tightly-packed RGB888 (drop alpha) via Accelerate — a
        // vectorized one-pass channel drop, identical bytes to the old per-pixel
        // loop but off the scalar path, freeing the CPU for the GPU/ANE-bound model.
        rgbaScratch.withUnsafeMutableBytes { srcRaw in
            rgbBuffer.withUnsafeMutableBytes { dstRaw in
                guard let sb = srcRaw.baseAddress, let db = dstRaw.baseAddress else { return }
                var src = vImage_Buffer(data: sb, height: vImagePixelCount(height),
                                        width: vImagePixelCount(width), rowBytes: width * 4)
                var dst = vImage_Buffer(data: db, height: vImagePixelCount(height),
                                        width: vImagePixelCount(width), rowBytes: width * 3)
                vImageConvert_RGBA8888toRGB888(&src, &dst, vImage_Flags(kvImageNoFlags))
            }
        }
        return rgbBuffer
    }
}

/// Synthetic camera-less backend: a fixed standing SMPL-24 pose in the model's
/// native camera space (+Y down, meters) so the pipeline runs headless in tests.
public final class StubNLFLandmarker {
    public init() {}
    public func reset() {}

    public func detect(_ pixelBuffer: CVPixelBuffer, at time: TimeInterval) async -> SMPLPose? {
        Self.standing(timestamp: time)
    }

    /// A plausible upright pose: +Y down (head≈0.4, feet≈1.8), shoulder width ~0.4 m.
    public static func standing(timestamp: Double) -> SMPLPose {
        func p(_ x: Float, _ y: Float, _ z: Float) -> SIMD3<Float> { SIMD3(x, y, z) }
        var j = [SIMD3<Float>](repeating: .zero, count: 24)
        j[SMPLJoint.pelvis.rawValue]       = p( 0.00, 1.00, 0.0)
        j[SMPLJoint.leftHip.rawValue]      = p( 0.10, 1.02, 0.0)
        j[SMPLJoint.rightHip.rawValue]     = p(-0.10, 1.02, 0.0)
        j[SMPLJoint.spine1.rawValue]       = p( 0.00, 0.88, 0.0)
        j[SMPLJoint.leftKnee.rawValue]     = p( 0.11, 1.40, 0.0)
        j[SMPLJoint.rightKnee.rawValue]    = p(-0.11, 1.40, 0.0)
        j[SMPLJoint.spine2.rawValue]       = p( 0.00, 0.76, 0.0)
        j[SMPLJoint.leftAnkle.rawValue]    = p( 0.11, 1.75, 0.0)
        j[SMPLJoint.rightAnkle.rawValue]   = p(-0.11, 1.75, 0.0)
        j[SMPLJoint.spine3.rawValue]       = p( 0.00, 0.66, 0.0)
        j[SMPLJoint.leftFoot.rawValue]     = p( 0.11, 1.82, -0.10)
        j[SMPLJoint.rightFoot.rawValue]    = p(-0.11, 1.82, -0.10)
        j[SMPLJoint.neck.rawValue]         = p( 0.00, 0.52, 0.0)
        j[SMPLJoint.leftCollar.rawValue]   = p( 0.06, 0.58, 0.0)
        j[SMPLJoint.rightCollar.rawValue]  = p(-0.06, 0.58, 0.0)
        j[SMPLJoint.head.rawValue]         = p( 0.00, 0.42, 0.0)
        j[SMPLJoint.leftShoulder.rawValue] = p( 0.18, 0.58, 0.0)
        j[SMPLJoint.rightShoulder.rawValue] = p(-0.18, 0.58, 0.0)
        j[SMPLJoint.leftElbow.rawValue]    = p( 0.22, 0.82, 0.0)
        j[SMPLJoint.rightElbow.rawValue]   = p(-0.22, 0.82, 0.0)
        j[SMPLJoint.leftWrist.rawValue]    = p( 0.24, 1.04, 0.0)
        j[SMPLJoint.rightWrist.rawValue]   = p(-0.24, 1.04, 0.0)
        j[SMPLJoint.leftHand.rawValue]     = p( 0.25, 1.10, 0.0)
        j[SMPLJoint.rightHand.rawValue]    = p(-0.25, 1.10, 0.0)
        let j2 = j.map { SIMD2<Float>(320 + $0.x * 200, $0.y * 200) }
        return SMPLPose(joints3D: j, joints2D: j2, hasTracked: 1, timestamp: timestamp)
    }
}

extension NLFPoseLandmarker: NLFPoseSource {}
extension StubNLFLandmarker: NLFPoseSource {
    public var isLive: Bool { false }   // canned pose — not real tracking
}

/// The live NLF backend, or the synthetic stub if the runtime isn't installed.
public func makeLiveNLFLandmarker() -> any NLFPoseSource {
    NLFPoseLandmarker() ?? StubNLFLandmarker()
}
