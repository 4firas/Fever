import CoreVideo
import CoreImage
import Foundation
import simd

/// Pose backend backed by the MediaPipe Python sidecar. Downscales each frame to
/// tightly-packed RGB, sends it to the sidecar, and converts the returned world
/// landmarks into the solver frame. Confined to the single serial inference worker.
public final class MediaPipePoseLandmarker: PoseLandmarker {
    private let service: PoseInferenceService
    private let latch = FloorOriginLatch()
    private let targetHeight: Int
    private let zSign: Float
    private let ci = CIContext(options: [.workingColorSpace: NSNull()])
    private var rgbBuffer = Data()
    private var rgbaScratch = [UInt8]()

    public init(service: PoseInferenceService,
                targetHeight: Int = 480,
                zSign: Float = MediaPipeFrame.defaultZSign) {
        self.service = service
        self.targetHeight = targetHeight
        self.zSign = zSign
    }

    /// Convenience: resolve the sidecar paths and build a PoseSidecar. Returns nil
    /// if the sidecar isn't installed (no embedded Resources and no dev venv).
    public convenience init?() {
        guard let paths = SidecarPaths.resolve(bundle: .main, projectRoot: nil) else { return nil }
        self.init(service: PoseSidecar(paths: paths))
    }

    public func reset() { latch.reset(); service.reset() }

    public func detect(_ pixelBuffer: CVPixelBuffer, at time: TimeInterval) async -> PoseResult? {
        let srcW = CVPixelBufferGetWidth(pixelBuffer), srcH = CVPixelBufferGetHeight(pixelBuffer)
        guard srcH > 0, srcW > 0 else { return nil }
        let h = targetHeight
        let w = max(1, Int((Float(srcW) / Float(srcH) * Float(h)).rounded()))
        guard let rgb = renderRGB(pixelBuffer, width: w, height: h) else { return nil }
        let t = UInt64((time * 1_000_000).rounded())
        guard let reply = await service.infer(rgb: rgb, width: w, height: h, tMicros: t) else { return nil }
        guard let pose = MediaPipeFrame.toSolverFrame(reply, latch: latch, zSign: zSign) else { return nil }
        // Stamp the caller's timestamp (toSolverFrame leaves it 0).
        return PoseResult(landmarks: pose.landmarks, timestamp: time, imagePoints: pose.imagePoints)
    }

    /// Downscale + convert to tightly-packed RGB888 (no row padding) via CoreImage.
    private func renderRGB(_ pb: CVPixelBuffer, width: Int, height: Int) -> Data? {
        let src = CIImage(cvPixelBuffer: pb)
        guard src.extent.width > 0, src.extent.height > 0 else { return nil }
        let sx = CGFloat(width) / src.extent.width
        let sy = CGFloat(height) / src.extent.height
        let scaled = src.transformed(by: CGAffineTransform(scaleX: sx, y: sy))
        let bytesPerRow = width * 4
        if rgbaScratch.count != bytesPerRow * height {
            rgbaScratch = [UInt8](repeating: 0, count: bytesPerRow * height)
        }
        if rgbBuffer.count != width * height * 3 {
            rgbBuffer = Data(count: width * height * 3)
        }
        let cs = CGColorSpaceCreateDeviceRGB()
        rgbaScratch.withUnsafeMutableBytes { ptr in
            ci.render(scaled, toBitmap: ptr.baseAddress!, rowBytes: bytesPerRow,
                      bounds: CGRect(x: 0, y: 0, width: width, height: height),
                      format: .RGBA8, colorSpace: cs)
        }
        // Pack RGBA -> RGB.
        rgbBuffer.withUnsafeMutableBytes { dst in
            let d = dst.bindMemory(to: UInt8.self)
            rgbaScratch.withUnsafeBufferPointer { s in
                for i in 0..<(width * height) {
                    d[i*3]   = s[i*4]
                    d[i*3+1] = s[i*4+1]
                    d[i*3+2] = s[i*4+2]
                }
            }
        }
        return rgbBuffer
    }
}

/// The live pose backend: the MediaPipe sidecar, or the synthetic stub if the
/// sidecar isn't installed (no embedded Resources and no dev venv) so the app
/// always runs.
public func makeLivePoseLandmarker() -> PoseLandmarker {
    if let mp = MediaPipePoseLandmarker() { return mp }
    return StubPoseLandmarker()
}
