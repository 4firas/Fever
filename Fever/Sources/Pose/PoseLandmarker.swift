import CoreVideo
import Foundation

/// Abstraction over the pose inference backend. The real backend is
/// `MediaPipePoseLandmarker` (BlazePose GHUM via the Python sidecar);
/// `StubPoseLandmarker` synthesizes a static T-pose so the solver / OSC path can
/// be exercised with no camera or hardware.
public protocol PoseLandmarker {
    func detect(_ pixelBuffer: CVPixelBuffer,
                at time: TimeInterval) async -> PoseResult?

    /// Clear any per-run temporal state (smoothed scale, depth-sign hysteresis).
    /// Called between tracking runs. Default no-op for stateless backends.
    func reset()
}

public extension PoseLandmarker {
    func reset() {}
}

/// Confidence thresholds matching the PinoFBT defaults.
public struct PoseLandmarkerConfig {
    public var minPoseDetectionConfidence: Float = 0.5
    public var minPosePresenceConfidence: Float = 0.5
    public var minTrackingConfidence: Float = 0.5
    public var numPoses: Int = 1
    public init() {}
}

// MARK: - Stub (no MediaPipe dependency required to compile/test the pipeline)

/// Returns a synthetic T-pose so the solver / OSC path can be exercised on
/// machines without a camera or the MediaPipe framework linked.
public final class StubPoseLandmarker: PoseLandmarker {

    public init() {}

    public func detect(_ pixelBuffer: CVPixelBuffer,
                       at time: TimeInterval) async -> PoseResult? {
        // 33 normalized landmarks forming an approximate standing pose, centered.
        // The limbs are SLIGHTLY ARTICULATED and gently ANIMATED off the time base
        // (knees/elbows softly bent and out of plane, feet pointing forward) so the
        // pose is NOT the degenerate fully-vertical/coplanar T-pose: that pathology
        // sits exactly on the ZXY euler gimbal seam and made the re-enabled rotation
        // wire read constant ±180 (a representation artifact, not a real rotation).
        // A realistic, well-conditioned, moving pose exercises the rotation path the
        // way live tracking does — bounded, motion-responsive euler on the wire.
        let c: Float = 0.5
        let s = Float(sin(time * 0.6))       // slow, smooth limb swing in [-1, 1]
        let d = Float(cos(time * 0.6))
        func p(_ x: Float, _ y: Float, _ z: Float) -> NormalizedLandmark {
            NormalizedLandmark(position: SIMD3<Float>(x, y, z), visibility: 0.9)
        }
        var lm: [NormalizedLandmark] = Array(repeating: p(c, c, 0), count: 33)
        lm[0]  = p(c, 0.35, 0.05)            // nose (forward of the ear line)
        lm[7]  = p(c - 0.06, 0.40, 0); lm[8] = p(c + 0.06, 0.40, 0)   // ears
        lm[11] = p(c - 0.18, 0.55, 0); lm[12] = p(c + 0.18, 0.55, 0)  // shoulders
        // Elbows/wrists: arms down at the sides, forearms bent CLEARLY forward
        // (well off the shoulder→elbow line) and swinging, so the elbow two-axis
        // frame is well-conditioned and articulates.
        lm[13] = p(c - 0.20, 0.72, 0.04); lm[14] = p(c + 0.20, 0.72, 0.04)        // elbows
        lm[15] = p(c - 0.22, 0.86, 0.22 + 0.06 * s); lm[16] = p(c + 0.22, 0.86, 0.22 + 0.06 * s)  // wrists
        lm[23] = p(c - 0.10, 0.75, 0); lm[24] = p(c + 0.10, 0.75, 0)  // hips
        // Knees softly bent forward; ankles a big step FORWARD of the knee so the
        // shank is held well off vertical (no gimbal seam) and SWINGS with time.
        lm[25] = p(c - 0.10, 0.86, 0.06); lm[26] = p(c + 0.10, 0.86, 0.06)        // knees
        lm[27] = p(c - 0.10, 0.95, 0.20 + 0.05 * s); lm[28] = p(c + 0.10, 0.95, 0.20 + 0.05 * s)  // ankles
        lm[29] = p(c - 0.10, 0.97, 0.14); lm[30] = p(c + 0.10, 0.97, 0.14)        // heels (behind toe)
        lm[31] = p(c - 0.10, 1.00, 0.30 + 0.04 * d); lm[32] = p(c + 0.10, 1.00, 0.30 + 0.04 * d)  // foot index (forward, swinging)
        return PoseResult(landmarks: lm, timestamp: time)
    }
}
