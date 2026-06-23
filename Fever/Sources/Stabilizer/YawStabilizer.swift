import simd

/// YawStabilizer — Fever's adaptation of PinoFBT's "Body Stabilizer".
///
/// PinoFBT keeps the avatar coherent on turns by deriving ONE body-facing yaw and
/// imposing it on the trackers (rather than letting each tracker's monocular yaw
/// jitter/flip independently — the classic depth ambiguity when you face away).
/// This does the same, safely:
///
///   1. takes the HIP's solved yaw (twist about world-up) as the body-facing
///      reference — already derived from the hip line, so no convention mismatch;
///   2. LOW-PASS smooths it across frames (wrap-safe at ±180°), killing the
///      per-frame yaw jitter/flips; then
///   3. IMPOSES that single smoothed yaw on the torso trackers (hip + chest),
///      preserving each one's pitch/roll (swing) — so the torso turns as one
///      coherent, stable unit. Limbs keep their own yaw (a wave/kick is real).
///
/// Reference type (cross-frame smoothing state), confined to the serial inference
/// worker like `RotationState`/`FootMotionState`. Opt-in via
/// `TrackingConfig.yawStabilizer` so it can never regress the working baseline.
public final class YawStabilizer {

    /// Fraction of the PREVIOUS yaw retained each frame (0 = raw/no smoothing,
    /// →1 = heavy lag). Tunable; the live default comes from the config.
    public var smoothing: Float

    private var smoothedYaw: Float?        // radians, low-pass state
    private let up = SIMD3<Float>(0, 1, 0)

    public init(smoothing: Float = 0.6) {
        self.smoothing = max(0, min(0.99, smoothing))
    }

    /// Drop the smoothing history (Recenter / run reset).
    public func reset() { smoothedYaw = nil }

    /// Feed the body-facing REFERENCE rotation (the hip); returns the smoothed
    /// coherent body-yaw as a quaternion about world-up. Wrap-safe across ±180°.
    public func update(reference q: simd_quatf) -> simd_quatf {
        let raw = swingTwist(q, axis: up).twistAngle
        let y: Float
        if let prev = smoothedYaw {
            y = prev + (1 - smoothing) * shortestAngleDelta(from: prev, to: raw)
        } else {
            y = raw
        }
        smoothedYaw = y
        return simd_quatf(angle: y, axis: up)
    }

    /// Replace `q`'s yaw (twist about world-up) with `yaw`, preserving the swing
    /// (pitch + roll). Used to impose the coherent body yaw on a torso tracker.
    public func imposeYaw(_ q: simd_quatf, yaw: simd_quatf) -> simd_quatf {
        let swing = swingTwist(q, axis: up).swing
        let out = swing * yaw
        let l = simd_length(out.vector)
        return l > 1e-6 ? simd_quatf(vector: out.vector / l) : q
    }
}
