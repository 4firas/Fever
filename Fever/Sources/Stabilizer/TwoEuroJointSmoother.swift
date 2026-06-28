import simd
import Foundation

/// PinoFBT 2.0's input-side landmark smoother: a vectorized **OneEuroFilter** applied
/// to the whole (24,3) model-joint array BEFORE the IK solve. Live-read off the real
/// instance: a SINGLE config over all 24 keypoints (no body/leg split), with the
/// `power` (a.k.a. `dollar`) SPEED EXPONENT baked to 2.
///
/// Exact live constants (`Q6_filter`): `power=2`, `min_cutoff=1`, `beta=400`,
/// `d_cutoff=1.0`. The big `beta` is correct: joint-stream speed is small and is
/// raised to `power` (`beta·speed²`), so it must NOT be sanity-clamped.
///
/// Per coordinate, per joint, standard One-Euro:
///   dt    = t - t_prev
///   deriv = (raw - lastRaw)/dt
///   edx   = lowpass(deriv, dx_prev, alpha(d_cutoff, dt))
///   speed = |edx|
///   cutoff = min_cutoff + beta * pow(speed, power)     // power = 2
///   x = lowpass(raw, x_prev, alpha(cutoff, dt))
///   alpha(c,dt) = 1 / (1 + tau/dt),  tau = 1/(2π·c)
public struct TwoEuroParams: Sendable {
    public var minCutoff: Double   // Hz
    public var beta: Double        // speed coefficient
    public var dCutoff: Double     // derivative cutoff Hz
    public var dollar: Double      // speed exponent (PinoFBT `power`; live = 2)
    public init(minCutoff: Double = 1.0, beta: Double = 400.0, dCutoff: Double = 1.0, dollar: Double = 2.0) {
        self.minCutoff = minCutoff; self.beta = beta; self.dCutoff = dCutoff; self.dollar = dollar
    }
}

public final class TwoEuroJointSmoother {

    private var params: TwoEuroParams
    private var x = [SIMD3<Float>](repeating: .zero, count: SMPLJoint.count)   // last filtered value
    private var dx = [SIMD3<Float>](repeating: .zero, count: SMPLJoint.count)  // last filtered derivative
    private var lastRaw = [SIMD3<Float>](repeating: .zero, count: SMPLJoint.count)
    private var lastT: Double = -1

    /// One config over ALL 24 joints (live-confirmed). The `legs:` parameter is
    /// retained in the signature for source compatibility but IGNORED — there is no
    /// body/leg split in this build.
    public init(body: TwoEuroParams = TwoEuroParams(),
                legs: TwoEuroParams = TwoEuroParams()) {
        self.params = body
        _ = legs
    }

    /// Reset filter state (call when tracking drops, so a re-acquire snaps cleanly).
    public func reset() { lastT = -1 }

    /// The current smoothed per-joint VELOCITY (units/second) — the OneEuro filtered
    /// derivative `dx` maintained internally. Valid after the first `smooth(_:)`
    /// call (zero before then). The forward-predicting upsampler extrapolates the
    /// smoothed joints along this velocity to fill the gap between inferences, which
    /// is what cancels the inference-rate latency without adding jitter (the velocity
    /// is already low-pass filtered, so it doesn't chatter).
    public func velocity() -> [SIMD3<Float>] { dx }

    private static func alpha(cutoff: Double, dt: Double) -> Float {
        let tau = 1.0 / (2.0 * Double.pi * max(cutoff, 1e-6))
        return Float(1.0 / (1.0 + tau / max(dt, 1e-6)))
    }

    /// Smooth one frame of model joints (24×3). `timestamp` in seconds.
    public func smooth(_ joints: [SIMD3<Float>], timestamp: Double) -> [SIMD3<Float>] {
        guard joints.count == SMPLJoint.count else { return joints }
        if lastT < 0 {                               // first frame — seed, no filtering
            x = joints; lastRaw = joints; lastT = timestamp
            dx = [SIMD3<Float>](repeating: .zero, count: SMPLJoint.count)
            return joints
        }
        let dt = max(timestamp - lastT, 1e-4); lastT = timestamp
        let rate = 1.0 / dt
        let p = params
        let edRate = Self.alpha(cutoff: p.dCutoff, dt: dt)
        var out = joints
        for i in 0..<SMPLJoint.count {
            let raw = joints[i]
            // filtered derivative
            let deriv = (raw - lastRaw[i]) * Float(rate)
            let edx = dx[i] + (deriv - dx[i]) * edRate
            dx[i] = edx
            // adaptive cutoff with the PinoFBT `power` speed exponent
            let speed = Double(simd_length(edx))
            let cutoff = p.minCutoff + p.beta * pow(speed, p.dollar)
            let a = Self.alpha(cutoff: cutoff, dt: dt)
            let fx = x[i] + (raw - x[i]) * a
            x[i] = fx; lastRaw[i] = raw
            out[i] = fx
        }
        return out
    }
}
