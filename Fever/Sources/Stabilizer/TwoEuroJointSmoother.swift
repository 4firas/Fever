import simd
import Foundation

/// PinoFBT's input-side landmark smoother: a vectorized **Two-Euro** filter applied
/// to the (24,3) world-space joint array BEFORE the IK solve (findings §4). It is a
/// One-Euro filter (Casiez et al.) plus the extra `dollar` exponent PinoFBT adds for
/// movement-jitter shaping, and it special-cases the legs (separate cutoffs).
///
/// Standing still → low speed → heavy smoothing (no jitter). Moving → light → low
/// lag. The numeric defaults are the documented [UNK] (Nuitka-baked in PinoFBT):
/// seeded sensibly here and exposed as live tunables (findings §4/§12).
public struct TwoEuroParams: Sendable {
    public var minCutoff: Double   // Hz
    public var beta: Double        // speed coefficient
    public var dCutoff: Double     // derivative cutoff Hz
    public var dollar: Double      // jitter exponent (Two-Euro feature; 1 == plain One-Euro)
    public init(minCutoff: Double = 1.0, beta: Double = 0.30, dCutoff: Double = 1.0, dollar: Double = 1.0) {
        self.minCutoff = minCutoff; self.beta = beta; self.dCutoff = dCutoff; self.dollar = dollar
    }
}

public final class TwoEuroJointSmoother {
    /// SMPL leg joints, smoothed with their own params (footwork is faster/jumpier).
    public static let legIndices: Set<Int> = [
        SMPLJoint.leftHip.rawValue, SMPLJoint.rightHip.rawValue,
        SMPLJoint.leftKnee.rawValue, SMPLJoint.rightKnee.rawValue,
        SMPLJoint.leftAnkle.rawValue, SMPLJoint.rightAnkle.rawValue,
        SMPLJoint.leftFoot.rawValue, SMPLJoint.rightFoot.rawValue,
    ]

    private var body: TwoEuroParams
    private var legs: TwoEuroParams
    private var x = [SIMD3<Float>](repeating: .zero, count: SMPLJoint.count)   // last filtered value
    private var dx = [SIMD3<Float>](repeating: .zero, count: SMPLJoint.count)  // last filtered derivative
    private var lastRaw = [SIMD3<Float>](repeating: .zero, count: SMPLJoint.count)
    private var lastT: Double = -1

    public init(body: TwoEuroParams = TwoEuroParams(minCutoff: 2.0, beta: 0.4),
                legs: TwoEuroParams = TwoEuroParams(minCutoff: 2.5, beta: 0.6)) {
        self.body = body; self.legs = legs
    }

    public func setParams(body: TwoEuroParams, legs: TwoEuroParams) { self.body = body; self.legs = legs }

    /// Reset filter state (call when tracking drops, so a re-acquire snaps cleanly).
    public func reset() { lastT = -1 }

    private static func alpha(cutoff: Double, dt: Double) -> Float {
        let tau = 1.0 / (2.0 * Double.pi * max(cutoff, 1e-6))
        return Float(1.0 / (1.0 + tau / max(dt, 1e-6)))
    }

    /// Smooth one frame of world-space joints. `timestamp` in seconds.
    public func smooth(_ world: [SIMD3<Float>], timestamp: Double) -> [SIMD3<Float>] {
        guard world.count == SMPLJoint.count else { return world }
        if lastT < 0 {                               // first frame — seed, no filtering
            x = world; lastRaw = world; lastT = timestamp
            dx = [SIMD3<Float>](repeating: .zero, count: SMPLJoint.count)
            return world
        }
        let dt = max(timestamp - lastT, 1e-4); lastT = timestamp
        let rate = 1.0 / dt
        var out = world
        for i in 0..<SMPLJoint.count {
            let p = Self.legIndices.contains(i) ? legs : body
            let raw = world[i]
            // filtered derivative
            let edRate = Self.alpha(cutoff: p.dCutoff, dt: dt)
            let deriv = (raw - lastRaw[i]) * Float(rate)
            let edx = dx[i] + (deriv - dx[i]) * edRate
            dx[i] = edx
            // adaptive cutoff with the Two-Euro 'dollar' exponent on speed
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
