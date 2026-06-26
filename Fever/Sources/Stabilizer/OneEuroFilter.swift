import Foundation

/// Generic scalar One-Euro filter — adaptive low-pass for jittery landmark streams.
/// Reference: Casiez et al. 2012. Lower `minCutoff` = smoother, higher `beta`
/// = more responsive to fast motion.
///
/// NOT the live smoother. The shipping pipeline smooths with
/// `TwoEuroJointSmoother`, which reproduces PinoFBT 2.0's EXACT constants
/// (speed exponent 2, beta 400). This struct uses the textbook formulation
/// (exponent 1) and is kept only as a tested reference — do NOT wire it into the
/// tracking path expecting PinoFBT parity; its output differs.
public struct OneEuroFilter: Sendable {
    public var minCutoff: Float
    public var beta: Float
    public let dCutoff: Float = 1.0

    private var xPrev: Float?
    private var dxPrev: Float = 0
    private var tPrev: TimeInterval?

    public init(minCutoff: Float = 1.0, beta: Float = 0.007) {
        self.minCutoff = minCutoff
        self.beta = beta
    }

    public mutating func filter(_ x: Float, at time: TimeInterval) -> Float {
        guard let tPrev, let xPrev else {
            self.xPrev = x
            self.tPrev = time
            return x
        }
        let dt = max(Float(time - tPrev), 1e-6)
        let dx = (x - xPrev) / dt
        let alphaD = smoothingFactor(cutoff: dCutoff, dt: dt)
        let dxHat = lowPass(dx, prev: dxPrev, alpha: alphaD)

        let cutoff = minCutoff + beta * abs(dxHat)
        let alpha = smoothingFactor(cutoff: cutoff, dt: dt)
        let xHat = lowPass(x, prev: xPrev, alpha: alpha)

        self.xPrev = xHat
        self.dxPrev = dxHat
        self.tPrev = time
        return xHat
    }

    private func smoothingFactor(cutoff: Float, dt: Float) -> Float {
        let r = 2 * Float.pi * cutoff * dt
        return r / (r + 1)
    }

    private func lowPass(_ value: Float, prev: Float, alpha: Float) -> Float {
        alpha * value + (1 - alpha) * prev
    }

    public mutating func reset() {
        xPrev = nil; dxPrev = 0; tPrev = nil
    }
}
