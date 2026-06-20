import Foundation

/// One-Euro filter — adaptive low-pass for jittery landmark streams.
/// Reference: Casiez et al. 2012. Lower `minCutoff` = smoother, higher `beta`
/// = more responsive to fast motion.
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

/// Applies One-Euro to every coordinate of every landmark (33 × 3 = 99 filters).
public final class LandmarkStabilizer {
    private var fx: [OneEuroFilter]
    private var fy: [OneEuroFilter]
    private var fz: [OneEuroFilter]

    /// Reused output buffer (single-worker confinement). Avoids a fresh
    /// 33-element array allocation on every frame.
    private var scratch: [NormalizedLandmark]

    public init(count: Int = 33, minCutoff: Float, beta: Float) {
        fx = Array(repeating: OneEuroFilter(minCutoff: minCutoff, beta: beta), count: count)
        fy = Array(repeating: OneEuroFilter(minCutoff: minCutoff, beta: beta), count: count)
        fz = Array(repeating: OneEuroFilter(minCutoff: minCutoff, beta: beta), count: count)
        scratch = Array(repeating: NormalizedLandmark(position: .zero, visibility: 0, presence: 0),
                        count: count)
    }

    public func stabilize(_ result: PoseResult) -> PoseResult {
        let lms = result.landmarks
        let t = result.timestamp
        // Size the reused buffer to the input once (normally a no-op at 33).
        if scratch.count != lms.count {
            scratch = Array(repeating: NormalizedLandmark(position: .zero, visibility: 0, presence: 0),
                            count: lms.count)
        }
        for i in lms.indices {
            let lm = lms[i]
            var p = lm.position
            p.x = fx[i].filter(p.x, at: t)
            p.y = fy[i].filter(p.y, at: t)
            p.z = fz[i].filter(p.z, at: t)
            scratch[i] = NormalizedLandmark(position: p,
                                            visibility: lm.visibility,
                                            presence: lm.presence)
        }
        return PoseResult(landmarks: scratch, timestamp: t)
    }
}
