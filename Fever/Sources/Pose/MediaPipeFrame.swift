import Foundation
import simd

/// One-time latches for the horizontal origin and floor plane, so OSC values stay
/// zero-centred (like PinoFBT) and feet seat at Y≈0. Reset on Recenter. Lifted from
/// the deleted MonocularDepthLift; semantics: the first call after `reset()` freezes
/// the value, every later call returns that frozen latch.
public final class FloorOriginLatch: @unchecked Sendable {
    private let lock = NSLock()
    private var originXZ: SIMD2<Float>?
    private var floor: Float?

    public init() {}

    public func latchOriginXZ(_ xz: SIMD2<Float>) -> SIMD2<Float> {
        lock.withLock { if let o = originXZ { return o }; originXZ = xz; return xz }
    }
    public func latchFloor(_ y: Float) -> Float {
        lock.withLock { if let f = floor { return f }; floor = y; return y }
    }
    public func reset() { lock.withLock { originXZ = nil; floor = nil } }
}

/// Converts a MediaPipe sidecar reply (world landmarks, hip-origin, y-down) into the
/// solver-frame `PoseResult` (+X right, +Y up, +Z toward camera, hip-relative metres).
public enum MediaPipeFrame {
    /// Sign applied to MediaPipe world Z to map it into the solver frame.
    /// Confirmed LIVE in VRChat (2026-06-20): with -1, raising a leg forward read
    /// as backward (depth inverted), so the correct sign is +1 — forward/back now
    /// matches real motion.
    public static let defaultZSign: Float = 1

    /// - Parameter zSign: sign applied to world Z (the backend passes its configured value).
    /// - Parameter level: gravity-leveling rotation (from `BodyStabilizer`/`LevelEstimator`)
    ///   applied to every landmark right after the axis-fix and BEFORE the origin/floor
    ///   latch, so floor-anchoring and the foot solver's `worldUp` are measured along true
    ///   gravity even under a tilted camera. Identity by default — no behavior change.
    public static func toSolverFrame(_ reply: SidecarReply, latch: FloorOriginLatch,
                                     zSign: Float = defaultZSign,
                                     level: simd_quatf = LevelEstimator.identity) -> PoseResult? {
        guard reply.found, reply.world.count == 33 else { return nil }
        let v = reply.visibility
        func present(_ l: BlazePose.Landmark) -> Bool { v[l.rawValue] > 0.5 }
        let haveShoulders = present(.leftShoulder) && present(.rightShoulder)
        let haveHip = present(.leftHip) || present(.rightHip)
        guard haveShoulders, haveHip else { return nil }

        // Axis fix: x stays, y negated (down->up), z * zSign. World is already hip-origin,
        // so `level` rotates the body about ~the hip into a true-vertical world frame
        // (identity → exact passthrough). This MUST precede the origin/floor latch below.
        var lm = [NormalizedLandmark](repeating: NormalizedLandmark(position: .zero, visibility: 0, presence: 0),
                                      count: 33)
        for i in 0..<33 {
            let w = reply.world[i]
            let p = level.act(SIMD3(w.x, -w.y, w.z * zSign))
            lm[i] = NormalizedLandmark(position: p,
                                       visibility: v[i], presence: reply.presence[i])
        }

        // Latch the horizontal origin from the hip midpoint so absolute XZ centres ~0.
        let hip = (lm[.leftHip].position + lm[.rightHip].position) * 0.5
        let origin = latch.latchOriginXZ(SIMD2(hip.x, hip.z))
        for i in 0..<33 { lm[i].position.x -= origin.x; lm[i].position.z -= origin.y }

        // Floor-anchor: shift the whole skeleton so the lowest foot sits at a latched Y0.
        let footSlots: [BlazePose.Landmark] = [.leftAnkle, .rightAnkle, .leftHeel, .rightHeel,
                                               .leftFootIndex, .rightFootIndex]
        var lowest: Float? = nil
        for s in footSlots where present(s) {
            lowest = lowest.map { Swift.min($0, lm[s].position.y) } ?? lm[s].position.y
        }
        if let lf = lowest {
            let floor = latch.latchFloor(lf)
            // Shift every landmark by the same floor (like the XZ subtraction above);
            // gating this on presence left low-presence joints ~1m out of place.
            for i in 0..<33 { lm[i].position.y -= floor }
        }

        let img = reply.image.count == 33
            ? reply.image
            : [SIMD2<Float>](repeating: SIMD2(.nan, .nan), count: 33)
        return PoseResult(landmarks: lm, timestamp: 0, imagePoints: img)
    }
}
