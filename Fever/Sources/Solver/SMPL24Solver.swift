import simd

/// One solved frame ready for the OSC assembler: per-tracker world positions and
/// per-tracker ZXY-euler rotations, plus the head anchor position.
public struct SolvedFrame: Sendable {
    public var slotPositions: [Int: SIMD3<Float>]   // tracker index (1...8) → world meters
    public var slotEulers: [Int: SIMD3<Float>]      // tracker index → ZXY euler degrees
    public var headPosition: SIMD3<Float>
    public var tracked: Bool

    /// Convenience for the debug tap.
    public var hipEulerZXY: SIMD3<Float> { slotEulers[1] ?? .zero }
}

/// Faithful PinoFBT IK: every tracker's rotation is built from a two-vector
/// orthonormal frame over the landmark constellation (`fast_kinematics.get_rotation`
/// / `calc_root/chest/arm/knee/ankle_rotation`), NOT a direct quaternion output.
/// Desktop PinoFBT 2.0 solves the full limb set (chest, feet, knees, elbows) — not
/// just the hip — so we do too. Operates on WORLD-space joints (quaternions already
/// in VRChat space). Per-slot hold-last through degenerate frames.
public final class SMPL24Solver {

    /// A tracker's rotation frame: the bone's main axis (pA→pB) and a lateral hint
    /// (sA→sB) that disambiguates the remaining two axes.
    private struct Bone { let pA, pB, sA, sB: SMPLJoint }
    private static let bones: [Int: Bone] = [
        1: Bone(pA: .pelvis,     pB: .spine1,    sA: .leftHip,      sB: .rightHip),       // hip (root)
        4: Bone(pA: .spine1,     pB: .neck,      sA: .leftShoulder, sB: .rightShoulder),  // chest (long, stable axis)
        5: Bone(pA: .leftKnee,   pB: .leftAnkle, sA: .leftHip,      sB: .rightHip),       // L knee (shin)
        6: Bone(pA: .rightKnee,  pB: .rightAnkle,sA: .leftHip,      sB: .rightHip),       // R knee
        2: Bone(pA: .leftAnkle,  pB: .leftFoot,  sA: .leftKnee,     sB: .leftAnkle),      // L foot (ankle→toe)
        3: Bone(pA: .rightAnkle, pB: .rightFoot, sA: .rightKnee,    sB: .rightAnkle),     // R foot
        7: Bone(pA: .leftElbow,  pB: .leftWrist, sA: .leftShoulder, sB: .leftElbow),      // L elbow (forearm)
        8: Bone(pA: .rightElbow, pB: .rightWrist,sA: .rightShoulder,sB: .rightElbow),     // R elbow
    ]

    private var holds: [Int: simd_quatf] = [:]
    private var restQuats: [Int: simd_quatf] = [:]   // per-slot rest orientation (rest-relative base)
    private var captureRest = true                    // latch the next tracked frame as rest
    private var headAnchor: HeadAnchorSource

    public init(headAnchor: HeadAnchorSource = .head15) { self.headAnchor = headAnchor }
    public func setHeadAnchor(_ a: HeadAnchorSource) { headAnchor = a }
    /// Re-latch the rest pose on the next tracked frame (Recenter / Start).
    public func requestRestCapture() { captureRest = true }
    public func reset() { holds.removeAll() }   // keep restQuats across momentary drops

    public func solve(world: [SIMD3<Float>], tracked: Bool) -> SolvedFrame {
        func j(_ joint: SMPLJoint) -> SIMD3<Float> { world[joint.rawValue] }
        let identity = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)

        var positions: [Int: SIMD3<Float>] = [:]
        var raws: [Int: simd_quatf] = [:]
        for slot in TrackerMapA.slots {
            positions[slot.index] = world[slot.joint.rawValue]
            guard let b = Self.bones[slot.index] else { continue }
            let primary = j(b.pB) - j(b.pA)
            let secondary = j(b.sB) - j(b.sA)
            let q = frameFromTwoAxes(primary: primary, secondary: secondary,
                                     holdLast: holds[slot.index] ?? identity)
            if tracked { holds[slot.index] = q }
            raws[slot.index] = q
        }

        // Latch rest on the first tracked frame after a request, then emit each
        // tracker's rotation RELATIVE to rest (identity at rest, clean deltas under
        // motion) — kills the rest offsets and the baseline overshoot.
        if captureRest && tracked { restQuats = raws; captureRest = false }
        var eulers: [Int: SIMD3<Float>] = [:]
        for (slot, qraw) in raws {
            let qOut = restQuats[slot].map { qraw * $0.inverse } ?? qraw
            eulers[slot] = quaternionToEulerZXYDegrees(qOut)
        }

        let head: SIMD3<Float>
        switch headAnchor {
        case .head15:           head = j(.head)
        case .neck12:           head = j(.neck)
        case .headNeckMidpoint: head = (j(.head) + j(.neck)) * 0.5
        }

        return SolvedFrame(slotPositions: positions, slotEulers: eulers,
                           headPosition: head, tracked: tracked)
    }
}
