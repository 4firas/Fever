import simd

/// One solved frame ready for the OSC assembler: per-tracker world positions, the
/// hip (root) rotation as ZXY euler degrees, and the head anchor position.
public struct SolvedFrame: Sendable {
    public var slotPositions: [Int: SIMD3<Float>]   // tracker index (1...8) → world meters
    public var hipEulerZXY: SIMD3<Float>            // degrees, ZXY (VRChat) — hip only
    public var headPosition: SIMD3<Float>           // world meters
    public var tracked: Bool
}

/// Faithful PinoFBT IK: rotations come from two-vector orthonormal frames over the
/// landmark constellation (`fast_kinematics.get_rotation` / `calc_root_rotation`),
/// NOT from a direct quaternion model output. The captured build sends rotation for
/// the HIP (root) only — every other slot is position-only — so that's all we solve
/// (findings §5/§7). Operates on WORLD-space joints, so the quaternion is already in
/// VRChat space. Holds the last good orientation through degenerate frames.
public final class SMPL24Solver {
    private var hipHold = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
    private var headAnchor: HeadAnchorSource

    public init(headAnchor: HeadAnchorSource = .head15) { self.headAnchor = headAnchor }
    public func setHeadAnchor(_ a: HeadAnchorSource) { headAnchor = a }
    public func reset() { hipHold = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1) }

    public func solve(world: [SIMD3<Float>], tracked: Bool) -> SolvedFrame {
        func j(_ joint: SMPLJoint) -> SIMD3<Float> { world[joint.rawValue] }

        // Root frame: spine direction (pelvis→spine1) is the primary/up axis; the
        // hip line (L→R) disambiguates the lateral axis. Two in-body vectors → no
        // world-up gauge, no fabricated roll (Math.frameFromTwoAxes).
        let spineUp  = j(.spine1) - j(.pelvis)
        let hipLine  = j(.rightHip) - j(.leftHip)
        let hipQuat  = frameFromTwoAxes(primary: spineUp, secondary: hipLine, holdLast: hipHold)
        if tracked { hipHold = hipQuat }
        let hipEuler = quaternionToEulerZXYDegrees(hipQuat)

        var positions: [Int: SIMD3<Float>] = [:]
        for slot in TrackerMapA.slots { positions[slot.index] = world[slot.joint.rawValue] }

        let head: SIMD3<Float>
        switch headAnchor {
        case .head15:           head = j(.head)
        case .neck12:           head = j(.neck)
        case .headNeckMidpoint: head = (j(.head) + j(.neck)) * 0.5
        }

        return SolvedFrame(slotPositions: positions, hipEulerZXY: hipEuler,
                           headPosition: head, tracked: tracked)
    }
}
