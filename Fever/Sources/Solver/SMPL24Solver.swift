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
    private var headAnchor: HeadAnchorSource

    public init(headAnchor: HeadAnchorSource = .head15) { self.headAnchor = headAnchor }
    public func setHeadAnchor(_ a: HeadAnchorSource) { headAnchor = a }
    public func reset() { holds.removeAll() }

    public func solve(world: [SIMD3<Float>], tracked: Bool) -> SolvedFrame {
        func j(_ joint: SMPLJoint) -> SIMD3<Float> { world[joint.rawValue] }
        let identity = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)

        // calc_root expects PinoFBT's landmark frame. Derived from a video diff vs a
        // PinoFBT OSC capture: Fever's world = (-jx,-jy,-jz) (a left-handed reflection)
        // makes calc's forward = cross(spine,hipline) antiparallel to the +Z reference
        // → get_rotation's degenerate 180° case → the wild yaw jitter we saw. Flipping
        // exactly ONE of X/Y restores a stable forward; this maps to (-world.x, world.y,
        // -world.z). That nails yaw + roll signs vs PinoFBT; pitch comes out inverted
        // (handled below) — no signed axis map can flip pitch alone (calc_root is
        // nonlinear), so we negate the hip pitch euler.
        let calcWorld = world.map { SIMD3<Float>(-$0.x, $0.y, -$0.z) }

        var positions: [Int: SIMD3<Float>] = [:]
        var raws: [Int: simd_quatf] = [:]
        for slot in TrackerMapA.slots {
            positions[slot.index] = world[slot.joint.rawValue]
            let q: simd_quatf
            switch slot.index {
            case 1:  q = calcRootRotation(calcWorld)    // hip — exact PinoFBT calc_root
            case 4:  q = calcChestRotation(calcWorld)   // chest — exact PinoFBT calc_chest
            default:
                guard let b = Self.bones[slot.index] else { continue }
                q = frameFromTwoAxes(primary: j(b.pB) - j(b.pA),
                                     secondary: j(b.sB) - j(b.sA),
                                     holdLast: holds[slot.index] ?? identity)
            }
            if tracked { holds[slot.index] = q }
            raws[slot.index] = q
        }

        // ABSOLUTE orientations (like PinoFBT — no in-app recenter; VRChat's own
        // T-pose calibration handles each part's fixed rest offset).
        var eulers: [Int: SIMD3<Float>] = [:]
        for (slot, qraw) in raws {
            eulers[slot] = quaternionToEulerZXYDegrees(qraw)
        }
        // Fever's space inverts pitch (bow reads as −); negate it for the hip AND the
        // chest (both solved in the same calc space). Yaw/roll come out correct.
        if var he = eulers[1] { he.x = -he.x; eulers[1] = he }
        if var ce = eulers[4] { ce.x = -ce.x; eulers[4] = ce }

        // Amplify the hip's rotation RELATIVE to the chest = pelvic articulation
        // (tilt/twist of the hips vs the torso). A whole-body turn moves hip+chest
        // together so their difference ≈ 0 → turns stay 1:1 (no over-rotation); only
        // isolated pelvic pitch/yaw/roll is boosted — the "hips don't pitch/yaw"
        // complaint. The hip rotation IS live (log: yaw→55° on a turn) but the
        // pelvis-vs-torso part is too subtle to read without this gain.
        if let chest = eulers[4], var hip = eulers[1] {
            let hipArticGain: Float = 2.0
            hip = chest + (hip - chest) * hipArticGain
            eulers[1] = hip
        }

        // Hip tracker sits at the SMPL root (≈ groin); raise its HEIGHT toward the
        // lower spine (waist) but keep full pelvis X/Z so lateral hip sway isn't damped.
        var hipPos = j(.pelvis)
        hipPos.y += (j(.spine1).y - j(.pelvis).y) * 0.5
        positions[1] = hipPos

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
