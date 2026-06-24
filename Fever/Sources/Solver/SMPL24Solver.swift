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

        // calc_root/calc_chest run on the RAW model joints (camera frame: +Y down,
        // +Z away), with forward = -Z (toward camera) and up = -Y. This is PinoFBT's
        // actual setup, recovered by searching right-handed transforms × axis signs
        // against a labelled video fixture: it's the ONLY one that gives all of
        // bow→pitch+, lean→pitch-, turnL→yaw-, turnR→yaw+, sbR→roll- with a stable
        // near-identity rest. Crucially it's a RIGHT-HANDED (identity) transform, so
        // the output is a valid rotation — the earlier left-handed flip-Y + pitch-negate
        // produced MIRRORED rotations that broke combined poses (side-bend "impossible
        // anatomy", facing-away snaps). world = (-jx,-jy,-jz), so raw = -world.
        let calcWorld = world.map { -$0 }
        let fwd = SIMD3<Float>(0, 0, -1)   // a1: forward toward camera
        let upRef = SIMD3<Float>(0, -1, 0) // a2: up (raw frame is +Y-down)

        // Forward normal of the hip(1)/chest(4) frame. When the subject faces away it
        // goes antiparallel to `fwd` — the shortest-arc singularity in calc_*Rotation
        // (axis undefined → wild spin). Detect it geometrically so we can freeze
        // through it (proven by simulation: calc_root explodes only within ~±15° of
        // facing-away; NLF's joints are fine — it's the frame math, not the model).
        func spineForward(_ slot: Int) -> SIMD3<Float> {
            if slot == 4 {
                let lc = calcWorld[13], rc = calcWorld[14]
                return simd_cross(lc - rc, (lc + rc) * 0.5 - calcWorld[9])
            }
            return simd_cross(calcWorld[6] - calcWorld[3], calcWorld[2] - calcWorld[1])
        }

        var positions: [Int: SIMD3<Float>] = [:]
        var raws: [Int: simd_quatf] = [:]
        for slot in TrackerMapA.slots {
            positions[slot.index] = world[slot.joint.rawValue]
            var q: simd_quatf
            switch slot.index {
            case 1:  q = calcRootRotation(calcWorld, a1: fwd, a2: upRef)   // hip
            case 4:  q = calcChestRotation(calcWorld, a1: fwd, a2: upRef)  // chest
            default:
                guard let b = Self.bones[slot.index] else { continue }
                q = frameFromTwoAxes(primary: j(b.pB) - j(b.pA),
                                     secondary: j(b.sB) - j(b.sA),
                                     holdLast: holds[slot.index] ?? identity)
            }
            // Spine stabilization (hip + chest): freeze the last-good orientation
            // through the facing-away singularity (forward ≈ antiparallel to the
            // reference, where calc_*Rotation spins), otherwise rate-limit to absorb
            // any stray spike. This is what kills the "hips break when I face away and
            // raise a leg/arm" — the band is held instead of allowed to spin.
            if slot.index == 1 || slot.index == 4 {
                let p = spineForward(slot.index)
                let n = simd_length(p)
                let singular = n < 1e-6 || simd_dot(p / n, fwd) < -0.95
                if singular, let last = holds[slot.index] {
                    q = last
                } else {
                    q = rateLimited(q, previous: holds[slot.index], maxDegrees: 25)
                }
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
        // No euler hacks: the right-handed raw-frame setup above gives correct signs
        // directly (bow→pitch+, lean→pitch-, turnL→yaw-, turnR→yaw+, sbR→roll-), so the
        // hip and chest are now EXACT PinoFBT calc_root/calc_chest with valid rotations.

        // Hip tracker sits at the SMPL root (≈ groin); raise its HEIGHT toward the
        // lower spine (waist) but keep full pelvis X/Z so lateral hip sway isn't damped.
        var hipPos = j(.pelvis)
        hipPos.y += (j(.spine1).y - j(.pelvis).y) * 0.5
        positions[1] = hipPos

        // Chest tracker is at spine3 (low — "under the rib-cage"); raise it toward the
        // neck so the working point lands on the sternum where VRChat's chest bone is.
        positions[4] = j(.spine3) + (j(.neck) - j(.spine3)) * 0.4

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
