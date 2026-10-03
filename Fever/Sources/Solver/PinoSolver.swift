import simd

/// The PinoFBT 2.0 desktop-faithful solver: a 1:1 port of the per-frame chain that
/// the real `main_UI.exe` + `fast_kinematics.pyd` run, producing the exact OSC wire
/// (17-message bundle) the captured PinoFBT build emits.
///
///   model joints3D (24,3, camera +Y-down, meters)   [already OneEuro-filtered]
///     → preprocess_joints (TRUE; pelvis-centred, 180°-X, torso-scaled, fixed legs)
///     → calc_root / calc_chest(+residual) / arm / knee / ankle IK
///     → per-tracker rotation = euler('zxy')[[1,2,0]] of the quat
///       per-tracker position = preprocess-out joint × user_height_ratio (hip=origin)
///
/// Slot map (desktop, live-confirmed):
///   1=chest 2=hip 3=L_elbow 4=R_elbow 5=L_knee 6=R_knee 7=L_ankle 8=R_ankle
/// head = position only (`preO[15] × 0.895`).
///
/// `user_height_ratio = height_cm / 175.0`.

/// One solved frame: per-tracker positions and ZXY-euler rotations (slot index
/// 1…8) plus the head anchor position.
public struct SolvedFrame: Sendable {
    public var slotPositions: [Int: SIMD3<Float>]   // tracker index (1...8) → meters
    public var slotEulers: [Int: SIMD3<Float>]      // tracker index → ZXY euler degrees
    public var headPosition: SIMD3<Float>
    public var tracked: Bool
}

public final class PinoSolver {

    /// Reference height (cm). `user_height_ratio = height_cm / 175`.
    public static let referenceHeightCm: Float = 175.0
    /// Head-specific position scale (NOT the body ratio) — live-confirmed.
    public static let headScale: Float = 0.895
    /// Elbow rest bones (preprocess-space): the upper-arm DIRECTION reference for the
    /// FK (PinoFBT's are length ~0.157, which parks the tracker at chest height).
    public static let restElbowL = SIMD3<Float>(0.05015, -0.14918, 0.00757)
    public static let restElbowR = SIMD3<Float>(-0.04863, -0.14904, 0.00918)
    /// Canonical upper-arm length (preprocess space; capture median |shoulder→elbow|
    /// = 0.240). We extend the elbow FK to this real length so the tracker lands at
    /// the ACTUAL elbow, while keeping PinoFBT's stable rotation-driven DIRECTION.
    public static let upperArmLength: Float = 0.240

    private var heightRatio: Float

    /// - Parameter heightCm: the user's height in centimeters (default 174 → 0.9943,
    ///   the captured session). `user_height_ratio = heightCm / 175`.
    public init(heightCm: Float = 174.0) {
        self.heightRatio = heightCm / Self.referenceHeightCm
    }

    public func setHeightCm(_ cm: Float) { heightRatio = cm / Self.referenceHeightCm }
    public func reset() { /* native PinoFBT 2.0 solver is stateless */ }

    /// Solve one frame. `joints` = 24 OneEuro-filtered model joints (camera +Y down).
    public func solve(joints: [SIMD3<Float>], tracked: Bool) -> SolvedFrame {
        let O = PinoKinematics.preprocessJoints(joints)

        // ── IK ────────────────────────────────────────────────────────────────
        let rootQ = PinoKinematics.calcRootRotation(O)                 // hip
        let (chestQ, chestResidual) = PinoKinematics.calcChestRotation(O)

        // ARMS (slots 3/4): pure 1:1 PinoFBT 2.0 calcPairedArmRotations
        let arms = PinoKinematics.calcPairedArmRotations(
            chestResidual: chestResidual,
            rShoulder: O[17], lShoulder: O[16],
            rElbow: O[19], lElbow: O[18],
            rWrist: O[21], lWrist: O[20]
        )

        // KNEE: blocks [R,L]; out[0]=L lane (from R bones), out[1]=R lane (from L bones).
        let lKneeQ = PinoKinematics.kneeRotation(hip: O[2], knee: O[5], ankle: O[8], toe: O[11])  // R bones → L lane
        let rKneeQ = PinoKinematics.kneeRotation(hip: O[1], knee: O[4], ankle: O[7], toe: O[10])  // L bones → R lane

        // ANKLE: same lane swap.
        let lAnkleQ = PinoKinematics.ankleRotation(knee: O[5], ankle: O[8], toe: O[11])  // R bones → L lane
        let rAnkleQ = PinoKinematics.ankleRotation(knee: O[4], ankle: O[7], toe: O[10])  // L bones → R lane

        // ── Euler (wire rotation) ───────────────────────────────────────────────
        let eulers: [Int: SIMD3<Float>] = [
            1: PinoKinematics.eulerZXY121Degrees(chestQ),
            2: PinoKinematics.eulerZXY121Degrees(rootQ),
            3: PinoKinematics.eulerZXY121Degrees(arms.lElbow),
            4: PinoKinematics.eulerZXY121Degrees(arms.rElbow),
            5: PinoKinematics.eulerZXY121Degrees(lKneeQ),
            6: PinoKinematics.eulerZXY121Degrees(rKneeQ),
            7: PinoKinematics.eulerZXY121Degrees(lAnkleQ),
            8: PinoKinematics.eulerZXY121Degrees(rAnkleQ),
        ]

        // ── Positions (preprocess-out × height_ratio; hip = origin) ──────────────
        // Wire L/R index swaps: chest→O[9]; L_knee→O[5], R_knee→O[4];
        // L_ankle→O[8], R_ankle→O[7]. Elbows are FK-reconstructed from the arm quat.
        let r = heightRatio
        var positions: [Int: SIMD3<Float>] = [:]
        positions[1] = O[9]  * r                                       // chest = spine3
        positions[2] = .zero                                           // hip = origin
        positions[5] = O[5]  * r                                       // L_knee
        positions[6] = O[4]  * r                                       // R_knee
        positions[7] = O[8]  * r                                       // L_ankle
        positions[8] = O[7]  * r                                       // R_ankle

        // Elbows (slots 3/4): pure 1:1 PinoFBT 2.0 forward kinematics
        // Reconstruct from shoulder joint + rotated rest bone offset
        positions[3] = (O[16] + PinoKinematics.quatApply(arms.lElbow, Self.restElbowL)) * r
        positions[4] = (O[17] + PinoKinematics.quatApply(arms.rElbow, Self.restElbowR)) * r

        // Head: position only, preO[15] × 0.895 (head-specific scale).
        let head = O[15] * Self.headScale

        return SolvedFrame(slotPositions: positions, slotEulers: eulers,
                           headPosition: head, tracked: tracked)
    }
}
