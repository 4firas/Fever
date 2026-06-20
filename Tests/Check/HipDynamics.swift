import Foundation
import simd
import FeverCore

/// HIP DYNAMICS — guards for the PinoFBT-style POSITION-space hip sway.
///
/// The pipeline is POSITION-ONLY (no /rotation, no head point). True pelvic
/// weight-shift is only a few cm — below the monocular noise/smoothing floor —
/// so `JointSolver.applyHipAdjustments` AMPLIFIES the hip's horizontal deviation
/// from the STANCE CENTER (the floor-projected ankle midpoint) by a tunable
/// gain (`hipExaggerateCoefficient` = lateral X, `hipTwistCoefficient` = fwd/back
/// Z). The amplification is on the SMOOTHED hip (landmarks are One-Euro filtered
/// upstream) and is CLAMPED so the hip can never leave the body.
///
/// These tests drive the REAL `JointSolver.solve()` in the solver frame, where
/// the hip-vs-ankle geometry is exactly controllable, plus an end-to-end
/// assembled-tracker check through `CoordinateMapper` + `TrackerAssembler`.
///
///   1. SWAY AMPLIFICATION
///      • centered stance → hip X deviation ≈ 0 (no false sway at rest)
///      • a small lateral weight-shift (hip over one foot) → assembled hip X
///        deviation from the stance baseline is AMPLIFIED by ~the configured
///        gain (output sway > input sway, ≈ gain×)
///      • a LARGE input is CLAMPED (hip never sent outside a sane range)
///
///   2. STABILITY
///      • a static centered pose over MANY frames does not grow / oscillate the
///        hip sway (no jitter amplification — deterministic, deadband holds)
///      • the hip still TRANSLATES with real whole-body translation (the prior
///        hip-translation fix is not broken by the sway exaggeration)
///      • vertical coherence (foot < knee < hip < chest) still holds
enum HipDynamics {

    // MARK: - Solver-frame synthetic pose

    /// Build a 33-landmark `PoseResult` directly in the SOLVER frame (the same
    /// right-handed, +Y-up, hip-root-relative METER frame `JointSolver.solve`
    /// consumes). This lets the test place the hip midpoint and the ankle
    /// midpoint INDEPENDENTLY — exactly the weight-shift / contrapposto geometry
    /// (hip translates over a planted foot) that the sway exaggeration targets.
    ///
    /// - `hipCenterX`:  X of the hip/upper-body centerline (the swaying part).
    /// - `ankleCenterX`: X of the ankle midpoint = the STANCE CENTER reference.
    /// - `stanceHalf`:  half the distance between the two ankles.
    /// - `hipZ`:        forward/back offset of the hip relative to the ankles.
    ///
    /// Y is +up: feet low (y≈0.0), hips mid (y≈0.95), chest/shoulders high. A
    /// gentle knee bend keeps the leg out of plane so nothing degenerates.
    static func solverPose(hipCenterX: Float,
                           ankleCenterX: Float,
                           stanceHalf: Float = 0.10,
                           hipZ: Float = 0.0) -> PoseResult {
        var lm = [NormalizedLandmark](
            repeating: NormalizedLandmark(position: .zero, visibility: 0.9), count: 33)
        func set(_ l: BlazePose.Landmark, _ x: Float, _ y: Float, _ z: Float = 0) {
            lm[l.rawValue] = NormalizedLandmark(position: SIMD3<Float>(x, y, z),
                                                visibility: 0.9)
        }
        let hw: Float = 0.10   // half hip width
        let sw: Float = 0.18   // half shoulder width

        // Head / face (high, follows the swaying upper body).
        set(.nose,      hipCenterX,        1.65, 0.05)
        set(.leftEye,   hipCenterX - 0.03, 1.66); set(.rightEye, hipCenterX + 0.03, 1.66)
        set(.leftEar,   hipCenterX - 0.05, 1.63); set(.rightEar, hipCenterX + 0.05, 1.63)
        // Shoulders / elbows / wrists (arms at sides), centered on the upper body.
        set(.leftShoulder, hipCenterX - sw, 1.45); set(.rightShoulder, hipCenterX + sw, 1.45)
        set(.leftElbow,    hipCenterX - sw, 1.20); set(.rightElbow,    hipCenterX + sw, 1.20)
        set(.leftWrist,    hipCenterX - sw, 0.98); set(.rightWrist,    hipCenterX + sw, 0.98)
        // Hips — the swaying segment, at hipCenterX, pushed forward by hipZ.
        set(.leftHip,  hipCenterX - hw, 0.95, hipZ); set(.rightHip, hipCenterX + hw, 0.95, hipZ)
        // Knees — slightly forward (bent) so the leg is not coplanar.
        set(.leftKnee,  ankleCenterX - stanceHalf, 0.50, 0.06)
        set(.rightKnee, ankleCenterX + stanceHalf, 0.50, 0.06)
        // Ankles — PLANTED at the stance center (the weight-bearing base).
        set(.leftAnkle,  ankleCenterX - stanceHalf, 0.08)
        set(.rightAnkle, ankleCenterX + stanceHalf, 0.08)
        // Heels / toes around the ankles (feet on the floor).
        set(.leftHeel,  ankleCenterX - stanceHalf, 0.06); set(.rightHeel, ankleCenterX + stanceHalf, 0.06)
        set(.leftFootIndex,  ankleCenterX - stanceHalf, 0.02, 0.05)
        set(.rightFootIndex, ankleCenterX + stanceHalf, 0.02, 0.05)
        return PoseResult(landmarks: lm, timestamp: 0)
    }

    static func joint(_ joints: [VRJoint], _ type: JointType) -> VRJoint? {
        joints.first { $0.type == type }
    }

    // MARK: - Run

    static func run(_ t: TestRunner) {
        runSwayAmplification(t)
        runStability(t)
    }

    // MARK: 1. SWAY AMPLIFICATION

    static func runSwayAmplification(_ t: TestRunner) {
        // Pin the gains so the assertions are exact (don't depend on persisted
        // UserDefaults). Default tasteful values: sway 2.0, lean 1.4.
        let cfg = TrackingConfig()
        cfg.hipExaggerateCoefficient = 2.0   // lateral sway gain
        cfg.hipTwistCoefficient = 1.4        // fwd/back lean gain
        cfg.hipLength = 0.0                  // no spine offset (isolate sway)
        cfg.jointSize = 1.0
        cfg.sendRotation = false            // position-only path (the real one)
        let solver = JointSolver(settings: cfg)

        let gainX = Float(cfg.hipExaggerateCoefficient)
        // The solver-frame stance center is the ankle midpoint; we place the
        // ankles symmetric about X=0 so stance center X = 0 and the hip's raw
        // deviation equals its centerline X. Deadband is 1.5 cm; pick an input
        // sway WELL above it so the deadband only trims a constant 1.5 cm.
        let deadband: Float = 0.015

        // ── 1a. Centered stance → ~zero sway (no false sway at rest) ─────────
        t.test("SWAY: centered stance produces ~zero hip sway") {
            let pose = solverPose(hipCenterX: 0.0, ankleCenterX: 0.0)
            let joints = solver.solve(pose)
            guard let hip = joint(joints, .hip) else { t.check(false, "no hip"); return }
            // Hip X must be at/near the stance center (0). Deadband swallows
            // sub-1.5 cm noise, so at dead center the amplified add is 0.
            t.check(abs(hip.position.x) < 1e-3,
                    "centered hip must not sway: x=\(hip.position.x)")
        }

        // ── 1b. A small weight-shift is AMPLIFIED by ~the gain ──────────────
        t.test("SWAY: lateral weight-shift is amplified by ~the configured gain") {
            // Input: the hip shifts 0.06 m to the right OVER the (planted) feet.
            let inputSway: Float = 0.06
            let pose = solverPose(hipCenterX: inputSway, ankleCenterX: 0.0)
            let joints = solver.solve(pose)
            guard let hip = joint(joints, .hip) else { t.check(false, "no hip"); return }
            let outputSway = hip.position.x          // stance center is X=0

            // The exaggeration ADDS (gain-1)*(dev-deadband) to the true hip dev.
            // outputSway = inputSway + (gainX-1)*(inputSway - deadband).
            let expected = inputSway + (gainX - 1) * (inputSway - deadband)
            print(String(format: "  [sway] input=%+.4f  output=%+.4f  expected≈%+.4f  (gain=%.2f, ratio=%.2f×)",
                         inputSway, outputSway, expected, gainX, outputSway / inputSway))

            // Output sway must EXCEED the input sway (amplified, gain > 1).
            t.check(outputSway > inputSway + 1e-4,
                    "output sway must exceed input (amplified): out=\(outputSway) in=\(inputSway)")
            // And match the configured-gain prediction closely.
            t.close(outputSway, expected, tol: 1e-3,
                    "amplified sway must match gain prediction")
            // Same direction as the shift (right stays right).
            t.check(outputSway > 0, "amplified sway keeps the shift direction")
            // Ratio is close to the gain (deadband makes it slightly under gain).
            let ratio = outputSway / inputSway
            t.check(ratio > 1.5 && ratio <= gainX + 1e-3,
                    "amplification ratio ~gain (deadband-trimmed): \(ratio)")
        }

        // ── 1c. Sub-deadband micro-deviation is NOT amplified ───────────────
        t.test("SWAY: sub-deadband micro-sway is suppressed (no jitter amplification)") {
            let tiny: Float = 0.008   // below the 1.5 cm deadband
            let pose = solverPose(hipCenterX: tiny, ankleCenterX: 0.0)
            let joints = solver.solve(pose)
            guard let hip = joint(joints, .hip) else { t.check(false, "no hip"); return }
            // Deadband zeroes the EXTRA amplification: output == the true (un-
            // amplified) deviation, i.e. NOT multiplied by the gain.
            t.close(hip.position.x, tiny, tol: 1e-4,
                    "sub-deadband sway must pass through un-amplified: \(hip.position.x)")
            t.check(hip.position.x < tiny * gainX,
                    "micro-sway must NOT be amplified by the gain")
        }

        // ── 1d. A LARGE input is CLAMPED to a sane range ────────────────────
        t.test("SWAY: a large weight-shift is clamped (hip stays in a sane range)") {
            // Slam the hip far to the side; the AMPLIFIED add is clamped to
            // max(0.12, halfStance+0.06). With a 0.10 half-stance → bound 0.16.
            let stanceHalf: Float = 0.10
            let bound = max(Float(0.12), stanceHalf + 0.06)   // 0.16
            let hugeInput: Float = 0.40
            let pose = solverPose(hipCenterX: hugeInput, ankleCenterX: 0.0,
                                  stanceHalf: stanceHalf)
            let joints = solver.solve(pose)
            guard let hip = joint(joints, .hip) else { t.check(false, "no hip"); return }
            let added = hip.position.x - hugeInput   // the exaggeration component
            print(String(format: "  [sway-clamp] input=%+.3f  output=%+.3f  added=%+.3f  bound=%.3f",
                         hugeInput, hip.position.x, added, bound))
            // The ADDED exaggeration is clamped at the bound (not gain×input).
            t.close(added, bound, tol: 1e-3,
                    "amplified add must be clamped to the bound: added=\(added)")
            t.check(added < (gainX - 1) * hugeInput,
                    "clamp must cap the add well below the un-clamped gain product")
            // Total hip X stays finite and within input + bound (never teleports).
            t.check(hip.position.x.isFinite, "clamped hip X finite")
            t.check(hip.position.x <= hugeInput + bound + 1e-4,
                    "hip must not exceed input + clamp bound: \(hip.position.x)")
        }

        // ── 1e. Forward/back lean is amplified on Z by the lean gain ────────
        t.test("SWAY: forward lean is amplified on Z by the lean gain") {
            let gainZ = Float(cfg.hipTwistCoefficient)
            let leanZ: Float = 0.05
            let pose = solverPose(hipCenterX: 0.0, ankleCenterX: 0.0, hipZ: leanZ)
            let joints = solver.solve(pose)
            guard let hip = joint(joints, .hip) else { t.check(false, "no hip"); return }
            // stance center Z = 0 (ankles at z=0); hip dev = leanZ.
            let expected = leanZ + (gainZ - 1) * (leanZ - deadband)
            t.check(hip.position.z > leanZ + 1e-4,
                    "forward lean must be amplified on Z: z=\(hip.position.z)")
            t.close(hip.position.z, expected, tol: 1e-3,
                    "lean amplification must match the lean gain")
        }
    }

    // MARK: 2. STABILITY

    static func runStability(_ t: TestRunner) {
        let cfg = TrackingConfig()
        cfg.hipExaggerateCoefficient = 2.0
        cfg.hipTwistCoefficient = 1.4
        cfg.hipLength = 0.0
        cfg.jointSize = 1.0
        cfg.mirrorTracking = false
        let solver = JointSolver(settings: cfg)

        // ── 2a. Static centered pose over MANY frames → no growth/oscillation ─
        t.test("STABILITY: static centered pose does not grow or oscillate hip sway") {
            let pose = solverPose(hipCenterX: 0.0, ankleCenterX: 0.0)
            var xs: [Float] = []
            for _ in 0..<300 {
                let joints = solver.solve(pose)
                guard let hip = joint(joints, .hip) else { t.check(false, "no hip"); return }
                xs.append(hip.position.x)
            }
            let lo = xs.min() ?? 0, hi = xs.max() ?? 0
            // Deterministic + deadbanded: the hip X must be effectively constant
            // (no integrator / feedback that could grow or ring).
            t.check(hi - lo < 1e-5,
                    "static hip sway must not grow/oscillate: spread \(hi - lo)")
            t.check(abs(hi) < 1e-3, "static hip must stay centered: \(hi)")
        }

        // ── 2b. A held off-center weight-shift is stable (no runaway) ───────
        t.test("STABILITY: a held weight-shift settles to a constant (no runaway)") {
            let pose = solverPose(hipCenterX: 0.06, ankleCenterX: 0.0)
            var xs: [Float] = []
            for _ in 0..<300 {
                let joints = solver.solve(pose)
                guard let hip = joint(joints, .hip) else { t.check(false, "no hip"); return }
                xs.append(hip.position.x)
            }
            let lo = xs.min() ?? 0, hi = xs.max() ?? 0
            t.check(hi - lo < 1e-5,
                    "held weight-shift must be a constant (no runaway): spread \(hi - lo)")
            // And it is amplified vs the 0.06 input (still > input, < clamp).
            t.check((xs.last ?? 0) > 0.06, "held shift stays amplified")
        }

        // ── 2c. Hip still TRANSLATES with whole-body translation ────────────
        // (Both hip AND ankles move together — the stance center moves with the
        // hip, so the sway deviation is ~0 and the hip tracks the body 1:1. This
        // proves the exaggeration did NOT break the prior hip-translation fix.)
        t.test("STABILITY: hip translates with whole-body shift (sway didn't pin it)") {
            let shift: Float = 0.15
            let a = solver.solve(solverPose(hipCenterX: 0.0,   ankleCenterX: 0.0))
            let b = solver.solve(solverPose(hipCenterX: shift, ankleCenterX: shift))
            guard let hipA = joint(a, .hip), let hipB = joint(b, .hip) else {
                t.check(false, "no hip"); return
            }
            let dx = hipB.position.x - hipA.position.x
            print(String(format: "  [hip-translate] hipX a=%+.4f b=%+.4f  Δ=%+.4f  (body shift=%.2f)",
                         hipA.position.x, hipB.position.x, dx, shift))
            // The whole body moved by `shift`; the hip must move ~1:1 with it
            // (the deviation from stance center stayed ~0, so no extra sway add).
            t.close(dx, shift, tol: 0.02,
                    "hip must translate ~1:1 with the whole body (not pinned): Δ=\(dx)")
        }

        // ── 2d. Vertical coherence (foot < knee < hip < chest) holds ────────
        t.test("STABILITY: vertical coherence foot<knee<hip<chest holds with sway on") {
            // Use a real off-center weight-shift so the sway path is active.
            let joints = solver.solve(solverPose(hipCenterX: 0.06, ankleCenterX: 0.0))
            guard let footL = joint(joints, .leftFoot), let footR = joint(joints, .rightFoot),
                  let kneeL = joint(joints, .leftKnee), let kneeR = joint(joints, .rightKnee),
                  let hip = joint(joints, .hip), let chest = joint(joints, .chest) else {
                t.check(false, "missing joints for coherence"); return
            }
            let foot = min(footL.position.y, footR.position.y)
            let knee = min(kneeL.position.y, kneeR.position.y)
            t.check(foot < knee, "foot.y (\(foot)) < knee.y (\(knee))")
            t.check(knee < hip.position.y, "knee.y (\(knee)) < hip.y (\(hip.position.y))")
            t.check(hip.position.y < chest.position.y,
                    "hip.y (\(hip.position.y)) < chest.y (\(chest.position.y))")
            // Sway is HORIZONTAL only: the hip Y must be unchanged by the X sway.
            let centered = solver.solve(solverPose(hipCenterX: 0.0, ankleCenterX: 0.0))
            if let hipC = joint(centered, .hip) {
                t.close(hip.position.y, hipC.position.y, tol: 1e-4,
                        "sway must not move the hip vertically (no bounce)")
            }
        }

        // ── 2e. End-to-end assembled HIP tracker amplifies sway on the wire ─
        // Drives the FULL CoordinateMapper + TrackerAssembler path so the
        // amplification is verified on the ASSEMBLED tracker, not just the joint.
        t.test("SWAY: assembled hip TRACKER X amplifies the weight-shift end-to-end") {
            let mapper = CoordinateMapper(userHeightMeters: 1.74,
                                          referenceHeightMeters: 1.8,
                                          mirrorHorizontally: false)
            let assembler = TrackerAssembler(enabled: cfg.enabledJoints,
                                             slotMap: cfg.slotMap)
            func hipTrackerX(hipCenterX: Float) -> Float? {
                let joints = solver.solve(solverPose(hipCenterX: hipCenterX, ankleCenterX: 0.0))
                let (body, _) = assembler.assemble(joints, mapper: mapper)
                return body.first { $0.slot == "1" }?.position.x
            }
            guard let centered = hipTrackerX(hipCenterX: 0.0),
                  let shifted = hipTrackerX(hipCenterX: 0.06) else {
                t.check(false, "no assembled hip tracker"); return
            }
            let mScale: Float = 1.74 / 1.8
            let inputOnWire: Float = 0.06 * mScale
            let swayOnWire = shifted - centered
            print(String(format: "  [sway-wire] centered=%+.4f shifted=%+.4f  inputOnWire≈%+.4f  swayOnWire=%+.4f",
                         centered, shifted, inputOnWire, swayOnWire))
            t.check(abs(centered) < 1e-3, "centered assembled hip ~0 (no false sway): \(centered)")
            t.check(swayOnWire > inputOnWire + 1e-4,
                    "assembled hip tracker must amplify the shift on the wire: \(swayOnWire) vs \(inputOnWire)")
        }
    }
}
