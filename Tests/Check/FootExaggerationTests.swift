import Foundation
import simd
import FeverCore

/// Step / stride exaggeration (JointSolver.applyFootAdjustments + FootMotionState):
/// a SWINGING foot's displacement is amplified, a PLANTED foot is not (stays glued
/// to the floor), and vertical lift is up-only.
enum FootExaggerationTests {

    /// Minimal standing pose in the solver frame (+Y up); the left ankle is settable.
    static func pose(leftAnkle a: SIMD3<Float>) -> PoseResult {
        var lm = [NormalizedLandmark](repeating: NormalizedLandmark(position: .zero, visibility: 0),
                                      count: 33)
        func set(_ l: BlazePose.Landmark, _ x: Float, _ y: Float, _ z: Float = 0) {
            lm[l.rawValue] = NormalizedLandmark(position: SIMD3(x, y, z), visibility: 0.95)
        }
        set(.leftShoulder, -0.18, 0.55); set(.rightShoulder, 0.18, 0.55)
        set(.leftHip, -0.10, 0.0); set(.rightHip, 0.10, 0.0)
        set(.leftKnee, -0.10, -0.42, 0.05); set(.rightKnee, 0.10, -0.42, 0.05)
        set(.leftAnkle, a.x, a.y, a.z); set(.rightAnkle, 0.10, -0.86, 0)
        set(.leftHeel, a.x, a.y - 0.04, a.z - 0.04); set(.leftFootIndex, a.x, a.y - 0.06, a.z + 0.12)
        set(.rightHeel, 0.10, -0.90, -0.04); set(.rightFootIndex, 0.10, -0.92, 0.12)
        return PoseResult(landmarks: lm, timestamp: 0)
    }

    static func leftFoot(_ joints: [VRJoint]) -> VRJoint? { joints.first { $0.type == .leftFoot } }

    static func run(_ t: TestRunner) {
        let base = SIMD3<Float>(-0.10, -0.86, 0.0)

        t.test("STEP: no FootMotionState → foot is literal (no exaggeration)") {
            let cfg = TrackingConfig(); cfg.jointSize = 1.0; cfg.footTrackersAtAnkle = true
            let solver = JointSolver(settings: cfg)   // no footMotionState injected
            guard let f = leftFoot(solver.solve(pose(leftAnkle: SIMD3(0.05, -0.70, 0.12)))) else {
                t.check(false, "no foot"); return
            }
            t.close(f.position.x, 0.05, tol: 1e-3, "x literal")
            t.close(f.position.y, -0.70, tol: 1e-3, "y literal")
            t.close(f.position.z, 0.12, tol: 1e-3, "z literal")
        }

        t.test("STEP: a planted foot is not exaggerated (swing ≈ 0, stays glued)") {
            let cfg = TrackingConfig(); cfg.jointSize = 1.0; cfg.footTrackersAtAnkle = true
            cfg.stepStrideCoefficient = 1.8; cfg.stepLiftCoefficient = 1.5
            let solver = JointSolver(settings: cfg, footMotionState: FootMotionState())
            _ = solver.solve(pose(leftAnkle: base))   // seed neutral + floor
            var f: VRJoint?
            // Foot shifts laterally but stays PLANTED (y == floor) over many frames.
            for _ in 0..<12 { f = leftFoot(solver.solve(pose(leftAnkle: SIMD3(-0.04, -0.86, 0.0)))) }
            guard let f else { t.check(false, "no foot"); return }
            t.check(abs(f.position.x - (-0.04)) < 0.03, "planted foot ≈ literal x (no stride amp): \(f.position.x)")
            t.check(abs(f.position.y - (-0.86)) < 0.01, "planted foot stays on the floor: \(f.position.y)")
        }

        t.test("STEP: a swinging foot's stride and lift are amplified") {
            let cfg = TrackingConfig(); cfg.jointSize = 1.0; cfg.footTrackersAtAnkle = true
            cfg.stepStrideCoefficient = 1.8; cfg.stepLiftCoefficient = 1.5
            let solver = JointSolver(settings: cfg, footMotionState: FootMotionState())
            _ = solver.solve(pose(leftAnkle: base))    // seed neutral (z=0) + floor (y=-0.86)
            // Step: foot lifts 0.15 (> liftNone 0.10 → full swing) and moves +0.12 in Z.
            let stepped = SIMD3<Float>(-0.10, -0.71, 0.12)
            var f: VRJoint?
            for _ in 0..<14 { f = leftFoot(solver.solve(pose(leftAnkle: stepped))) }
            guard let f else { t.check(false, "no foot"); return }
            t.check(f.position.z > 0.12 + 1e-3, "stride amplified on Z (> raw 0.12): \(f.position.z)")
            t.check(f.position.y > -0.71 + 1e-3, "lift amplified up (> raw -0.71): \(f.position.y)")
            // And bounded by the clamps (never thrown away): stride ≤ 0.22 over raw.
            t.check(f.position.z < 0.12 + 0.22 + 1e-3, "stride stays clamped")
            print(String(format: "  [step] z 0.12->%.3f  y -0.71->%.3f", f.position.z, f.position.y))
        }
    }
}
