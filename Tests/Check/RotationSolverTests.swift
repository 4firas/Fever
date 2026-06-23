import Foundation
import simd
import FeverCore

/// Per-bone rotation-solver correctness, pinned against the PinoFBT capture:
///   • CHEST must follow the torso SPINE, not the head/neck tilt (PinoFBT chest ≈0
///     at neutral; Fever was carrying a ~20° forward pitch from the shoulder→ear up).
///   • FOOT must follow the heel→toe direction — PITCH (toe up/down) + YAW (turn) —
///     with ROLL locked (PinoFBT foot ≈ pitch-only; Fever had pitch LOCKED OUT and
///     yaw/roll wild because the frame forced +Y vertical).
enum RotationSolverTests {

    /// Build a full 33-landmark upright pose in the SOLVER frame (x-right, y-UP,
    /// +z toward camera). `headForwardZ` tilts the ears/nose forward (head looking
    /// down) WITHOUT moving the spine — used to prove the chest ignores neck tilt.
    /// `toe`/`heel` override the foot landmarks to test foot pitch/yaw.
    static func uprightPose(headForwardZ: Float = 0,
                            toeL: SIMD3<Float>? = nil, heelL: SIMD3<Float>? = nil) -> PoseResult {
        var lm = [NormalizedLandmark](repeating: NormalizedLandmark(position: .zero), count: 33)
        func set(_ i: BlazePose.Landmark, _ x: Float, _ y: Float, _ z: Float) {
            lm[i.rawValue] = NormalizedLandmark(position: SIMD3<Float>(x, y, z), visibility: 1)
        }
        set(.nose, 0, 1.55, 0.10 + headForwardZ)
        set(.leftEar, -0.06, 1.52, 0.02 + headForwardZ)
        set(.rightEar, 0.06, 1.52, 0.02 + headForwardZ)
        set(.leftShoulder, -0.18, 1.35, 0); set(.rightShoulder, 0.18, 1.35, 0)
        set(.leftHip, -0.10, 0.90, 0); set(.rightHip, 0.10, 0.90, 0)
        set(.leftElbow, -0.22, 1.10, 0); set(.rightElbow, 0.22, 1.10, 0)
        set(.leftWrist, -0.24, 0.85, 0); set(.rightWrist, 0.24, 0.85, 0)
        set(.leftKnee, -0.10, 0.50, 0); set(.rightKnee, 0.10, 0.50, 0)
        set(.leftAnkle, -0.10, 0.10, 0); set(.rightAnkle, 0.10, 0.10, 0)
        // Feet: heel behind (−z), toe forward (+z) and slightly down (toe.y<heel.y).
        let hl = heelL ?? SIMD3<Float>(-0.10, 0.08, -0.05)
        let tl = toeL  ?? SIMD3<Float>(-0.10, 0.05,  0.10)
        set(.leftHeel, hl.x, hl.y, hl.z); set(.rightHeel, 0.10, 0.08, -0.05)
        set(.leftFootIndex, tl.x, tl.y, tl.z); set(.rightFootIndex, 0.10, 0.05, 0.10)
        return PoseResult(landmarks: lm, timestamp: 0)
    }

    static func joint(_ pose: PoseResult, _ type: JointType) -> VRJoint {
        JointSolver(settings: TrackingConfig()).solve(pose).first { $0.type == type }!
    }

    static func run(_ t: TestRunner) {
        // ── CHEST follows the spine, not the head ──────────────────────────────
        t.test("CHEST rotation follows the torso SPINE, not head/neck tilt") {
            // Ears/nose tilted hard forward (looking down); spine stays vertical.
            let chest = joint(uprightPose(headForwardZ: 0.25), .chest)
            let yAxis = chest.rotation.act(SIMD3<Float>(0, 1, 0))   // chest local +Y in world
            t.check(yAxis.y > 0.95,
                    "chest +Y must point up the vertical spine, got \(yAxis)")
            t.check(abs(yAxis.z) < 0.20,
                    "chest +Y must NOT tilt forward with the head, got z=\(yAxis.z)")
        }

        // ── FOOT follows heel→toe (pitch + yaw), roll locked ───────────────────
        t.test("FOOT rotation +Y follows the heel→toe direction (not forced vertical)") {
            let lf = joint(uprightPose(), .leftFoot)
            let yAxis = lf.rotation.act(SIMD3<Float>(0, 1, 0))
            let footDir = simd_normalize(SIMD3<Float>(-0.10, 0.05, 0.10) - SIMD3<Float>(-0.10, 0.08, -0.05))
            let dot = simd_dot(yAxis, footDir)
            t.check(dot > 0.97, "foot +Y must follow heel→toe: yAxis=\(yAxis) footDir=\(footDir) dot=\(dot)")
        }

        t.test("FOOT pitch responds to toe down (was locked out by the vertical-up frame)") {
            // Flat foot (toe level with heel) vs toe pointed down.
            let flat = joint(uprightPose(toeL: SIMD3<Float>(-0.10, 0.08, 0.15),
                                         heelL: SIMD3<Float>(-0.10, 0.08, -0.05)), .leftFoot)
            let down = joint(uprightPose(toeL: SIMD3<Float>(-0.10, 0.00, 0.15),
                                         heelL: SIMD3<Float>(-0.10, 0.08, -0.05)), .leftFoot)
            let yFlat = flat.rotation.act(SIMD3<Float>(0, 1, 0))
            let yDown = down.rotation.act(SIMD3<Float>(0, 1, 0))
            t.check(yDown.y < yFlat.y - 0.05,
                    "toe-down must pitch the foot +Y downward: flat.y=\(yFlat.y) down.y=\(yDown.y)")
        }

        t.test("FOOT yaw responds to a foot turn (heel→toe rotated horizontally)") {
            let straight = joint(uprightPose(), .leftFoot).rotation.act(SIMD3<Float>(0, 1, 0))
            // Turn the foot: toe swings in +x while staying level/forward.
            let turned = joint(uprightPose(toeL: SIMD3<Float>(0.05, 0.05, 0.10),
                                           heelL: SIMD3<Float>(-0.10, 0.08, -0.05)), .leftFoot)
                .rotation.act(SIMD3<Float>(0, 1, 0))
            t.check(turned.x > straight.x + 0.05,
                    "foot turn must yaw the foot +Y in x: straight.x=\(straight.x) turned.x=\(turned.x)")
        }

        // ── HIP stays the clean reference (regression guard) ───────────────────
        t.test("HIP rotation +Y follows the vertical spine at neutral (regression)") {
            let hip = joint(uprightPose(headForwardZ: 0.25), .hip)
            let yAxis = hip.rotation.act(SIMD3<Float>(0, 1, 0))
            t.check(yAxis.y > 0.95, "hip +Y must stay vertical at neutral, got \(yAxis)")
        }
    }
}
