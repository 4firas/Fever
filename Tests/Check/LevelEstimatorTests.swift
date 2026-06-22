import Foundation
import simd
import FeverCore

/// Direct unit coverage for `LevelEstimator` — the IMU-free gravity/up math.
///
/// Assertions are written against the rotation's ACTION on vectors (`q.act`),
/// not its raw components, so they are robust to quaternion double-cover sign.
/// The load-bearing invariants:
///   • leveling sends the (tilted) spine onto vertical +Y,
///   • a vertical spine yields identity (YAW-PRESERVED — body turns untouched),
///   • roll-off leaves the spine's X lean intact, roll-on removes it,
///   • leveling is a length-preserving proper rotation,
///   • the upright gate distinguishes standing from a crouch.
enum LevelEstimatorTests {
    static func run(_ t: TestRunner) {

        // Apply leveling to the (normalized) spine and report the result.
        func leveled(neck: SIMD3<Float>, midHip: SIMD3<Float>, roll: Bool) -> SIMD3<Float> {
            let q = LevelEstimator.levelingQuaternion(neck: neck, midHip: midHip, includeRoll: roll)
            return q.act(simd_normalize(neck - midHip))
        }

        t.test("LEVEL: an untilted (vertical) spine yields identity") {
            // spine straight up → no correction → action is the identity on any vector.
            let q = LevelEstimator.levelingQuaternion(neck: SIMD3(0, 1.5, 0),
                                                       midHip: SIMD3(0, 0, 0),
                                                       includeRoll: true)
            let probe = SIMD3<Float>(0.3, 0.4, -0.7)
            let r = q.act(probe)
            t.close(r.x, probe.x, tol: 1e-4, "identity leaves x")
            t.close(r.y, probe.y, tol: 1e-4, "identity leaves y")
            t.close(r.z, probe.z, tol: 1e-4, "identity leaves z")
        }

        t.test("LEVEL: forward camera pitch — spine leveled to vertical (z→0)") {
            // Spine leans toward the camera (+z): pitch-only leveling must zero z.
            let s = leveled(neck: SIMD3(0, 1.0, 0.4), midHip: .zero, roll: false)
            t.close(s.z, 0, tol: 1e-4, "forward-pitched spine z zeroed")
            t.close(s.y, 1, tol: 1e-4, "leveled spine points up")
            t.close(s.x, 0, tol: 1e-4, "no x introduced by pitch-only")
        }

        t.test("LEVEL: backward camera pitch — spine leveled to vertical (z→0)") {
            let s = leveled(neck: SIMD3(0, 1.0, -0.45), midHip: .zero, roll: false)
            t.close(s.z, 0, tol: 1e-4, "back-pitched spine z zeroed")
            t.close(s.y, 1, tol: 1e-4, "leveled spine points up")
        }

        t.test("LEVEL: roll OFF leaves the spine's sideways lean intact") {
            // Spine leans in +x with no z. Pitch is 0 and roll disabled → identity,
            // so the x lean survives (we did NOT correct roll).
            let s = leveled(neck: SIMD3(0.3, 1.0, 0), midHip: .zero, roll: false)
            let n = simd_normalize(SIMD3<Float>(0.3, 1.0, 0))
            t.close(s.x, n.x, tol: 1e-4, "roll-off keeps x lean")
            t.close(s.y, n.y, tol: 1e-4, "roll-off keeps y")
        }

        t.test("LEVEL: roll ON removes the sideways lean (x→0)") {
            let s = leveled(neck: SIMD3(0.3, 1.0, 0), midHip: .zero, roll: true)
            t.close(s.x, 0, tol: 1e-4, "roll-on zeroes x")
            t.close(s.y, 1, tol: 1e-4, "leveled spine points up")
        }

        t.test("LEVEL: combined pitch+roll — spine fully vertical") {
            let s = leveled(neck: SIMD3(0.3, 1.0, 0.4), midHip: .zero, roll: true)
            t.close(s.x, 0, tol: 1e-4, "combined zeroes x")
            t.close(s.z, 0, tol: 1e-4, "combined zeroes z")
            t.close(s.y, 1, tol: 1e-4, "leveled spine points up")
        }

        t.test("LEVEL: YAW-PRESERVED — a vertical spine is identity regardless of turn") {
            // A standing user who has TURNED keeps a vertical spine (yaw doesn't
            // tilt it). Leveling must therefore be identity so body yaw is never
            // cancelled. We verify a lateral hip-line vector keeps its heading.
            let q = LevelEstimator.levelingQuaternion(neck: SIMD3(0, 1.2, 0),
                                                       midHip: .zero, includeRoll: true)
            let hipLineTurned = SIMD3<Float>(0, 0, 1)   // user turned 90°: hips along Z
            let r = q.act(hipLineTurned)
            t.close(r.x, 0, tol: 1e-4, "turned hip-line heading preserved (x)")
            t.close(r.y, 0, tol: 1e-4, "turned hip-line heading preserved (y)")
            t.close(r.z, 1, tol: 1e-4, "turned hip-line heading preserved (z)")
        }

        t.test("LEVEL: leveling is a length-preserving proper rotation") {
            let q = LevelEstimator.levelingQuaternion(neck: SIMD3(0.25, 1.0, 0.35),
                                                       midHip: .zero, includeRoll: true)
            t.close(simd_length(q.vector), 1, tol: 1e-4, "unit quaternion")
            let v = SIMD3<Float>(0.7, -0.2, 0.5)
            t.close(simd_length(q.act(v)), simd_length(v), tol: 1e-4, "rotation preserves length")
        }

        t.test("LEVEL: an ill-conditioned (too short) spine returns identity") {
            let q = LevelEstimator.levelingQuaternion(neck: SIMD3(0, 0.1, 0.05),
                                                       midHip: .zero, includeRoll: true)
            let probe = SIMD3<Float>(0.2, 0.9, 0.3)
            let r = q.act(probe)
            t.close(r.x, probe.x, tol: 1e-4, "short spine → identity x")
            t.close(r.y, probe.y, tol: 1e-4, "short spine → identity y")
            t.close(r.z, probe.z, tol: 1e-4, "short spine → identity z")
        }

        t.test("LEVEL: no NaN under a hard tilt") {
            let q = LevelEstimator.levelingQuaternion(neck: SIMD3(0.9, 0.3, 0.9),
                                                       midHip: .zero, includeRoll: true)
            let v = q.vector
            t.check(v.x.isFinite && v.y.isFinite && v.z.isFinite && v.w.isFinite,
                    "hard tilt produced non-finite quat: \(v)")
        }

        t.test("UPRIGHT: standing (head up, a foot below hips) passes the gate") {
            let ok = LevelEstimator.uprightSanity(nose: SIMD3(0, 0.7, 0),
                                                  midHip: .zero,
                                                  leftAnkle: SIMD3(-0.1, -0.85, 0),
                                                  rightAnkle: SIMD3(0.1, -0.85, 0))
            t.check(ok, "standing pose must be upright")
        }

        t.test("UPRIGHT: a head-down crouch fails the gate (reference lost)") {
            // Nose dropped to/below the hips (crouched into the lens) → not upright.
            let ok = LevelEstimator.uprightSanity(nose: SIMD3(0, -0.1, 0.4),
                                                  midHip: .zero,
                                                  leftAnkle: SIMD3(-0.1, -0.85, 0),
                                                  rightAnkle: SIMD3(0.1, -0.85, 0))
            t.check(!ok, "head-down crouch must NOT be upright")
        }

        t.test("UPRIGHT: feet not below the hips fails the gate") {
            let ok = LevelEstimator.uprightSanity(nose: SIMD3(0, 0.7, 0),
                                                  midHip: .zero,
                                                  leftAnkle: SIMD3(-0.1, 0.2, 0),
                                                  rightAnkle: SIMD3(0.1, 0.2, 0))
            t.check(!ok, "feet above hips must NOT be upright")
        }
    }
}
