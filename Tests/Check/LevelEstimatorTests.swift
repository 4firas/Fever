import Foundation
import simd
import FeverCore

/// Direct unit coverage for `LevelEstimator` — the IMU-free gravity/up math.
///
/// The reference vector is now FOOT-DERIVED (midHip − footMid), not the spine,
/// so natural torso lean no longer bakes a pitch into the leveling datum.
///
/// Assertions are written against the rotation's ACTION on vectors (`q.act`),
/// not its raw components, so they are robust to quaternion double-cover sign.
/// The load-bearing invariants:
///   • leveling sends the foot-derived vertical onto +Y,
///   • a vertical foot-to-hip vector yields identity (YAW-PRESERVED),
///   • roll-off leaves the foot vector's X component intact,
///   • roll-on removes it (centres the body laterally),
///   • leveling is a length-preserving proper rotation,
///   • the upright gate distinguishes standing from a crouch,
///   • natural torso LEAN does NOT change the leveling datum.
enum LevelEstimatorTests {
    static func run(_ t: TestRunner) {

        // Apply leveling to the foot-derived up vector and return the result.
        // midHip is the hip centre (origin); footMid is the foot centre (below).
        func leveled(midHip: SIMD3<Float>, footMid: SIMD3<Float>, roll: Bool) -> SIMD3<Float> {
            let q = LevelEstimator.levelingQuaternion(midHip: midHip, footMid: footMid,
                                                      includeRoll: roll)
            let up = midHip - footMid
            guard simd_length(up) > 1e-5 else { return SIMD3(0, 1, 0) }
            return q.act(simd_normalize(up))
        }

        t.test("LEVEL: a vertical foot-to-hip vector yields identity") {
            let q = LevelEstimator.levelingQuaternion(midHip: SIMD3(0, 0, 0),
                                                       footMid: SIMD3(0, -0.9, 0),
                                                       includeRoll: true)
            let probe = SIMD3<Float>(0.3, 0.4, -0.7)
            let r = q.act(probe)
            t.close(r.x, probe.x, tol: 1e-4, "identity leaves x")
            t.close(r.y, probe.y, tol: 1e-4, "identity leaves y")
            t.close(r.z, probe.z, tol: 1e-4, "identity leaves z")
        }

        t.test("LEVEL: forward camera pitch — foot-to-hip leveled to vertical (z→0)") {
            // Camera tilted such that foot appears shifted in z relative to hip.
            let s = leveled(midHip: .zero, footMid: SIMD3(0, -0.9, 0.36), roll: false)
            t.close(s.z, 0, tol: 1e-4, "pitched foot-to-hip z zeroed")
            t.close(s.y, 1, tol: 1e-4, "leveled vector points up")
            t.close(s.x, 0, tol: 1e-4, "no x introduced by pitch-only")
        }

        t.test("LEVEL: backward camera pitch — foot-to-hip leveled to vertical (z→0)") {
            let s = leveled(midHip: .zero, footMid: SIMD3(0, -0.9, -0.36), roll: false)
            t.close(s.z, 0, tol: 1e-4, "back-pitched z zeroed")
            t.close(s.y, 1, tol: 1e-4, "leveled vector points up")
        }

        t.test("LEVEL: roll OFF leaves the lateral component intact") {
            // Foot laterally offset (camera rolled) but no z offset → pitch is 0,
            // roll disabled → identity, so the x lean survives.
            let s = leveled(midHip: .zero, footMid: SIMD3(0.3, -0.9, 0), roll: false)
            let n = simd_normalize(SIMD3<Float>(0, 0, 0) - SIMD3<Float>(0.3, -0.9, 0))
            t.close(s.x, n.x, tol: 1e-4, "roll-off keeps x lean")
            t.close(s.y, n.y, tol: 1e-4, "roll-off keeps y")
        }

        t.test("LEVEL: roll ON removes the lateral lean (x→0)") {
            let s = leveled(midHip: .zero, footMid: SIMD3(0.3, -0.9, 0), roll: true)
            t.close(s.x, 0, tol: 1e-4, "roll-on zeroes x")
            t.close(s.y, 1, tol: 1e-4, "leveled vector points up")
        }

        t.test("LEVEL: combined pitch+roll — fully vertical") {
            let s = leveled(midHip: .zero, footMid: SIMD3(0.3, -0.9, 0.36), roll: true)
            t.close(s.x, 0, tol: 1e-4, "combined zeroes x")
            t.close(s.z, 0, tol: 1e-4, "combined zeroes z")
            t.close(s.y, 1, tol: 1e-4, "leveled vector points up")
        }

        t.test("LEVEL: YAW-PRESERVED — a vertical foot-to-hip is identity regardless of body turn") {
            // A user turning has their hips above their feet regardless of yaw.
            // The leveling quaternion must be identity (no yaw correction introduced).
            let q = LevelEstimator.levelingQuaternion(midHip: SIMD3(0, 0, 0),
                                                       footMid: SIMD3(0, -0.9, 0),
                                                       includeRoll: true)
            let hipLineTurned = SIMD3<Float>(0, 0, 1)  // user turned 90°
            let r = q.act(hipLineTurned)
            t.close(r.x, 0, tol: 1e-4, "turned hip-line heading preserved (x)")
            t.close(r.y, 0, tol: 1e-4, "turned hip-line heading preserved (y)")
            t.close(r.z, 1, tol: 1e-4, "turned hip-line heading preserved (z)")
        }

        t.test("LEVEL: leveling is a length-preserving proper rotation") {
            let q = LevelEstimator.levelingQuaternion(midHip: .zero,
                                                       footMid: SIMD3(0.25, -0.9, 0.35),
                                                       includeRoll: true)
            t.close(simd_length(q.vector), 1, tol: 1e-4, "unit quaternion")
            let v = SIMD3<Float>(0.7, -0.2, 0.5)
            t.close(simd_length(q.act(v)), simd_length(v), tol: 1e-4, "rotation preserves length")
        }

        t.test("LEVEL: an ill-conditioned (too short) foot-to-hip vector returns identity") {
            let q = LevelEstimator.levelingQuaternion(midHip: SIMD3(0, 0.05, 0),
                                                       footMid: SIMD3(0, -0.05, 0.03),
                                                       includeRoll: true)
            let probe = SIMD3<Float>(0.2, 0.9, 0.3)
            let r = q.act(probe)
            t.close(r.x, probe.x, tol: 1e-4, "short up vector → identity x")
            t.close(r.y, probe.y, tol: 1e-4, "short up vector → identity y")
            t.close(r.z, probe.z, tol: 1e-4, "short up vector → identity z")
        }

        t.test("LEVEL: no NaN under a hard tilt") {
            let q = LevelEstimator.levelingQuaternion(midHip: .zero,
                                                       footMid: SIMD3(0.7, -0.3, -0.5),
                                                       includeRoll: true)
            let v = q.vector
            t.check(v.x.isFinite && v.y.isFinite && v.z.isFinite && v.w.isFinite,
                    "hard tilt produced non-finite quat: \(v)")
        }

        t.test("LEVEL: natural torso LEAN does NOT affect the datum (key fix)") {
            // A user stands upright with a natural ~15° forward lean of the SPINE
            // (shoulders slightly in front of hips — normal posture). The foot-derived
            // up vector is still purely vertical because feet are directly below hips.
            // Leveling must be identity so the torso lean is NOT baked into the datum.
            let footMid = SIMD3<Float>(0, -0.9, 0)  // feet directly below hip
            let q = LevelEstimator.levelingQuaternion(midHip: .zero, footMid: footMid,
                                                      includeRoll: false)
            let probe = SIMD3<Float>(0.3, 0.4, -0.7)
            let r = q.act(probe)
            t.close(r.x, probe.x, tol: 1e-4, "natural torso lean → leveling is identity x")
            t.close(r.y, probe.y, tol: 1e-4, "natural torso lean → leveling is identity y")
            t.close(r.z, probe.z, tol: 1e-4, "natural torso lean → leveling is identity z")
        }

        t.test("UPRIGHT: standing (head up, a foot below hips) passes the gate") {
            let ok = LevelEstimator.uprightSanity(nose: SIMD3(0, 0.7, 0),
                                                  midHip: .zero,
                                                  leftAnkle: SIMD3(-0.1, -0.85, 0),
                                                  rightAnkle: SIMD3(0.1, -0.85, 0))
            t.check(ok, "standing pose must be upright")
        }

        t.test("UPRIGHT: a head-down crouch fails the gate (reference lost)") {
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
