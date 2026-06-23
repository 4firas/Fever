import Foundation
import simd
import FeverCore

/// Coverage for `BodyStabilizer` — the stateful datum-freeze + continuous
/// re-leveling around `LevelEstimator`. Pins the PinoQuest-parity behaviors:
/// datum frozen at capture, baseline leveling held while the toggle is OFF,
/// re-arm on reset (Re-center), lost-on-crouch, and gentle continuous re-leveling
/// when the toggle is ON.
enum BodyStabilizerTests {

    /// Build a sidecar reply for a standing body; `shoulderZ` leans the spine
    /// toward (+) / away from (−) the camera (simulated camera pitch). `noseY` /
    /// `ankleY` are in MediaPipe y-DOWN (smaller = higher).
    ///
    /// Ankle z is derived from shoulderZ: with shoulder 0.5 m above the hip and
    /// ankle 0.9 m below, the same camera tilt pushes ankles by -1.8×shoulderZ in
    /// the opposite z direction. This gives exact leveling for `spineDir(shoulderZ)`.
    static func reply(shoulderZ: Float, noseY: Float = -0.7, ankleY: Float = 0.9) -> SidecarReply {
        let ankleZ = -1.8 * shoulderZ
        var world = [SIMD3<Float>](repeating: .zero, count: 33)
        let vis = [Float](repeating: 1, count: 33)
        func put(_ l: BlazePose.Landmark, _ v: SIMD3<Float>) { world[l.rawValue] = v }
        put(.nose, SIMD3(0, noseY, shoulderZ))
        put(.leftShoulder, SIMD3(-0.2, -0.5, shoulderZ))
        put(.rightShoulder, SIMD3(0.2, -0.5, shoulderZ))
        put(.leftHip, SIMD3(-0.1, 0, 0))
        put(.rightHip, SIMD3(0.1, 0, 0))
        put(.leftAnkle, SIMD3(-0.1, ankleY, ankleZ))
        put(.rightAnkle, SIMD3(0.1, ankleY, ankleZ))
        return SidecarReply(found: true, world: world, visibility: vis, presence: vis, image: [])
    }

    /// The axis-fixed (solver-frame) spine direction for a reply, for checking
    /// that a returned leveling rotation makes it vertical.
    static func spineDir(_ shoulderZ: Float) -> SIMD3<Float> {
        let neck = SIMD3<Float>(0, 0.5, shoulderZ)   // axis-fixed shoulder mid
        return simd_normalize(neck)                   // midHip is at origin
    }

    static func run(_ t: TestRunner) {
        let z: Float = MediaPipeFrame.defaultZSign

        t.test("BODYSTAB: untilted standing → identity leveling") {
            let s = BodyStabilizer()
            let q = s.levelRotation(reply: reply(shoulderZ: 0), zSign: z, enabled: false, dt: 0)
            let probe = SIMD3<Float>(0.3, 0.5, -0.6)
            let r = q.act(probe)
            t.close(r.x, probe.x, tol: 1e-4, "identity x")
            t.close(r.y, probe.y, tol: 1e-4, "identity y")
            t.close(r.z, probe.z, tol: 1e-4, "identity z")
        }

        t.test("BODYSTAB: tilted camera → datum levels the spine to vertical") {
            let s = BodyStabilizer()
            let q = s.levelRotation(reply: reply(shoulderZ: 0.3), zSign: z, enabled: false, dt: 0)
            let leveled = q.act(spineDir(0.3))
            t.close(leveled.z, 0, tol: 1e-3, "datum leveling zeroes spine z")
            t.close(leveled.y, 1, tol: 1e-3, "leveled spine points up")
        }

        t.test("BODYSTAB: datum is FROZEN — a later differing frame reuses the first datum") {
            let s = BodyStabilizer()
            let q1 = s.levelRotation(reply: reply(shoulderZ: 0.3), zSign: z, enabled: false, dt: 0.03)
            // A second frame with a DIFFERENT tilt; with the toggle off the datum holds.
            let q2 = s.levelRotation(reply: reply(shoulderZ: 0.5), zSign: z, enabled: false, dt: 0.03)
            t.close(q2.vector.x, q1.vector.x, tol: 1e-5, "datum held x")
            t.close(q2.vector.y, q1.vector.y, tol: 1e-5, "datum held y")
            t.close(q2.vector.z, q1.vector.z, tol: 1e-5, "datum held z")
            t.close(q2.vector.w, q1.vector.w, tol: 1e-5, "datum held w")
        }

        t.test("BODYSTAB: not-upright before any capture → identity, no datum") {
            let s = BodyStabilizer()
            // Nose dropped below the hips (head-down crouch): never captures a datum.
            let q = s.levelRotation(reply: reply(shoulderZ: 0.3, noseY: 0.2), zSign: z, enabled: false, dt: 0)
            let probe = SIMD3<Float>(0.2, 0.7, 0.4)
            let r = q.act(probe)
            t.close(r.x, probe.x, tol: 1e-4, "no datum → identity x")
            t.close(r.z, probe.z, tol: 1e-4, "no datum → identity z")
            t.check(s.isLevelLost(), "non-upright frame reports level lost")
        }

        t.test("BODYSTAB: reset() re-arms the datum capture") {
            let s = BodyStabilizer()
            let q1 = s.levelRotation(reply: reply(shoulderZ: 0.3), zSign: z, enabled: false, dt: 0)
            s.reset()
            // New datum from a different tilt after reset.
            let q2 = s.levelRotation(reply: reply(shoulderZ: -0.4), zSign: z, enabled: false, dt: 0)
            let differs = abs(q2.vector.x - q1.vector.x) + abs(q2.vector.z - q1.vector.z) > 1e-3
            t.check(differs, "reset re-captured a fresh datum (q2 ≠ q1)")
            // And the fresh datum actually levels the new tilt.
            let leveled = q2.act(spineDir(-0.4))
            t.close(leveled.z, 0, tol: 1e-3, "fresh datum levels the new tilt")
        }

        t.test("BODYSTAB: a crouch AFTER a datum holds the datum + reports lost") {
            let s = BodyStabilizer()
            let datum = s.levelRotation(reply: reply(shoulderZ: 0.3), zSign: z, enabled: false, dt: 0)
            t.check(!s.isLevelLost(), "upright capture is not lost")
            let q = s.levelRotation(reply: reply(shoulderZ: 0.3, noseY: 0.2), zSign: z, enabled: false, dt: 0)
            t.close(q.vector.x, datum.vector.x, tol: 1e-5, "crouch holds datum x")
            t.close(q.vector.w, datum.vector.w, tol: 1e-5, "crouch holds datum w")
            t.check(s.isLevelLost(), "crouch reports level lost (box should vanish)")
        }

        t.test("BODYSTAB: continuous re-leveling (toggle ON) tracks a new tilt over time") {
            let s = BodyStabilizer()
            // Datum captured untilted (≈identity).
            _ = s.levelRotation(reply: reply(shoulderZ: 0), zSign: z, enabled: false, dt: 0)
            // One tiny enabled step toward a tilt: should move PART of the way (gentle).
            let small = s.levelRotation(reply: reply(shoulderZ: 0.4), zSign: z, enabled: true, dt: 0.1)
            let zSmall = small.act(spineDir(0.4)).z
            // Then many seconds of re-leveling: should converge near vertical.
            var qBig = small
            for _ in 0..<60 { qBig = s.levelRotation(reply: reply(shoulderZ: 0.4), zSign: z, enabled: true, dt: 0.1) }
            let zBig = qBig.act(spineDir(0.4)).z
            let zRaw = spineDir(0.4).z   // un-leveled lean
            t.check(abs(zSmall) < abs(zRaw), "one gentle step reduces the tilt but not fully")
            t.check(abs(zSmall) > 0.05, "one step does NOT snap fully (continuous, not instant)")
            t.close(zBig, 0, tol: 5e-2, "sustained re-leveling converges to vertical")
        }
    }
}
