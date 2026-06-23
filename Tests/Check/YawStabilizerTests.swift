import Foundation
import simd
import FeverCore

/// YawStabilizer (PinoFBT-style Body Stabilizer) + the swing–twist math it rests
/// on: smooth ONE body-facing yaw and impose it coherently on the torso while
/// preserving each tracker's pitch/roll.
enum YawStabilizerTests {

    static let up = SIMD3<Float>(0, 1, 0)
    static func deg(_ r: Float) -> Float { r * 180 / .pi }
    static func yawDeg(_ q: simd_quatf) -> Float { deg(swingTwist(q, axis: up).twistAngle) }

    /// Twisted-torso pose: shoulders yawed about vertical relative to the hips, so
    /// chest yaw ≠ hip yaw when the stabilizer is OFF.
    static func twistedPose() -> PoseResult {
        var lm = [NormalizedLandmark](repeating: NormalizedLandmark(position: .zero), count: 33)
        func set(_ i: BlazePose.Landmark, _ x: Float, _ y: Float, _ z: Float) {
            lm[i.rawValue] = NormalizedLandmark(position: SIMD3<Float>(x, y, z), visibility: 1)
        }
        set(.nose, 0, 1.55, 0.10); set(.leftEar, -0.06, 1.52, 0.02); set(.rightEar, 0.06, 1.52, 0.02)
        // Shoulders YAWED ~27° about vertical (left back −z, right forward +z).
        set(.leftShoulder, -0.16, 1.35, -0.08); set(.rightShoulder, 0.16, 1.35, 0.08)
        // Hips square to the camera (no yaw).
        set(.leftHip, -0.10, 0.90, 0); set(.rightHip, 0.10, 0.90, 0)
        set(.leftElbow, -0.22, 1.10, 0); set(.rightElbow, 0.22, 1.10, 0)
        set(.leftWrist, -0.24, 0.85, 0); set(.rightWrist, 0.24, 0.85, 0)
        set(.leftKnee, -0.10, 0.50, 0); set(.rightKnee, 0.10, 0.50, 0)
        set(.leftAnkle, -0.10, 0.10, 0); set(.rightAnkle, 0.10, 0.10, 0)
        set(.leftHeel, -0.10, 0.08, -0.05); set(.rightHeel, 0.10, 0.08, -0.05)
        set(.leftFootIndex, -0.10, 0.05, 0.10); set(.rightFootIndex, 0.10, 0.05, 0.10)
        return PoseResult(landmarks: lm, timestamp: 0)
    }

    static func run(_ t: TestRunner) {
        // ── swing–twist math ──────────────────────────────────────────────────
        t.test("swingTwist round-trips: swing · twist(angle) reconstructs q") {
            let q = simd_quatf(angle: 0.8, axis: simd_normalize(SIMD3<Float>(0.3, 0.5, 0.8)))
            let (sw, ang) = swingTwist(q, axis: up)
            let recon = sw * simd_quatf(angle: ang, axis: up)
            t.check(abs(simd_dot(recon.vector, q.vector)) > 0.999, "round-trip dot \(simd_dot(recon.vector, q.vector))")
        }
        t.test("swingTwist of a pure YAW → swing ≈ identity, angle = the yaw") {
            let q = simd_quatf(angle: 50 * .pi / 180, axis: up)
            let (sw, ang) = swingTwist(q, axis: up)
            t.close(deg(ang), 50, tol: 0.5, "twist angle")
            t.check(abs(sw.real) > 0.999, "swing ≈ identity, real=\(sw.real)")
        }
        t.test("swingTwist of a pure PITCH → twist angle ≈ 0") {
            let q = simd_quatf(angle: 30 * .pi / 180, axis: SIMD3<Float>(1, 0, 0))
            t.close(deg(swingTwist(q, axis: up).twistAngle), 0, tol: 0.5, "no yaw in a pure pitch")
        }
        t.test("imposeYaw replaces yaw, preserves pitch/roll") {
            let pitch = simd_quatf(angle: 30 * .pi / 180, axis: SIMD3<Float>(1, 0, 0))
            let ys = YawStabilizer()
            let out = ys.imposeYaw(pitch, yaw: simd_quatf(angle: 90 * .pi / 180, axis: up))
            t.close(yawDeg(out), 90, tol: 1, "yaw set to 90")
            let swing = swingTwist(out, axis: up).swing
            t.check(abs(simd_dot(swing.vector, pitch.vector)) > 0.99, "pitch swing preserved")
        }

        // ── temporal smoothing (wrap-safe) ────────────────────────────────────
        t.test("update eases toward a yaw step and converges") {
            let ys = YawStabilizer(smoothing: 0.5)
            _ = ys.update(reference: simd_quatf(angle: 0, axis: up))           // prime at 0
            let step = simd_quatf(angle: .pi / 2, axis: up)                    // target 90°
            let a1 = yawDeg(ys.update(reference: step))
            t.check(a1 > 5 && a1 < 85, "first step eases between 0 and 90: \(a1)")
            var last: Float = 0
            for _ in 0..<60 { last = yawDeg(ys.update(reference: step)) }
            t.close(last, 90, tol: 1, "converges to target")
        }
        t.test("update is wrap-safe across the ±180° seam") {
            let ys = YawStabilizer(smoothing: 0.5)
            _ = ys.update(reference: simd_quatf(angle: 170 * .pi / 180, axis: up))
            // target −170° is only 20° away across the seam, NOT 340°.
            let a = yawDeg(ys.update(reference: simd_quatf(angle: -170 * .pi / 180, axis: up)))
            // eased ~10° past 170 → ~180/−180 region, never swinging back toward 0.
            t.check(abs(a) > 150, "wrap-safe smoothing stays near the seam, got \(a)")
        }

        // ── default + solver integration ──────────────────────────────────────
        t.test("yawStabilizer defaults OFF (opt-in)") {
            UserDefaults.standard.removeObject(forKey: "yawStabilizer")
            t.check(TrackingConfig().yawStabilizer == false, "default must be off")
        }
        t.test("ON: torso (hip+chest) share one coherent yaw on a twisted torso") {
            let pose = twistedPose()
            let off = JointSolver(settings: { let c = TrackingConfig(); c.yawStabilizer = false; return c }())
                .solve(pose)
            let hipOff = yawDeg(off.first { $0.type == .hip }!.rotation)
            let chestOff = yawDeg(off.first { $0.type == .chest }!.rotation)
            t.check(abs(shortestAngleDelta(from: hipOff * .pi / 180, to: chestOff * .pi / 180)) * 180 / .pi > 8,
                    "OFF: twisted torso → hip yaw \(hipOff) differs from chest yaw \(chestOff)")

            let cfg = TrackingConfig(); cfg.yawStabilizer = true
            let on = JointSolver(settings: cfg, yawStabilizer: YawStabilizer(smoothing: 0)).solve(pose)
            let hipOn = yawDeg(on.first { $0.type == .hip }!.rotation)
            let chestOn = yawDeg(on.first { $0.type == .chest }!.rotation)
            t.close(hipOn, chestOn, tol: 0.5, "ON: hip and chest share the coherent body yaw")
        }
    }
}
