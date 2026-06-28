import simd
import FeverCore

/// Locks the PredictiveUpsampler contract: on the first tick (seed) it is
/// byte-identical to the raw byte-exact PinoSolver (1:1 wire preserved); the
/// critically-damped follower EASES toward a changed pose (smooth, no snap) and
/// always CONVERGES to it (no missed movements); the optional forward `lead` leads
/// and is clamped (no fly-off) — the user's "silky, don't miss movements" goal.
enum PredictiveUpsamplerTests {

    static func run(_ t: TestRunner) {
        let joints = StubNLFLandmarker.standing(timestamp: 0).joints3D
        let zeroVel = [SIMD3<Float>](repeating: .zero, count: 24)
        let st: Float = 0.06, dt: Float = 1.0 / 120

        func rawBundle(_ tracked: Bool, _ elbows: Bool) -> (body: [OSCTracker], head: OSCTracker) {
            assemblePinoBundle(PinoSolver(heightCm: 174).solve(joints: joints, tracked: tracked),
                               sendElbows: elbows)
        }
        func chestY(_ r: (body: [OSCTracker], head: OSCTracker?)) -> Float {
            r.body.first { $0.slot == "1" }?.position.y ?? .nan
        }

        // 1) First tick (seed) ⇒ byte-identical to the raw solver (1:1 wire preserved).
        t.test("PredictiveUpsampler seed tick == raw PinoSolver (byte-identical)") {
            let up = PredictiveUpsampler(heightCm: 174)
            let (rawBody, rawHead) = rawBundle(true, true)
            let (body, head) = up.step(joints: joints, velocity: zeroVel, smoothTime: st, dt: dt,
                                       lead: 0, tracked: true, heightCm: 174, sendElbows: true)
            t.check(body.count == rawBody.count, "slot count matches (\(body.count))")
            for (a, b) in zip(body, rawBody) {
                t.check(a.slot == b.slot, "slot order \(a.slot)")
                t.check(simd_length(a.position - b.position) < 1e-6, "pos identical \(a.slot)")
                t.check(simd_length(a.eulerDegrees - b.eulerDegrees) < 1e-4, "euler identical \(a.slot)")
            }
            t.check(simd_length((head?.position ?? .zero) - rawHead.position) < 1e-6, "head pos identical")
        }

        // 2) lead > 0 but zero velocity ⇒ no extrapolation (seed == raw).
        t.test("PredictiveUpsampler zero velocity (lead>0) == raw") {
            let up = PredictiveUpsampler(heightCm: 174)
            let (rawBody, _) = rawBundle(true, false)
            let (body, _) = up.step(joints: joints, velocity: zeroVel, smoothTime: st, dt: dt,
                                    lead: 0.08, tracked: true, heightCm: 174, sendElbows: false)
            for (a, b) in zip(body, rawBody) {
                t.check(simd_length(a.position - b.position) < 1e-6, "zero-vel pos == raw \(a.slot)")
            }
        }

        // 3) tracked = false ⇒ no extrapolation even with velocity + lead.
        t.test("PredictiveUpsampler untracked ⇒ no extrapolation") {
            let up = PredictiveUpsampler(heightCm: 174)
            let vel = [SIMD3<Float>](repeating: SIMD3(1, 0, 0), count: 24)
            let (rawBody, _) = rawBundle(false, false)
            let (body, _) = up.step(joints: joints, velocity: vel, smoothTime: st, dt: dt,
                                    lead: 0.08, tracked: false, heightCm: 174, sendElbows: false)
            for (a, b) in zip(body, rawBody) {
                t.check(simd_length(a.position - b.position) < 1e-6, "untracked == raw \(a.slot)")
            }
        }

        // 4) a velocity SPIKE through `lead` is clamped — finite + bounded (no fly-off).
        t.test("PredictiveUpsampler clamps a velocity spike (no fly-off)") {
            let up = PredictiveUpsampler(heightCm: 174)
            let spike = [SIMD3<Float>](repeating: SIMD3(100, 100, 100), count: 24)  // absurd m/s
            let (body, head) = up.step(joints: joints, velocity: spike, smoothTime: st, dt: dt,
                                       lead: 0.08, tracked: true, heightCm: 174, sendElbows: true)
            for tr in body {
                t.check(tr.position.x.isFinite && tr.position.y.isFinite && tr.position.z.isFinite,
                        "finite \(tr.slot)")
                t.check(simd_length(tr.position) < 50, "bounded, no fly-off \(tr.slot): \(tr.position)")
            }
            t.check(head != nil, "head present")
        }

        // 5) with lead > 0 + differential velocity it LEADS (output moves vs raw).
        t.test("PredictiveUpsampler lead>0 with velocity leads (differs from raw)") {
            let up = PredictiveUpsampler(heightCm: 174)
            var vel = [SIMD3<Float>](repeating: SIMD3(0, 0.4, 0), count: 24)
            vel[SMPLJoint.pelvis.rawValue] = .zero
            let (rawBody, _) = rawBundle(true, false)
            let (body, _) = up.step(joints: joints, velocity: vel, smoothTime: st, dt: dt,
                                    lead: 0.08, tracked: true, heightCm: 174, sendElbows: false)
            let moved = zip(body, rawBody).contains { simd_length($0.0.position - $0.1.position) > 1e-4 }
            t.check(moved, "differential velocity + lead must change the output")
        }

        // 6) the damped follower EASES toward a changed pose (no snap) then CONVERGES
        // to it (no permanent lag, no missed movement). Seed at rest, then feed a
        // clearly different pose: first tick only PARTWAY, settles AT the target.
        t.test("PredictiveUpsampler follower eases toward a changed pose then converges") {
            let damped = PredictiveUpsampler(heightCm: 174)
            let rest = chestY(damped.step(joints: joints, velocity: zeroVel, smoothTime: st, dt: dt,
                                          lead: 0, tracked: true, heightCm: 174, sendElbows: false))
            var shifted = joints
            for i in 0..<24 where i != SMPLJoint.pelvis.rawValue { shifted[i].y -= 0.25 }
            let target = chestY(assemblePinoBundle(PinoSolver(heightCm: 174).solve(joints: shifted, tracked: true),
                                                   sendElbows: false))
            let first = chestY(damped.step(joints: shifted, velocity: zeroVel, smoothTime: st, dt: dt,
                                           lead: 0, tracked: true, heightCm: 174, sendElbows: false))
            guard abs(target - rest) > 1e-3 else { t.check(true, "no measurable chest delta"); return }
            t.check(abs(first - rest) < abs(target - rest),
                    "first tick EASES (partway), not a snap: rest=\(rest) first=\(first) target=\(target)")
            var last = first
            for _ in 0..<240 {
                last = chestY(damped.step(joints: shifted, velocity: zeroVel, smoothTime: st, dt: dt,
                                          lead: 0, tracked: true, heightCm: 174, sendElbows: false))
            }
            t.check(abs(last - target) < 0.02, "follower CONVERGES to target (no missed movement): \(last) vs \(target)")
        }

        // 7) assemblePinoBundle slot filtering (shared by UI snapshot + the wire).
        t.test("assemblePinoBundle: 6-point excludes elbows, 8-point includes, head always present") {
            let solved = PinoSolver(heightCm: 174).solve(joints: joints, tracked: true)
            let (body6, head6) = assemblePinoBundle(solved, sendElbows: false)
            let (body8, head8) = assemblePinoBundle(solved, sendElbows: true)
            t.check(body6.count == 6, "6-point body has 6 slots, got \(body6.count)")
            t.check(body8.count == 8, "8-point body has 8 slots, got \(body8.count)")
            t.check(!body6.contains { $0.slot == "3" || $0.slot == "4" }, "6-point excludes elbow slots 3/4")
            t.check(body8.contains { $0.slot == "3" } && body8.contains { $0.slot == "4" },
                    "8-point includes elbow slots 3/4")
            t.check(head6.eulerDegrees == .zero && head8.eulerDegrees == .zero, "head euler is zero")
        }

        // 8) reset() re-seeds: after converging on one pose, reset() makes the NEXT tick
        //    byte-identical to the raw solver again (drops the eased dampJoints state).
        //    This is the contract the mid-session mirror-flip reset relies on.
        t.test("PredictiveUpsampler reset() re-seeds (post-reset tick == raw, no eased state carried)") {
            let up = PredictiveUpsampler(heightCm: 174)
            let shifted = joints.map { $0 + SIMD3<Float>(0.2, -0.1, 0.05) }
            for _ in 0..<120 {   // converge the follower onto the shifted pose (holds eased state)
                _ = up.step(joints: shifted, velocity: zeroVel, smoothTime: st, dt: dt,
                            lead: 0, tracked: true, heightCm: 174, sendElbows: true)
            }
            up.reset()
            let (rawBody, rawHead) = rawBundle(true, true)
            let (body, head) = up.step(joints: joints, velocity: zeroVel, smoothTime: st, dt: dt,
                                       lead: 0, tracked: true, heightCm: 174, sendElbows: true)
            t.check(body.count == rawBody.count, "slot count matches post-reset")
            for (a, b) in zip(body, rawBody) {
                t.check(simd_length(a.position - b.position) < 1e-6, "post-reset pos == raw \(a.slot)")
                t.check(simd_length(a.eulerDegrees - b.eulerDegrees) < 1e-4, "post-reset euler == raw \(a.slot)")
            }
            t.check(simd_length((head?.position ?? .zero) - rawHead.position) < 1e-6, "post-reset head == raw")
        }
    }
}
