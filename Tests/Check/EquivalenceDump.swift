import Foundation
import simd
import FeverCore

/// Cross-language equivalence harness: dumps the on-device (Swift) solve / smoother
/// output as JSON so the PC daemon's pure-Python port (`pc-daemon/fbt_server.py`) can be
/// diffed against the reference, proving PC-mode output equals on-device-mode output.
///
/// Driven from `FeverCheck --solve-dump` / `--smoother-dump`. Pure I/O + the live
/// `PinoSolver` / `TwoEuroJointSmoother` — no new math, so it can't drift from the wire.
enum EquivalenceDump {

    private static func readPoses(_ path: String) -> [[SIMD3<Float>]] {
        guard let data = FileManager.default.contents(atPath: path),
              let arr = (try? JSONSerialization.jsonObject(with: data)) as? [[[Double]]] else { return [] }
        return arr.map { pose in pose.map { SIMD3<Float>(Float($0[0]), Float($0[1]), Float($0[2])) } }
    }

    private static func write(_ obj: Any, to path: String) {
        let data = (try? JSONSerialization.data(withJSONObject: obj)) ?? Data()
        try? data.write(to: URL(fileURLWithPath: path))
    }

    /// Run `PinoSolver.solve` (height 175 → ratio 1.0) on each pose and emit per-slot
    /// position + ZXY euler + the head anchor.
    static func solve(in inPath: String, out outPath: String) {
        let solver = PinoSolver(heightCm: 175)
        var rows: [[String: Any]] = []
        for joints in readPoses(inPath) where joints.count == 24 {
            solver.reset()   // clear the elbow hold-last so each pose is independent
            let f = solver.solve(joints: joints, tracked: true)
            var slots: [String: Any] = [:]
            for n in 1...8 {
                let p = f.slotPositions[n] ?? .zero
                let e = f.slotEulers[n] ?? .zero
                slots["\(n)"] = ["pos": [Double(p.x), Double(p.y), Double(p.z)],
                                 "rot": [Double(e.x), Double(e.y), Double(e.z)]]
            }
            let h = f.headPosition
            rows.append(["slots": slots, "head": [Double(h.x), Double(h.y), Double(h.z)]])
        }
        write(rows, to: outPath)
    }

    /// Run `PredictiveUpsampler` over a stateful sequence of ticks (each carries its own
    /// target joints, velocity, smoothTime, dt, lead, tracked) and emit the resulting
    /// per-tick slot-1 (chest) position, slot-3 (L-elbow) euler, and head position. This
    /// exercises the critically-damped SmoothDamp follower + lead clamp + re-solve as a
    /// stateful whole, so the Python `Upsampler` port can be diffed end-to-end.
    static func upsampler(in inPath: String, out outPath: String) {
        guard let data = FileManager.default.contents(atPath: inPath),
              let seq = (try? JSONSerialization.jsonObject(with: data)) as? [[String: Any]] else {
            write([], to: outPath); return
        }
        let up = PredictiveUpsampler(heightCm: 175)
        var rows: [[String: Any]] = []
        for tick in seq {
            guard let jraw = tick["j"] as? [[Double]], let vraw = tick["vel"] as? [[Double]] else { continue }
            let joints = jraw.map { SIMD3<Float>(Float($0[0]), Float($0[1]), Float($0[2])) }
            let vel = vraw.map { SIMD3<Float>(Float($0[0]), Float($0[1]), Float($0[2])) }
            let st = Float((tick["smoothTime"] as? Double) ?? 0.07)
            let dt = Float((tick["dt"] as? Double) ?? (1.0 / 120))
            let lead = Float((tick["lead"] as? Double) ?? 0)
            let tracked = (tick["tracked"] as? Bool) ?? true
            let (body, head) = up.step(joints: joints, velocity: vel, smoothTime: st, dt: dt,
                                       lead: lead, tracked: tracked, heightCm: 175, sendElbows: true)
            let chest = body.first { $0.slot == "1" }?.position ?? .zero
            let elbow = body.first { $0.slot == "3" }?.eulerDegrees ?? .zero
            let h = head?.position ?? .zero
            rows.append(["chest": [Double(chest.x), Double(chest.y), Double(chest.z)],
                         "elbow": [Double(elbow.x), Double(elbow.y), Double(elbow.z)],
                         "head": [Double(h.x), Double(h.y), Double(h.z)]])
        }
        write(rows, to: outPath)
    }

    /// Run `TwoEuroJointSmoother` over a timestamped sequence and emit the smoothed
    /// joints + the filtered velocity per frame.
    static func smoother(in inPath: String, out outPath: String) {
        guard let data = FileManager.default.contents(atPath: inPath),
              let seq = (try? JSONSerialization.jsonObject(with: data)) as? [[String: Any]] else {
            write([], to: outPath); return
        }
        let sm = TwoEuroJointSmoother()
        var rows: [[String: Any]] = []
        for frame in seq {
            guard let t = frame["t"] as? Double, let jraw = frame["j"] as? [[Double]] else { continue }
            let joints = jraw.map { SIMD3<Float>(Float($0[0]), Float($0[1]), Float($0[2])) }
            let out = sm.smooth(joints, timestamp: t)
            let vel = sm.velocity()
            rows.append(["sm": out.map { [Double($0.x), Double($0.y), Double($0.z)] },
                         "vel": vel.map { [Double($0.x), Double($0.y), Double($0.z)] }])
        }
        write(rows, to: outPath)
    }
}
