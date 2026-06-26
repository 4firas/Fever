import Foundation
import simd
import CoreVideo
import FeverCore

@main
struct FeverCheck {

    static func main() async {
        // Headless benchmark mode: `swift run FeverCheck --bench [N]`. Runs the
        // live solve→encode chain N times and prints avg microseconds/frame.
        // Non-failing (always exits 0) so it is CI-safe.
        if CommandLine.arguments.contains("--bench") {
            let n = CommandLine.arguments.compactMap { Int($0) }.first ?? 2000
            await runBenchmark(iterations: n)
            exit(0)
        }

        // ROTATION STUB OSC WIRE: stream the live solved trackers with `/rotation`
        // ENABLED through the real OSCSender to `--host`/`--port` (default
        // 127.0.0.1:9000) and verify the wire contract (rotation present for slots
        // 1-8, no head/rotation, finite in-range euler, no (0,0,0) positions). Used
        // by the installed-binary stub-wire verification. `swift run FeverCheck
        // --rot-stub [--host H] [--port P]`.
        if CommandLine.arguments.contains("--rot-stub") {
            let ok = await RotationWireStub.runCLI()
            exit(ok ? 0 : 1)
        }

        // CROSS-LANGUAGE EQUIVALENCE HARNESS (on-device vs PC daemon). Reads a JSON file
        // of raw model joint sets, runs the on-device solve / smoother, and writes the
        // results as JSON so the Python daemon port can be diffed against it byte-for-byte.
        //   --solve-dump <in.json> <out.json>     in: [[[x,y,z]×24], …]  out: [{slots,head}, …]
        //   --smoother-dump <in.json> <out.json>  in: [{t,j:[[x,y,z]×24]}, …]  out: [{sm,vel}, …]
        if let i = CommandLine.arguments.firstIndex(of: "--solve-dump"),
           i + 2 < CommandLine.arguments.count {
            EquivalenceDump.solve(in: CommandLine.arguments[i + 1], out: CommandLine.arguments[i + 2])
            exit(0)
        }
        if let i = CommandLine.arguments.firstIndex(of: "--smoother-dump"),
           i + 2 < CommandLine.arguments.count {
            EquivalenceDump.smoother(in: CommandLine.arguments[i + 1], out: CommandLine.arguments[i + 2])
            exit(0)
        }
        if let i = CommandLine.arguments.firstIndex(of: "--upsampler-dump"),
           i + 2 < CommandLine.arguments.count {
            EquivalenceDump.upsampler(in: CommandLine.arguments[i + 1], out: CommandLine.arguments[i + 2])
            exit(0)
        }

        let t = TestRunner()

        // Inline unit tests (live path).
        testOSCMessage(t)
        testQuaternionEulerRoundTrip(t)
        testOneEuroFilter(t)
        NLFProtocolTests.run(t)
        SlotMapTests.run(t)

        // Suites (all exercise the live PinoSolver / OSC wire / config).
        await WireDefenseTests.run(t)
        await FinalizeWireTests.run(t)
        await RotationWireStub.run(t: t)
        MathTests.run(t)
        SpineKinematicsTests.run(t)
        PinoKinematicsTests.run(t)
        PredictiveUpsamplerTests.run(t)
        WireParityTests.run(t)
        ConfigPersistenceTests.run(t)
        PCOscRouteTests.run(t)
        PCModelLaunchTests.run(t)

        let summary = t.finalSummary()
        print(summary)
        if t.failed > 0 {
            FileHandle.standardError.write(Data("FAILED: \(t.failed) assertion(s) failed\n".utf8))
            exit(1)
        }
        exit(0)
    }

    // MARK: - 1. OSCMessage.encoded()

    static func testOSCMessage(_ t: TestRunner) {
        t.test("OSCMessage.encoded /position fff") {
            let msg = OSCMessage(address: "/tracking/trackers/1/position",
                                 arguments: [.float(0), .float(1), .float(0)])
            let data = msg.encoded()
            let bytes = [UInt8](data)

            // 4-byte aligned overall.
            t.check(bytes.count % 4 == 0, "packet length not 4-aligned: \(bytes.count)")

            // Address block: NUL-terminated and zero-padded to a 4-byte boundary.
            let addr = Array("/tracking/trackers/1/position".utf8)
            // Address length 29 → padded block must be 32 (next multiple of 4 that
            // leaves room for at least one NUL terminator).
            let addrBlockLen = ((addr.count / 4) + 1) * 4
            t.check(addrBlockLen == 32, "address block length expected 32, got \(addrBlockLen)")
            t.check(Array(bytes[0..<addr.count]) == addr, "address bytes mismatch")
            // Every byte from the address end through the block boundary is NUL.
            var addrPadAllZero = true
            for i in addr.count..<addrBlockLen where bytes[i] != 0 { addrPadAllZero = false }
            t.check(addrPadAllZero, "address padding not all NUL")
            t.check(bytes[addr.count] == 0, "address not NUL-terminated")

            // Type tag block: ",fff" then NUL pad to 4-byte boundary → 8 bytes.
            let tagStart = addrBlockLen
            let expectedTag: [UInt8] = Array(",fff".utf8) + [0, 0, 0, 0] // 4 + 4 pad = 8
            let tagBlock = Array(bytes[tagStart..<(tagStart + 8)])
            t.check(tagBlock == expectedTag, "type tag block mismatch: \(tagBlock)")

            // Three big-endian float32 args.
            let argStart = tagStart + 8
            let a0 = Array(bytes[argStart..<(argStart + 4)])
            let a1 = Array(bytes[(argStart + 4)..<(argStart + 8)])
            let a2 = Array(bytes[(argStart + 8)..<(argStart + 12)])
            t.check(a0 == [0x00, 0x00, 0x00, 0x00], "arg0 (0.0) bytes: \(a0)")
            t.check(a1 == [0x3F, 0x80, 0x00, 0x00], "arg1 (1.0) bytes: \(a1)")
            t.check(a2 == [0x00, 0x00, 0x00, 0x00], "arg2 (0.0) bytes: \(a2)")

            // Total = 32 + 8 + 12 = 52.
            t.check(bytes.count == 52, "total length expected 52, got \(bytes.count)")
        }

        t.test("OSCMessage.encoded int arg") {
            let msg = OSCMessage(address: "/x", arguments: [.int(1)])
            let bytes = [UInt8](msg.encoded())
            t.check(bytes.count % 4 == 0, "int packet not 4-aligned")
            // address "/x" (2) + NUL + pad = 4 ; tag ",i" + NUL + pad = 4 ; int 4.
            // big-endian int32(1) = 0x00000001
            let arg = Array(bytes.suffix(4))
            t.check(arg == [0x00, 0x00, 0x00, 0x01], "int32(1) big-endian: \(arg)")
        }
    }

    // MARK: - 2. quaternionToEulerZXYDegrees round-trip

    /// Build q = Ry(y)*Rx(x)*Rz(z) from Euler degrees (VRChat ZXY composition).
    static func zxyQuat(_ x: Float, _ y: Float, _ z: Float) -> simd_quatf {
        let d = Float.pi / 180
        let qx = simd_quatf(angle: x * d, axis: SIMD3<Float>(1, 0, 0))
        let qy = simd_quatf(angle: y * d, axis: SIMD3<Float>(0, 1, 0))
        let qz = simd_quatf(angle: z * d, axis: SIMD3<Float>(0, 0, 1))
        return qy * qx * qz
    }

    static func testQuaternionEulerRoundTrip(_ t: TestRunner) {
        let cases: [(Float, Float, Float)] = [
            (0, 0, 0),
            (0, 90, 0),     // pure +90 yaw
            (30, 0, 0),     // pure +30 about X (pitch)
            (0, 0, 30),     // pure +30 roll about Z
            (15, -40, 25),  // combo
        ]
        for (x, y, z) in cases {
            t.test("ZXY euler round-trip (\(x),\(y),\(z))") {
                let q = zxyQuat(x, y, z)
                let e = quaternionToEulerZXYDegrees(q)
                t.check(e.x.isFinite && e.y.isFinite && e.z.isFinite,
                        "euler produced NaN/inf: \(e)")
                let qBack = zxyQuat(e.x, e.y, e.z)
                // Compare rotations via |dot| (sign-agnostic), > 0.999.
                let dot = abs(simd_dot(q.vector, qBack.vector))
                t.check(dot > 0.999, "round-trip dot too low: \(dot) for euler \(e)")
            }
        }

        // Gimbal-lock guard: x ≈ +90° (sinX ≈ 1) must not NaN.
        t.test("ZXY euler gimbal lock no NaN") {
            let q = zxyQuat(90, 35, 0)
            let e = quaternionToEulerZXYDegrees(q)
            t.check(e.x.isFinite && e.y.isFinite && e.z.isFinite,
                    "gimbal-lock euler NaN/inf: \(e)")
            // x should be recovered near +90.
            t.close(e.x, 90, tol: 0.5, "gimbal-lock pitch")
            // Reconstruction should still reproduce the rotation.
            let qBack = zxyQuat(e.x, e.y, e.z)
            let dot = abs(simd_dot(q.vector, qBack.vector))
            t.check(dot > 0.999, "gimbal-lock round-trip dot: \(dot)")
        }
    }

    // MARK: - 3. OneEuroFilter

    static func testOneEuroFilter(_ t: TestRunner) {
        t.test("OneEuroFilter converges on constant") {
            var f = OneEuroFilter(minCutoff: 1.0, beta: 0.007)
            let value: Float = 3.5
            var out: Float = 0
            var time: TimeInterval = 0
            for _ in 0..<200 {
                out = f.filter(value, at: time)
                time += 1.0 / 60.0
                t.check(out.isFinite, "filter produced non-finite: \(out)")
            }
            t.close(out, value, tol: 1e-3, "constant convergence")
        }

        t.test("OneEuroFilter step moves monotonically toward target") {
            var f = OneEuroFilter(minCutoff: 1.0, beta: 0.007)
            var time: TimeInterval = 0
            // Prime at 0.
            _ = f.filter(0, at: time); time += 1.0 / 60.0
            for _ in 0..<10 { _ = f.filter(0, at: time); time += 1.0 / 60.0 }
            // Apply a step to 10 and verify each output increases toward it.
            var prev: Float = 0
            for i in 0..<60 {
                let out = f.filter(10, at: time); time += 1.0 / 60.0
                t.check(out.isFinite, "step filter non-finite: \(out)")
                t.check(out >= prev - 1e-5,
                        "step output not monotonic at \(i): \(out) < \(prev)")
                t.check(out <= 10 + 1e-3, "step overshoot: \(out)")
                prev = out
            }
            t.check(prev > 5, "step did not approach target, last=\(prev)")
        }
    }

    // MARK: - Benchmark (headless, non-failing) — live PinoSolver chain

    /// Runs the live per-frame solve→encode chain N times and reports avg µs/frame:
    /// StubNLFLandmarker standing pose → TwoEuroJointSmoother → PinoSolver →
    /// OSC encode (position + rotation for all 8 slots). Sidecar inference is
    /// intentionally excluded — this measures the CPU solve/encode budget that must
    /// fit comfortably inside the upsampled output tick (≤120 Hz → 8.3 ms).
    static func runBenchmark(iterations n: Int) async {
        let pose = StubNLFLandmarker.standing(timestamp: 0)
        let smoother = TwoEuroJointSmoother()
        let solver = PinoSolver(heightCm: 174)
        var sink = 0   // consume output so the chain isn't optimized away.

        func frame(_ tt: Double) {
            let sm = smoother.smooth(pose.joints3D, timestamp: tt)
            let solved = solver.solve(joints: sm, tracked: true)
            for slot in TrackerMapPino.slots {
                let p = solved.slotPositions[slot.index] ?? .zero
                let e = solved.slotEulers[slot.index] ?? .zero
                sink &+= encodeTracker(OSCTracker(slot: slot.path, position: p, eulerDegrees: e))
            }
        }

        for i in 0..<200 { frame(Double(i) / 30.0) }                 // warm up
        let start = DispatchTime.now().uptimeNanoseconds
        for i in 0..<n { frame(Double(i + 1000) / 30.0) }
        let end = DispatchTime.now().uptimeNanoseconds

        let totalNs = Double(end - start)
        let perFrameUs = (totalNs / Double(n)) / 1000.0
        let budgetPct = (perFrameUs / 1000.0) / 8.3 * 100.0

        print("BENCH: \(n) iterations, live solve→encode chain (sidecar inference excluded)")
        print(String(format: "BENCH: avg %.3f µs/frame  (%.4f%% of the 8.3 ms 120fps output tick)",
                     perFrameUs, budgetPct))
        print("BENCH: sink=\(sink)  (anti-DCE checksum)")
    }

    /// Encode one tracker's /position + /rotation OSC messages, returning the
    /// total encoded byte count (consumed as an anti-dead-code sink).
    static func encodeTracker(_ t: OSCTracker) -> Int {
        let pos = OSCMessage(address: "/tracking/trackers/\(t.slot)/position",
                             arguments: [.float(t.position.x),
                                         .float(t.position.y),
                                         .float(t.position.z)]).encoded()
        let rot = OSCMessage(address: "/tracking/trackers/\(t.slot)/rotation",
                             arguments: [.float(t.eulerDegrees.x),
                                         .float(t.eulerDegrees.y),
                                         .float(t.eulerDegrees.z)]).encoded()
        return pos.count + rot.count
    }
}
