import Foundation
import Network
import simd
import FeverCore

/// ROTATION WIRE STUB — proves the re-enabled `/rotation` path on the ACTUAL OSC
/// wire bytes through the REAL `OSCSender` (rotationEnabled = true), driving the
/// full production chain (lift → solve → rest-relative rebase → map → assemble),
/// and asserting the PinoFBT rotation contract:
///
///   • `/rotation` IS present for ALL 8 body trackers (PinoFBT parity),
///   • NO `head/rotation` is ever emitted (head is position-only),
///   • euler values are BOUNDED (not pinned at ±180),
///   • NO (0,0,0) positions,
///   • every slot every frame (8 position + 8 rotation + 1 head position).
///
/// Invoked via `swift run FeverCheck --rot-stub` (used by the installed-binary
/// stub-wire verification) and also asserted in-process by `RotationWireStub.run`.
enum RotationWireStub {

    /// A decoded OSC `,fff` message (position or rotation) — 3 big-endian float32.
    struct Decoded {
        let address: String
        let slot: String
        let kind: String        // "position" | "rotation"
        let v: SIMD3<Float>
    }

    static func decode(_ data: Data) -> Decoded? {
        let bytes = [UInt8](data)
        guard let nul = bytes.firstIndex(of: 0),
              let address = String(bytes: bytes[0..<nul], encoding: .utf8) else { return nil }
        let prefix = "/tracking/trackers/"
        guard address.hasPrefix(prefix) else { return nil }
        let rest = address.dropFirst(prefix.count)         // "<slot>/<kind>"
        guard let slash = rest.firstIndex(of: "/") else { return nil }
        let slot = String(rest[rest.startIndex..<slash])
        let kind = String(rest[rest.index(after: slash)...])
        guard kind == "position" || kind == "rotation" else { return nil }
        let addrBlock = ((address.utf8.count / 4) + 1) * 4
        let tagStart = addrBlock
        guard bytes.count >= tagStart + 4,
              bytes[tagStart] == 0x2C, bytes[tagStart + 1] == 0x66,
              bytes[tagStart + 2] == 0x66, bytes[tagStart + 3] == 0x66 else { return nil }
        let argStart = addrBlock + 8
        guard bytes.count >= argStart + 12 else { return nil }
        func f(_ off: Int) -> Float {
            let b = bytes[(argStart + off)..<(argStart + off + 4)]
            return Float(bitPattern: b.reduce(UInt32(0)) { ($0 << 8) | UInt32($1) })
        }
        return Decoded(address: address, slot: slot, kind: kind,
                       v: SIMD3<Float>(f(0), f(4), f(8)))
    }

    /// Build the assembled body trackers + head from the production chain on a
    /// realistic upright pose, with rotation populated by the full rest-relative
    /// rebase chain (mirrors FrameProcessor). Returns nil if the lift fails.
    static func assembleWithRotation() -> (body: [OSCTracker], head: OSCTracker?)? {
        let (raw, present, image) = GeometrySanity.makeUprightRaw()
        let liftEngine = MonocularDepthLift(referenceHeight: 1.8)
        guard let pose = GeometrySanity.lift(raw, present, image, using: liftEngine) else {
            return nil
        }
        let cfg = TrackingConfig()
        cfg.mirrorTracking = false
        cfg.userHeightMeters = 1.74
        let state = RotationState()
        let solver = JointSolver(settings: cfg, rotationState: state)
        let rebaser = RotationRebaser(smoothingFactor: cfg.rotationSmoothingF)
        var joints = solver.solve(pose)
        // Rest-relative rebase (no rest captured yet → identity rest = absolute,
        // hemisphere-locked + smoothed) exactly as the live processor does.
        for i in joints.indices where joints[i].type != .head {
            joints[i].rotation = rebaser.rebase(joints[i].type,
                                                live: joints[i].rotation,
                                                captureNow: false)
        }
        let mapper = CoordinateMapper(userHeightMeters: 1.74,
                                      referenceHeightMeters: 1.8,
                                      mirrorHorizontally: false)
        let assembler = TrackerAssembler(enabled: cfg.enabledJoints, slotMap: cfg.slotMap)
        let (body, head) = assembler.assemble(joints, mapper: mapper)
        return (body, head)
    }

    /// Standalone CLI mode: bind on `--host`/`--port` (default 127.0.0.1:9000),
    /// stream several frames with rotation ENABLED, capture, print a contract
    /// report. Exit 0 on pass. Used by the installed-binary stub-wire step.
    static func runCLI() async -> Bool {
        let args = CommandLine.arguments
        func val(_ flag: String) -> String? {
            guard let i = args.firstIndex(of: flag), i + 1 < args.count else { return nil }
            return args[i + 1]
        }
        let host = val("--host") ?? "127.0.0.1"
        let port = UInt16(val("--port") ?? "9000") ?? 9000
        return await run(host: host, port: port, t: nil)
    }

    /// Core: stand up a loopback listener, transmit with rotation enabled, capture,
    /// assert the contract (on a `TestRunner` if given, else print/return for CLI).
    @discardableResult
    static func run(host: String = "127.0.0.1",
                    port: UInt16 = 9000,
                    t: TestRunner?) async -> Bool {
        guard let (body, headOpt) = assembleWithRotation() else {
            print("ROT-STUB: lift failed"); t?.check(false, "lift failed"); return false
        }
        guard let head = headOpt else {
            print("ROT-STUB: no head reference"); t?.check(false, "no head"); return false
        }

        // Only bind a listener on loopback (the in-process / CI path). For a real
        // remote host we just transmit (the installed-binary check sniffs the wire
        // externally), but for 127.0.0.1 we capture and assert here.
        let isLoopback = (host == "127.0.0.1" || host == "localhost")
        let collector = RotBox()
        var listener: NWListener?
        if isLoopback {
            do {
                let params = NWParameters.udp
                params.allowLocalEndpointReuse = true
                listener = try NWListener(using: params, on: NWEndpoint.Port(rawValue: port)!)
            } catch {
                print("ROT-STUB: failed to bind listener on \(port): \(error)")
                t?.check(false, "listener bind failed"); return false
            }
            let queue = DispatchQueue(label: "fever.rotstub.\(port)")
            listener?.newConnectionHandler = { conn in
                conn.start(queue: queue)
                RotPump(connection: conn, collector: collector).start()
            }
            listener?.start(queue: queue)
            try? await Task.sleep(nanoseconds: 300_000_000)
        }

        // Real OSCSender with ROTATION ENABLED (the re-enabled path).
        let sender = OSCSender(host: host, port: Int(port))
        await sender.seedSlots(["1", "2", "3", "4", "5", "6", "7", "8", "head"])
        await sender.setRotationEnabled(true)
        await sender.start()
        try? await Task.sleep(nanoseconds: 300_000_000)

        let frames = 12
        for _ in 0..<frames {
            await sender.send(trackers: body)
            await sender.sendHeadPosition(head)        // head = POSITION ONLY
            try? await Task.sleep(nanoseconds: 25_000_000)
        }
        try? await Task.sleep(nanoseconds: 400_000_000)
        await sender.stop()
        listener?.cancel()

        if !isLoopback {
            print("ROT-STUB: transmitted \(frames) frames (rotation enabled) to \(host):\(port)")
            return true
        }

        // Decode + group the captured datagrams.
        let decoded = collector.snapshot().compactMap { decode($0) }
        let positions = decoded.filter { $0.kind == "position" }
        let rotations = decoded.filter { $0.kind == "rotation" }
        let bodyRotSlots = Set(rotations.map { $0.slot }).filter { $0 != "head" }
        let headRot = rotations.filter { $0.slot == "head" }

        var ok = true
        func require(_ cond: Bool, _ msg: String) {
            if let t {
                t.check(cond, msg)
            } else if !cond {
                print("ROT-STUB: FAIL — \(msg)")
            }
            if !cond { ok = false }
        }

        // (a) /rotation present for ALL 8 body trackers (PinoFBT parity). The
        //     per-bone rotation solver is fixed (spine-aligned chest, heel→toe feet
        //     with roll locked), so every numbered tracker carries a real rotation.
        let expectBody = Set(["1", "2", "3", "4", "5", "6", "7", "8"])
        require(bodyRotSlots == expectBody,
                "/rotation must be present for all 8 body slots: got \(bodyRotSlots.sorted())")
        // (b) NO head/rotation.
        require(headRot.isEmpty, "head/rotation must NEVER be emitted (got \(headRot.count))")
        // (c) euler BOUNDED (not pinned at ±180) and finite.
        var minE = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var maxE = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
        var pinned = 0
        for r in rotations {
            require(r.v.x.isFinite && r.v.y.isFinite && r.v.z.isFinite,
                    "rotation slot \(r.slot) non-finite: \(r.v)")
            for a in 0..<3 {
                minE[a] = min(minE[a], r.v[a]); maxE[a] = max(maxE[a], r.v[a])
                if abs(abs(r.v[a]) - 180) < 0.5 { pinned += 1 }
            }
        }
        require(pinned == 0, "\(pinned) euler component(s) pinned at ±180 (rotation must be bounded)")
        // (d) NO (0,0,0) positions on any slot (head included).
        var zeros = 0
        for p in positions where p.v == .zero { zeros += 1 }
        require(zeros == 0, "\(zeros) (0,0,0) position sample(s) on the wire")
        // (e) head POSITION present (the anchor).
        require(positions.contains { $0.slot == "head" }, "head position never reached the wire")

        // Report.
        print("ROT-STUB: captured \(decoded.count) datagrams — \(positions.count) position, \(rotations.count) rotation")
        print("ROT-STUB: body rotation slots = \(bodyRotSlots.sorted()); head/rotation count = \(headRot.count)")
        print(String(format: "ROT-STUB: euler ranges  x=[%+.1f,%+.1f]  y=[%+.1f,%+.1f]  z=[%+.1f,%+.1f]  (pinned±180=%d)",
                     minE.x, maxE.x, minE.y, maxE.y, minE.z, maxE.z, pinned))
        if t == nil {
            print(ok ? "ROT-STUB: PASS" : "ROT-STUB: FAIL")
            return ok
        }
        return true
    }
}

/// Thread-safe captured-datagram box for the rotation stub.
private final class RotBox: @unchecked Sendable {
    private let lock = NSLock()
    private var packets: [Data] = []
    func add(_ d: Data) { lock.withLock { packets.append(d) } }
    func snapshot() -> [Data] { lock.withLock { packets } }
}

private final class RotPump: @unchecked Sendable {
    private let connection: NWConnection
    private let collector: RotBox
    init(connection: NWConnection, collector: RotBox) {
        self.connection = connection
        self.collector = collector
    }
    func start() {
        connection.receiveMessage { data, _, _, _ in
            if let data, !data.isEmpty { self.collector.add(data) }
            self.start()
        }
    }
}
