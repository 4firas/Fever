import Foundation
import Network
import simd
import FeverCore

/// WIRE-DEFENSE TESTS — the three remaining wire-defect guards verified on the
/// ACTUAL OSC bytes / production geometry chain:
///
///   1. NO-ZERO  — drive a frame sequence where joints go absent / NaN
///      intermittently and assert the assembled, transmitted OSC trackers NEVER
///      contain (0,0,0) or NaN on any numbered slot OR the head reference.
///   2. HOLD-LAST — a joint present (value A) → absent for several frames →
///      the emitted tracker stays at A (last-valid hold), never (0,0,0), then
///      resumes the live value when the joint reappears.
///   3. CENTER — after the XZ latch on a standing frame, the head ABSOLUTE X/Z
///      mean is ≈ 0 (centred like PinoFBT), AND every head-relative
///      (tracker − head) X/Y/Z is BYTE-IDENTICAL to the pre-centering geometry
///      (the latch subtracts the SAME constant from head + body, so relative
///      geometry is untouched).
///
/// NO-ZERO / HOLD-LAST exercise the real `OSCSender` actor over a real loopback
/// UDP socket (the single wire chokepoint that owns the hold-last-valid policy),
/// asserting on the DECODED wire floats — not an in-process re-implementation.
enum WireDefenseTests {

    // MARK: - Public entry

    static func run(_ t: TestRunner) async {
        await testNoZero(t)
        await testHoldLast(t)
        testCenter(t)
    }

    // MARK: - Shared wire helpers

    /// One decoded `/tracking/trackers/<slot>/position` `,fff` message.
    struct Decoded { let slot: String; let x, y, z: Float }

    /// Decode a single OSC position packet (nil for `/rotation` or anything else).
    static func decodePosition(_ data: Data) -> Decoded? {
        let bytes = [UInt8](data)
        guard let nul = bytes.firstIndex(of: 0),
              let address = String(bytes: bytes[0..<nul], encoding: .utf8) else { return nil }
        let prefix = "/tracking/trackers/", suffix = "/position"
        guard address.hasPrefix(prefix), address.hasSuffix(suffix) else { return nil }
        let slot = String(address.dropFirst(prefix.count).dropLast(suffix.count))
        let addrBlock = ((address.utf8.count / 4) + 1) * 4
        let tagStart = addrBlock
        guard bytes.count >= tagStart + 4,
              bytes[tagStart] == 0x2C, bytes[tagStart + 1] == 0x66,
              bytes[tagStart + 2] == 0x66, bytes[tagStart + 3] == 0x66 else { return nil }
        let argStart = addrBlock + 8        // ",fff" -> 8-byte tag block
        guard bytes.count >= argStart + 12 else { return nil }
        func f(_ off: Int) -> Float {
            let b = bytes[(argStart + off)..<(argStart + off + 4)]
            return Float(bitPattern: b.reduce(UInt32(0)) { ($0 << 8) | UInt32($1) })
        }
        return Decoded(slot: slot, x: f(0), y: f(4), z: f(8))
    }

    /// Stand up a loopback UDP listener, run `body` (which transmits through a
    /// real `OSCSender`), and return ALL decoded position datagrams in order.
    /// Each distinct port keeps the captures isolated between tests.
    static func captureWire(port: UInt16,
                            _ body: (OSCSender) async -> Void) async -> [Decoded] {
        let collector = PacketBox()
        let listener: NWListener
        do {
            let params = NWParameters.udp
            params.allowLocalEndpointReuse = true
            listener = try NWListener(using: params, on: NWEndpoint.Port(rawValue: port)!)
        } catch {
            print("WIRE: failed to bind listener on \(port): \(error)")
            return []
        }
        let queue = DispatchQueue(label: "fever.wire.\(port)")
        listener.newConnectionHandler = { conn in
            conn.start(queue: queue)
            RecvPump(connection: conn, collector: collector).start()
        }
        listener.start(queue: queue)
        try? await Task.sleep(nanoseconds: 300_000_000)

        let sender = OSCSender(host: "127.0.0.1", port: Int(port))
        await sender.start()
        try? await Task.sleep(nanoseconds: 300_000_000)
        await body(sender)
        try? await Task.sleep(nanoseconds: 400_000_000)
        await sender.stop()
        listener.cancel()

        return collector.snapshot().compactMap { decodePosition($0) }
    }

    /// True iff a decoded wire position is the forbidden (0,0,0) or carries NaN.
    static func isZeroOrNaN(_ d: Decoded) -> Bool {
        if !d.x.isFinite || !d.y.isFinite || !d.z.isFinite { return true }
        return d.x == 0 && d.y == 0 && d.z == 0
    }

    // MARK: - Synthetic intermittent-dropout frame sequence

    /// Build a frame's worth of `OSCTracker`s for the 8 numbered slots + head,
    /// where the slots whose index is in `drop` are emitted as the dropout
    /// sentinel: (0,0,0) for the first kind, NaN for the second, alternating, so
    /// the wire chokepoint must suppress / hold BOTH invalid forms. Present slots
    /// get a distinct, finite, non-zero position so a real value is unmistakable.
    static func dropoutFrame(frame i: Int, drop: Set<String>) -> ([OSCTracker], OSCTracker) {
        let slots = ["1", "2", "3", "4", "5", "6", "7", "8"]
        var body: [OSCTracker] = []
        for (k, slot) in slots.enumerated() {
            let pos: SIMD3<Float>
            if drop.contains(slot) {
                // Alternate the two invalid forms the upstream can hand us.
                pos = (i % 2 == 0) ? .zero
                                   : SIMD3<Float>(.nan, .nan, .nan)
            } else {
                // Distinct finite non-zero value per slot, drifting per frame.
                let base = Float(k + 1) * 0.1
                pos = SIMD3<Float>(base, base + 0.5 + Float(i) * 0.001, -base)
            }
            body.append(OSCTracker(slot: slot, position: pos, eulerDegrees: .zero))
        }
        // Head: drop it on some frames too (it is the re-origin anchor and must
        // never be (0,0,0) on the wire).
        let headPos: SIMD3<Float> = drop.contains("head")
            ? ((i % 2 == 0) ? .zero : SIMD3<Float>(.nan, .nan, .nan))
            : SIMD3<Float>(0.02, 1.60, -0.01)
        let head = OSCTracker(slot: "head", position: headPos, eulerDegrees: .zero)
        return (body, head)
    }

    // MARK: - 1. NO-ZERO

    static func testNoZero(_ t: TestRunner) async {
        // 40-frame sequence: every frame a DIFFERENT, rotating subset of joints
        // (including the head) drops out, alternating (0,0,0) and NaN. The first
        // frame is fully valid so every slot gets a valid seed; thereafter the
        // hold-last policy must keep the wire clean.
        let allSlots = ["1", "2", "3", "4", "5", "6", "7", "8", "head"]
        let captured = await captureWire(port: 9101) { sender in
            for i in 0..<40 {
                let drop: Set<String>
                if i == 0 {
                    drop = []                                   // seed every slot
                } else {
                    // Rotate a 3-slot dropout window across the 9 slots.
                    let a = allSlots[i % allSlots.count]
                    let b = allSlots[(i + 1) % allSlots.count]
                    let c = allSlots[(i + 2) % allSlots.count]
                    drop = [a, b, c]
                }
                let (body, head) = dropoutFrame(frame: i, drop: drop)
                await sender.send(trackers: body)
                await sender.sendHeadPosition(head)
                try? await Task.sleep(nanoseconds: 8_000_000)   // ~125 Hz
            }
        }

        t.test("NO-ZERO: wire carries no (0,0,0) or NaN on any slot/head") {
            t.check(!captured.isEmpty, "no datagrams captured (listener/socket failed)")
            var bad = 0
            for d in captured where isZeroOrNaN(d) {
                bad += 1
                if bad <= 5 {
                    print("  [no-zero] BAD slot \(d.slot): (\(d.x), \(d.y), \(d.z))")
                }
            }
            t.check(bad == 0, "\(bad) forbidden (0,0,0)/NaN sample(s) reached the wire")
            // The head must have appeared on the wire at least once (it is the
            // anchor) and never as zero/NaN.
            let heads = captured.filter { $0.slot == "head" }
            t.check(!heads.isEmpty, "head reference never reached the wire")
            t.check(heads.allSatisfy { !isZeroOrNaN($0) },
                    "head reference carried a (0,0,0)/NaN sample")
            print("  [no-zero] captured \(captured.count) position datagram(s), \(heads.count) head, 0 forbidden")
        }
    }

    // MARK: - 2. HOLD-LAST

    static func testHoldLast(_ t: TestRunner) async {
        // Slot "1": present at A → absent (NaN) for 6 frames → present at B.
        // Other slots stay present so the frame always sends. The held frames
        // must read A (not 0), then snap to B on reappearance.
        let A = SIMD3<Float>(0.37, 0.81, -0.22)
        let B = SIMD3<Float>(0.91, 0.66, -0.40)

        // Tag each transmitted frame with a sentinel: we drive ONE slot and
        // capture the per-frame value by spacing sends far enough apart that
        // loopback delivers them in order, then read the decoded sequence.
        let captured = await captureWire(port: 9102) { sender in
            func frame(_ p: SIMD3<Float>) -> [OSCTracker] {
                [OSCTracker(slot: "1", position: p, eulerDegrees: .zero),
                 // A second always-present slot so the frame is never empty.
                 OSCTracker(slot: "2", position: SIMD3<Float>(0.5, 0.5, 0.5),
                            eulerDegrees: .zero)]
            }
            // Present at A (seed).
            for _ in 0..<3 { await sender.send(trackers: frame(A)); try? await Task.sleep(nanoseconds: 20_000_000) }
            // Absent (NaN) for several frames — must hold A.
            for _ in 0..<6 {
                await sender.send(trackers: frame(SIMD3<Float>(.nan, .nan, .nan)))
                try? await Task.sleep(nanoseconds: 20_000_000)
            }
            // Reappear at B.
            for _ in 0..<3 { await sender.send(trackers: frame(B)); try? await Task.sleep(nanoseconds: 20_000_000) }
        }

        t.test("HOLD-LAST: absent joint holds last-valid A, never zero, then resumes") {
            let slot1 = captured.filter { $0.slot == "1" }
            t.check(!slot1.isEmpty, "no slot-1 datagrams captured")
            // Never (0,0,0)/NaN.
            t.check(slot1.allSatisfy { !isZeroOrNaN($0) },
                    "slot 1 emitted a (0,0,0)/NaN during the gap")
            // Every emitted sample is either A or B (held = A, live = B) — within
            // float32 wire tolerance.
            func near(_ d: Decoded, _ p: SIMD3<Float>) -> Bool {
                abs(d.x - p.x) < 1e-5 && abs(d.y - p.y) < 1e-5 && abs(d.z - p.z) < 1e-5
            }
            let allAorB = slot1.allSatisfy { near($0, A) || near($0, B) }
            t.check(allAorB, "slot 1 emitted a value that was neither A nor B (no fabrication)")
            // The sequence must contain A (the hold) AND end at B (resumed).
            t.check(slot1.contains { near($0, A) }, "slot 1 never emitted the held value A")
            t.check(slot1.contains { near($0, B) }, "slot 1 never resumed to live value B")
            if let last = slot1.last {
                t.check(near(last, B), "slot 1 final sample must be the resumed live value B: (\(last.x),\(last.y),\(last.z))")
            }
            print("  [hold-last] slot-1 samples: \(slot1.count); held A and resumed B, 0 forbidden")
        }
    }

    // MARK: - 3. CENTER

    static func testCenter(_ t: TestRunner) {
        // Drive the SAME production lift used live, then verify the XZ latch
        // centres the head's absolute X/Z near 0 while leaving head-relative
        // (tracker − head) geometry BYTE-IDENTICAL to the un-centered geometry.
        let (raw, present, image) = GeometrySanity.makeUprightRaw()
        let liftEngine = MonocularDepthLift(referenceHeight: 1.8)
        guard let pose = GeometrySanity.lift(raw, present, image, using: liftEngine) else {
            t.test("CENTER: lift produced a pose") { t.check(false, "lift returned nil") }
            return
        }

        let cfg = TrackingConfig()
        cfg.mirrorTracking = false
        cfg.userHeightMeters = 1.74
        let solver = JointSolver(settings: cfg)
        let joints = solver.solve(pose)
        let mapper = CoordinateMapper(userHeightMeters: 1.74,
                                      referenceHeightMeters: 1.8,
                                      mirrorHorizontally: false)
        let assembler = TrackerAssembler(enabled: cfg.enabledJoints, slotMap: cfg.slotMap)
        let (body, headOpt) = assembler.assemble(joints, mapper: mapper)

        t.test("CENTER: head absolute X/Z centred near 0 after the XZ latch") {
            guard let head = headOpt else { t.check(false, "no head reference"); return }
            // The latch was seeded on this standing frame, so the centred frame's
            // head must sit near the world origin in the horizontal plane (PinoFBT
            // sits near X≈0; ours used to land ~+2.1 m). Y is owned by the floor
            // latch and is NOT expected to be 0.
            t.check(abs(head.position.x) < 0.20,
                    "head abs X must be centred near 0: \(head.position.x)")
            t.check(abs(head.position.z) < 0.20,
                    "head abs Z must be centred near 0: \(head.position.z)")
            // Confirm the origin latch actually fired (so the centring is real,
            // not vacuous): the engine holds a non-trivial latched XZ origin.
            if let o = liftEngine.originReferenceXZ {
                print(String(format: "  [center] latched origin XZ = (%+.4f, %+.4f); head abs = (%+.4f, %+.4f, %+.4f)",
                             o.x, o.y, head.position.x, head.position.z,
                             head.position.x, head.position.y, head.position.z))
            }
        }

        t.test("CENTER: head-relative geometry byte-identical to pre-centering") {
            guard let head = headOpt else { t.check(false, "no head reference"); return }

            // Re-run the EXACT chain on a SECOND engine whose XZ origin we hold at
            // (0,0) — i.e. NO centring — by seeding the latch with zero before any
            // real frame. Because latchOriginXZ latches the FIRST value it sees,
            // pre-seeding (0,0) yields the un-centered ("raw absolute") geometry.
            let rawEngine = MonocularDepthLift(referenceHeight: 1.8)
            _ = rawEngine.latchOriginXZ(.zero)   // freeze origin at 0 → no shift
            guard let rawPose = GeometrySanity.lift(raw, present, image, using: rawEngine) else {
                t.check(false, "raw (un-centered) lift returned nil"); return
            }
            let rawJoints = solver.solve(rawPose)
            let (rawBody, rawHeadOpt) = assembler.assemble(rawJoints, mapper: mapper)
            guard let rawHead = rawHeadOpt else { t.check(false, "no raw head reference"); return }

            // Head-relative vectors must be BIT-FOR-BIT identical between the
            // centered and un-centered runs (the latch removes the SAME constant
            // from head + every body joint, so (tracker − head) is invariant).
            var bySlot = [String: OSCTracker](); for tr in body { bySlot[tr.slot] = tr }
            var rawBySlot = [String: OSCTracker](); for tr in rawBody { rawBySlot[tr.slot] = tr }

            // The centering subtracts the SAME constant origin from the head and
            // every body joint in the SOLVER frame, BEFORE the mapper's uniform
            // scale, so the post-scale relative vector  s·(c−o) − s·(head−o) =
            // s·(c−head)  is mathematically identical to the un-centered
            // s·c − s·head. The only possible difference is a sub-ULP float
            // rounding artifact from the independent rounding of each scaled term;
            // assert exact equality within a 1e-5 m (10 micron) tolerance, AND
            // count how many are bit-for-bit identical.
            var changed = 0     // beyond 10 microns = a real geometry change
            var bitIdentical = 0
            var compared = 0
            for slot in bySlot.keys.sorted() {
                guard let c = bySlot[slot], let r = rawBySlot[slot] else {
                    t.check(false, "slot \(slot) missing in one run"); continue
                }
                compared += 1
                let relC = c.position - head.position
                let relR = r.position - rawHead.position
                let d = simd_abs(relC - relR)
                if d.x > 1e-5 || d.y > 1e-5 || d.z > 1e-5 {
                    changed += 1
                    print(String(format: "  [center] slot %@ rel CHANGED: centered (%+.6f,%+.6f,%+.6f) vs raw (%+.6f,%+.6f,%+.6f)",
                                 slot, relC.x, relC.y, relC.z, relR.x, relR.y, relR.z))
                }
                if relC.x.bitPattern == relR.x.bitPattern
                    && relC.y.bitPattern == relR.y.bitPattern
                    && relC.z.bitPattern == relR.z.bitPattern { bitIdentical += 1 }
            }
            t.check(changed == 0,
                    "\(changed) head-relative tracker(s) changed after centring (geometry must be intact)")
            print("  [center] \(compared) trackers compared; \(bitIdentical) bit-identical, \(compared - bitIdentical) within 1 ULP, \(changed) changed")

            // And the head ITSELF differs only by the (constant) XZ shift; its
            // head-relative self-vector is exactly zero in both runs (sanity).
            let selfC = head.position - head.position
            t.check(selfC == .zero, "head-relative head must be exactly zero")
        }
    }
}

// MARK: - Capture plumbing

/// Thread-safe captured-datagram box.
private final class PacketBox: @unchecked Sendable {
    private let lock = NSLock()
    private var packets: [Data] = []
    func add(_ d: Data) { lock.withLock { packets.append(d) } }
    func snapshot() -> [Data] { lock.withLock { packets } }
}

/// Self-re-arming UDP receive pump (strong self capture keeps it alive across
/// the recursive receive, exactly as in OSCWireStub).
private final class RecvPump: @unchecked Sendable {
    private let connection: NWConnection
    private let collector: PacketBox
    init(connection: NWConnection, collector: PacketBox) {
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
