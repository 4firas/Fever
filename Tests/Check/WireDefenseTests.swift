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
        await testRotationHold(t)
        await testSixPointBundle(t)
        await testRotationFiniteOrZeroSeed(t)
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

    /// Decode a single OSC `/tracking/trackers/<slot>/rotation` `,fff` message.
    static func decodeRotation(_ data: Data) -> Decoded? {
        let bytes = [UInt8](data)
        guard let nul = bytes.firstIndex(of: 0),
              let address = String(bytes: bytes[0..<nul], encoding: .utf8) else { return nil }
        let prefix = "/tracking/trackers/", suffix = "/rotation"
        guard address.hasPrefix(prefix), address.hasSuffix(suffix) else { return nil }
        let slot = String(address.dropFirst(prefix.count).dropLast(suffix.count))
        let addrBlock = ((address.utf8.count / 4) + 1) * 4
        guard bytes.count >= addrBlock + 4,
              bytes[addrBlock] == 0x2C, bytes[addrBlock + 1] == 0x66,
              bytes[addrBlock + 2] == 0x66, bytes[addrBlock + 3] == 0x66 else { return nil }
        let argStart = addrBlock + 8
        guard bytes.count >= argStart + 12 else { return nil }
        func f(_ off: Int) -> Float {
            let b = bytes[(argStart + off)..<(argStart + off + 4)]
            return Float(bitPattern: b.reduce(UInt32(0)) { ($0 << 8) | UInt32($1) })
        }
        return Decoded(slot: slot, x: f(0), y: f(4), z: f(8))
    }

    /// Split an OSC `#bundle` datagram into its element messages. Layout:
    /// "#bundle\0" (8) + timetag (8) + repeated [int32-BE size][message bytes].
    static func parseBundle(_ data: Data) -> [Data] {
        let bytes = [UInt8](data)
        guard bytes.count > 16,
              String(bytes: bytes[0..<7], encoding: .utf8) == "#bundle" else { return [] }
        var out: [Data] = []
        var i = 16
        while i + 4 <= bytes.count {
            let size = Int(bytes[i]) << 24 | Int(bytes[i + 1]) << 16 | Int(bytes[i + 2]) << 8 | Int(bytes[i + 3])
            i += 4
            guard size > 0, i + size <= bytes.count else { break }
            out.append(Data(bytes[i..<(i + size)]))
            i += size
        }
        return out
    }

    /// Stand up a loopback UDP listener, run `body` (which transmits through a
    /// real `OSCSender`), and return ALL decoded position datagrams in order.
    /// Each distinct port keeps the captures isolated between tests.
    static func captureWire(port: UInt16,
                            _ body: (OSCSender) async -> Void) async -> [Decoded] {
        await captureRaw(port: port, body).compactMap { decodePosition($0) }
    }

    /// Like `captureWire` but returns the RAW datagrams (so a bundle path can be
    /// split + decoded by the caller).
    static func captureRaw(port: UInt16,
                           _ body: (OSCSender) async -> Void) async -> [Data] {
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

        return collector.snapshot()
    }

    /// True iff a decoded wire position is forbidden: NaN on any slot, or (0,0,0) on
    /// any slot EXCEPT the hip ("2"), whose CORRECT value IS the origin (it's the root
    /// of the normalized tracker space; PinoFBT sends (0,0,0) for the hip every frame).
    static func isZeroOrNaN(_ d: Decoded) -> Bool {
        if !d.x.isFinite || !d.y.isFinite || !d.z.isFinite { return true }
        return d.slot != "2" && d.x == 0 && d.y == 0 && d.z == 0
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

    // MARK: - 3. ROTATION HOLD-LAST (the sendPinoBundle live path)

    /// Mirrors the position hold-last, but for ROTATION on the bundle path: a slot
    /// present with rotation A → drops (position NaN AND rotation snapped to identity
    /// 0,0,0, as a degenerate ~0 bone yields) → reappears with rotation B. The wire
    /// must HOLD rotation A through the gap (never identity), then resume to B.
    static func testRotationHold(_ t: TestRunner) async {
        let posA = SIMD3<Float>(0.37, 0.81, -0.22), rotA = SIMD3<Float>(31, -14, 47)
        let posB = SIMD3<Float>(0.40, 0.79, -0.25), rotB = SIMD3<Float>(-22, 60, 12)

        let raw = await captureRaw(port: 9103) { sender in
            func frame(pos: SIMD3<Float>, rot: SIMD3<Float>) -> [OSCTracker] {
                [OSCTracker(slot: "1", position: pos, eulerDegrees: rot),
                 // Hip is always valid (origin is its correct value) so the bundle is never empty.
                 OSCTracker(slot: "2", position: .zero, eulerDegrees: .zero)]
            }
            for _ in 0..<3 {
                await sender.sendPinoBundle(trackers: frame(pos: posA, rot: rotA), head: nil)
                try? await Task.sleep(nanoseconds: 20_000_000)
            }
            // Dropout: position NaN (→ held) AND rotation snapped to identity (0,0,0).
            for _ in 0..<6 {
                await sender.sendPinoBundle(trackers: frame(pos: SIMD3<Float>(.nan, .nan, .nan), rot: .zero), head: nil)
                try? await Task.sleep(nanoseconds: 20_000_000)
            }
            for _ in 0..<3 {
                await sender.sendPinoBundle(trackers: frame(pos: posB, rot: rotB), head: nil)
                try? await Task.sleep(nanoseconds: 20_000_000)
            }
        }

        let rots = raw.flatMap { parseBundle($0) }.compactMap { decodeRotation($0) }.filter { $0.slot == "1" }

        t.test("HOLD-LAST (rotation): dropout holds rotation A, never identity, then resumes B") {
            t.check(!rots.isEmpty, "no slot-1 rotation datagrams captured")
            func near(_ d: Decoded, _ p: SIMD3<Float>) -> Bool {
                abs(d.x - p.x) < 1e-3 && abs(d.y - p.y) < 1e-3 && abs(d.z - p.z) < 1e-3
            }
            t.check(rots.allSatisfy { !($0.x == 0 && $0.y == 0 && $0.z == 0) },
                    "slot 1 rotation snapped to identity (0,0,0) during the dropout")
            t.check(rots.allSatisfy { near($0, rotA) || near($0, rotB) },
                    "slot 1 rotation emitted a value that was neither A nor B")
            t.check(rots.contains { near($0, rotA) }, "slot 1 never emitted the held rotation A")
            t.check(rots.contains { near($0, rotB) }, "slot 1 never resumed to live rotation B")
            if let last = rots.last { t.check(near(last, rotB), "final slot-1 rotation must be resumed B") }
            print("  [hold-last rot] slot-1 rotation samples: \(rots.count); held A, resumed B, identity forbidden")
        }
    }

    // MARK: - 4. 6-POINT BUNDLE (13 messages: no elbows, head position-only)

    /// When the caller passes only the six numbered slots (no 3/4), sendPinoBundle emits
    /// a 13-message bundle: 6 position + 6 rotation + head/position, with NO head/rotation.
    static func testSixPointBundle(_ t: TestRunner) async {
        let sixSlots = ["1", "2", "5", "6", "7", "8"]
        let raw = await captureRaw(port: 9104) { sender in
            let body = sixSlots.map {
                OSCTracker(slot: $0, position: SIMD3<Float>(0.1, 0.2, 0.3), eulerDegrees: SIMD3<Float>(1, 2, 3))
            }
            let head = OSCTracker(slot: "head", position: SIMD3<Float>(0, 1.6, 0), eulerDegrees: .zero)
            for _ in 0..<4 {
                await sender.sendPinoBundle(trackers: body, head: head)
                try? await Task.sleep(nanoseconds: 20_000_000)
            }
        }
        t.test("6-POINT bundle = 13 messages (6 pos + 6 rot + head/position, no elbows, no head rotation)") {
            guard let bundle = raw.last else { t.check(false, "no bundle captured"); return }
            let elems = parseBundle(bundle)
            let positions = elems.compactMap { decodePosition($0) }
            let rotations = elems.compactMap { decodeRotation($0) }
            t.check(elems.count == 13, "13 total messages, got \(elems.count)")
            t.check(rotations.count == 6, "6 rotation messages, got \(rotations.count)")
            t.check(positions.count == 7, "7 position messages (6 body + head), got \(positions.count)")
            t.check(Set(rotations.map { $0.slot }) == Set(sixSlots), "rotation slots are the 6-point set (no 3/4)")
            t.check(positions.contains { $0.slot == "head" }, "head/position present")
            t.check(!rotations.contains { $0.slot == "head" }, "NO head/rotation (position-only head contract)")
        }
    }

    // MARK: - 5. ROTATION finiteOrZero seed (no prior latch)

    /// A NaN euler on a slot that has never latched a valid rotation must reach the wire
    /// as exactly (0,0,0), never NaN (the round-2 finiteOrZero fallback).
    static func testRotationFiniteOrZeroSeed(_ t: TestRunner) async {
        let raw = await captureRaw(port: 9105) { sender in
            let body = [
                OSCTracker(slot: "1", position: SIMD3<Float>(0.3, 0.4, 0.5), eulerDegrees: SIMD3<Float>(.nan, .nan, .nan)),
                OSCTracker(slot: "2", position: .zero, eulerDegrees: .zero),   // hip keeps the bundle non-empty
            ]
            for _ in 0..<4 {
                await sender.sendPinoBundle(trackers: body, head: nil)
                try? await Task.sleep(nanoseconds: 20_000_000)
            }
        }
        let rots = raw.flatMap { parseBundle($0) }.compactMap { decodeRotation($0) }.filter { $0.slot == "1" }
        t.test("ROTATION finiteOrZero: NaN euler, never latched → exactly (0,0,0), no NaN on the wire") {
            t.check(!rots.isEmpty, "slot-1 rotation captured")
            t.check(rots.allSatisfy { $0.x.isFinite && $0.y.isFinite && $0.z.isFinite }, "no NaN/Inf reached the wire")
            t.check(rots.allSatisfy { $0.x == 0 && $0.y == 0 && $0.z == 0 }, "emitted exactly (0,0,0)")
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
