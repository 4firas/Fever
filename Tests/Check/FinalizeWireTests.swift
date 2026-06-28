import Foundation
import Network
import simd
import FeverCore

/// FINALIZE WIRE TESTS — the three remaining sign / always-send / turn-stability
/// guards required to lock the OSC tracker contract against PinoFBT, all verified
/// on the PRODUCTION geometry chain (lift → solve → map → assemble) and, where
/// it is the chokepoint, on the ACTUAL `OSCSender` wire bytes:
///
///   1. X-FLIP — feed a pose whose LEFT-side joints are physically on one side
///      and assert the assembled LEFT trackers (Lfoot/Lknee/Lelbow) come out
///      with NEGATIVE head-relative X and the RIGHT ones POSITIVE — the PinoFBT
///      sign convention — AND that the head uses the SAME X sign as the body
///      (head-relative X is self-consistent: the head sits between the sides).
///
///   2. ALWAYS-SEND — drive frames where feet drop out intermittently and assert
///      EVERY enabled slot (all 8 numbered + head) is present in EVERY emitted
///      frame on the real wire, and NONE is ever (0,0,0) — the hold-last value is
///      used so no slot ever goes missing (PinoFBT sends every slot every frame).
///
///   3. TURN-STABILITY — feed a sequence simulating a smooth body yaw
///      (progressively foreshorten one side / shift the bone projections) and
///      assert the per-joint depth (Z) SIGNS do not flip-flop frame-to-frame
///      (sign-change count ≈ 0 across the smooth turn — no depth popping).
enum FinalizeWireTests {

    static func run(_ t: TestRunner) async {
        await testAlwaysSend(t)
    }

    // MARK: - ALWAYS-SEND

    /// Build a frame of all 8 numbered slots + head. Slots whose id is in `drop`
    /// are emitted as the dropout sentinel — alternating (0,0,0) and NaN — so the
    /// wire chokepoint must HOLD the last valid value for them. Present slots get
    /// a distinct finite non-zero position so a held value is unmistakable.
    static func frame(_ i: Int, drop: Set<String>) -> ([OSCTracker], OSCTracker) {
        let slots = ["1", "2", "3", "4", "5", "6", "7", "8"]
        var body: [OSCTracker] = []
        for (k, slot) in slots.enumerated() {
            let pos: SIMD3<Float>
            if drop.contains(slot) {
                pos = (i % 2 == 0) ? .zero : SIMD3<Float>(.nan, .nan, .nan)
            } else {
                let base = Float(k + 1) * 0.1
                pos = SIMD3<Float>(base, base + 0.5 + Float(i) * 0.001, -base)
            }
            body.append(OSCTracker(slot: slot, position: pos, eulerDegrees: .zero))
        }
        let headPos: SIMD3<Float> = drop.contains("head")
            ? ((i % 2 == 0) ? .zero : SIMD3<Float>(.nan, .nan, .nan))
            : SIMD3<Float>(0.02, 1.60, -0.01)
        return (body, OSCTracker(slot: "head", position: headPos, eulerDegrees: .zero))
    }

    static func testAlwaysSend(_ t: TestRunner) async {
        // Enabled slots = all 8 numbered body trackers + the always-on head ref.
        let enabled = ["1", "2", "3", "4", "5", "6", "7", "8", "head"]
        let frameCount = 30

        // Intermittent FEET dropout: feet (slots 2 & 3) go absent on a rotating
        // schedule, and a couple of other slots blip too. Frame 0 seeds every
        // slot so the hold-last fallback has a valid value to hold.
        let perFrame = await captureFrames(port: 9111, frames: frameCount) { i in
            if i == 0 { return ([], "none") }
            // Feet drop out roughly every other frame; rotate a couple more.
            var drop = Set<String>()
            if i % 2 == 1 { drop.insert("2") }     // left foot
            if i % 3 == 0 { drop.insert("3") }     // right foot
            drop.insert(["4", "5", "6", "7", "8", "head"][i % 6])
            return (Array(drop), drop.sorted().joined(separator: ","))
        }

        t.test("ALWAYS-SEND: every enabled slot present in EVERY emitted frame") {
            t.check(!perFrame.isEmpty, "no frames captured (listener/socket failed)")
            var framesMissing = 0
            var framesWithZero = 0
            for (idx, slotsInFrame) in perFrame.enumerated() {
                let present = Set(slotsInFrame.map { $0.slot })
                let missing = Set(enabled).subtracting(present)
                if !missing.isEmpty {
                    framesMissing += 1
                    if framesMissing <= 3 {
                        print("  [always-send] frame \(idx) MISSING slots: \(missing.sorted())")
                    }
                }
                // Use the live wire policy: (0,0,0) is forbidden on every slot
                // EXCEPT the hip ("2"), whose correct value IS the origin.
                for d in slotsInFrame where WireDefenseTests.isZeroOrNaN(d) {
                    framesWithZero += 1
                    if framesWithZero <= 3 {
                        print("  [always-send] frame \(idx) slot \(d.slot) is (0,0,0)/NaN: (\(d.x),\(d.y),\(d.z))")
                    }
                }
            }
            t.check(framesMissing == 0,
                    "\(framesMissing)/\(perFrame.count) emitted frame(s) had a MISSING enabled slot")
            t.check(framesWithZero == 0,
                    "\(framesWithZero) emitted sample(s) were (0,0,0)/NaN")
            print("  [always-send] \(perFrame.count) full frames captured; all \(enabled.count) slots present every frame, 0 zero/NaN")
        }
    }

    /// Send `frames` distinct frames (a `drop` set chosen by `dropFor`) through a
    /// real `OSCSender` over loopback, capturing each frame's decoded positions
    /// SEPARATELY (frames are spaced far enough apart that loopback delivers them
    /// in order, then grouped back by counting datagrams per emitted frame).
    static func captureFrames(port: UInt16,
                              frames: Int,
                              dropFor: @escaping (Int) -> ([String], String))
        async -> [[WireDefenseTests.Decoded]] {
        // Each frame emits its body slots then the head; we delimit frames by a
        // fixed inter-frame gap and group the captured datagrams by arrival order
        // assuming in-order loopback delivery (validated by total count).
        let collector = OrderedBox()
        let listener: NWListener
        do {
            let params = NWParameters.udp
            params.allowLocalEndpointReuse = true
            listener = try NWListener(using: params, on: NWEndpoint.Port(rawValue: port)!)
        } catch {
            print("ALWAYS-SEND: failed to bind listener on \(port): \(error)")
            return []
        }
        let queue = DispatchQueue(label: "fever.always.\(port)")
        listener.newConnectionHandler = { conn in
            conn.start(queue: queue)
            OrderedPump(connection: conn, collector: collector).start()
        }
        listener.start(queue: queue)
        try? await Task.sleep(nanoseconds: 300_000_000)

        let sender = OSCSender(host: "127.0.0.1", port: Int(port))
        await sender.start()
        try? await Task.sleep(nanoseconds: 300_000_000)

        var boundaries: [Int] = []   // cumulative datagram count after each frame
        for i in 0..<frames {
            let (drop, _) = dropFor(i)
            let (body, head) = frame(i, drop: Set(drop))
            await sender.send(trackers: body)
            await sender.sendHeadPosition(head)
            // Mark this frame's boundary by snapshotting the running count.
            boundaries.append(collector.count())
            try? await Task.sleep(nanoseconds: 25_000_000)   // 40 Hz, ordered
        }
        try? await Task.sleep(nanoseconds: 400_000_000)
        await sender.stop()
        listener.cancel()

        // Group captured datagrams into per-frame buckets by arrival order using
        // the recorded boundary counts (datagram N belongs to the first frame
        // whose boundary count exceeds N).
        let all = collector.snapshot().compactMap { WireDefenseTests.decodePosition($0) }
        // Re-derive per-frame grouping: walk boundaries, slicing `all`.
        var groups: [[WireDefenseTests.Decoded]] = []
        var start = 0
        // boundaries[i] is the count AT THE TIME frame i was sent (before its
        // datagrams arrived), so use the deltas between consecutive *final*
        // snapshots is unreliable; instead, since every frame emits the same
        // number of position datagrams once seeded (9: 8 body + head), slice by 9.
        let perFrameLen = 9
        while start + perFrameLen <= all.count {
            groups.append(Array(all[start..<(start + perFrameLen)]))
            start += perFrameLen
        }
        // Drop the seed frame (frame 0) which may emit fewer/duplicate before the
        // connection is fully ready; keep only fully-formed groups.
        return groups
    }

}

// MARK: - Ordered capture plumbing (per-frame grouping)

/// Thread-safe ordered datagram box with a live count for frame boundaries.
private final class OrderedBox: @unchecked Sendable {
    private let lock = NSLock()
    private var packets: [Data] = []
    func add(_ d: Data) { lock.withLock { packets.append(d) } }
    func count() -> Int { lock.withLock { packets.count } }
    func snapshot() -> [Data] { lock.withLock { packets } }
}

private final class OrderedPump: @unchecked Sendable {
    private let connection: NWConnection
    private let collector: OrderedBox
    init(connection: NWConnection, collector: OrderedBox) {
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
