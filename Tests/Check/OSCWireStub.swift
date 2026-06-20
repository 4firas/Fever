import Foundation
import Network
import simd
import FeverCore

/// STUB OSC WIRE — a concrete geometry check on the ACTUAL bytes on the wire.
///
/// Spins up a real UDP listener on 127.0.0.1:9000, runs the synthetic upright
/// pose through the full lift→solve→map→assemble chain, transmits the assembled
/// trackers through the REAL `OSCSender` (real `NWConnection` UDP send), captures
/// the datagrams, decodes the `/tracking/trackers/<slot>/position` messages, and
/// prints the decoded leg-tracker positions (knee vs foot). The assertion is the
/// same geometry, but verified on the decoded WIRE FLOATS rather than in-process
/// values: foot below knee, and feet near the floor.
///
/// Invoked via `swift run FeverCheck --osc-stub`.
enum OSCWireStub {

    /// One decoded OSC position message.
    struct Decoded {
        let slot: String
        let x, y, z: Float
    }

    /// Decode a single OSC packet IF it is a `/tracking/trackers/<slot>/position`
    /// `,fff` message. Returns nil for anything else (e.g. `/rotation`).
    static func decodePosition(_ data: Data) -> Decoded? {
        let bytes = [UInt8](data)
        // Address string: NUL-terminated, then NUL-padded to a 4-byte boundary.
        guard let nul = bytes.firstIndex(of: 0) else { return nil }
        guard let address = String(bytes: bytes[0..<nul], encoding: .utf8) else { return nil }
        let prefix = "/tracking/trackers/"
        let suffix = "/position"
        guard address.hasPrefix(prefix), address.hasSuffix(suffix) else { return nil }
        let slot = String(address.dropFirst(prefix.count).dropLast(suffix.count))

        let addrBlock = ((address.utf8.count / 4) + 1) * 4
        // Type tag block: ",fff" NUL-padded to 4-byte boundary (= 8 bytes here).
        let tagStart = addrBlock
        guard bytes.count >= tagStart + 4 else { return nil }
        guard bytes[tagStart] == 0x2C,           // ','
              bytes[tagStart + 1] == 0x66,        // 'f'
              bytes[tagStart + 2] == 0x66,        // 'f'
              bytes[tagStart + 3] == 0x66 else {  // 'f'
            return nil
        }
        let tagBlock = ((4 / 4) + 1) * 4          // ",fff" -> 8
        let argStart = addrBlock + tagBlock
        guard bytes.count >= argStart + 12 else { return nil }

        func f(_ off: Int) -> Float {
            let b = bytes[(argStart + off)..<(argStart + off + 4)]
            let be = b.reduce(UInt32(0)) { ($0 << 8) | UInt32($1) }  // big-endian
            return Float(bitPattern: be)
        }
        return Decoded(slot: slot, x: f(0), y: f(4), z: f(8))
    }

    /// Run the stub: listen, send, capture, decode, assert, print. Returns true
    /// on success (foot below knee + feet near floor on the decoded wire bytes).
    static func run() async -> Bool {
        let port: UInt16 = 9000

        // 1. Build the assembled trackers via the FULL geometry chain.
        let (raw, present, image) = GeometrySanity.makeUprightRaw()
        let liftEngine = MonocularDepthLift(referenceHeight: 1.8)
        guard let pose = GeometrySanity.lift(raw, present, image, using: liftEngine) else {
            print("OSC-STUB: lift failed"); return false
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
        let (body, _) = assembler.assemble(joints, mapper: mapper)

        // 2. Stand up a real UDP listener on 127.0.0.1:9000 to capture the wire.
        let collector = PacketCollector()
        let listener: NWListener
        do {
            let params = NWParameters.udp
            params.allowLocalEndpointReuse = true
            listener = try NWListener(using: params,
                                      on: NWEndpoint.Port(rawValue: port)!)
        } catch {
            print("OSC-STUB: failed to bind listener on \(port): \(error)")
            return false
        }
        let queue = DispatchQueue(label: "fever.osc.stub.listener")
        listener.newConnectionHandler = { conn in
            conn.start(queue: queue)
            // Recursive UDP receive pump. Hold the continuation in a box so the
            // @Sendable receiveMessage completion can re-arm itself without
            // capturing a non-Sendable local function.
            let pump = ReceivePump(connection: conn, collector: collector)
            pump.start()
        }
        listener.start(queue: queue)
        // Let the listener finish binding before we transmit.
        try? await Task.sleep(nanoseconds: 300_000_000)

        print("OSC-STUB: assembled \(body.count) body tracker(s) to transmit")

        // 3. Send through the REAL OSCSender (real NWConnection UDP send). Give
        // the NWConnection time to reach `.ready` first — UDP datagrams sent
        // while it is still `.preparing` can be silently dropped on loopback.
        let sender = OSCSender(host: "127.0.0.1", port: Int(port))
        await sender.start()
        try? await Task.sleep(nanoseconds: 300_000_000)
        // Send the frame several times so at least one datagram per slot lands
        // (UDP on loopback is reliable in practice, but repeat for robustness).
        for _ in 0..<10 {
            await sender.send(trackers: body)
            try? await Task.sleep(nanoseconds: 40_000_000)  // 40 ms
        }
        // Give the loopback receive a moment to drain.
        try? await Task.sleep(nanoseconds: 400_000_000)
        await sender.stop()
        listener.cancel()

        // 4. Decode the captured datagrams → latest position per slot.
        var positions = [String: Decoded]()
        for pkt in collector.snapshot() {
            if let d = decodePosition(pkt) { positions[d.slot] = d }
        }

        print("OSC-STUB: captured \(collector.snapshot().count) datagram(s); decoded \(positions.count) position slot(s)")
        func dump(_ slot: String, _ name: String) {
            if let d = positions[slot] {
                print(String(format: "  [wire] %-10@ slot %@  pos = (%+.4f, %+.4f, %+.4f)",
                             name as NSString, slot, d.x, d.y, d.z))
            } else {
                print("  [wire] \(name) slot \(slot): NOT RECEIVED")
            }
        }
        // Leg trackers: 5=leftKnee 6=rightKnee 2=leftFoot 3=rightFoot.
        dump("5", "leftKnee"); dump("6", "rightKnee")
        dump("2", "leftFoot"); dump("3", "rightFoot")
        dump("1", "hip")

        // 5. Concrete geometry assertion on the decoded WIRE floats.
        guard let footL = positions["2"], let footR = positions["3"],
              let kneeL = positions["5"], let kneeR = positions["6"] else {
            print("OSC-STUB: FAIL — missing leg trackers on the wire")
            return false
        }
        var ok = true
        let lowestFoot = min(footL.y, footR.y)
        let lowestKnee = min(kneeL.y, kneeR.y)
        if !(footL.y < kneeL.y) {
            print("OSC-STUB: FAIL — leftFoot.y (\(footL.y)) not below leftKnee.y (\(kneeL.y))"); ok = false
        }
        if !(footR.y < kneeR.y) {
            print("OSC-STUB: FAIL — rightFoot.y (\(footR.y)) not below rightKnee.y (\(kneeR.y))"); ok = false
        }
        if !(lowestFoot > -0.10 && lowestFoot < 0.20) {
            print("OSC-STUB: FAIL — feet not near floor: lowest \(lowestFoot)"); ok = false
        }
        if ok {
            print(String(format: "OSC-STUB: PASS — feet below knees on the wire (foot %.3f < knee %.3f) and feet near floor (%.3f)",
                         lowestFoot, lowestKnee, lowestFoot))
        }
        return ok
    }
}

/// Thread-safe collector for captured UDP datagrams.
private final class PacketCollector: @unchecked Sendable {
    private let lock = NSLock()
    private var packets: [Data] = []
    func add(_ d: Data) { lock.withLock { packets.append(d) } }
    func snapshot() -> [Data] { lock.withLock { packets } }
}

/// Self-re-arming UDP receive pump. Encapsulating the recursion in a `@Sendable`
/// reference type lets the `receiveMessage` completion re-arm without capturing a
/// non-Sendable local function. The completion captures `self` STRONGLY so the
/// pump stays alive across the recursive receive (it is created as a transient
/// local in the listener's connection handler and would otherwise be released
/// immediately, silently dropping every datagram).
private final class ReceivePump: @unchecked Sendable {
    private let connection: NWConnection
    private let collector: PacketCollector
    init(connection: NWConnection, collector: PacketCollector) {
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
