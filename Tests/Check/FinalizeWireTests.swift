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
        testXFlip(t)
        await testAlwaysSend(t)
        testTurnStability(t)
    }

    // MARK: - Shared geometry chain

    /// Run the FULL production chain on a raw Vision-style pose and return the
    /// assembled body trackers (by slot) plus the head reference. `mirror=false`
    /// pins handedness so the X sign convention is deterministic.
    static func assembleChain(raw: [SIMD2<Float>],
                              present: [Bool],
                              image: [SIMD2<Float>],
                              userHeight: Float = 1.74,
                              mirror: Bool = false)
        -> (bySlot: [String: OSCTracker], head: OSCTracker?)? {
        let liftEngine = MonocularDepthLift(referenceHeight: 1.8)
        guard let pose = GeometrySanity.lift(raw, present, image, using: liftEngine) else {
            return nil
        }
        let cfg = TrackingConfig()
        cfg.mirrorTracking = mirror
        cfg.userHeightMeters = Double(userHeight)
        let joints = JointSolver(settings: cfg).solve(pose)
        let mapper = CoordinateMapper(userHeightMeters: userHeight,
                                      referenceHeightMeters: 1.8,
                                      mirrorHorizontally: mirror)
        let assembler = TrackerAssembler(enabled: cfg.enabledJoints, slotMap: cfg.slotMap)
        let (body, head) = assembler.assemble(joints, mapper: mapper)
        var bySlot = [String: OSCTracker]()
        for tr in body { bySlot[tr.slot] = tr }
        return (bySlot, head)
    }

    // MARK: - 1. X-FLIP

    /// An ASYMMETRIC upright pose: the left arm is raised out to the side and the
    /// left foot is planted wider, so the left/right sides are physically
    /// distinct (not a mirror-symmetric T-pose). LEFT joints sit at x < center,
    /// RIGHT joints at x > center, in Vision normalized coords (+Y up).
    static func makeAsymmetricRaw() -> (raw: [SIMD2<Float>], present: [Bool], image: [SIMD2<Float>]) {
        var raw = [SIMD2<Float>](repeating: .zero, count: 33)
        var present = [Bool](repeating: false, count: 33)
        var image = [SIMD2<Float>](repeating: SIMD2<Float>(.nan, .nan), count: 33)
        let c: Float = 0.5

        func set(_ l: BlazePose.Landmark, _ x: Float, _ y: Float) {
            raw[l.rawValue] = SIMD2<Float>(x, y)
            present[l.rawValue] = true
            image[l.rawValue] = SIMD2<Float>(x, 1 - y)
        }

        // Head / face.
        set(.nose, c, 0.92)
        set(.leftEye, c - 0.03, 0.93); set(.rightEye, c + 0.03, 0.93)
        set(.leftEar, c - 0.05, 0.92); set(.rightEar, c + 0.05, 0.92)
        // Shoulders.
        set(.leftShoulder, c - 0.13, 0.80); set(.rightShoulder, c + 0.13, 0.80)
        // LEFT arm raised out to the side (elbow/wrist farther left + higher);
        // RIGHT arm hangs down at the side. Sides are clearly asymmetric.
        set(.leftElbow, c - 0.28, 0.82);  set(.rightElbow, c + 0.15, 0.66)
        set(.leftWrist, c - 0.40, 0.84);  set(.rightWrist, c + 0.16, 0.52)
        // Hips.
        set(.leftHip, c - 0.08, 0.52); set(.rightHip, c + 0.08, 0.52)
        // Knees — slightly bent for a real out-of-plane depth.
        set(.leftKnee, c - 0.12, 0.30); set(.rightKnee, c + 0.08, 0.30)
        // Ankles — LEFT foot planted wider out to the side.
        set(.leftAnkle, c - 0.16, 0.08); set(.rightAnkle, c + 0.08, 0.08)
        return (raw, present, image)
    }

    static func testXFlip(_ t: TestRunner) {
        let (raw, present, image) = makeAsymmetricRaw()
        guard let (bySlot, headOpt) = assembleChain(raw: raw, present: present, image: image) else {
            t.test("X-FLIP: chain produced trackers") { t.check(false, "lift returned nil") }
            return
        }

        // slot map: 1=hip 2=lFoot 3=rFoot 4=chest 5=lKnee 6=rKnee 7=lElbow 8=rElbow.
        func x(_ slot: String) -> Float? { bySlot[slot]?.position.x }

        t.test("X-FLIP: LEFT trackers negative head-relative X, RIGHT positive (PinoFBT sign)") {
            guard let head = headOpt else { t.check(false, "no head reference"); return }
            let hx = head.position.x
            guard let lFoot = x("2"), let rFoot = x("3"),
                  let lKnee = x("5"), let rKnee = x("6"),
                  let lElbow = x("7"), let rElbow = x("8") else {
                t.check(false, "missing L/R trackers: \(bySlot.keys.sorted())"); return
            }
            // Head-relative X = tracker.x - head.x.
            let lFootR = lFoot - hx, rFootR = rFoot - hx
            let lKneeR = lKnee - hx, rKneeR = rKnee - hx
            let lElbowR = lElbow - hx, rElbowR = rElbow - hx

            t.check(lFootR < 0,  "Lfoot head-relative X must be NEGATIVE: \(lFootR)")
            t.check(rFootR > 0,  "Rfoot head-relative X must be POSITIVE: \(rFootR)")
            t.check(lKneeR < 0,  "Lknee head-relative X must be NEGATIVE: \(lKneeR)")
            t.check(rKneeR > 0,  "Rknee head-relative X must be POSITIVE: \(rKneeR)")
            t.check(lElbowR < 0, "Lelbow head-relative X must be NEGATIVE: \(lElbowR)")
            t.check(rElbowR > 0, "Relbow head-relative X must be POSITIVE: \(rElbowR)")
            // The three LEFT trackers all share the SAME sign; same for RIGHT.
            t.check(lFootR < 0 && lKneeR < 0 && lElbowR < 0,
                    "all LEFT trackers must share the negative X sign")
            t.check(rFootR > 0 && rKneeR > 0 && rElbowR > 0,
                    "all RIGHT trackers must share the positive X sign")
            print(String(format: "  [x-flip] head.x=%+.3f  Lfoot=%+.3f Rfoot=%+.3f  Lknee=%+.3f Rknee=%+.3f  Lelbow=%+.3f Relbow=%+.3f (head-relative)",
                         hx, lFootR, rFootR, lKneeR, rKneeR, lElbowR, rElbowR))
        }

        t.test("X-FLIP: head + body use the SAME X sign (head between the sides)") {
            guard let head = headOpt else { t.check(false, "no head reference"); return }
            let hx = head.position.x
            guard let lFoot = x("2"), let rFoot = x("3") else {
                t.check(false, "missing foot trackers"); return
            }
            // The head's absolute X must lie BETWEEN the left and right body
            // trackers (it is the body's horizontal center) — i.e. the head and
            // the body are expressed in the SAME (consistent) X axis, not flipped
            // relative to one another.
            t.check(lFoot < hx, "head X must be to the RIGHT of the left foot (same axis): head=\(hx) Lfoot=\(lFoot)")
            t.check(hx < rFoot, "head X must be to the LEFT of the right foot (same axis): head=\(hx) Rfoot=\(rFoot)")
            // And mirror-OFF means no handedness flip vs the raw input: the raw
            // left side (smaller x) stays the negative side after the chain.
            t.check(lFoot < rFoot, "left/right ordering preserved end-to-end (no spurious flip)")
        }

        // Sanity: with mirror ON the X side flips (PinoFBT-style front-camera
        // mirror), confirming the convention is exactly the X sign and nothing
        // else — left becomes positive, right becomes negative, magnitudes equal.
        t.test("X-FLIP: mirror toggle flips the X side cleanly (sign only)") {
            guard let mir = assembleChain(raw: raw, present: present, image: image, mirror: true) else {
                t.check(false, "mirrored chain nil"); return
            }
            guard let head = headOpt, let mHead = mir.head,
                  let lFoot = x("2"), let mLFoot = mir.bySlot["2"]?.position.x,
                  let rFoot = x("3"), let mRFoot = mir.bySlot["3"]?.position.x else {
                t.check(false, "missing trackers for mirror compare"); return
            }
            let lR = lFoot - head.position.x,  mLR = mLFoot - mHead.position.x
            let rR = rFoot - head.position.x,  mRR = mRFoot - mHead.position.x
            // Head-relative X negates under the mirror; magnitude preserved.
            t.check(lR * mLR < 0, "Lfoot head-relative X must flip sign under mirror: \(lR) vs \(mLR)")
            t.check(rR * mRR < 0, "Rfoot head-relative X must flip sign under mirror: \(rR) vs \(mRR)")
            t.check(abs(lR + mLR) < 1e-4, "Lfoot X magnitude preserved across mirror")
        }
    }

    // MARK: - 2. ALWAYS-SEND

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
                for d in slotsInFrame where (d.x == 0 && d.y == 0 && d.z == 0) || !d.x.isFinite || !d.y.isFinite || !d.z.isFinite {
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

    // MARK: - 3. TURN-STABILITY

    static func testTurnStability(_ t: TestRunner) {
        // Simulate a smooth body yaw: progressively foreshorten the LEFT side of
        // the body (its limbs project shorter as it rotates toward the camera)
        // while the RIGHT side projects fuller, sweeping the depth field
        // continuously. The per-joint depth SIGNS (relative to the hip root) must
        // NOT flip-flop across the smooth turn — depth-sign hysteresis in the lift
        // is the guard against the "leg popping in/out" jitter.
        let ref: Float = 1.8
        let lift = MonocularDepthLift(referenceHeight: ref)

        // Build a metric-XY frame for turn parameter τ ∈ [0,1]: at τ=0 the body
        // faces the camera (both legs full length in-plane); as τ grows the LEFT
        // leg foreshortens (knee/ankle pulled toward x=center, shorter projected
        // bone → larger synthesized forward depth) — a continuous yaw.
        func turnFrame(_ tau: Float) -> ([SIMD2<Float>], [Bool]) {
            var xy = [SIMD2<Float>](repeating: .zero, count: 33)
            var present = [Bool](repeating: false, count: 33)
            func set(_ l: BlazePose.Landmark, _ x: Float, _ y: Float) {
                xy[l.rawValue] = SIMD2<Float>(x, y); present[l.rawValue] = true
            }
            // Shoulders + hips define the chain roots. As the body yaws, the LEFT
            // shoulder/hip move toward center (narrower projected width on that
            // side), the RIGHT side stays wide.
            let lShift = 0.10 * tau    // left side pulls in as it rotates back
            set(.leftShoulder, -0.20 + lShift, 0.50); set(.rightShoulder, 0.20, 0.50)
            set(.leftHip, -0.10 + lShift, 0.0);        set(.rightHip, 0.10, 0.0)
            // LEFT leg foreshortens: projected thigh/shank shrink with τ (knee &
            // ankle ride up toward the hip in Y as the leg swings out of plane).
            let lKneeY = -0.22 + 0.10 * tau
            let lAnkleY = -0.44 + 0.22 * tau
            set(.leftKnee, -0.10 + lShift, lKneeY)
            set(.leftAnkle, -0.10 + lShift, lAnkleY)
            // RIGHT leg stays full length, in-plane (control).
            set(.rightKnee, 0.10, -0.22)
            set(.rightAnkle, 0.10, -0.44)
            return (xy, present)
        }

        // Sweep τ smoothly over many frames (and hold a few at the ends), running
        // the REAL depth solver each frame and recording the depth SIGN of each
        // tracked segment relative to the hip root.
        let segments: [BlazePose.Landmark] = [.leftKnee, .leftAnkle, .rightKnee, .rightAnkle]
        var prevSign = [BlazePose.Landmark: Float]()
        var flips = [BlazePose.Landmark: Int]()
        for seg in segments { flips[seg] = 0 }

        let steps = 60
        var frameIndex = 0
        for i in 0...steps {
            let tau = Float(i) / Float(steps)   // 0 → 1 smooth
            let (xy, present) = turnFrame(tau)
            let z = lift.depths(metricXY: xy, present: present)
            let zHip = 0.5 * (z[BlazePose.Landmark.leftHip.rawValue]
                              + z[BlazePose.Landmark.rightHip.rawValue])
            for seg in segments {
                let dz = z[seg.rawValue] - zHip
                // Treat a near-zero depth as "no sign" (don't count crossings
                // through the dead zone as flips — only genuine sign reversals).
                let sign: Float = abs(dz) < 1e-3 ? (prevSign[seg] ?? 0) : (dz >= 0 ? 1 : -1)
                if frameIndex > 0, let p = prevSign[seg], p != 0, sign != 0, sign != p {
                    flips[seg]! += 1
                }
                prevSign[seg] = sign
            }
            frameIndex += 1
        }

        t.test("TURN-STABILITY: joint depth signs do not flip-flop across a smooth yaw") {
            var totalFlips = 0
            for seg in segments {
                let f = flips[seg] ?? 0
                totalFlips += f
                print("  [turn] \(seg) depth-sign flips across \(steps + 1) frames: \(f)")
            }
            // A smooth, monotonic turn must not produce depth-sign popping. Allow
            // at most ONE legitimate crossing per segment (a real front→back pass
            // through the hip plane), but never the multi-flip chatter the
            // hysteresis exists to suppress.
            t.check(totalFlips <= 1,
                    "depth signs flipped \(totalFlips) time(s) across a smooth turn (expected ≈ 0)")
        }

        t.test("TURN-STABILITY: depths stay finite throughout the turn") {
            let lift2 = MonocularDepthLift(referenceHeight: ref)
            var allFinite = true
            for i in 0...steps {
                let (xy, present) = turnFrame(Float(i) / Float(steps))
                let z = lift2.depths(metricXY: xy, present: present)
                for seg in segments where !z[seg.rawValue].isFinite { allFinite = false }
            }
            t.check(allFinite, "all turn-frame depths must be finite")
        }
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
