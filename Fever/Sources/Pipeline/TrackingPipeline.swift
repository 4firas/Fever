import AVFoundation
import CoreImage
import CoreVideo
import Foundation
import Observation
import simd

/// TrackingPipeline — the tracking-graph orchestrator (PinoFBT 2.0 1:1 port).
///
///   FrameSource → single-slot mailbox (latest frame) → dedicated inference worker:
///       NLFPoseLandmarker (onnxruntime sidecar) → SMPLPose(24, camera +Y-down)
///       → TwoEuroJointSmoother (OneEuro over the RAW (24,3), power=2/β=400, BEFORE IK)
///       → PinoSolver (preprocess_joints → calc_root/chest/arm/knee/ankle IK)
///       → OSC #bundle (17 msgs: pos+rot for slots 1..8 + head/position) → :9000
///   → throttled telemetry → @MainActor @Observable (the UI)
///
/// Capture/preview stay smooth: the camera delegate only stashes the latest frame;
/// the worker pulls latest-only off the capture queue. This is the BYTE-EXACT desktop
/// chain: slot map 1=chest..8=R_ankle, positions = preprocess-out × height_cm/175
/// (hip at origin), rotation = euler('zxy')[[1,2,0]] of the solver quats; VRChat
/// re-origins the tracker space via the continuous head/position. The default
/// behavior is a pure 1:1 PinoFBT match.
public struct LiveTracker: Sendable, Identifiable {
    /// OSC slot path: "1"…"8" for the numbered body trackers, or "head".
    public let slot: String
    /// Semantic label for the UI.
    public let joint: JointType
    /// World-space tracker position in meters (hip-origin space).
    public let position: SIMD3<Float>
    /// Wire `/rotation` euler degrees (ZXY); `.zero` for the head anchor.
    public let eulerDegrees: SIMD3<Float>
    public var id: String { slot }
    public init(slot: String, joint: JointType, position: SIMD3<Float>, eulerDegrees: SIMD3<Float>) {
        self.slot = slot; self.joint = joint; self.position = position; self.eulerDegrees = eulerDegrees
    }
}

@MainActor
@Observable
public final class TrackingPipeline {

    // MARK: - Published telemetry (main-actor)
    public private(set) var isRunning = false
    public private(set) var fps: Double = 0
    public private(set) var outputFPS: Double = 0     // fps-mux OSC send rate (Hz)
    public private(set) var fpsMultiplier: Int = 7    // current fps-mux multiplier (1–10×)
    /// The latest solved body trackers (slot-keyed), published at the throttled
    /// telemetry rate so the UI can show real per-tracker position/rotation. Empty
    /// while stopped or before the first solved frame.
    public private(set) var liveTrackers: [LiveTracker] = []
    public private(set) var previewPoints: [SIMD2<Float>] = []
    /// The exact frame inference last ran on (published with previewPoints) so the
    /// on-screen preview updates at the inference rate, not the faster camera rate.
    public private(set) var previewImage: CGImage?
    public private(set) var droppedFrames = 0
    public private(set) var previewSession: AVCaptureSession?
    public private(set) var cameraAuthorized = false
    /// A non-fatal on-device health warning surfaced to the UI (e.g. running on the
    /// canned DEMO pose because the NLF runtime is missing, or no camera frames are
    /// arriving). nil = healthy. Distinct from a hard stop so the user isn't left on a
    /// green "Running" over a dead/fake pipeline.
    public private(set) var healthNote: String?
    static let demoNote = "Demo pose — NLF runtime not found (not real tracking)"
    static let noFramesNote = "No camera frames — check the camera or its permission"

    // MARK: - Dependencies
    private let config: TrackingConfig
    private let source: FrameSource
    private let landmarker: any NLFPoseSource
    private let camera: CameraCapture?
    private var osc: OSCSender?
    private let processor: FrameProcessor
    private let runtime = RuntimeBox()
    private let stubMailbox = FrameMailbox()
    private var worker: Task<Void, Never>?
    private var oscRepeater: Task<Void, Never>?
    private var healthWatchdog: Task<Void, Never>?     // one-shot no-frames check; cancelled on stop
    /// Nonisolated mirror of worker+oscRepeater so deinit can cancel them.
    private let loops = LoopHandles()

    private static let telemetryInterval: TimeInterval = 1.0 / 12.0
    private static let oscRepeatInterval: TimeInterval = 0.1
    private static let idlePollInterval: TimeInterval = 0.004
    private var lastTelemetryPublish: TimeInterval = 0

    public init(config: TrackingConfig, source: FrameSource, landmarker: any NLFPoseSource) {
        self.config = config
        self.source = source
        self.landmarker = landmarker
        self.processor = FrameProcessor(config: config)
        self.camera = source as? CameraCapture
        if let camera {
            self.previewSession = camera.session
            self.cameraAuthorized = (camera.authorization == .authorized)
        } else {
            self.previewSession = nil
            self.cameraAuthorized = true
        }
    }

    // MARK: - Lifecycle

    public func start() {
        guard !isRunning else { return }
        isRunning = true
        // Surface a DEMO session immediately (canned pose, not real tracking). A live
        // landmarker starts healthy; the no-frames watchdog below may flag it later.
        healthNote = landmarker.isLive ? nil : Self.demoNote

        processor.rebuild(from: config)
        landmarker.reset()

        let sender = OSCSender(host: config.oscHost, port: config.oscPort)
        var seedSlots = TrackerMapPino.slots
            .filter { config.sendElbows || ($0.index != 3 && $0.index != 4) }   // 6-point default
            .map(\.path)
        seedSlots.append("head")   // 1:1 PinoFBT always streams head/position
        Task {
            await sender.setRotationEnabled(true)   // PinoFBT emits /rotation for all 8 body slots
            await sender.seedSlots(seedSlots)
            await sender.start()
        }
        osc = sender
        runtime.setSender(sender)
        runtime.setSnapshot(nil)

        if let camera {
            camera.onAuthorizationChange = { [weak self] state in
                Task { @MainActor in self?.cameraAuthorized = (state == .authorized) }
            }
            camera.onFrame = nil
        } else {
            let mailbox = stubMailbox
            source.onFrame = { pixelBuffer, time in mailbox.store(pixelBuffer, at: time) }
        }

        let pull = makeFramePuller()
        let landmarker = NLFLandmarkerBox(self.landmarker)
        let processor = self.processor
        let runtime = self.runtime
        let dropCounter = DropCounter(source: source)
        let previewInterval = Self.telemetryInterval   // captured here (main-actor) for the worker

        // [weak self] so the detached worker does NOT retain the pipeline (otherwise the
        // pipeline<->Task cycle would keep it — and the worker — alive forever, so deinit
        // could never cancel it). The hot path uses the separately-captured runtime/
        // processor; only the preview/telemetry publish needs self, and skips if gone.
        worker = Task.detached(priority: .userInitiated) { [weak self] in
            let ciContext = CIContext(options: [.useSoftwareRenderer: false])
            var lastPreviewBuild: TimeInterval = 0
            while !Task.isCancelled {
                if runtime.takeRebaselineRequest() { processor.recenter() }

                guard let (pixelBuffer, time) = pull() else {
                    try? await Task.sleep(nanoseconds: UInt64(Self.idlePollInterval * 1_000_000_000))
                    continue
                }
                guard let pose = await landmarker.detect(pixelBuffer, at: time) else { continue }
                if Task.isCancelled { break }   // a Stop landed during detect → don't touch the (possibly restarted) shared state

                let frame = processor.process(pose, droppedFrames: dropCounter.current)
                // Hand the smoothed joints + velocity to the predictive upsampler
                // (the high-rate output loop below extrapolates + solves + sends).
                // Stamp it with the monotonic store time so the loop knows how stale
                // the pose is and how far forward to predict.
                runtime.setSnapshot(PoseSnapshot(joints: frame.smoothedJoints,
                                                 velocity: frame.velocity,
                                                 tracked: frame.tracked,
                                                 heightCm: frame.heightCm,
                                                 sendElbows: frame.sendElbows,
                                                 mirror: frame.mirror))
                runtime.setFPS(frame.telemetry.fps)
                runtime.setFpsMultiplier(frame.telemetry.fpsMultiplier)
                runtime.setPredictionLeadMs(frame.telemetry.predictionLeadMs)

                // Preview is for the human, not the tracker — cap it at the telemetry
                // rate (12 Hz) so the costly full-res CGImage isn't built on inference
                // frames that would only be dropped. The publish hop is a fire-and-
                // forget Task (NOT a blocking `await MainActor.run`) so the worker can
                // pull the next frame immediately instead of stalling on the main thread.
                let now = ProcessInfo.processInfo.systemUptime
                let telemetry = frame.telemetry
                let liveTrackers = frame.liveTrackers
                // Build an IMMUTABLE preview payload (let), so the fire-and-forget Task
                // captures a fixed value — never a `var` the loop could reassign before
                // the Task runs (which would be a data race).
                let previewPayload: (points: [SIMD2<Float>], image: CGImage?)?
                if now - lastPreviewBuild >= previewInterval {
                    lastPreviewBuild = now
                    let ci = CIImage(cvImageBuffer: pixelBuffer)
                    previewPayload = (frame.preview, ciContext.createCGImage(ci, from: ci.extent))
                } else {
                    previewPayload = nil
                }
                Task { @MainActor in
                    guard let self else { return }
                    if let p = previewPayload { self.publishPreview(p.points, image: p.image) }
                    self.publishTelemetry(telemetry, trackers: liveTrackers)
                }
            }
        }

        // FPS-MUX (LEADING predictor): upsample the low inference rate up to N× and
        // re-run the exact PinoSolver IK on every tick over a critically-damped
        // follower of the latest smoothed joints, PLUS a forward lead that anticipates
        // motion. The lead is a constant horizon (the user-tunable predictionLeadMs, in
        // ms), NOT the snapshot's actual age — so this trims follow-delay while moving but
        // does not chase a stale pose during an inference stall. Rotations are re-derived
        // by the IK (no euler wrap); overshoot is bounded (capped lead + per-joint clamp).
        // This is the sole sender; the worker only stores the snapshot. Output rate =
        // clamp(N × inferenceFPS, inferenceFPS, 120) Hz.
        oscRepeater = Task.detached(priority: .userInitiated) {
            let up = PredictiveUpsampler(heightCm: 174)
            var nextTick = ProcessInfo.processInfo.systemUptime
            var lastMirror: Bool?                         // resets the follower across a handedness flip
            while !Task.isCancelled {
                let fps = runtime.fps() > 1 ? runtime.fps() : 12
                let mul = Double(max(1, runtime.fpsMultiplier()))
                let rate = min(120.0, max(fps, fps * mul))   // 1× = inference rate … up to 10×, capped 120 Hz
                let dt = 1.0 / rate
                // Phase-lock the cadence to a wall-clock deadline so per-tick work time
                // doesn't accumulate into drift/jitter. `dt` passed to step() below is
                // unchanged (byte-identical solve); only the WAKE timing is steadier.
                nextTick += dt
                let now = ProcessInfo.processInfo.systemUptime
                let remaining = nextTick - now
                if remaining > 0 {
                    try? await Task.sleep(nanoseconds: UInt64(remaining * 1_000_000_000))
                } else if remaining < -dt {
                    nextTick = now                            // fell far behind (a stall) → resync, don't burst-catch-up
                }
                if Task.isCancelled { break }                 // a Stop landed during the sleep → don't send into a torn-down session
                guard let s = runtime.sender() else { continue }
                guard let snap = runtime.snapshot(), !snap.joints.isEmpty else { up.reset(); continue }
                // A mid-session mirror flip swaps L/R and negates x — damping ACROSS that
                // discontinuity would warp the whole body for ~smoothTime. Reset the
                // follower so it re-seeds cleanly on the new handedness (matches the
                // inference-side smoother reset).
                if lastMirror != snap.mirror { up.reset(); lastMirror = snap.mirror }
                // Silky-glide: damp toward the latest pose with a time-constant scaled
                // to the inference period, so the follower is still gliding when the
                // next frame lands (continuous, never a freeze-then-jump). PLUS a small
                // forward LEAD that anticipates motion — it keeps fast moves crisp (not
                // mushy) and shrinks the follow-delay while you're moving; the damping
                // absorbs any direction-reversal overshoot, so it never rubberbands.
                // smoothTime scales with the inference period (tune the 0.7 factor); the
                // lead is the user's predictionLeadMs (ms → s) — see UserSettings.
                let smoothTime = Float(min(0.10, max(0.03, 0.7 / fps)))   // ~0.07 s at 10 fps
                let lead = Float(runtime.predictionLeadMs()) / 1000.0     // user-tunable forward-lead (ms → s)
                let (body, head) = up.step(joints: snap.joints, velocity: snap.velocity,
                                           smoothTime: smoothTime, dt: Float(dt), lead: lead,
                                           tracked: snap.tracked,
                                           heightCm: snap.heightCm, sendElbows: snap.sendElbows)
                await s.sendPinoBundle(trackers: body, head: head)
            }
        }
        loops.set(worker: worker, osc: oscRepeater)   // mirror for the nonisolated deinit

        source.start()

        // No-frames watchdog: a LIVE landmarker that has produced no snapshot within 5s
        // means the camera/permission is dead — flag it instead of a green "Running"
        // over nothing. (The stub always produces frames, so it's exempt.)
        if self.landmarker.isLive {
            healthWatchdog = Task { [weak self] in
                try? await Task.sleep(for: .seconds(5))
                guard let self, !Task.isCancelled, self.isRunning,
                      self.runtime.snapshot() == nil, self.healthNote == nil else { return }
                self.healthNote = Self.noFramesNote
            }
        }
    }

    public func stop() {
        guard isRunning else { return }
        isRunning = false
        worker?.cancel(); worker = nil
        oscRepeater?.cancel(); oscRepeater = nil
        healthWatchdog?.cancel(); healthWatchdog = nil
        loops.set(worker: nil, osc: nil)
        source.stop()
        source.onFrame = nil
        camera?.onAuthorizationChange = nil
        stubMailbox.clear()
        runtime.setSender(nil); runtime.setSnapshot(nil)
        if let sender = osc { Task { await sender.stop() } }
        osc = nil
        processor.reset()
        fps = 0; previewPoints = []; previewImage = nil; liveTrackers = []; healthNote = nil
    }

    /// If the pipeline is dropped without an explicit stop(), still cancel its two
    /// detached loops so they don't run on (the [weak self] worker capture is what lets
    /// this deinit run at all). The @MainActor-isolated `worker`/`oscRepeater` aren't
    /// reachable from a nonisolated deinit, so we mirror them into `loops` (a nonisolated
    /// Sendable box) and cancel through that.
    deinit { loops.cancelAll() }

    /// Recenter: clear the smoother/solver state so a fresh standing pose re-seeds
    /// cleanly. (VRChat does the real re-origin via the head stream — no rest pose.)
    public func calibrate() { runtime.requestRebaseline() }

    /// Whether the given semantic tracker is currently streaming live solved data
    /// (the session is running and a solved value exists for its slot this window).
    public func isLive(_ joint: JointType) -> Bool {
        isRunning && liveTrackers.contains { $0.joint == joint }
    }

    private func makeFramePuller() -> @Sendable () -> (CVPixelBuffer, TimeInterval)? {
        if let camera {
            let box = CameraPuller(camera: camera)
            return { box.next() }
        } else {
            let mailbox = stubMailbox
            return { mailbox.take() }
        }
    }

    private func publishPreview(_ points: [SIMD2<Float>], image: CGImage?) {
        guard isRunning else { return }
        previewPoints = points
        previewImage = image
    }

    private func publishTelemetry(_ t: Telemetry, trackers: [LiveTracker]) {
        guard isRunning else { return }
        let now = ProcessInfo.processInfo.systemUptime
        guard now - lastTelemetryPublish >= Self.telemetryInterval else { return }
        lastTelemetryPublish = now
        fps = t.fps
        droppedFrames = t.droppedFrames
        fpsMultiplier = t.fpsMultiplier
        outputFPS = min(120, max(t.fps, t.fps * Double(max(1, t.fpsMultiplier))))
        liveTrackers = trackers
        // Frames are flowing → clear a prior "no frames" warning (but NOT the demo note).
        if healthNote == Self.noFramesNote { healthNote = nil }
    }
}

// MARK: - Sendable handoff types

private struct Telemetry: Sendable {
    let fps: Double
    let droppedFrames: Int
    let fpsMultiplier: Int
    let predictionLeadMs: Int
}
private struct AssembledFrame: Sendable {
    let smoothedJoints: [SIMD3<Float>]
    let velocity: [SIMD3<Float>]
    let tracked: Bool
    let heightCm: Float
    let sendElbows: Bool
    let mirror: Bool            // current capture handedness (flips L/R + sign on toggle)
    let liveTrackers: [LiveTracker]
    let preview: [SIMD2<Float>]
    let telemetry: Telemetry
}

/// The latest smoothed-joint snapshot handed to the predictive upsampler: the
/// OneEuro-smoothed (24,3) joints + their smoothed velocity, the tracked flag, the
/// capture handedness, and the live solve params.
private struct PoseSnapshot: Sendable {
    var joints: [SIMD3<Float>]
    var velocity: [SIMD3<Float>]
    var tracked: Bool
    var heightCm: Float
    var sendElbows: Bool
    var mirror: Bool             // handedness — output loop resets the upsampler on a flip
}

// MARK: - Single-slot mailbox + Sendable adapters

/// Nonisolated, Sendable holder for the pipeline's two detached loops, so a
/// nonisolated `deinit` can cancel them (the @MainActor `worker`/`oscRepeater`
/// stored properties are unreachable from deinit).
private final class LoopHandles: @unchecked Sendable {
    private let lock = NSLock()
    private var worker: Task<Void, Never>?
    private var osc: Task<Void, Never>?
    func set(worker: Task<Void, Never>?, osc: Task<Void, Never>?) {
        lock.withLock { self.worker = worker; self.osc = osc }
    }
    func cancelAll() {
        let (w, o): (Task<Void, Never>?, Task<Void, Never>?) = lock.withLock { (worker, osc) }
        w?.cancel(); o?.cancel()
    }
}

private final class FrameMailbox: @unchecked Sendable {
    private let lock = NSLock()
    private var buffer: CVPixelBuffer?
    private var time: TimeInterval = 0
    func store(_ b: CVPixelBuffer, at t: TimeInterval) { lock.withLock { buffer = b; time = t } }
    func take() -> (CVPixelBuffer, TimeInterval)? {
        lock.withLock { guard let b = buffer else { return nil }; buffer = nil; return (b, time) }
    }
    func clear() { lock.withLock { buffer = nil } }
}

private final class CameraPuller: @unchecked Sendable {
    private let camera: CameraCapture
    init(camera: CameraCapture) { self.camera = camera }
    func next() -> (CVPixelBuffer, TimeInterval)? { camera.nextFrame() }
}

private final class NLFLandmarkerBox: @unchecked Sendable {
    private let landmarker: any NLFPoseSource
    init(_ landmarker: any NLFPoseSource) { self.landmarker = landmarker }
    func detect(_ pixelBuffer: CVPixelBuffer, at time: TimeInterval) async -> SMPLPose? {
        await landmarker.detect(pixelBuffer, at: time)
    }
}

private final class RuntimeBox: @unchecked Sendable {
    private let lock = NSLock()
    private var _sender: OSCSender?
    private var _snapshot: PoseSnapshot?
    private var _rebaseline = false
    func requestRebaseline() { lock.withLock { _rebaseline = true } }
    func takeRebaselineRequest() -> Bool {
        lock.withLock { guard _rebaseline else { return false }; _rebaseline = false; return true }
    }
    func setSender(_ s: OSCSender?) { lock.withLock { _sender = s } }
    func sender() -> OSCSender? { lock.withLock { _sender } }
    func setSnapshot(_ s: PoseSnapshot?) { lock.withLock { _snapshot = s } }
    func snapshot() -> PoseSnapshot? { lock.withLock { _snapshot } }
    private var _fps: Double = 0
    func setFPS(_ f: Double) { lock.withLock { _fps = f } }
    func fps() -> Double { lock.withLock { _fps } }
    private var _fpsMul: Int = 7
    func setFpsMultiplier(_ m: Int) { lock.withLock { _fpsMul = m } }
    func fpsMultiplier() -> Int { lock.withLock { _fpsMul } }
    private var _leadMs: Int = 50
    func setPredictionLeadMs(_ m: Int) { lock.withLock { _leadMs = m } }
    func predictionLeadMs() -> Int { lock.withLock { _leadMs } }
}

private final class DropCounter: @unchecked Sendable {
    private let camera: CameraCapture?
    init(source: FrameSource) { self.camera = source as? CameraCapture }
    var current: Int { camera?.droppedFrames ?? 0 }
}

// MARK: - FrameProcessor (inference-worker confined)

/// The faithful NLF chain: camera→world transform → Two-Euro smooth → hip IK →
/// OSC trackers (map A). Serialized by `lock` (worker is strictly serial; rebuild/
/// reset run only while stopped).
private final class FrameProcessor: @unchecked Sendable {
    private let lock = NSLock()
    private var smoother: TwoEuroJointSmoother
    private let solver: PinoSolver
    private let cfg: TrackingConfig   // live source for fpsMultiplier (shared object)
    /// Reflect the model skeleton (swap L/R labels + negate x) to convert the Mac
    /// webcam's native handedness to PinoFBT's capture handedness. A PROPER
    /// reflection of the input geometry — the IK re-derives correct-chirality
    /// quats, so the spine/limb solvers AND the elbow FK stay 1:1. (An image flip
    /// would be an IMPROPER reflection and corrupt the rotation-based FK.)
    /// Last-applied mirror state, so a live toggle of `cfg.mirrorTracking` can be
    /// detected (and the smoother reset across the handedness discontinuity).
    private var mirrorEnabled = true

    private var frameCount = 0
    private var windowStart: TimeInterval = 0
    private var measuredFPS: Double = 0

    init(config: TrackingConfig) {
        // 1:1 PinoFBT chain: OneEuro over raw (24,3) → preprocess → IK → bundle.
        cfg = config
        smoother = TwoEuroJointSmoother()
        solver = PinoSolver(heightCm: Float(config.userHeightMeters * 100))
        mirrorEnabled = config.mirrorTracking
    }

    /// Proper L/R mirror of the SMPL-24 skeleton: each joint takes the reflected
    /// (x-negated) position of its left/right counterpart. Geometry-exact, so the
    /// downstream IK produces valid capture-handed rotations.
    static func mirrorSkeleton(_ j: [SIMD3<Float>]) -> [SIMD3<Float>] {
        // L↔R SMPL index swap (central joints map to themselves).
        let s: [Int] = [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10,
                        12, 14, 13, 15, 17, 16, 19, 18, 21, 20, 23, 22]
        guard j.count == 24 else { return j }
        var o = [SIMD3<Float>](repeating: .zero, count: 24)
        for i in 0..<24 { let v = j[s[i]]; o[i] = SIMD3<Float>(-v.x, v.y, v.z) }
        return o
    }

    // Debug tap: while playing to VRChat (which owns :9000), append the real solved
    // values to ~/fever_osc.log at ~3 Hz so they can be read live without a port.
    // OPT-IN only (set FEVER_OSC_DEBUG): otherwise it's a silent side-effect that
    // writes solved body data into the user's home dir on every session.
    private static let debugEnabled = ProcessInfo.processInfo.environment["FEVER_OSC_DEBUG"] != nil
    private static let debugPath = (NSHomeDirectory() as NSString).appendingPathComponent("fever_osc.log")
    private var lastDebugWrite: TimeInterval = 0

    func rebuild(from config: TrackingConfig) {
        lock.withLock {
            smoother = TwoEuroJointSmoother()
            solver.setHeightCm(Float(config.userHeightMeters * 100))
            solver.reset()
            mirrorEnabled = config.mirrorTracking
            frameCount = 0; windowStart = 0; measuredFPS = 0
        }
        if Self.debugEnabled {
            try? "fever osc debug log\n".write(toFile: Self.debugPath, atomically: false, encoding: .utf8)
        }
        lastDebugWrite = 0
    }

    private func debugTap(_ solved: SolvedFrame, ht: Float, t: TimeInterval) {
        guard Self.debugEnabled, t - lastDebugWrite >= 0.33 else { return }
        lastDebugWrite = t
        func e(_ slot: Int) -> String {   // euler (pitch X, yaw Y, roll Z) degrees
            let v = solved.slotEulers[slot] ?? .zero
            return String(format: "(%.0f,%.0f,%.0f)", v.x, v.y, v.z)
        }
        func p(_ slot: Int) -> String {   // position (x,y,z) meters
            let v = solved.slotPositions[slot] ?? .zero
            return String(format: "(%.2f,%.2f,%.2f)", v.x, v.y, v.z)
        }
        let hp = solved.headPosition
        // All tracker dot positions (to diagnose mis-placed/floating dots in VRChat).
        // Slot map: 1=chest 2=hip 3=L_elbow 4=R_elbow 5=L_knee 6=R_knee 7=L_ankle 8=R_ankle.
        let line = String(format: "head=(%.2f,%.2f,%.2f) chest=%@ hip=%@ Lel=%@ Rel=%@ Lkn=%@ Rkn=%@ Lan=%@ Ran=%@\n",
                          hp.x, hp.y, hp.z, p(1), p(2), p(3), p(4), p(5), p(6), p(7), p(8))
        if let h = FileHandle(forWritingAtPath: Self.debugPath) {
            h.seekToEndOfFile(); h.write(Data(line.utf8)); try? h.close()
        }
    }

    func reset() { lock.withLock { smoother.reset(); solver.reset(); measuredFPS = 0; frameCount = 0; windowStart = 0 } }
    func recenter() { lock.withLock { smoother.reset(); solver.reset() } }

    func process(_ pose: SMPLPose, droppedFrames: Int) -> AssembledFrame {
        lock.lock(); defer { lock.unlock() }

        // 1:1 PinoFBT: OneEuro over the RAW (24,3) model joints (camera +Y down;
        // preprocess_joints does the FY flip & scale), BEFORE IK. Proper skeleton
        // mirror (capture handedness) BEFORE filtering. The actual IK + slot map run
        // in the high-rate output loop on the FORWARD-PREDICTED joints; here we keep
        // the smoothed joints + velocity for that loop, and solve the CURRENT (un-
        // predicted) frame only for the UI snapshot + debug tap.
        // Live config (no Stop/Start needed): mirror handedness, body height, and the
        // 6-/8-point elbow toggle are read EVERY frame. A mirror flip changes
        // handedness, so reset the smoother to avoid a one-frame velocity spike across
        // that discontinuity.
        let mirror = cfg.mirrorTracking
        if mirror != mirrorEnabled { smoother.reset(); mirrorEnabled = mirror }
        let elbows = cfg.sendElbows
        let heightCm = Float(cfg.userHeightMeters * 100)
        solver.setHeightCm(heightCm)

        let src = mirror ? Self.mirrorSkeleton(pose.joints3D) : pose.joints3D
        let smoothed = smoother.smooth(src, timestamp: pose.timestamp)
        let velocity = smoother.velocity()
        let tracked = pose.isTracked

        let solved = solver.solve(joints: smoothed, tracked: tracked)
        let (body, head) = assemblePinoBundle(solved, sendElbows: elbows)
        debugTap(solved, ht: pose.hasTracked, t: pose.timestamp)

        // FPS over a 1 s sliding window (real inference rate).
        let now = pose.timestamp
        if windowStart == 0 { windowStart = now }
        frameCount += 1
        let elapsed = now - windowStart
        if elapsed >= 1.0 { measuredFPS = Double(frameCount) / elapsed; frameCount = 0; windowStart = now }

        // UI snapshot: the real solved trackers, labelled by semantic joint.
        var live: [LiveTracker] = []
        live.reserveCapacity(body.count + 1)
        for t in body {
            if let jt = JointType.forPinoSlot(t.slot) {
                live.append(LiveTracker(slot: t.slot, joint: jt,
                                        position: t.position, eulerDegrees: t.eulerDegrees))
            }
        }
        live.append(LiveTracker(slot: "head", joint: .head,
                                position: head.position, eulerDegrees: .zero))

        return AssembledFrame(smoothedJoints: smoothed, velocity: velocity, tracked: tracked,
                              heightCm: heightCm, sendElbows: elbows, mirror: mirror, liveTrackers: live,
                              preview: pose.normalizedPoints(),
                              telemetry: Telemetry(fps: measuredFPS, droppedFrames: droppedFrames,
                                                   fpsMultiplier: cfg.fpsMultiplier,
                                                   predictionLeadMs: cfg.predictionLeadMs))
    }
}
