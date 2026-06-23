import AVFoundation
import CoreVideo
import Foundation
import Observation
import simd

/// TrackingPipeline — the tracking-graph orchestrator.
///
/// Wires the whole tracking graph together:
///
///   FrameSource (capture queue)  →  single-slot MAILBOX (latest frame only)
///     → inference WORKER (dedicated serial queue, OFF the capture queue)
///         → PoseLandmarker      (MediaPipe sidecar inference, ~12 fps ceiling)
///         → LandmarkStabilizer  (One-Euro, 33×3)
///         → JointSolver         (9 VRJoints in the solver frame)
///         → RotationRebaser     (rest-relative + SLERP smoothing)
///         → CoordinateMapper    (single VRChat conversion)
///         → TrackerAssembler    (fixed numbered slots + head ref)
///         → OSCSender (actor)   (/position + /rotation → host:9000)
///     → THROTTLED telemetry (≈12 Hz) → @MainActor @Observable (the UI)
///
/// ── Why this layout (the performance fix) ────────────────────────────────────
/// MediaPipe pose inference (the Python sidecar, over length-prefixed IPC) has a
/// hard ~78 ms (~12 fps) ceiling on this machine. Running it INLINE on the camera
/// frames queue (the old design, with a `DispatchSemaphore` wait) stalled the
/// capture graph and the preview to ~5 fps. The fix DECOUPLES the two:
///
///   • The camera delegate only stashes the latest `CVPixelBuffer` into a
///     single-slot mailbox and returns immediately — the capture queue and the
///     `AVCaptureVideoPreviewLayer` preview stay smooth (full camera fps).
///   • A dedicated inference worker (one long-lived serial queue) pulls the
///     LATEST frame, runs the sidecar, and processes it. Any frames that arrive
///     while it is busy overwrite the mailbox slot and are dropped (process-
///     latest-only). Pose therefore updates at the sidecar's ~12 fps without ever
///     backing up capture.
///   • Telemetry is coalesced and published to the `@MainActor @Observable` at
///     most ~12 Hz, so SwiftUI never re-renders per frame.
///   • OSC streams on every new pose AND on a steady repeat tick (the last solved
///     trackers are re-sent) so VRChat receives a continuous tracker stream even
///     between inference frames.
///
/// Only `Sendable` values cross isolation boundaries: `[OSCTracker]` to the
/// `OSCSender` actor, and a small `Telemetry` snapshot to the main actor.
@MainActor
@Observable
public final class TrackingPipeline {

    // MARK: - Published telemetry (main-actor, @Observable)

    /// Whether the capture + inference loop is currently running.
    public private(set) var isRunning: Bool = false

    /// Throttled measured inference frame rate (Hz) — the REAL pose update rate.
    public private(set) var fps: Double = 0

    /// The most recent frame's solved + enabled joints (solver frame),
    /// republished on the throttle tick for UI overlays / inspectors.
    public private(set) var lastFrameJoints: [VRJoint] = []

    /// RAW 2D detected landmarks for the live preview overlay (33-slot, SCREEN-
    /// normalized, `SIMD2(.nan,.nan)` = absent). UNSMOOTHED, published on EVERY
    /// processed frame at FULL inference rate (NOT gated by the telemetry
    /// throttle) so the preview skeleton tracks instantly like VISO. This is the
    /// raw detection, NOT the smoothed solved joints — the OSC/tracker path keeps
    /// its One-Euro + SLERP smoothing untouched.
    public private(set) var previewPoints: [SIMD2<Float>] = []

    /// PinoQuest-style leveled reference box for the overlay (screen-normalized
    /// corners + a validity flag for the vanish-on-crouch behavior). Published at
    /// full rate alongside `previewPoints`.
    public private(set) var leveledBox: LeveledBox = .invalid

    /// Running total of dropped frames, mirrored from the source on each tick.
    public private(set) var droppedFrames: Int = 0

    /// The live camera session for the preview layer, or `nil` in stub mode.
    /// Part of the shared UI contract.
    public private(set) var previewSession: AVCaptureSession?

    /// Whether the camera is authorized. `false` until access is resolved /
    /// granted; stays `false` when denied (the UI surfaces this, no crash).
    /// In stub mode this is reported `true` (no camera permission involved).
    public private(set) var cameraAuthorized: Bool = false

    // MARK: - Dependencies

    private let config: TrackingConfig
    private let source: FrameSource
    private let landmarker: PoseLandmarker

    /// The live camera, if `source` is one (enables the mailbox + preview path).
    private let camera: CameraCapture?

    /// The OSC output actor. Recreated on `start()` from the live config so
    /// host / port edits take effect on the next run.
    private var osc: OSCSender?

    /// Per-frame processing state, confined to the inference worker (+ lock).
    private let processor: FrameProcessor

    /// Sendable handle the worker reads to reach the OSC actor.
    private let runtime = RuntimeBox()

    /// The single-slot mailbox the stub path feeds (the camera has its own
    /// internal mailbox; this one bridges `onFrame`-only sources like the stub).
    private let stubMailbox = FrameMailbox()

    /// The dedicated inference worker. One long-lived `Task` per run; cancelled
    /// on `stop()`. Runs OFF the capture queue.
    private var worker: Task<Void, Never>?

    /// Steady OSC repeat ticker: re-sends the last solved trackers so VRChat
    /// gets a continuous stream even between Vision frames.
    private var oscRepeater: Task<Void, Never>?

    // MARK: - Throttle bookkeeping

    /// Minimum wall-clock gap between telemetry publishes (≈12 Hz).
    private static let telemetryInterval: TimeInterval = 1.0 / 12.0

    /// Steady OSC repeat cadence (≈10 Hz) when no new pose has arrived.
    private static let oscRepeatInterval: TimeInterval = 0.1

    /// Idle poll interval for the worker when the mailbox is empty.
    private static let idlePollInterval: TimeInterval = 0.004  // 4 ms

    private var lastTelemetryPublish: TimeInterval = 0

    // MARK: - Init

    /// - Parameters:
    ///   - config: live, observable tuning + network settings.
    ///   - source: the frame source (`CameraCapture` live, `StubFrameSource` headless).
    ///   - landmarker: the pose backend (`MediaPipePoseLandmarker` live, `StubPoseLandmarker` test).
    public init(config: TrackingConfig,
                source: FrameSource,
                landmarker: PoseLandmarker) {
        self.config = config
        self.source = source
        self.landmarker = landmarker
        self.processor = FrameProcessor(config: config)
        self.camera = source as? CameraCapture

        if let camera {
            // Surface the live session immediately so the preview layer can
            // attach even before authorization resolves (it shows black until
            // the session runs, then fills in — never a crash).
            self.previewSession = camera.session
            self.cameraAuthorized = (camera.authorization == .authorized)
        } else {
            // Stub / headless synthetic source: no camera, no preview, no
            // permission gate.
            self.previewSession = nil
            self.cameraAuthorized = true
        }
    }

    // MARK: - Lifecycle

    /// Start capture + inference. Idempotent.
    public func start() {
        guard !isRunning else { return }
        isRunning = true

        // Rebuild tuned objects from current config (live tuning on start).
        processor.rebuild(from: config)
        // Clear the pose backend's per-run temporal state (smoothed scale +
        // depth-sign hysteresis) so a new run starts from a clean estimate.
        landmarker.reset()
        // Sync gravity-leveling (Body Stabilizer) config to the backend for this run.
        landmarker.setLeveling(enabled: config.bodyStabilizer, includeRoll: config.levelIncludeRoll)

        // Fresh OSC connection from the live host/port.
        let sender = OSCSender(host: config.oscHost, port: config.oscPort)
        let wantRotation = config.sendRotation
        // Every slot we will EVER transmit, pre-listed so the sender can seed a
        // neutral hold-last-valid fallback for each: every enabled, mapped body
        // slot, plus the head reference when enabled. This guarantees every slot is
        // emitted EVERY frame (PinoFBT parity) even before its joint first produces
        // a valid sample — the ~5% early-session foot dropout came from feet having
        // no prior held value, so an invalid first sample was skipped entirely.
        var seedSlots = config.enabledJoints.compactMap { config.slotMap[$0] }
        if config.sendHeadReference { seedSlots.append("head") }
        Task {
            await sender.setRotationEnabled(wantRotation)
            await sender.seedSlots(seedSlots)
            await sender.start()
        }
        osc = sender
        runtime.setSender(sender)
        runtime.setLastBody([])
        runtime.setLastHead(nil)

        let sendHead = config.sendHeadReference
        // ABSOLUTE rotation by default (PinoFBT ground truth): a Recenter re-latches
        // scale/floor but does NOT capture a rotation rest pose unless the user has
        // explicitly opted into rest-relative mode. Snapshot for the detached worker.
        let captureRotationRest = config.rotationRestRelative

        // Wire the frame intake. The CAMERA path uses CameraCapture's internal
        // single-slot mailbox (the delegate stashes; we pull via `nextFrame()`).
        // The STUB path has no mailbox of its own, so its `onFrame` feeds OUR
        // mailbox; either way the worker pulls the LATEST frame off-queue.
        if let camera {
            // Observe authorization changes (after the permission prompt is
            // answered) and republish on the main actor. Capture self WEAKLY so
            // the pipeline → camera → closure → pipeline cycle never forms; this
            // no longer relies on stop() being called to avoid a leak.
            camera.onAuthorizationChange = { [weak self] state in
                Task { @MainActor in self?.cameraAuthorized = (state == .authorized) }
            }
            camera.onFrame = nil
        } else {
            let mailbox = stubMailbox
            source.onFrame = { pixelBuffer, time in
                // Runs on the stub's emit queue: stash latest only, return.
                mailbox.store(pixelBuffer, at: time)
            }
        }

        // Build a Sendable "pull latest frame" closure for the worker.
        let pull = makeFramePuller()

        // Snapshot collaborators for the worker. The non-Sendable landmarker is
        // confined inside a Sendable box used only by this single serial worker.
        let landmarker = LandmarkerBox(self.landmarker)
        let processor = self.processor
        let runtime = self.runtime
        let dropCounter = DropCounter(source: source)
        let telemetrySink = self

        // The dedicated inference worker: one long-lived loop, OFF the capture
        // queue. Pulls the latest frame, runs Vision, processes, sends OSC, and
        // publishes throttled telemetry.
        worker = Task.detached(priority: .userInitiated) {
            while !Task.isCancelled {
                // Recenter: re-latch the landmarker's scale/floor (on this worker
                // queue, preserving confinement). Rotation is ABSOLUTE by default
                // (PinoFBT streams absolute limb orientations), so a Recenter does
                // NOT zero the rotations — it only captures a rest pose when the user
                // has explicitly enabled rest-relative mode.
                if runtime.takeRebaselineRequest() {
                    landmarker.reset()
                    if captureRotationRest {
                        processor.requestRestCapture()
                    }
                }

                guard let (pixelBuffer, time) = pull() else {
                    // No new frame yet — yield briefly and poll again.
                    try? await Task.sleep(nanoseconds: UInt64(Self.idlePollInterval * 1_000_000_000))
                    continue
                }

                // 1. Vision 3D inference, OFF the capture queue. `detect` is the
                //    ~78 ms hot path; running it here keeps capture/preview smooth.
                guard let detected = await landmarker.detect(pixelBuffer, at: time) else {
                    continue
                }

                // 1a. PREDICTIVE GAP-FILL — right after detect(), BEFORE One-Euro
                //     / the solver. Synthesizes brief per-landmark dropouts
                //     (fast moves, momentary occlusion) so joints ease through a
                //     gap instead of vanishing/snapping. Operates on the WHOLE
                //     PoseResult, so it benefits BOTH the metric `landmarks` (the
                //     OSC/solver path) AND the `imagePoints` (the preview).
                let raw = processor.predict(detected)

                // 1b. Publish the RAW 2D detected landmarks to the preview overlay
                //     on EVERY processed frame at FULL rate — a dedicated, light
                //     main-actor hop NOT gated by the ~12 Hz telemetry throttle.
                //     This drives the instant, VISO-like preview skeleton from the
                //     raw detection; the smoothed tracker path is untouched.
                let previewPoints = raw.imagePoints
                // Leveled reference box, built from the same leveled landmarks +
                // image points that drive the preview (pure, off the main actor).
                let leveledBox = LeveledBoxBuilder.build(landmarks: raw.landmarks,
                                                         imagePoints: raw.imagePoints)
                await MainActor.run {
                    telemetrySink.publishPreviewPoints(previewPoints)
                    telemetrySink.publishLeveledBox(leveledBox)
                }

                // 2..6. One-Euro → solve → fuse → SLERP → map → assemble.
                let frame = processor.process(raw, droppedFrames: dropCounter.current)

                // 7. Stream the new pose to OSC and remember it for the steady
                //    repeat ticker.
                let body = frame.body
                let head = sendHead ? frame.head : nil
                runtime.setLastBody(body)
                runtime.setLastHead(head)
                if let s = runtime.sender() {
                    await s.send(trackers: body)
                    // Stream the head POSITION continuously (PinoFBT-style) — the
                    // anchor VRChat re-origins to, cancelling the body's absolute
                    // frame offset. Position-only: never head rotation.
                    if let head { await s.sendHeadPosition(head) }
                }

                // 8. THROTTLED telemetry hop to the main actor.
                let telemetry = frame.telemetry
                await MainActor.run {
                    telemetrySink.publishTelemetry(telemetry)
                }
            }
        }

        // Steady OSC repeat: re-send the last solved trackers at a fixed cadence
        // so VRChat receives a continuous stream between Vision frames. Holds the
        // last pose; correctness (slots, head ref) is preserved because it just
        // re-sends the already-assembled trackers.
        oscRepeater = Task.detached(priority: .utility) {
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: UInt64(Self.oscRepeatInterval * 1_000_000_000))
                guard let s = runtime.sender() else { continue }
                let body = runtime.lastBody()
                guard !body.isEmpty else { continue }
                await s.send(trackers: body)
                if let head = runtime.lastHead() {
                    await s.sendHeadPosition(head)
                }
            }
        }

        source.start()
    }

    /// Stop capture + inference and tear down the OSC connection. Idempotent.
    public func stop() {
        guard isRunning else { return }
        isRunning = false

        worker?.cancel()
        worker = nil
        oscRepeater?.cancel()
        oscRepeater = nil

        source.stop()
        source.onFrame = nil
        camera?.onAuthorizationChange = nil
        stubMailbox.clear()

        runtime.setSender(nil)
        runtime.setLastBody([])
        runtime.setLastHead(nil)
        if let sender = osc {
            Task { await sender.stop() }
        }
        osc = nil

        processor.reset()
        fps = 0
        previewPoints = []
        leveledBox = .invalid
    }

    /// Recenter: re-baseline the body from the user's CURRENT standing pose.
    ///
    /// We deliberately send NO head OSC point — ever. The user wears a Quest HMD,
    /// which is the authoritative head; VRChat aligns the body to the real HMD head
    /// and auto-centers our feet (the two lowest trackers) under it. (PinoFBT works
    /// exactly this way and sends no head point.) The old behaviour fired a head
    /// snap pulse here, which snapped VRChat's OSC space to OUR estimated head and
    /// yanked the body — removed.
    ///
    /// Instead, Recenter re-latches the landmarker's scale + floor reference from
    /// the current frame, so the user can stand up straight, hit Recenter, and
    /// re-seed a clean baseline. The worker performs the reset on its own queue.
    /// Apply Body Stabilizer / leveling config to the running backend. Called live
    /// from the UI when the toggle or the roll setting changes; safe at any time
    /// (thread-safe inside the backend) and a no-op for backends without leveling.
    public func applyLevelingConfig() {
        landmarker.setLeveling(enabled: config.bodyStabilizer, includeRoll: config.levelIncludeRoll)
    }

    public func calibrate() {
        runtime.requestRebaseline()
    }

    // MARK: - Frame puller

    /// Builds the Sendable "pull the latest frame, non-blocking" closure the
    /// worker polls. Camera path drains CameraCapture's internal mailbox; stub
    /// path drains our own mailbox.
    private func makeFramePuller() -> @Sendable () -> (CVPixelBuffer, TimeInterval)? {
        if let camera {
            let box = CameraPuller(camera: camera)
            return { box.next() }
        } else {
            let mailbox = stubMailbox
            return { mailbox.take() }
        }
    }

    // MARK: - Main-actor helpers

    /// Full-rate publish of the RAW preview landmarks (one per processed frame,
    /// NOT throttled) so the preview overlay tracks at the inference rate.
    private func publishPreviewPoints(_ points: [SIMD2<Float>]) {
        guard isRunning else { return }
        previewPoints = points
    }

    /// Full-rate publish of the leveled reference box (paired with the preview).
    private func publishLeveledBox(_ box: LeveledBox) {
        guard isRunning else { return }
        leveledBox = box
    }

    /// Throttled publish of the latest telemetry snapshot (≈12 Hz max).
    private func publishTelemetry(_ t: Telemetry) {
        guard isRunning else { return }
        let now = ProcessInfo.processInfo.systemUptime
        guard now - lastTelemetryPublish >= Self.telemetryInterval else { return }
        lastTelemetryPublish = now

        fps = t.fps
        lastFrameJoints = t.joints
        droppedFrames = t.droppedFrames
    }
}

// MARK: - Sendable handoff types

/// Immutable per-frame telemetry crossing the worker → main actor.
private struct Telemetry: Sendable {
    let fps: Double
    let joints: [VRJoint]
    let droppedFrames: Int
}

/// One assembled frame's outputs: Sendable trackers + the telemetry snapshot.
private struct AssembledFrame: Sendable {
    let body: [OSCTracker]
    let head: OSCTracker?
    let telemetry: Telemetry
}

/// A captured, already-VRChat-mapped head pose for the calibration snap pulse.
private struct HeadPose: Sendable {
    let position: SIMD3<Float>
    let eulerDegrees: SIMD3<Float>
}

// MARK: - FrameMailbox (single-slot, latest-only)

/// Lock-guarded single-slot mailbox for `onFrame`-only sources (the stub). Holds
/// at most one `CVPixelBuffer`; a new arrival overwrites the previous one
/// (process-latest-only). `CVPixelBuffer` is a reference-counted CoreVideo
/// object, so holding one reference here is safe. `@unchecked Sendable` because
/// the `NSLock` provides the synchronization the compiler cannot see.
private final class FrameMailbox: @unchecked Sendable {
    private let lock = NSLock()
    private var buffer: CVPixelBuffer?
    private var time: TimeInterval = 0

    func store(_ b: CVPixelBuffer, at t: TimeInterval) {
        lock.withLock { buffer = b; time = t }
    }

    func take() -> (CVPixelBuffer, TimeInterval)? {
        lock.withLock {
            guard let b = buffer else { return nil }
            buffer = nil
            return (b, time)
        }
    }

    func clear() { lock.withLock { buffer = nil } }
}

/// Sendable adapter that pulls the latest frame from a live `CameraCapture`'s
/// internal mailbox. `@unchecked Sendable`: it only calls `nextFrame()`, which
/// is itself lock-guarded and thread-safe; the non-Sendable camera reference is
/// confined to this read.
private final class CameraPuller: @unchecked Sendable {
    private let camera: CameraCapture
    init(camera: CameraCapture) { self.camera = camera }
    func next() -> (CVPixelBuffer, TimeInterval)? { camera.nextFrame() }
}

/// Sendable wrapper confining the non-Sendable `any PoseLandmarker` so it can be
/// captured by the single, strictly-serial inference worker. `@unchecked
/// Sendable`: the landmarker is touched only from that one worker loop (never
/// concurrently), and `detect` consumes the pixel buffer entirely within the
/// call. The CVPixelBuffer it receives is non-Sendable, so `detect` is invoked
/// here with `nonisolated(unsafe)` confinement of both the detector and buffer.
private final class LandmarkerBox: @unchecked Sendable {
    private let landmarker: PoseLandmarker
    init(_ landmarker: PoseLandmarker) { self.landmarker = landmarker }
    func detect(_ pixelBuffer: CVPixelBuffer, at time: TimeInterval) async -> PoseResult? {
        await landmarker.detect(pixelBuffer, at: time)
    }
    /// Re-baseline the landmarker's latched scale/floor (called ON the worker
    /// queue so the single-serial confinement is preserved).
    func reset() { landmarker.reset() }
}

// MARK: - RuntimeBox (worker ↔ main-actor OSC handle + last trackers)

/// Lock-protected holder for the live `OSCSender` actor reference plus the last
/// solved trackers (for the steady OSC repeat ticker). `@unchecked Sendable`:
/// the lock provides the synchronization. `OSCSender` is an actor and
/// `[OSCTracker]` / `OSCTracker` are `Sendable`.
private final class RuntimeBox: @unchecked Sendable {
    private let lock = NSLock()
    private var _sender: OSCSender?
    private var _lastBody: [OSCTracker] = []
    private var _lastHead: OSCTracker?
    /// Wall-clock (`systemUptime`) deadline until which continuous head re-sends
    /// must be suppressed, so a calibration snap pulse stays >300 ms isolated and
    /// VRChat treats it as a snap, not a lerp. 0 = no suppression.
    private var _headSuppressedUntil: TimeInterval = 0
    /// Set by `calibrate()` (Recenter); consumed once by the worker to re-latch
    /// the landmarker's scale/floor from the user's current standing pose.
    private var _rebaselineRequested = false

    /// Request a one-shot re-baseline of the landmarker (scale + floor latch).
    func requestRebaseline() { lock.withLock { _rebaselineRequested = true } }
    /// Worker consumes the request exactly once (returns true then clears it).
    func takeRebaselineRequest() -> Bool {
        lock.withLock {
            guard _rebaselineRequested else { return false }
            _rebaselineRequested = false
            return true
        }
    }

    func setSender(_ s: OSCSender?) { lock.withLock { _sender = s } }
    func sender() -> OSCSender? { lock.withLock { _sender } }

    func setLastBody(_ b: [OSCTracker]) { lock.withLock { _lastBody = b } }
    func lastBody() -> [OSCTracker] { lock.withLock { _lastBody } }

    func setLastHead(_ h: OSCTracker?) { lock.withLock { _lastHead = h } }
    func lastHead() -> OSCTracker? { lock.withLock { _lastHead } }

    /// Suppress continuous head sends until `deadline` (systemUptime seconds).
    func suppressHead(until deadline: TimeInterval) {
        lock.withLock { _headSuppressedUntil = max(_headSuppressedUntil, deadline) }
    }
    /// Whether the worker/repeater should currently skip sending the head slot.
    func isHeadSuppressed() -> Bool {
        lock.withLock { ProcessInfo.processInfo.systemUptime < _headSuppressedUntil }
    }
}

/// Reads the current dropped-frame count from a source that exposes one
/// (`CameraCapture`), else reports 0. `@unchecked Sendable`: it only reads
/// `CameraCapture.droppedFrames`, itself lock-guarded and thread-safe.
private final class DropCounter: @unchecked Sendable {
    private let camera: CameraCapture?
    init(source: FrameSource) { self.camera = source as? CameraCapture }
    var current: Int { camera?.droppedFrames ?? 0 }
}

// MARK: - FrameProcessor (inference-worker confined)

/// Owns all mutable per-frame processing state and runs the solve → smooth →
/// fuse → map → assemble chain. Touched ONLY from the inference worker (one
/// frame at a time — the worker loop is strictly serial), plus `rebuild` /
/// `reset` / `latestHeadPose` from the main actor while the loop is stopped. All
/// access is serialized by `lock`, so the type is soundly `@unchecked Sendable`.
private final class FrameProcessor: @unchecked Sendable {

    private let lock = NSLock()

    private let predictor: JointPredictor
    private var stabilizer: LandmarkStabilizer
    /// Per-joint hold-last for the two-axis OSC rotation frames (degenerate frame
    /// → last good orientation, never a fabricated world-up roll). Owned here and
    /// injected into the solver so it survives across frames (the solver is a
    /// value type rebuilt on every config change).
    private let rotationState: RotationState
    /// Rest-relative rotation rebaser (PinoFBT delta-from-rest): captures the rest
    /// pose on Recenter and emits `inverse(qRest)*qLive`, hemisphere-locked +
    /// SLERP-smoothed, so the wire is bounded & zero-centered. Replaces the old
    /// absolute-quaternion `QuaternionStabilizer` on the OSC rotation path.
    private let rotationRebaser: RotationRebaser
    /// Per-foot slow-EMA state for the step/stride exaggeration (injected into the
    /// value-type solver) + raw-landmark cleanup (L/R anti-swap + occlusion gating)
    /// that runs before gap-fill. Reference types confined to this worker; reset on
    /// Recenter / run.
    private let footMotion = FootMotionState()
    /// PinoFBT-style yaw Body-Stabilizer (cross-frame yaw smoothing), injected into
    /// the value-type solver; gated by `TrackingConfig.yawStabilizer`.
    private let yawStab = YawStabilizer()
    private let consistency = LandmarkConsistency()
    private var solver: JointSolver
    private var mapper: CoordinateMapper
    private var assembler: TrackerAssembler
    private let angleTracker: AngleTracker

    // FPS measurement over a 1 s sliding window.
    private var frameCount: Int = 0
    private var windowStart: TimeInterval = 0
    private var measuredFPS: Double = 0

    // Latest VRChat-mapped head pose for the calibration snap pulse.
    private var latestHead: HeadPose?

    init(config: TrackingConfig) {
        self.predictor = JointPredictor()
        self.stabilizer = LandmarkStabilizer(minCutoff: config.stabilizerMinCutoffF,
                                             beta: config.stabilizerBetaF)
        self.rotationState = RotationState()
        self.rotationRebaser = RotationRebaser(smoothingFactor: config.rotationSmoothingF)
        self.solver = JointSolver(settings: config, rotationState: self.rotationState,
                                  footMotionState: self.footMotion, yawStabilizer: self.yawStab)
        self.mapper = CoordinateMapper(userHeightMeters: config.userHeightMetersF,
                                       referenceHeightMeters: 1.8,
                                       mirrorHorizontally: config.mirrorTracking)
        self.assembler = TrackerAssembler(enabled: config.enabledJoints,
                                          slotMap: config.slotMap)
        self.angleTracker = AngleTracker()
    }

    /// Rebuild all tuned objects from the current config (called on start while
    /// the loop is stopped). Also starts IMU fusion (identity no-op on macOS).
    func rebuild(from config: TrackingConfig) {
        lock.withLock {
            stabilizer = LandmarkStabilizer(minCutoff: config.stabilizerMinCutoffF,
                                            beta: config.stabilizerBetaF)
            rotationRebaser.smoothingFactor = config.rotationSmoothingF
            rotationRebaser.reset()
            rotationState.reset()
            yawStab.reset()
            solver = JointSolver(settings: config, rotationState: rotationState,
                                 footMotionState: footMotion, yawStabilizer: yawStab)
            mapper = CoordinateMapper(userHeightMeters: config.userHeightMetersF,
                                      referenceHeightMeters: 1.8,
                                      mirrorHorizontally: config.mirrorTracking)
            assembler = TrackerAssembler(enabled: config.enabledJoints,
                                         slotMap: config.slotMap)
            predictor.reset()
            footMotion.reset()
            consistency.reset()
            frameCount = 0
            windowStart = 0
            measuredFPS = 0
            latestHead = nil
        }
        angleTracker.start()
    }

    /// Reset between runs (called on stop while the loop is stopped).
    func reset() {
        angleTracker.stop()
        lock.withLock {
            predictor.reset()
            footMotion.reset()
            consistency.reset()
            rotationRebaser.reset()
            rotationState.reset()
            yawStab.reset()
            frameCount = 0
            windowStart = 0
            measuredFPS = 0
            latestHead = nil
        }
    }

    /// Predictive per-landmark gap-fill for one raw inference result, run BEFORE
    /// One-Euro / the solver. Returns a `PoseResult` with both the metric
    /// `landmarks` and the `imagePoints` synthesized where landmarks dropped out.
    /// Runs on the worker; the lock guards against a concurrent rebuild/reset.
    func predict(_ raw: PoseResult) -> PoseResult {
        // Raw-landmark cleanup (L/R anti-swap + occlusion gating) BEFORE the
        // predictive gap-fill, so the predictor holds-last on gated-out limbs.
        lock.withLock { predictor.predict(consistency.process(raw)) }
    }

    /// Request a one-shot rest-pose capture for the OSC rotation rebaser, latched
    /// on the next processed frame. Driven by the SAME Recenter that re-latches
    /// the landmarker's scale/floor, so one T/I-pose Recenter does scale + floor +
    /// rest-rotation together. Called on the worker queue (lock guards a rebuild).
    func requestRestCapture() {
        lock.withLock { rotationRebaser.requestRestCapture() }
    }

    /// The full solve → smooth → fuse → map → assemble chain for one inference
    /// result. Runs on the worker; the lock guards against a concurrent
    /// rebuild/reset.
    func process(_ raw: PoseResult, droppedFrames: Int) -> AssembledFrame {
        lock.lock()
        defer { lock.unlock() }

        // 1. One-Euro landmark smoothing.
        let stabilized = stabilizer.stabilize(raw)

        // 2. Solve 9 joints in the solver/Vision frame. Each non-head joint's
        //    rotation is now an ABSOLUTE solver-frame orientation built from TWO
        //    in-body axes (no world-up gauge); the foot's is locked-roll yaw+pitch.
        var joints = solver.solve(stabilized)

        // 3. ROTATION: rest-relative rebase (PinoFBT delta-from-rest) + IMU fuse +
        //    hemisphere-lock + SLERP. Capture the rest pose on the post-Recenter
        //    frame (same Recenter that re-latches scale/floor). The head is
        //    POSITION-ONLY downstream, so it is intentionally NOT rebased here
        //    (its absolute orientation is harmless; the assembler ignores it for
        //    the head reference, which streams position only).
        let captureRest = rotationRebaser.consumeCapturePending()
        for i in joints.indices where joints[i].type != .head {
            let fused = angleTracker.fuseWorldRotation(joints[i].rotation)
            joints[i].rotation = rotationRebaser.rebase(joints[i].type,
                                                        live: fused,
                                                        captureNow: captureRest)
        }

        // 4+5. Single authoritative VRChat conversion + fixed-slot assembly.
        let assembled = assembler.assemble(joints, mapper: mapper)

        // Remember the mapped head pose for calibrate() (snap pulse).
        if let head = assembled.head {
            latestHead = HeadPose(position: head.position, eulerDegrees: head.eulerDegrees)
        }

        // FPS measurement over a 1 s sliding window — reflects the REAL inference
        // rate (the worker calls this once per processed Vision frame).
        let now = raw.timestamp
        if windowStart == 0 { windowStart = now }
        frameCount += 1
        let elapsed = now - windowStart
        if elapsed >= 1.0 {
            measuredFPS = Double(frameCount) / elapsed
            frameCount = 0
            windowStart = now
        }

        // Only the enabled joints are republished for the UI (matches what is
        // actually transmitted by the assembler).
        let enabledJoints = joints.filter { assembler.enabled.contains($0.type) }
        let telemetry = Telemetry(fps: measuredFPS,
                                  joints: enabledJoints,
                                  droppedFrames: droppedFrames)

        return AssembledFrame(body: assembled.body,
                              head: assembled.head,
                              telemetry: telemetry)
    }

    /// The most-recent VRChat-mapped head pose, for the calibration snap pulse.
    func latestHeadPose() -> HeadPose? {
        lock.withLock { latestHead }
    }
}
