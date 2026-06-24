import AVFoundation
import CoreVideo
import Foundation
import Observation
import simd

/// TrackingPipeline — the tracking-graph orchestrator (NLF / SMPL-24 rebuild).
///
///   FrameSource → single-slot mailbox (latest frame) → dedicated inference worker:
///       NLFPoseLandmarker (onnxruntime sidecar) → SMPLPose(24, camera +Y-down)
///       → CameraToWorldTransform → TwoEuroJointSmoother → SMPL24Solver
///       → OSC bundle (map A: positions 1..8 + head, HIP-only rotation) → :9000
///   → throttled telemetry → @MainActor @Observable (the UI)
///
/// Capture/preview stay smooth: the camera delegate only stashes the latest frame;
/// the worker pulls latest-only off the capture queue. Faithful to PinoFBT (findings):
/// floor-anchored absolute positions, hip-only rotation, VRChat re-origins via head.
@MainActor
@Observable
public final class TrackingPipeline {

    // MARK: - Published telemetry (main-actor)
    public private(set) var isRunning = false
    public private(set) var fps: Double = 0
    public private(set) var lastFrameJoints: [VRJoint] = []      // retained surface (unused in NLF path)
    public private(set) var previewPoints: [SIMD2<Float>] = []
    public private(set) var leveledBox: LeveledBox = .invalid    // retained surface; always invalid (no leveling)
    public private(set) var droppedFrames = 0
    public private(set) var previewSession: AVCaptureSession?
    public private(set) var cameraAuthorized = false

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

        processor.rebuild(from: config)
        landmarker.reset()

        let sender = OSCSender(host: config.oscHost, port: config.oscPort)
        var seedSlots = TrackerMapA.slots.map(\.path)
        if config.sendHeadReference { seedSlots.append("head") }
        Task {
            await sender.setRotationEnabled(true)   // hip (slot 1) carries rotation — PinoFBT
            await sender.seedSlots(seedSlots)
            await sender.start()
        }
        osc = sender
        runtime.setSender(sender)
        runtime.setLastBody([])
        runtime.setLastHead(nil)

        let sendHead = config.sendHeadReference

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
        let telemetrySink = self

        worker = Task.detached(priority: .userInitiated) {
            while !Task.isCancelled {
                if runtime.takeRebaselineRequest() { processor.recenter() }

                guard let (pixelBuffer, time) = pull() else {
                    try? await Task.sleep(nanoseconds: UInt64(Self.idlePollInterval * 1_000_000_000))
                    continue
                }
                guard let pose = await landmarker.detect(pixelBuffer, at: time) else { continue }

                let frame = processor.process(pose, droppedFrames: dropCounter.current)
                let body = frame.body
                let head = sendHead ? frame.head : nil
                runtime.setLastBody(body)
                runtime.setLastHead(head)
                if let s = runtime.sender() {
                    await s.send(trackers: body)
                    if let head { await s.sendHeadPosition(head) }
                }

                // preview at full rate; fps/telemetry throttled
                let preview = frame.preview
                let telemetry = frame.telemetry
                await MainActor.run {
                    telemetrySink.publishPreview(preview)
                    telemetrySink.publishTelemetry(telemetry)
                }
            }
        }

        // steady re-send so VRChat gets a continuous stream between inference ticks
        oscRepeater = Task.detached(priority: .utility) {
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: UInt64(Self.oscRepeatInterval * 1_000_000_000))
                guard let s = runtime.sender() else { continue }
                let body = runtime.lastBody()
                guard !body.isEmpty else { continue }
                await s.send(trackers: body)
                if let head = runtime.lastHead() { await s.sendHeadPosition(head) }
            }
        }

        source.start()
    }

    public func stop() {
        guard isRunning else { return }
        isRunning = false
        worker?.cancel(); worker = nil
        oscRepeater?.cancel(); oscRepeater = nil
        source.stop()
        source.onFrame = nil
        camera?.onAuthorizationChange = nil
        stubMailbox.clear()
        runtime.setSender(nil); runtime.setLastBody([]); runtime.setLastHead(nil)
        if let sender = osc { Task { await sender.stop() } }
        osc = nil
        processor.reset()
        fps = 0; previewPoints = []; leveledBox = .invalid
    }

    /// Recenter: clear the smoother/solver state so a fresh standing pose re-seeds
    /// cleanly. (VRChat does the real re-origin via the head stream — no rest pose.)
    public func calibrate() { runtime.requestRebaseline() }

    /// Retained for the existing UI binding; the NLF path has no leveling stage.
    public func applyLevelingConfig() {}

    private func makeFramePuller() -> @Sendable () -> (CVPixelBuffer, TimeInterval)? {
        if let camera {
            let box = CameraPuller(camera: camera)
            return { box.next() }
        } else {
            let mailbox = stubMailbox
            return { mailbox.take() }
        }
    }

    private func publishPreview(_ points: [SIMD2<Float>]) {
        guard isRunning else { return }
        previewPoints = points
    }

    private func publishTelemetry(_ t: Telemetry) {
        guard isRunning else { return }
        let now = ProcessInfo.processInfo.systemUptime
        guard now - lastTelemetryPublish >= Self.telemetryInterval else { return }
        lastTelemetryPublish = now
        fps = t.fps
        droppedFrames = t.droppedFrames
    }
}

// MARK: - Sendable handoff types

private struct Telemetry: Sendable {
    let fps: Double
    let droppedFrames: Int
}
private struct AssembledFrame: Sendable {
    let body: [OSCTracker]
    let head: OSCTracker?
    let preview: [SIMD2<Float>]
    let telemetry: Telemetry
}

// MARK: - Single-slot mailbox + Sendable adapters

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
    private var _lastBody: [OSCTracker] = []
    private var _lastHead: OSCTracker?
    private var _rebaseline = false
    func requestRebaseline() { lock.withLock { _rebaseline = true } }
    func takeRebaselineRequest() -> Bool {
        lock.withLock { guard _rebaseline else { return false }; _rebaseline = false; return true }
    }
    func setSender(_ s: OSCSender?) { lock.withLock { _sender = s } }
    func sender() -> OSCSender? { lock.withLock { _sender } }
    func setLastBody(_ b: [OSCTracker]) { lock.withLock { _lastBody = b } }
    func lastBody() -> [OSCTracker] { lock.withLock { _lastBody } }
    func setLastHead(_ h: OSCTracker?) { lock.withLock { _lastHead = h } }
    func lastHead() -> OSCTracker? { lock.withLock { _lastHead } }
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
    private var transform: CameraToWorldTransform
    private var smoother: TwoEuroJointSmoother
    private let solver: SMPL24Solver

    private var frameCount = 0
    private var windowStart: TimeInterval = 0
    private var measuredFPS: Double = 0

    init(config: TrackingConfig) {
        transform = CameraToWorldTransform(mirrorX: config.mirrorTracking, heightScale: 1)
        smoother = TwoEuroJointSmoother()
        solver = SMPL24Solver(headAnchor: .head15)
    }

    // Debug tap: while playing to VRChat (which owns :9000), append the real solved
    // values to ~/fever_osc.log at ~3 Hz so they can be read live without a port.
    private static let debugPath = (NSHomeDirectory() as NSString).appendingPathComponent("fever_osc.log")
    private var lastDebugWrite: TimeInterval = 0

    func rebuild(from config: TrackingConfig) {
        lock.withLock {
            transform = CameraToWorldTransform(mirrorX: config.mirrorTracking, heightScale: 1)
            smoother = TwoEuroJointSmoother()
            solver.reset()
            frameCount = 0; windowStart = 0; measuredFPS = 0
        }
        try? "fever osc debug log\n".write(toFile: Self.debugPath, atomically: false, encoding: .utf8)
        lastDebugWrite = 0
    }

    private func debugTap(_ solved: SolvedFrame, ht: Float, t: TimeInterval) {
        guard t - lastDebugWrite >= 0.33 else { return }
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
        let line = String(format: "head=(%.2f,%.2f,%.2f) hip=%@ chest=%@ Lf=%@ Rf=%@ Lk=%@ Rk=%@  HIProt=%@\n",
                          hp.x, hp.y, hp.z, p(1), p(4), p(2), p(3), p(5), p(6), e(1))
        if let h = FileHandle(forWritingAtPath: Self.debugPath) {
            h.seekToEndOfFile(); h.write(Data(line.utf8)); try? h.close()
        }
    }

    func reset() { lock.withLock { smoother.reset(); solver.reset(); measuredFPS = 0; frameCount = 0; windowStart = 0 } }
    func recenter() { lock.withLock { smoother.reset(); solver.reset() } }

    func process(_ pose: SMPLPose, droppedFrames: Int) -> AssembledFrame {
        lock.lock(); defer { lock.unlock() }

        let world = transform.apply(pose.joints3D)
        let smoothed = smoother.smooth(world, timestamp: pose.timestamp)
        let solved = solver.solve(world: smoothed, tracked: pose.isTracked)

        var body: [OSCTracker] = []
        body.reserveCapacity(TrackerMapA.slots.count)
        for slot in TrackerMapA.slots {
            let pos = solved.slotPositions[slot.index] ?? .zero
            let euler = solved.slotEulers[slot.index] ?? .zero
            body.append(OSCTracker(slot: slot.path, position: pos, eulerDegrees: euler))
        }
        let head = OSCTracker(slot: "head", position: solved.headPosition, eulerDegrees: .zero)
        debugTap(solved, ht: pose.hasTracked, t: pose.timestamp)

        // FPS over a 1 s sliding window (real inference rate).
        let now = pose.timestamp
        if windowStart == 0 { windowStart = now }
        frameCount += 1
        let elapsed = now - windowStart
        if elapsed >= 1.0 { measuredFPS = Double(frameCount) / elapsed; frameCount = 0; windowStart = now }

        return AssembledFrame(body: body, head: head, preview: pose.normalizedPoints(),
                              telemetry: Telemetry(fps: measuredFPS, droppedFrames: droppedFrames))
    }
}
