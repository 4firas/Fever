import AVFoundation
import CoreVideo
import Foundation

/// Camera authorization state, surfaced to the UI via the pipeline.
public enum CameraAuthorization: Sendable, Equatable {
    /// The user has not yet been asked (we will request on `start()`).
    case notDetermined
    /// Access granted — capture can run.
    case authorized
    /// Access denied or restricted by policy — capture cannot run, but the
    /// app must keep functioning (no crash, just no preview / pose).
    case denied
}

/// Live `FrameSource` backed by an `AVCaptureSession` over the built-in
/// front-facing webcam.
///
/// ── Concurrency / performance model (REWORKED) ──────────────────────────────
/// The sample-buffer delegate runs on the nonisolated capture `outputQueue` and
/// does the ABSOLUTE MINIMUM: it copies the latest `CVPixelBuffer` reference into
/// a single-slot mailbox and returns IMMEDIATELY. It NEVER runs Vision inference
/// inline and NEVER blocks on a semaphore. A previously stashed-but-unconsumed
/// frame is overwritten (process-latest-only) — that is intentional and keeps the
/// next inference starting from the freshest frame; it is NOT counted as a drop.
/// This keeps the capture graph — and therefore the `AVCaptureVideoPreviewLayer`
/// preview — perfectly smooth regardless of how slow downstream inference is.
///
/// A dedicated inference worker (owned by `TrackingPipeline`) pulls the latest
/// buffer out of the mailbox via `nextFrame()` and runs Vision off the capture
/// queue. `onFrame` is retained for protocol compatibility but is NO LONGER the
/// inference entry point in the live path.
///
/// `CVPixelBuffer` is a reference-counted CoreVideo object; stashing exactly one
/// in the mailbox (and dropping the previous one) is safe and does not exhaust
/// the capture pool because `alwaysDiscardsLateVideoFrames` plus the single-slot
/// replacement keep at most one buffer alive outside the pool at a time.
public final class CameraCapture: NSObject, FrameSource {

    /// Default capture frame-rate cap (fps). 30 by default — smooth, light, and plenty
    /// for body pose, and it keeps a GoPro/Continuity/UVC cam from spinning the capture
    /// graph (and the PC stream) faster than we need. Overridable per-session via the
    /// `maxFPS` property (driven by `TrackingConfig.cameraMaxFPS`).
    public static let defaultCaptureFPS: CMTimeScale = 30

    /// Capture frame-rate cap (fps). The camera runs FASTER than the model infers
    /// (~10–15 fps): the inference worker pulls latest-only, so a higher capture rate
    /// means the next inference always starts from a FRESHER frame — which cuts the
    /// IRL→VR latency floor (frame age). `applyFrameRate` clamps this target into
    /// whatever the device actually supports (a value outside the device's real ranges
    /// would abort the process), so any camera — built-in, GoPro, Continuity, UVC — is
    /// pinned to at most `maxFPS`. DEFAULT 30 (`defaultCaptureFPS`); set from
    /// `TrackingConfig.cameraMaxFPS` before/while running via `setMaxFPS(_:)`.
    public var maxFPS: CMTimeScale {
        get { lock.withLock { _maxFPS } }
        set { lock.withLock { _maxFPS = newValue } }
    }
    private var _maxFPS: CMTimeScale = CameraCapture.defaultCaptureFPS

    /// The underlying session. Exposed so a SwiftUI `AVCaptureVideoPreviewLayer`
    /// can attach to it for the full-bleed live preview, per the shared contract.
    public var session: AVCaptureSession { _session }
    private let _session = AVCaptureSession()

    /// Serial queue owning all session mutation / start / stop.
    private let sessionQueue = DispatchQueue(label: "com.fir4s.fever.camera.session")

    /// Serial queue receiving sample-buffer callbacks.
    private let outputQueue = DispatchQueue(label: "com.fir4s.fever.camera.frames",
                                            qos: .userInitiated)

    private let videoOutput = AVCaptureVideoDataOutput()

    /// Preferred camera `uniqueID` (e.g. a GoPro/USB webcam chosen in Settings).
    /// nil → auto-pick the BUILT-IN camera (lowest latency). Set before start to force one.
    public var preferredDeviceID: String?

    /// All connected video cameras (built-in + external/USB/Continuity), for the
    /// Settings camera picker. CACHED: a SwiftUI Settings re-render (e.g. each keystroke
    /// in a text field) calls this, and a synchronous `DiscoverySession` enumeration can
    /// hitch the main thread (notably while a Continuity camera wakes). The cache is
    /// invalidated when a device is connected/disconnected.
    private static let camCacheLock = NSLock()
    nonisolated(unsafe) private static var camCache: [AVCaptureDevice]?
    nonisolated(unsafe) private static var camObserversInstalled = false

    public static func availableCameras() -> [AVCaptureDevice] {
        camCacheLock.withLock {
            if let cached = camCache { return cached }
            installCamObserversLocked()
            var types: [AVCaptureDevice.DeviceType] = [.builtInWideAngleCamera]
            types.append(.external)
            types.append(.continuityCamera)
            let devices = AVCaptureDevice.DiscoverySession(deviceTypes: types, mediaType: .video,
                                                           position: .unspecified).devices
            camCache = devices
            return devices
        }
    }

    /// Register (once) for device connect/disconnect so the camera list cache is dropped
    /// when the hardware actually changes. Called under `camCacheLock`.
    private static func installCamObserversLocked() {
        guard !camObserversInstalled else { return }
        camObserversInstalled = true
        let clear: @Sendable (Notification) -> Void = { _ in camCacheLock.withLock { camCache = nil } }
        let nc = NotificationCenter.default
        nc.addObserver(forName: AVCaptureDevice.wasConnectedNotification, object: nil, queue: nil, using: clear)
        nc.addObserver(forName: AVCaptureDevice.wasDisconnectedNotification, object: nil, queue: nil, using: clear)
    }

    /// Choose the capture device: the explicitly-preferred one if connected, else the
    /// BUILT-IN camera. External/virtual cams (OBS Virtual, Continuity, action-cam webcam
    /// bridges) register as `.external` but are higher-latency or simply wrong for tracking
    /// — measured: a GoPro in webcam mode is ~430ms glass→frame vs the built-in's ~60–100ms.
    /// So auto-pick the built-in and only use an external when the user picks it in Settings.
    private func pickDevice() -> AVCaptureDevice? {
        let cams = Self.availableCameras()
        if let id = preferredDeviceID, let d = cams.first(where: { $0.uniqueID == id }) { return d }
        return AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .unspecified)
            ?? AVCaptureDevice.default(for: .video)
            ?? cams.first
    }

    /// Legacy frame sink (FrameSource protocol). Retained for compatibility; in
    /// the live path the inference worker pulls frames from the mailbox via
    /// `nextFrame()` instead, so this is normally left `nil`.
    ///
    /// Lock-backed: written on the main actor (pipeline start/stop) but read on
    /// the capture `outputQueue`, so the same `NSLock` that guards the mailbox
    /// serializes both ends (the compiler cannot see this because `self` crosses
    /// the boundary via `nonisolated(unsafe)`).
    public var onFrame: ((CVPixelBuffer, TimeInterval) -> Void)? {
        get { lock.withLock { _onFrame } }
        set { lock.withLock { _onFrame = newValue } }
    }

    /// Observed when the resolved authorization state changes (e.g. after the
    /// system permission prompt is answered). Invoked on an arbitrary queue;
    /// the pipeline hops to the main actor before publishing.
    ///
    /// Lock-backed for the same reason as `onFrame`: written on the main actor,
    /// read from the AVFoundation `requestAccess` completion queue inside
    /// `setAuthorization`.
    public var onAuthorizationChange: ((CameraAuthorization) -> Void)? {
        get { lock.withLock { _onAuthorizationChange } }
        set { lock.withLock { _onAuthorizationChange = newValue } }
    }

    // MARK: - FlowLimiter (retained for FrameSource conformance)
    //
    // The new design does not rely on the delegate blocking, so these are no
    // longer used to gate inline inference. They remain to satisfy the protocol
    // and the stub; the live worker uses the mailbox instead.

    public var isInferring: Bool {
        get { lock.withLock { _isInferring } }
        set { lock.withLock { _isInferring = newValue } }
    }

    public func tryBeginInferring() -> Bool {
        lock.withLock {
            if _isInferring { return false }
            _isInferring = true
            return true
        }
    }

    public func endInferring() {
        lock.withLock { _isInferring = false }
    }

    /// Running count of OS-dropped (late) frames. Intentional process-latest-only
    /// mailbox overwrites are NOT counted — skipping a stale frame to keep the
    /// freshest one is by design (the source of the low-latency behavior), not a defect.
    public var droppedFrames: Int {
        lock.withLock { _droppedFrames }
    }

    /// Current resolved authorization state.
    public var authorization: CameraAuthorization {
        lock.withLock { _authorization }
    }

    // MARK: - Single-slot mailbox + lock-protected state

    private let lock = NSLock()
    private var _isInferring = false
    private var _droppedFrames = 0
    private var isConfigured = false
    private var isAuthorized = false
    private var _authorization: CameraAuthorization = .notDetermined

    /// Lock-guarded backing storage for the public closure properties (see
    /// `onFrame` / `onAuthorizationChange`).
    private var _onFrame: ((CVPixelBuffer, TimeInterval) -> Void)?
    private var _onAuthorizationChange: ((CameraAuthorization) -> Void)?

    /// The single-slot mailbox: the most recent unconsumed frame. A new arrival
    /// overwrites (and drop-counts) any frame still sitting here.
    private var pendingBuffer: CVPixelBuffer?
    private var pendingTime: TimeInterval = 0

    public override init() {
        super.init()
    }

    // MARK: - Lifecycle

    /// Start capture. Resolves camera authorization first:
    ///  - `.authorized`     → configure + run immediately.
    ///  - `.notDetermined`  → request access; run if/when granted.
    ///  - `.denied`/`.restricted` → publish `.denied`, do NOT run, do NOT crash.
    public func start() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            setAuthorization(.authorized)
            beginSession()
        case .notDetermined:
            setAuthorization(.notDetermined)
            // `self` is non-Sendable; confine the continuation to a captured
            // reference. AVFoundation invokes the handler on an internal queue.
            nonisolated(unsafe) let capture = self
            AVCaptureDevice.requestAccess(for: .video) { granted in
                if granted {
                    capture.setAuthorization(.authorized)
                    capture.beginSession()
                } else {
                    capture.setAuthorization(.denied)
                }
            }
        case .denied, .restricted:
            setAuthorization(.denied)
        @unknown default:
            setAuthorization(.denied)
        }
    }

    /// Configure (once) and start the session on the session queue.
    private func beginSession() {
        nonisolated(unsafe) let capture = self
        sessionQueue.async {
            capture.configureIfNeeded()
            guard capture.lock.withLock({ capture._isAuthorizedConfigured }) else { return }
            if !capture._session.isRunning {
                capture._session.startRunning()
            }
        }
    }

    public func stop() {
        nonisolated(unsafe) let capture = self
        sessionQueue.async {
            if capture._session.isRunning {
                capture._session.stopRunning()
            }
        }
        // Drop any buffer still waiting in the mailbox so it does not pin a
        // CoreVideo buffer after stop.
        lock.withLock {
            pendingBuffer = nil
            _isInferring = false
        }
    }

    /// Bind a preview layer to this camera's session ON the session queue, so the
    /// (session-mutating) `layer.session =` assignment serializes with `startRunning()` /
    /// `stopRunning()`.
    ///
    /// CRITICAL: `AVCaptureSession` is not safe to mutate from one thread while another
    /// enumerates it. `AVCaptureVideoPreviewLayer.session = …` internally does
    /// `addVideoPreviewLayer:` → `commitConfiguration` (a session mutation). If that runs on
    /// the main thread at the same moment `startRunning()` enumerates the session's
    /// connections on the session queue, the process ABORTS with
    /// `__NSFastEnumerationMutationHandler` ("collection mutated while being enumerated").
    /// That exact race aborted on-device Start. The layer is created/configured (gravity,
    /// frame, mirror transform) on the main thread by the caller — only the session bind,
    /// the one operation that touches the session, is deferred onto this queue.
    public func attachPreview(_ layer: AVCaptureVideoPreviewLayer, to session: AVCaptureSession) {
        // Both are non-Sendable AVFoundation objects; this hop deliberately serializes
        // their use with the rest of the session's lifecycle on `sessionQueue`.
        nonisolated(unsafe) let lyr = layer
        nonisolated(unsafe) let sess = session
        sessionQueue.async {
            if lyr.session !== sess { lyr.session = sess }
        }
    }

    /// Unbind a preview layer from this session on the session queue, BEFORE the layer is
    /// deallocated. `layer.session = nil` runs `removeVideoPreviewLayer:`, which cleanly
    /// tears the layer's `CAImageQueue` out of the capture graph. Without it, dismantling the
    /// SwiftUI preview (a mode switch / `session`-change / Stop) frees the layer while the
    /// session still references its image queue, and the next `stopRunning()` SEGFAULTS in
    /// `CAImageQueueInvalidate` — especially with an external GoPro/virtual camera, whose CMIO
    /// graph teardown is fragile. The async closure retains `layer` until the unbind runs, so
    /// the layer can't deallocate mid-detach; serialized on `sessionQueue` with start/stop.
    public func detachPreview(_ layer: AVCaptureVideoPreviewLayer) {
        nonisolated(unsafe) let lyr = layer
        sessionQueue.async {
            if lyr.session !== nil { lyr.session = nil }
        }
    }

    // MARK: - Mailbox API (consumed by the inference worker)

    /// Pull the latest unconsumed frame, clearing the mailbox. Returns `nil`
    /// when no new frame has arrived since the last call. Non-blocking.
    public func nextFrame() -> (CVPixelBuffer, TimeInterval)? {
        lock.withLock {
            guard let buf = pendingBuffer else { return nil }
            pendingBuffer = nil
            return (buf, pendingTime)
        }
    }

    // MARK: - Authorization plumbing

    /// Whether the session graph is configured AND access was granted. Read
    /// inside the lock; written from `configureIfNeeded` / `setAuthorization`.
    private var _isAuthorizedConfigured: Bool {
        isAuthorized && isConfigured
    }

    private func setAuthorization(_ state: CameraAuthorization) {
        // Snapshot the handler under the SAME lock that the main-actor setter
        // uses, so a concurrent write to `onAuthorizationChange` cannot race this
        // background read. Invoke it AFTER releasing the lock.
        let (changed, cb): (Bool, ((CameraAuthorization) -> Void)?) = lock.withLock {
            let was = _authorization
            _authorization = state
            isAuthorized = (state == .authorized)
            return (was != state, _onAuthorizationChange)
        }
        if changed { cb?(state) }
    }

    // MARK: - Session configuration

    /// One-time session graph configuration. Runs on `sessionQueue`.
    private func configureIfNeeded() {
        guard !isConfigured else { return }

        _session.beginConfiguration()
        _session.sessionPreset = .hd1280x720   // 720p: ample for body pose, fast.

        // Preferred external camera (GoPro/USB) if present, else built-in default.
        guard let device = pickDevice(),
              let input = try? AVCaptureDeviceInput(device: device),
              _session.canAddInput(input) else {
            _session.commitConfiguration()
            return
        }
        _session.addInput(input)

        // Force a STEADY capture rate. Locking min == max frame duration stops
        // auto-exposure from dropping the capture rate in lower light (that
        // throttling once pinned the pipeline to ~17fps). We do NOT touch
        // activeFormat (the preset already gives 720p and changing the format
        // manually breaks frame delivery on this SDK); we only pin the frame
        // duration. Capture is capped at `maxFPS` (default 30) — still FASTER than
        // the model infers — so the latest-only worker always grabs the freshest
        // frame and the IRL→VR latency floor is minimized.
        applyFrameRate(to: device)
        minimizeLatencyEffects(on: device)

        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.setSampleBufferDelegate(self, queue: outputQueue)

        guard _session.canAddOutput(videoOutput) else {
            _session.commitConfiguration()
            return
        }
        _session.addOutput(videoOutput)

        _session.commitConfiguration()
        lock.withLock { isConfigured = true }
    }

    /// Pin the capture device toward `maxFPS`, but ONLY to a frame duration the device's
    /// active format actually supports. Setting `activeVideoMin/MaxFrameDuration` to a
    /// value outside `videoSupportedFrameRateRanges` throws an UNCATCHABLE ObjC
    /// `NSException` → the process aborts. Virtual/DAL cameras (DroidCam, OBS,
    /// Continuity) and many external webcams cap below 30, so we clamp our desired
    /// 1/maxFPS into the device's real CMTime bounds (and skip entirely if it advertises
    /// no ranges, letting the device run free).
    private func applyFrameRate(to device: AVCaptureDevice) {
        guard (try? device.lockForConfiguration()) != nil else { return }
        defer { device.unlockForConfiguration() }

        // Prefer the range with the highest capability (fastest supported fps).
        guard let range = device.activeFormat.videoSupportedFrameRateRanges
                .max(by: { $0.maxFrameRate < $1.maxFrameRate }) else {
            return   // device exposes no frame-rate control → leave it alone
        }

        // Desired = 1/maxFPS, clamped into [minFrameDuration, maxFrameDuration] using the
        // device's EXACT CMTime bounds (so the value is always in-range and can never
        // trigger the exception). minFrameDuration = the SHORTEST duration = the FASTEST
        // fps the device supports — so clamping our (longer) 1/30 against it caps the cam
        // at 30 even when it can do 60+, while never asking for more than it can deliver.
        var duration = CMTime(value: 1, timescale: max(1, maxFPS))
        if CMTimeCompare(duration, range.minFrameDuration) < 0 { duration = range.minFrameDuration }
        if CMTimeCompare(duration, range.maxFrameDuration) > 0 { duration = range.maxFrameDuration }
        guard duration.isValid, duration.isNumeric else { return }

        device.activeVideoMinFrameDuration = duration
        device.activeVideoMaxFrameDuration = duration
    }

    /// Kill the macOS camera "video effects" that add per-frame processing latency.
    /// On Apple-silicon built-in cameras these run on the live feed and are the main
    /// controllable latency adder (the 30fps frame period is fixed — the sensor has no
    /// faster format). Center Stage reframes/crops through an extra pass: we take APP
    /// control of it and force it OFF so the system can't silently re-enable it for
    /// tracking. Reactions runs continuous hand-gesture detection on every frame but is
    /// system-owned (no app setter) — we can only detect it and tell the user to turn it
    /// off in Control Center ▸ Video Effects, which is a real latency win when it's on.
    private func minimizeLatencyEffects(on device: AVCaptureDevice) {
        if device.activeFormat.isCenterStageSupported {
            AVCaptureDevice.centerStageControlMode = .app
            AVCaptureDevice.isCenterStageEnabled = false
        }
        if #available(macOS 14.0, *), AVCaptureDevice.reactionEffectsEnabled {
            NSLog("[Fever camera] ⚠︎ Reactions video-effect is ON — it runs per-frame gesture "
                + "detection and adds latency. Turn it OFF in Control Center ▸ Video Effects for lowest lag.")
        }
    }

    /// Update the capture frame-rate cap and re-apply it to the live device (if the
    /// session is already configured), so a Settings change takes effect without a
    /// restart. Serialized on the session queue alongside the other session mutations.
    public func setMaxFPS(_ fps: Int) {
        let ts = CMTimeScale(min(240, max(1, fps)))
        nonisolated(unsafe) let capture = self
        sessionQueue.async {
            capture.maxFPS = ts
            guard capture.lock.withLock({ capture.isConfigured }), let device = capture.pickDevice() else { return }
            capture.applyFrameRate(to: device)
        }
    }

    /// Switch the active camera live (Settings picker). Swaps the session's video
    /// input on the session queue; if not yet configured, just records the choice
    /// so the next `start()` uses it.
    public func selectCamera(_ deviceID: String?) {
        sessionQueue.async { [weak self] in
            guard let self else { return }
            self.preferredDeviceID = deviceID
            guard self.lock.withLock({ self.isConfigured }) else { return }
            // Build the NEW input BEFORE removing the working one — if the requested
            // device can't be opened, keep the current camera running instead of
            // tearing it down into a frameless (silently green) session.
            guard let device = self.pickDevice(),
                  let input = try? AVCaptureDeviceInput(device: device) else { return }
            self._session.beginConfiguration()
            let previous = self._session.inputs
            for old in previous { self._session.removeInput(old) }
            if self._session.canAddInput(input) {
                self._session.addInput(input)
                self.applyFrameRate(to: device)
            } else {
                // Roll back to the previous input(s) so we never end input-less.
                for old in previous where self._session.canAddInput(old) { self._session.addInput(old) }
            }
            self._session.commitConfiguration()
        }
    }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate (nonisolated)

extension CameraCapture: AVCaptureVideoDataOutputSampleBufferDelegate {

    /// Delivered on `outputQueue`. Does the minimum: stash the latest pixel
    /// buffer into the single-slot mailbox and return immediately. NEVER runs
    /// inference and NEVER blocks the capture queue. A previously stashed,
    /// still-unconsumed frame is overwritten and counted as dropped
    /// (process-latest-only).
    public nonisolated func captureOutput(_ output: AVCaptureOutput,
                                          didOutput sampleBuffer: CMSampleBuffer,
                                          from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        let pts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        let time = pts.isValid ? CMTimeGetSeconds(pts) : Date().timeIntervalSince1970

        // Retain the CoreVideo pixel buffer in the mailbox. The CMSampleBuffer
        // itself is NOT retained past this call; the CVPixelBuffer is a separate
        // reference-counted CV object and is safe to hold.
        let sink: ((CVPixelBuffer, TimeInterval) -> Void)? = lock.withLock {
            // Overwriting an unconsumed frame is INTENTIONAL (process-latest-only:
            // always keep the freshest frame for the lowest latency), so it is NOT a
            // dropped frame and is not counted — only OS-dropped late frames (the
            // didDrop callback) are real drops worth surfacing.
            //
            // When a live sink is set (PC-offload siphons frames straight to the
            // encoder), the pull mailbox is never drained — so don't pin a second pool
            // buffer in it; that only raises late-drop pressure for no consumer.
            if _onFrame == nil {
                pendingBuffer = pixelBuffer
                pendingTime = time
            }
            return _onFrame
        }
        // Secondary tap: in PC-offload mode the controller sets `onFrame` to siphon
        // frames to the ffmpeg encoder. nil in the local path, so this is a no-op
        // there. The sink MUST be non-blocking (it stashes for an async encoder) —
        // it must never run inference or block the capture queue.
        sink?(pixelBuffer, time)
    }

    /// The OS dropped a late frame before delivery — count it.
    public nonisolated func captureOutput(_ output: AVCaptureOutput,
                                          didDrop sampleBuffer: CMSampleBuffer,
                                          from connection: AVCaptureConnection) {
        lock.withLock { _droppedFrames += 1 }
    }
}
