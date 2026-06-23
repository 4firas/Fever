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
/// frame is overwritten (process-latest-only) and counted as dropped. This keeps
/// the capture graph — and therefore the `AVCaptureVideoPreviewLayer` preview —
/// perfectly smooth regardless of how slow downstream inference is.
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

    /// Capture frame rate, pinned to the NLF inference ceiling so preview == inference.
    public static let captureFPS: CMTimeScale = 15

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

    /// Running count of dropped frames: OS-dropped (late) frames, plus frames
    /// overwritten in the single-slot mailbox before the worker consumed them.
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

        // Built-in / front camera. Default video device, NOT a back-position
        // lookup (which is nil on a Mac).
        guard let device = AVCaptureDevice.default(for: .video),
              let input = try? AVCaptureDeviceInput(device: device),
              _session.canAddInput(input) else {
            _session.commitConfiguration()
            return
        }
        _session.addInput(input)

        // Force a STEADY 30fps (this webcam's hardware max). Locking min == max
        // frame duration stops auto-exposure from dropping the capture rate in
        // lower light — that throttling was pinning the pipeline to ~17fps. We
        // do NOT touch activeFormat (the preset already gives 720p and changing
        // the format manually breaks frame delivery on this SDK); we only pin the
        // frame duration, so frame delivery is unchanged and there's no accuracy
        // cost.
        // Pin the camera to the inference rate so the preview, the data output, and
        // the model all run at the SAME fps: every captured frame gets inferred and
        // the skeleton sits on exactly the frame shown (no 30 fps video with a 15 fps
        // skeleton floating on top). The NLF model sustains ~15 fps on this machine.
        if (try? device.lockForConfiguration()) != nil {
            let target = CMTime(value: 1, timescale: CameraCapture.captureFPS)
            device.activeVideoMinFrameDuration = target
            device.activeVideoMaxFrameDuration = target
            device.unlockForConfiguration()
        }

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
        lock.withLock {
            if pendingBuffer != nil {
                // The worker hadn't consumed the previous frame yet — drop it.
                _droppedFrames += 1
            }
            pendingBuffer = pixelBuffer
            pendingTime = time
        }
    }

    /// The OS dropped a late frame before delivery — count it.
    public nonisolated func captureOutput(_ output: AVCaptureOutput,
                                          didDrop sampleBuffer: CMSampleBuffer,
                                          from connection: AVCaptureConnection) {
        lock.withLock { _droppedFrames += 1 }
    }
}
