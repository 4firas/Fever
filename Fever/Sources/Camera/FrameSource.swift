import CoreVideo
import Foundation

/// Abstract frame source feeding the pipeline. `CameraCapture` is the live
/// impl; `StubFrameSource` synthesizes frames on a timer for headless tests.
public protocol FrameSource: AnyObject {
    var onFrame: ((CVPixelBuffer, TimeInterval) -> Void)? { get set }
    var isInferring: Bool { get set }

    /// Atomically claim the single in-flight inference slot. Returns `true` if
    /// the caller now owns it (must pair with `endInferring()`), `false` if a
    /// previous frame is still in flight and this one should be dropped. This
    /// single lock region enforces the FlowLimiter "exactly one frame in flight"
    /// invariant that the synchronous inference bridge depends on, rather than a
    /// non-atomic check-then-set straddling two queues.
    func tryBeginInferring() -> Bool
    /// Release the in-flight inference slot claimed by `tryBeginInferring()`.
    func endInferring()

    func start()
    func stop()
}

/// Synthesizes a steady stream of blank BGRA CVPixelBuffers at ~30 fps so the
/// pipeline (landmarker → solver → OSC) can be exercised with no camera.
/// Pair with `StubPoseLandmarker` for a fully hardware-free run.
public final class StubFrameSource: FrameSource {

    public var onFrame: ((CVPixelBuffer, TimeInterval) -> Void)?

    private let lock = NSLock()
    private var _isInferring = false
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

    private var timer: DispatchSourceTimer?
    private let queue = DispatchQueue(label: "fever.stub.frames", qos: .userInitiated)
    private let width: Int
    private let height: Int

    public init(width: Int = 256, height: Int = 256) {
        self.width = width
        self.height = height
    }

    public func start() {
        guard timer == nil else { return }
        let t = DispatchSource.makeTimerSource(queue: queue)
        t.schedule(deadline: .now(), repeating: 1.0 / 30.0)
        t.setEventHandler { [weak self] in self?.emit() }
        t.resume()
        timer = t
    }

    public func stop() {
        timer?.cancel()
        timer = nil
    }

    private func emit() {
        // Atomically claim the single in-flight slot for the span of the
        // synchronous callback, mirroring CameraCapture — the FlowLimiter
        // invariant is enforced by this one lock region, not by incidental
        // same-queue serialization.
        guard tryBeginInferring() else { return }
        defer { endInferring() }
        guard let pb = Self.makeBlankBuffer(width: width, height: height) else { return }
        onFrame?(pb, Date().timeIntervalSince1970)
    }

    private static func makeBlankBuffer(width: Int, height: Int) -> CVPixelBuffer? {
        var pb: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: false,
            kCVPixelBufferCGBitmapContextCompatibilityKey: false
        ]
        let status = CVPixelBufferCreate(kCFAllocatorDefault,
                                         width, height,
                                         kCVPixelFormatType_32BGRA,
                                         attrs as CFDictionary, &pb)
        guard status == kCVReturnSuccess, let buf = pb else { return nil }
        CVPixelBufferLockBaseAddress(buf, [])
        let ptr = CVPixelBufferGetBaseAddress(buf)
        let bytes = CVPixelBufferGetDataSize(buf)
        if let ptr { memset(ptr, 0, bytes) }
        CVPixelBufferUnlockBaseAddress(buf, [])
        return buf
    }
}
