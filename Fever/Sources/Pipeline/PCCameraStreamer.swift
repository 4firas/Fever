import Foundation
import CoreVideo
#if canImport(Darwin)
import Darwin
#endif

/// In PC-offload mode Fever owns the camera (so it can show the live preview), so
/// ffmpeg can't open the camera itself — instead Fever pipes the captured frames
/// into ffmpeg's `rawvideo` stdin, and ffmpeg encodes low-delay H.264 and streams
/// it (mpegts/udp) to the PC.
///
/// ffmpeg is launched LAZILY on the first frame so its `-video_size` is set to the
/// camera's ACTUAL delivered dimensions — not a guess. (External / Continuity /
/// virtual cams don't always honour the 720p preset, and a wrong rawvideo size
/// desyncs every frame into garbage on frame 1.)
///
/// `feed(_:)` runs on the camera's capture queue and must never block it: it only
/// RETAINS the latest pixel buffer (latest-wins) and signals the writer — the
/// expensive BGRA pack happens on the writer thread, NOT on the capture queue that
/// also feeds the live preview. If ffmpeg falls behind, intermediate frames are
/// dropped — exactly the process-latest-only behavior that keeps latency low.
///
/// Internally synchronized (NSLock + a serial writer queue woken by a semaphore),
/// hence `@unchecked Sendable` so the @MainActor controller can hand `feed` to the
/// nonisolated capture callback.
final class PCCameraStreamer: @unchecked Sendable {

    private let ffmpegPath: String
    private let host: String
    private let port: Int
    private let outW: Int, outH: Int, fps: Int, bitrateMbps: Int
    private let flip: Bool
    private let onExit: @Sendable (Int32) -> Void

    private let writeQueue = DispatchQueue(label: "com.fir4s.fever.pc.stream", qos: .userInitiated)
    private let lock = NSLock()
    private let frameReady = DispatchSemaphore(value: 0)   // signalled per stored frame; wakes the writer
    private var latestPB: CVPixelBuffer?                    // retained latest frame (latest-wins)
    private var running = true
    private var ffmpeg: Process?
    private var stdin: FileHandle?
    private var camW = 0, camH = 0          // pinned from the FIRST frame
    private var packBuffer = Data()         // reused on the writer queue (no per-frame malloc)
    private var lastFeedUptime: TimeInterval = 0   // wall-clock of the last accepted frame

    /// - outW/outH/fps: the ENCODED stream size/rate (ffmpeg scales/resamples).
    /// ffmpeg's input geometry is taken from the first camera frame, not assumed.
    init(ffmpegPath: String, host: String, port: Int,
         outW: Int, outH: Int, fps: Int, bitrateMbps: Int, flip: Bool,
         onExit: @escaping @Sendable (Int32) -> Void) {
        self.ffmpegPath = ffmpegPath; self.host = host; self.port = port
        self.outW = outW; self.outH = outH; self.fps = fps
        self.bitrateMbps = bitrateMbps; self.flip = flip; self.onExit = onExit
        startWriter()
    }

    /// True once ffmpeg has been launched — i.e. at least one camera frame arrived
    /// and the encoder spun up. The controller's first-frame watchdog reads this to
    /// catch a silent no-frames stall (camera authorized but delivering nothing).
    var hasLaunched: Bool { lock.withLock { ffmpeg != nil } }

    /// Seconds since the last frame was accepted (0 before the first frame). The
    /// controller's heartbeat reads this to catch a mid-stream camera stall — a frozen
    /// pipeline the one-shot first-frame watchdog can't see.
    var secondsSinceLastFeed: Double {
        lock.withLock { lastFeedUptime == 0 ? 0 : ProcessInfo.processInfo.systemUptime - lastFeedUptime }
    }

    /// Capture-queue entry point. Stays cheap: launch ffmpeg on the first frame
    /// (sized to it), retain the latest pixel buffer (ARC retain — drops the previous,
    /// latest-wins), and wake the writer. The pack/copy is deferred to the writer.
    func feed(_ pb: CVPixelBuffer) {
        let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)
        let stored: Bool = lock.withLock {
            guard running else { return false }
            if ffmpeg == nil { launchLocked(camW: w, camH: h) }
            // Ignore a mid-stream resolution change (a fixed rawvideo size can't follow it).
            guard w == camW, h == camH else { return false }
            latestPB = pb
            lastFeedUptime = ProcessInfo.processInfo.systemUptime
            return true
        }
        if stored { frameReady.signal() }
    }

    /// Build + launch ffmpeg with the real input geometry. Called under `lock`.
    private func launchLocked(camW: Int, camH: Int) {
        self.camW = camW; self.camH = camH
        var vf = "scale=\(outW):\(outH),fps=\(fps)"
        if flip { vf = "hflip,\(vf)" }                    // PinoFBT cv2.flip parity

        let p = Process()
        p.executableURL = URL(fileURLWithPath: ffmpegPath)
        p.arguments = [
            "-hide_banner", "-loglevel", "warning", "-stats",
            "-fflags", "+nobuffer",                       // don't buffer input frames
            "-f", "rawvideo", "-pixel_format", "bgra",
            "-video_size", "\(camW)x\(camH)", "-framerate", "\(fps)",
            "-i", "-",
            "-an",                                        // no audio track
            "-vf", vf,
            "-c:v", "h264_videotoolbox", "-realtime", "1",
            "-b:v", "\(bitrateMbps)M",
            "-g", "\(max(1, fps / 2))", "-bf", "0", "-pix_fmt", "yuv420p",
            // Push each packet out immediately instead of letting the mpegts muxer
            // accumulate — shaves buffering latency off the Mac→PC hop.
            "-flush_packets", "1", "-muxdelay", "0", "-muxpreload", "0",
            "-f", "mpegts", "udp://\(host):\(port)?pkt_size=1316"
        ]
        let inPipe = Pipe()
        p.standardInput = inPipe
        p.standardOutput = FileHandle.nullDevice
        let logPath = (NSTemporaryDirectory() as NSString).appendingPathComponent("fever_pc_stream.log")
        FileManager.default.createFile(atPath: logPath, contents: nil)
        p.standardError = FileHandle(forWritingAtPath: logPath) ?? FileHandle.nullDevice
        p.terminationHandler = { [onExit] proc in onExit(proc.terminationStatus) }
        do { try p.run() } catch { onExit(-1); return }
        let wfh = inPipe.fileHandleForWriting
        // Without this, a write after ffmpeg closes the read-end raises SIGPIPE,
        // which (unhandled in the GUI) kills the whole app. Turn it into a normal
        // EPIPE error the writer can catch.
        _ = fcntl(wfh.fileDescriptor, F_SETNOSIGPIPE, 1)
        self.ffmpeg = p
        self.stdin = wfh
    }

    private func startWriter() {
        // Capture the semaphore + lock STRONGLY (they must outlive `self` so a parked
        // writer can be woken from deinit), but hold `self` WEAKLY and re-bind it AFTER
        // the park each iteration — otherwise the parked thread would pin the streamer
        // forever and it could never deinit.
        let frameReady = self.frameReady
        let lock = self.lock
        writeQueue.async { [weak self] in
            while true {
                frameReady.wait()                         // block until a frame lands (or stop/deinit signals)
                guard let self else { return }            // streamer deinited → exit cleanly
                guard lock.withLock({ self.running }) else { break }
                let (pb, fh): (CVPixelBuffer?, FileHandle?) = lock.withLock {
                    let x = self.latestPB; self.latestPB = nil; return (x, self.stdin)
                }
                guard let pb, let fh else { continue }    // spurious wake / ffmpeg not up yet
                guard let data = self.packLatest(pb) else { continue }
                do { try fh.write(contentsOf: data) }
                catch { break }                           // ffmpeg closed the pipe → done
            }
        }
    }

    deinit {
        // Safety net if stop() was never called: wake the parked writer so it observes
        // the dead streamer and exits instead of leaking its thread.
        lock.withLock { running = false; latestPB = nil }
        frameReady.signal()
    }

    func stop() {
        let proc: Process? = lock.withLock { running = false; latestPB = nil; return ffmpeg }
        frameReady.signal()                               // wake an idle writer so it exits wait()
        // Terminate ffmpeg DIRECTLY (Process.terminate is thread-safe). This closes
        // ffmpeg's stdin read-end, so a writer wedged inside a blocking `write()` (a
        // stalled-but-alive encoder) gets EPIPE and breaks — otherwise the close below,
        // queued behind that same wedged write on the serial queue, could never run.
        if let proc, proc.isRunning { proc.terminate() }
        // Serialize the handle close on the writer queue so it can't race a mid-flight write.
        writeQueue.async { [self] in
            try? stdin?.close()
            if let f = ffmpeg, f.isRunning { f.terminate() }
        }
    }

    /// Copy a BGRA pixel buffer into tightly-packed bytes (stripping any row-stride
    /// padding) so ffmpeg's rawvideo reader gets exactly width*4*height bytes. Runs
    /// ONLY on the writer queue and reuses `packBuffer`, so the malloc + zero-fill is
    /// paid once (on the first frame) instead of every frame.
    private func packLatest(_ pb: CVPixelBuffer) -> Data? {
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pb) else { return nil }
        let w = CVPixelBufferGetWidth(pb)
        let h = CVPixelBufferGetHeight(pb)
        let srcStride = CVPixelBufferGetBytesPerRow(pb)
        let dstStride = w * 4
        let needed = dstStride * h
        if packBuffer.count != needed { packBuffer = Data(count: needed) }   // one-time alloc
        packBuffer.withUnsafeMutableBytes { (dst: UnsafeMutableRawBufferPointer) in
            guard let dstBase = dst.baseAddress else { return }
            if srcStride == dstStride {
                memcpy(dstBase, base, needed)              // no padding → single copy
            } else {
                for row in 0..<h {
                    memcpy(dstBase + row * dstStride, base + row * srcStride, dstStride)
                }
            }
        }
        return packBuffer
    }
}
