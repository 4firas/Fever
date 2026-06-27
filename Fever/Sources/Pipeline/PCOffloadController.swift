import Foundation
import Observation
import simd
import AVFoundation
#if canImport(Darwin)
import Darwin
#endif

/// Drives "Inference on PC" mode: wake the GPU PC (Wake-on-LAN), launch the
/// headless byte-exact PinoFBT daemon on it over SSH, and stream this Mac's
/// camera to it as low-latency H.264. In this mode the **PC** runs the model +
/// IK and emits the VRChat OSC; the Mac is only the camera + the orchestrator.
///
/// All blocking work (WoL wait, SSH) runs off the main actor via `bg(_:)`; the
/// observable `phase`/`status` are updated on the main actor for SwiftUI. The
/// local `ffmpeg` capture/stream process is owned here and terminated on stop.
///
/// Nothing here is destructive or visible on the PC: the daemon is launched
/// `-WindowStyle Hidden` with `pythonw.exe` (no console), and stop kills only
/// our own `fbt_daemon` process.
@MainActor
@Observable
public final class PCOffloadController {

    public enum Phase: Sendable, Equatable {
        case idle, waking, starting, streaming, stopping, error
    }

    public private(set) var phase: Phase = .idle
    public private(set) var status: String = "Idle"

    public var isActive: Bool { phase != .idle && phase != .error }

    private var streamer: PCCameraStreamer?
    private var camera: CameraCapture?
    private var worker: Task<Void, Never>?
    private var current: PCOffloadConfig?
    private var skelReceiver: PCSkeletonReceiver?
    private var oscRelay: PCOscRelay?                 // Mac-side OSC bounce (relay route only)
    private var watchdog: Task<Void, Never>?

    /// Skeleton points (the PC's solved joints2D, normalized) sent back for the live
    /// preview overlay in PC mode. Empty until the PC's return channel is delivering.
    public private(set) var previewPoints: [SIMD2<Float>] = []

    public nonisolated init() {}

    /// Begin PC-offload: wake → launch daemon → start streaming. Idempotent while
    /// already active (no-op). Errors surface in `phase`/`status`.
    public func start(_ cfg: PCOffloadConfig, camera: CameraCapture) {
        guard phase == .idle || phase == .error else { return }
        current = cfg
        self.camera = camera
        phase = .waking
        status = "Waking \(cfg.host)…"
        worker = Task { [weak self] in await self?.run(cfg) }
    }

    /// Stop streaming and kill the remote daemon; return to idle.
    public func stop() {
        guard phase != .idle, phase != .stopping else { return }
        let cfg = current
        let inFlight = worker          // may still be mid-wake / mid-startDaemon
        phase = .stopping
        status = "Stopping…"
        worker?.cancel()
        worker = nil
        teardownLocal()
        Task { [weak self] in
            // Let any in-flight launch unwind FIRST, so a daemon that starts during
            // the wake→start window is killed by the stopDaemon below rather than
            // surviving the Stop (it would keep the PC's GPU busy headlessly).
            await inFlight?.value
            var killed = true
            if let cfg {
                killed = (try? await PCOffloadController.bg { PCOrchestrator.stopDaemon(cfg) }) ?? false
            }
            await MainActor.run {
                guard let self else { return }
                self.phase = .idle
                // Don't claim a clean stop if we couldn't reach the PC to kill the daemon.
                self.status = killed ? "Idle"
                    : "Stopped here — couldn't reach PC to stop the daemon"
                self.current = nil
            }
        }
    }

    /// Tear down the Mac-side pieces: stop siphoning frames, stop the encoder, stop
    /// the camera, clear the overlay.
    private func teardownLocal() {
        watchdog?.cancel(); watchdog = nil
        camera?.onFrame = nil
        streamer?.stop(); streamer = nil
        camera?.stop(); camera = nil
        skelReceiver?.stop(); skelReceiver = nil
        oscRelay?.stop(); oscRelay = nil
        previewPoints = []
    }

    /// A failure that originates AFTER we reached `.streaming` (ffmpeg died, or no
    /// frames ever flowed). Surface it, release the Mac-side pieces, and kill the
    /// remote daemon so the PC doesn't keep running the model headlessly. Guarded so
    /// it can't clobber a normal stop()/restart already in progress.
    private func failStreaming(_ message: String) {
        guard phase == .streaming else { return }
        phase = .error
        status = message
        let cfg = current
        teardownLocal()
        Task { if let cfg { _ = try? await PCOffloadController.bg { PCOrchestrator.stopDaemon(cfg) } } }
    }

    /// Synchronous-friendly shutdown for app termination: stop the Mac-side pieces
    /// immediately, then kill the remote daemon off-thread and call `done`. Returns
    /// false if nothing was running (the caller can terminate at once). Without this,
    /// quitting Fever leaves the PC daemon running forever — GPU busy, PC never sleeps.
    @discardableResult
    public func shutdownForQuit(_ done: @escaping @Sendable () -> Void) -> Bool {
        guard phase != .idle else { return false }
        let cfg = current
        phase = .stopping
        worker?.cancel(); worker = nil
        teardownLocal()
        current = nil
        DispatchQueue.global(qos: .userInitiated).async {
            if let cfg { _ = PCOrchestrator.stopDaemon(cfg) }
            done()
        }
        return true
    }

    /// Listen for the PC's returned skeleton and publish it as `previewPoints`.
    /// `flipX` mirrors x to align with the on-screen mirrored preview. The socket +
    /// DispatchSource live in a non-isolated helper (see PCSkeletonReceiver) so the
    /// background handlers don't trip a main-actor isolation assert.
    private func startSkeletonReceiver(port: UInt16, flipX: Bool) {
        skelReceiver = PCSkeletonReceiver(port: port, flipX: flipX) { [weak self] pts in
            Task { @MainActor in self?.previewPoints = pts }
        }
    }

    // MARK: - Orchestration

    private func run(_ cfg: PCOffloadConfig) async {
        do {
            try await PCOffloadController.bg { try PCOrchestrator.wakeAndWait(cfg) }
            if Task.isCancelled { return }
            phase = .starting
            status = "Starting inference on \(cfg.host)…"
            // Skeleton return channel: receive the PC's joints2D for the live overlay.
            // The camera image is streamed RAW (no hflip) and the skeleton is mirrored on
            // the PC, so the returned joints2D are in raw-frame space; x-flip them to align
            // with the ALWAYS-mirrored preview layer (matches the on-device overlay flip).
            let skelPort: UInt16 = 5001
            startSkeletonReceiver(port: skelPort, flipX: true)
            let skelBack = PCOrchestrator.localIP(reaching: cfg.host).map { "\($0):\(skelPort)" }

            // OSC route: Direct → the daemon sends straight to the Quest (cfg.oscIP:Port);
            // Relay via Mac → the daemon sends to THIS Mac's LAN IP on relayPort and a
            // PCOscRelay forwards each datagram on to the Quest (for when the wired PC
            // can't reach the Quest's Wi-Fi subnet but the Mac can). The daemon's immediate
            // target is computed here as immutable `let`s so the detached launch can capture
            // them safely.
            let daemonOsc: (ip: String, port: Int)
            if cfg.relayViaMac {
                guard let macIP = PCOrchestrator.localIP(reaching: cfg.host) else {
                    throw PCOffloadError.stream("couldn't determine this Mac's LAN IP for the OSC relay")
                }
                guard let relay = PCOscRelay(port: UInt16(cfg.relayPort),
                                             forwardHost: cfg.oscIP, forwardPort: UInt16(cfg.oscPort)) else {
                    throw PCOffloadError.stream("couldn't open the OSC relay on port \(cfg.relayPort)")
                }
                oscRelay = relay
                daemonOsc = (macIP, cfg.relayPort)
            } else {
                daemonOsc = (cfg.oscIP, cfg.oscPort)
            }
            try await PCOffloadController.bg {
                try PCOrchestrator.startDaemon(cfg, oscIP: daemonOsc.ip, oscPort: daemonOsc.port, skeletonBack: skelBack)
            }
            if Task.isCancelled { return }

            // Fever owns the camera (so the live preview works); ffmpeg can't open it
            // too, so we start the camera here and pipe its frames into the encoder.
            guard let cam = camera else { throw PCOffloadError.stream("no camera") }
            // Don't claim "Streaming" if the camera is blocked — that's a silent dead end.
            let auth = AVCaptureDevice.authorizationStatus(for: .video)
            if auth == .denied || auth == .restricted {
                throw PCOffloadError.stream("camera access denied — enable it in System Settings ▸ Privacy ▸ Camera")
            }
            // Preflight the encoder: if ffmpeg isn't where we expect, fail with an
            // actionable message NOW rather than spawning, failing with exit -1, and
            // looping a relaunch storm on every frame.
            guard let ffmpeg = PCOrchestrator.resolveFFmpeg(cfg.ffmpeg) else {
                throw PCOffloadError.stream("ffmpeg not found (looked in \(cfg.ffmpeg)) — install it with: brew install ffmpeg")
            }
            cam.start()
            // ffmpeg is launched lazily inside the streamer on the first frame, sized to
            // the camera's ACTUAL dimensions (any camera, not just 720p).
            let s = PCCameraStreamer(
                ffmpegPath: ffmpeg, host: cfg.host, port: cfg.streamPort,
                outW: cfg.streamW, outH: cfg.streamH,
                fps: cfg.streamFPS, bitrateMbps: cfg.bitrateMbps,
                onExit: { [weak self] code in
                    // ffmpeg early-exit (encoder/UDP failure, spawn failure=-1) → surface it
                    // and tear down (release the camera, kill the remote daemon) instead of
                    // falsely reading "Streaming". No-op on a normal stop().
                    Task { @MainActor in self?.failStreaming("Camera stream ended (ffmpeg exit \(code))") }
                })
            streamer = s
            cam.onFrame = { [weak s] pb, _ in s?.feed(pb) }   // siphon frames to the encoder
            phase = .streaming
            let route = cfg.relayViaMac ? "   ·   OSC → \(cfg.oscIP):\(cfg.oscPort) (via Mac)"
                                        : "   ·   OSC → \(cfg.oscIP):\(cfg.oscPort)"
            status = "Streaming \(cfg.model == .gvhmr ? "GVHMR" : "NLF") → \(cfg.host)\(route)"
            // First-frame watchdog: a camera can be authorized yet deliver nothing
            // (held by another app, asleep Continuity cam). If no frame has launched
            // ffmpeg within a few seconds, flip to .error instead of showing a green
            // "Running on PC" over a dead pipeline.
            watchdog = Task { [weak self] in
                // (1) First frame: ffmpeg must launch within 4s. Scoped `if let` so self
                // goes weak again afterwards and the heartbeat loop can re-bind it.
                try? await Task.sleep(for: .seconds(4))
                if let self, !Task.isCancelled, self.phase == .streaming, self.streamer?.hasLaunched != true {
                    self.failStreaming("No camera frames after 4s — the camera may be in use by another app."); return
                }
                // (2) Heartbeat: catch a MID-stream camera stall (frozen pipeline) and a
                // PC daemon that died — both of which the one-shot check above can't see,
                // and both of which would otherwise sit on a green "Running on PC".
                var sinceProbe = 0.0
                while !Task.isCancelled {
                    try? await Task.sleep(for: .seconds(2))
                    guard let self, !Task.isCancelled, self.phase == .streaming else { return }
                    if let st = self.streamer, st.secondsSinceLastFeed > 3 {
                        self.failStreaming("Camera stalled — no frames reached the encoder for 3s."); return
                    }
                    sinceProbe += 2
                    if sinceProbe >= 10, let cfg = self.current {   // probe the PC daemon ~every 10s (invisible SSH)
                        sinceProbe = 0
                        let alive = (try? await PCOffloadController.bg { PCOrchestrator.daemonAlive(cfg) }) ?? true
                        if self.phase == .streaming, !alive {
                            self.failStreaming("PC daemon stopped — it may have crashed loading the model/IK."); return
                        }
                    }
                }
            }
        } catch is CancellationError {
            // stop() already handled teardown
        } catch {
            // A late error from a worker that stop() cancelled (e.g. wakeAndWait
            // throwing .unreachable mid-cancel) must NOT clobber an already-idle or
            // freshly-restarted controller.
            if Task.isCancelled { return }
            phase = .error
            status = "Error: \(error)"
            teardownLocal()
            if let cfg = current { _ = try? await PCOffloadController.bg { PCOrchestrator.stopDaemon(cfg) } }
        }
    }

    /// Run blocking work off the main actor. Forwards the awaiter's cancellation to
    /// the detached task so the worker's between-attempt `Task.isCancelled` checks
    /// actually fire (a bare `Task.detached.value` does NOT propagate cancellation,
    /// which would leave stop() unable to interrupt the wake loop).
    private static func bg<T: Sendable>(_ work: @Sendable @escaping () throws -> T) async throws -> T {
        let task = Task.detached(priority: .userInitiated) { try work() }
        return try await withTaskCancellationHandler {
            try await task.value
        } onCancel: {
            task.cancel()
        }
    }
}

// MARK: - Immutable config snapshot

/// Which pose model the PC daemon runs in PC mode. `.nlf` (default) launches the
/// byte-exact PinoFBT 2.0 NLF daemon (`fbt_daemon.py`, trtlib py311 env); `.gvhmr`
/// launches the world-grounded GVHMR daemon (`gvhmr_daemon.py`, its own `.venv`).
public enum PCModel: String, Sendable, Equatable {
    case nlf
    case gvhmr
}

/// Everything the orchestration needs, captured once at start (Sendable so it can
/// cross into the detached worker). Paths target the deep hidden PC bridge.
public struct PCOffloadConfig: Sendable, Equatable {
    public var host: String          // PC IP for SSH + the UDP video target
    public var user: String          // SSH user
    public var mac: String           // PC NIC MAC for Wake-on-LAN
    public var model: PCModel        // which pose model the daemon runs (NLF default, or GVHMR)
    public var oscIP: String         // the FINAL VRChat OSC target (127.0.0.1 = PCVR, else Quest IP)
    public var oscPort: Int
    /// OSC route. false = Direct: the daemon sends straight to `oscIP:oscPort`. true =
    /// Relay via Mac: the daemon sends to this Mac's LAN IP on `relayPort`, and a Mac-side
    /// `PCOscRelay` forwards every datagram on to `oscIP:oscPort` (the Quest). Use relay
    /// when the wired PC can't reach the Quest's Wi-Fi subnet but the Mac can.
    public var relayViaMac: Bool
    public var relayPort: Int         // local UDP port the Mac relay listens on (relay route)
    public var heightCm: Int
    public var sendElbows: Bool      // 8-point (PinoFBT desktop default)
    public var mirror: Bool          // proper L/R skeleton mirror on the PC (on-device mirrorTracking; NLF)
    public var predictionLeadMs: Int // forward-lead horizon (ms) → --lead-ms; cancels pipeline latency (both models)
    // -- GVHMR-only knobs (ignored for NLF) --
    public var gvhmrK: Int           // GVHMR readout lookahead (frames); latency↔smoothness, default 5
    public var gvhmrMirror: Bool     // GVHMR facing tune → --mirror
    public var gvhmrFlipX: Bool      // GVHMR facing tune → --flip-x
    public var gvhmrMoving: Bool     // GVHMR camera mode: false = Static (fast), true = Moving → --moving
    public var gvhmrFootContact: Bool // GVHMR foot-contact damping (less foot-slide) → --foot-contact
    public var gvhmrNativeRot: Bool  // GVHMR native joint rotations (richer twist/roll) → --native-rot
    public var streamW: Int          // H.264 stream width
    public var streamH: Int          // H.264 stream height
    public var streamFPS: Int        // capture/encode fps
    public var bitrateMbps: Int      // H.264 bitrate
    public var politeMode: Bool      // run the PC daemon below-normal priority when sharing
    public var fpsCap: Int           // PC-side processing cap (0 = unlimited)
    public var cameraName: String?   // avfoundation device name for ffmpeg (nil → "0" = default cam)
    public var streamPort: Int       // UDP port the PC daemon listens on (daemon default = 5000)
    public var interpreter: String   // PC python interpreter: NLF = pythonw.exe (no console); GVHMR = its .venv python.exe
    public var daemon: String        // PC daemon script (fbt_daemon.py or gvhmr_daemon.py)
    public var daemonMatch: String   // command-line substring to find this daemon for alive/stop (must NOT match the other model)
    public var ffmpeg: String        // local ffmpeg

    public init(host: String, user: String, mac: String, model: PCModel = .nlf,
                oscIP: String, oscPort: Int,
                relayViaMac: Bool, relayPort: Int,
                heightCm: Int, sendElbows: Bool, mirror: Bool, predictionLeadMs: Int = 50,
                gvhmrK: Int = 5, gvhmrMirror: Bool = false, gvhmrFlipX: Bool = false, gvhmrMoving: Bool = false,
                gvhmrFootContact: Bool = false, gvhmrNativeRot: Bool = false,
                streamW: Int, streamH: Int,
                streamFPS: Int, bitrateMbps: Int, politeMode: Bool, fpsCap: Int,
                cameraName: String?, streamPort: Int = 5000,
                interpreter: String, daemon: String, daemonMatch: String, ffmpeg: String) {
        self.host = host; self.user = user; self.mac = mac; self.model = model
        self.oscIP = oscIP; self.oscPort = oscPort
        self.relayViaMac = relayViaMac; self.relayPort = relayPort
        self.heightCm = heightCm; self.sendElbows = sendElbows; self.mirror = mirror
        self.predictionLeadMs = predictionLeadMs
        self.gvhmrK = gvhmrK; self.gvhmrMirror = gvhmrMirror
        self.gvhmrFlipX = gvhmrFlipX; self.gvhmrMoving = gvhmrMoving
        self.gvhmrFootContact = gvhmrFootContact; self.gvhmrNativeRot = gvhmrNativeRot
        self.streamW = streamW; self.streamH = streamH; self.streamFPS = streamFPS
        self.bitrateMbps = bitrateMbps; self.politeMode = politeMode; self.fpsCap = fpsCap
        self.cameraName = cameraName; self.streamPort = streamPort
        self.interpreter = interpreter; self.daemon = daemon; self.daemonMatch = daemonMatch
        self.ffmpeg = ffmpeg
    }

    /// Default for the GPU PC (the deep hidden `\.rtcache\…\.bridge`). The Windows
    /// home username is taken from the SSH `user` the operator entered — NOT hardcoded
    /// (this is a public repo). Override the whole base path with `FEVER_PC_BRIDGE`.
    ///
    /// `model` selects which daemon to launch under the SAME `.bridge` base:
    ///   • `.nlf`   → trtlib `\py311\pythonw.exe` running `\work\fbt_daemon.py`.
    ///   • `.gvhmr` → the GVHMR `\gvhmr\.venv\Scripts\python.exe` running `\gvhmr\gvhmr_daemon.py`.
    /// `daemonMatch` is the per-model command-line token used to find/stop ONLY that
    /// daemon (so GVHMR's stop never kills NLF's, and vice-versa).
    public static func make(host: String, user: String, mac: String, model: PCModel = .nlf,
                            oscIP: String, oscPort: Int,
                            relayViaMac: Bool, relayPort: Int,
                            heightCm: Int, sendElbows: Bool, mirror: Bool, predictionLeadMs: Int = 50,
                            gvhmrK: Int = 5, gvhmrMirror: Bool = false, gvhmrFlipX: Bool = false,
                            gvhmrMoving: Bool = false,
                            gvhmrFootContact: Bool = false, gvhmrNativeRot: Bool = false,
                            streamW: Int, streamH: Int,
                            streamFPS: Int, bitrateMbps: Int, politeMode: Bool, fpsCap: Int,
                            streamPort: Int, cameraName: String?) -> PCOffloadConfig {
        let base = ProcessInfo.processInfo.environment["FEVER_PC_BRIDGE"]
            ?? #"C:\Users\\#(user)\AppData\Local\.rtcache\runtime\v8\store\.bridge"#
        let interpreter: String, daemon: String, daemonMatch: String
        switch model {
        case .nlf:
            interpreter = base + #"\py311\pythonw.exe"#
            daemon = base + #"\work\fbt_daemon.py"#
            daemonMatch = "fbt_daemon"
        case .gvhmr:
            interpreter = base + #"\gvhmr\.venv\Scripts\python.exe"#
            daemon = base + #"\gvhmr\gvhmr_daemon.py"#
            daemonMatch = "gvhmr_daemon"
        }
        return PCOffloadConfig(
            host: host, user: user, mac: mac, model: model, oscIP: oscIP, oscPort: oscPort,
            relayViaMac: relayViaMac, relayPort: relayPort,
            heightCm: heightCm, sendElbows: sendElbows, mirror: mirror, predictionLeadMs: predictionLeadMs,
            gvhmrK: gvhmrK, gvhmrMirror: gvhmrMirror, gvhmrFlipX: gvhmrFlipX, gvhmrMoving: gvhmrMoving,
            gvhmrFootContact: gvhmrFootContact, gvhmrNativeRot: gvhmrNativeRot,
            streamW: streamW, streamH: streamH, streamFPS: streamFPS, bitrateMbps: bitrateMbps,
            politeMode: politeMode, fpsCap: fpsCap, cameraName: cameraName, streamPort: streamPort,
            interpreter: interpreter, daemon: daemon, daemonMatch: daemonMatch,
            ffmpeg: "/opt/homebrew/bin/ffmpeg")
    }

    /// The WMI `Win32_Process.Create` CommandLine that launches this config's daemon.
    /// Pure / string-only (no SSH, no side effects) so the exact launch arguments are
    /// unit-testable. `oscIP`/`oscPort` are the daemon's IMMEDIATE OSC target (the Quest
    /// for Direct, this Mac for Relay — resolved by the controller); `skeletonBack` is the
    /// optional `host:port` return channel (NLF only — GVHMR's contract has no such flag).
    ///
    /// • NLF   → `--osc-ip --osc-port --height [--six-point] [--no-mirror] [--polite]
    ///   [--fps-cap N] [--skeleton-back sb] --lead-ms <predictionLeadMs>`.
    /// • GVHMR → `--listen udp://0.0.0.0:<streamPort>?overrun_nonfatal=1&fifo_size=5000000
    ///   --osc-ip --osc-port --height --k [--six-point] [--mirror] [--flip-x] [--moving]
    ///   [--foot-contact] [--native-rot] --out-hz 120 --lead-ms <predictionLeadMs>`. The `&`
    ///   is literal here (this string is passed to `Win32_Process.Create` verbatim, never
    ///   through cmd.exe) and the URL has no spaces, so it stays a single argv token.
    ///
    /// `--lead-ms` is the user-tunable forward-lead in MILLISECONDS (the daemon divides by
    /// 1000 for its lead horizon), applied to BOTH models to cancel pipeline latency.
    public func daemonCommandLine(oscIP: String, oscPort: Int, skeletonBack: String?) -> String {
        // Double-quote the (space-capable) program + script paths; flags/values are simple
        // tokens. Strip any embedded double-quote so the quoting can't be broken out of.
        func dq(_ s: String) -> String { "\"" + s.replacingOccurrences(of: "\"", with: "") + "\"" }
        var cmd = "\(dq(interpreter)) \(dq(daemon))"
        switch model {
        case .nlf:
            cmd += " --osc-ip \(oscIP) --osc-port \(oscPort) --height \(heightCm)"
            if !sendElbows { cmd += " --six-point" }
            // Proper L/R skeleton mirror on the PC (NOT an image flip), matching on-device
            // mirrorTracking. The daemon defaults to mirror-on, so only pass --no-mirror.
            if !mirror { cmd += " --no-mirror" }
            if politeMode { cmd += " --polite" }
            if fpsCap > 0 { cmd += " --fps-cap \(fpsCap)" }
            if let sb = skeletonBack { cmd += " --skeleton-back \(sb)" }
            // User-tunable forward-lead (ms) to cancel pipeline latency (shared with on-device).
            cmd += " --lead-ms \(predictionLeadMs)"
        case .gvhmr:
            cmd += " --listen udp://0.0.0.0:\(streamPort)?overrun_nonfatal=1&fifo_size=16384"
            cmd += " --osc-ip \(oscIP) --osc-port \(oscPort) --height \(heightCm) --k \(gvhmrK)"
            if !sendElbows { cmd += " --six-point" }
            if gvhmrMirror { cmd += " --mirror" }
            if gvhmrFlipX { cmd += " --flip-x" }
            if gvhmrMoving { cmd += " --moving" }
            // Experimental tunes: foot-contact damping (less foot-slide) and GVHMR's own
            // joint rotations (richer twist/roll, in place of position-derived rotations).
            if gvhmrFootContact { cmd += " --foot-contact" }
            if gvhmrNativeRot { cmd += " --native-rot" }
            // Skeleton return channel: the daemon projects the in-camera joints to 2D and
            // sends them back for the live preview overlay (same 24x2 f32 contract as NLF).
            if let sb = skeletonBack { cmd += " --skeleton-back \(sb)" }
            cmd += " --out-hz 120 --lead-ms \(predictionLeadMs)"
        }
        return cmd
    }
}

public enum PCOffloadError: Error, CustomStringConvertible {
    case badMAC, unreachable, daemonLaunch(String), stream(String)
    public var description: String {
        switch self {
        case .badMAC:            return "Invalid PC MAC address"
        case .unreachable:       return "PC didn't wake / SSH unreachable"
        case .daemonLaunch(let s): return "daemon launch failed (\(s))"
        case .stream(let s):     return "camera stream failed (\(s))"
        }
    }
}

// MARK: - The actual shell/socket work (nonisolated, Sendable-safe)

enum PCOrchestrator {

    /// The Mac's own LAN IP on the route to `host` (so the PC can send the skeleton
    /// back). Connects a UDP socket (no packets actually sent) and reads the bound
    /// local address the OS chose for that route.
    static func localIP(reaching host: String) -> String? {
        let fd = socket(AF_INET, SOCK_DGRAM, 0); guard fd >= 0 else { return nil }
        defer { close(fd) }
        var addr = sockaddr_in()
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = in_port_t(9).bigEndian
        // Accept a numeric IP directly; otherwise resolve a hostname via getaddrinfo
        // (without this, a DNS-named PC silently skips the skeleton return channel).
        if inet_pton(AF_INET, host, &addr.sin_addr) != 1 {
            var hints = addrinfo(ai_flags: 0, ai_family: AF_INET, ai_socktype: SOCK_DGRAM,
                                 ai_protocol: 0, ai_addrlen: 0, ai_canonname: nil, ai_addr: nil, ai_next: nil)
            var res: UnsafeMutablePointer<addrinfo>?
            guard getaddrinfo(host, nil, &hints, &res) == 0, let info = res else { return nil }
            defer { freeaddrinfo(info) }
            guard let sa = info.pointee.ai_addr else { return nil }
            addr.sin_addr = sa.withMemoryRebound(to: sockaddr_in.self, capacity: 1) { $0.pointee.sin_addr }
        }
        let connected = withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                connect(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        guard connected == 0 else { return nil }
        var local = sockaddr_in()
        var len = socklen_t(MemoryLayout<sockaddr_in>.size)
        let got = withUnsafeMutablePointer(to: &local) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { getsockname(fd, $0, &len) }
        }
        guard got == 0 else { return nil }
        var buf = [CChar](repeating: 0, count: Int(INET_ADDRSTRLEN))
        inet_ntop(AF_INET, &local.sin_addr, &buf, socklen_t(INET_ADDRSTRLEN))
        return String(decoding: buf.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
    }

    // -- Wake-on-LAN: raw UDP broadcast of the magic packet (no external deps) --
    static func sendMagicPacket(mac: String) throws {
        let hex = mac.uppercased().filter { $0.isHexDigit }
        guard hex.count == 12 else { throw PCOffloadError.badMAC }
        var macBytes = [UInt8]()
        var i = hex.startIndex
        while i < hex.endIndex {
            let j = hex.index(i, offsetBy: 2)
            guard let b = UInt8(hex[i..<j], radix: 16) else { throw PCOffloadError.badMAC }
            macBytes.append(b); i = j
        }
        var packet = [UInt8](repeating: 0xFF, count: 6)
        for _ in 0..<16 { packet.append(contentsOf: macBytes) }

        let fd = socket(AF_INET, SOCK_DGRAM, 0)
        guard fd >= 0 else { throw PCOffloadError.unreachable }
        defer { close(fd) }
        var on: Int32 = 1
        setsockopt(fd, SOL_SOCKET, SO_BROADCAST, &on, socklen_t(MemoryLayout<Int32>.size))
        var addr = sockaddr_in()
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = in_port_t(9).bigEndian
        addr.sin_addr.s_addr = in_addr_t(0xFFFF_FFFF)   // 255.255.255.255
        _ = packet.withUnsafeBytes { raw in
            withUnsafePointer(to: &addr) { ap in
                ap.withMemoryRebound(to: sockaddr.self, capacity: 1) { sa in
                    sendto(fd, raw.baseAddress, raw.count, 0, sa, socklen_t(MemoryLayout<sockaddr_in>.size))
                }
            }
        }
    }

    /// First existing executable among the configured path and the usual Homebrew /
    /// system locations, else whatever `which ffmpeg` resolves on the login PATH.
    /// Prevents a hardcoded `/opt/homebrew/bin/ffmpeg` from silently failing on an
    /// Intel brew (`/usr/local`) or a Mac without ffmpeg.
    static func resolveFFmpeg(_ preferred: String) -> String? {
        let fm = FileManager.default
        let candidates = [preferred,
                          "/opt/homebrew/bin/ffmpeg",
                          "/usr/local/bin/ffmpeg",
                          "/usr/bin/ffmpeg"]
        if let hit = candidates.first(where: { fm.isExecutableFile(atPath: $0) }) { return hit }
        if let (st, out) = try? runProcess("/usr/bin/which", ["ffmpeg"], timeout: 4), st == 0 {
            let path = out.trimmingCharacters(in: .whitespacesAndNewlines)
            if !path.isEmpty, fm.isExecutableFile(atPath: path) { return path }
        }
        return nil
    }

    /// Box that carries the child's output across the read thread back to the caller.
    /// The DispatchSemaphore provides the happens-before, so the unchecked Sendable is
    /// safe (the box is only read after the semaphore is signalled / the child reaped).
    private final class OutputBox: @unchecked Sendable { var data = Data() }

    // -- SSH (passwordless key auth; the Mac's ssh + the user's key) --
    /// Runs a child and returns (exitStatus, combined stdout+stderr). Bounded by a
    /// hard wall-clock `timeout`: the read happens on a background thread, and if the
    /// child wedges (e.g. SSH connected then the network dropped), we terminate it and
    /// throw `.unreachable` instead of blocking the worker forever in `.starting`.
    @discardableResult
    static func runProcess(_ launch: String, _ args: [String], timeout: TimeInterval = 25) throws -> (Int32, String) {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: launch)
        p.arguments = args
        let pipe = Pipe()
        p.standardOutput = pipe
        p.standardError = pipe
        try p.run()

        let box = OutputBox()
        let done = DispatchSemaphore(value: 0)
        DispatchQueue.global(qos: .userInitiated).async {
            box.data = pipe.fileHandleForReading.readDataToEndOfFile()   // EOFs when the child exits
            done.signal()
        }
        if done.wait(timeout: .now() + timeout) == .timedOut {
            p.terminate()                         // closes the pipe → the read returns
            _ = done.wait(timeout: .now() + 1)
            throw PCOffloadError.unreachable
        }
        p.waitUntilExit()
        return (p.terminationStatus, String(data: box.data, encoding: .utf8) ?? "")
    }

    static func ssh(_ c: PCOffloadConfig, _ remote: String, timeout: TimeInterval = 25) throws -> (Int32, String) {
        try runProcess("/usr/bin/ssh", [
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ConnectTimeout=8",
            // Give up a post-connect stall in ~9s rather than hanging — pairs with the
            // runProcess wall-clock timeout as a second line of defence.
            "-o", "ServerAliveInterval=3",
            "-o", "ServerAliveCountMax=3",
            "-o", "BatchMode=yes",
            "\(c.user)@\(c.host)", remote
        ], timeout: timeout)
    }

    static func wakeAndWait(_ c: PCOffloadConfig, attempts: Int = 12, interval: TimeInterval = 5) throws {
        try sendMagicPacket(mac: c.mac)
        for n in 0..<attempts {
            if Task.isCancelled { throw CancellationError() }
            if let (st, _) = try? ssh(c, "exit", timeout: 12), st == 0 { return }
            if n % 3 == 2 { try? sendMagicPacket(mac: c.mac) }   // nudge again
            // Pace the retries: a cold PC needs 30–60 s to reach sshd, and a
            // "connection refused" during boot returns instantly — without a delay
            // all 12 attempts would burn in ~1 s and falsely report unreachable.
            // Runs in the detached worker, so a blocking sleep is fine; check cancel
            // first so stop() is honored within one interval.
            if n < attempts - 1 {
                if Task.isCancelled { throw CancellationError() }
                Thread.sleep(forTimeInterval: interval)
            }
        }
        throw PCOffloadError.unreachable
    }

    static func startDaemon(_ c: PCOffloadConfig, oscIP: String? = nil, oscPort: Int? = nil,
                            skeletonBack: String? = nil) throws {
        // Launch the daemon DETACHED from the SSH session, via WMI Win32_Process.Create.
        //
        // CRITICAL: a plain `Start-Process` child is part of the SSH session's process
        // tree, and Windows OpenSSH KILLS that tree the moment the (one-shot) SSH session
        // closes — which is immediately after this returns. That silently killed the
        // daemon mid-startup (it printed its boot banner, then died during model load),
        // so PC inference never actually ran: no tracking, no skeleton, no OSC, while the
        // Mac still showed its local camera preview and a green "Streaming". WMI spawns
        // the process under WmiPrvSE, OUTSIDE the SSH session, so it survives the close.
        // NLF uses pythonw.exe (GUI-subsystem, no window); GVHMR uses its .venv python.exe,
        // which under the non-interactive WMI host raises no window on the user's desktop
        // either — so nothing appears on the PC for either model.
        //
        // The exact argv is built by the model-aware `daemonCommandLine` (NLF = the
        // original launch unchanged; GVHMR = its own --listen/--k/--moving contract).
        // The daemon's IMMEDIATE OSC target: the Quest directly (Direct route) or this
        // Mac's relay (Relay route). Defaults to the config's final target for Direct.
        let dOscIP = oscIP ?? c.oscIP
        let dOscPort = oscPort ?? c.oscPort
        let cmd = c.daemonCommandLine(oscIP: dOscIP, oscPort: dOscPort, skeletonBack: skeletonBack)
        // CommandLine is single-quoted for PowerShell (double any embedded single quote).
        let cmdLit = cmd.replacingOccurrences(of: "'", with: "''")
        // Defensive: reap any stale Fever daemon (either model) BEFORE launching, so an
        // earlier session that didn't clean up (app crash / lost SSH) can't leave an orphan
        // pinning the GPU alongside the new one. Then WMI-Create the new daemon.
        let ps = "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'gvhmr_daemon|fbt_daemon|frame_source' } "
            + "| ForEach-Object { Stop-Process -Id $_.ProcessId -Force }; "
            + "$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create "
            + "-Arguments @{CommandLine='\(cmdLit)'}; "
            + "if ($r.ReturnValue -ne 0) { Write-Error \"WMI Create rc=$($r.ReturnValue)\"; exit 1 }"
        // Pass the script as base64 UTF-16LE via -EncodedCommand so NO quoting has to
        // survive the SSH→PowerShell hop (the old nested-quote form was brittle).
        let enc = Data(ps.utf16.flatMap { [UInt8($0 & 0xff), UInt8($0 >> 8)] }).base64EncodedString()
        let (st, out) = try ssh(c, "powershell -NoProfile -EncodedCommand \(enc)")
        if st != 0 { throw PCOffloadError.daemonLaunch(out.trimmingCharacters(in: .whitespacesAndNewlines)) }
    }

    /// Whether our daemon is still running on the PC. Returns false ONLY when the SSH
    /// probe itself succeeded AND found zero matching processes — a transient SSH
    /// failure returns true so a network hiccup can't kill a working session.
    static func daemonAlive(_ c: PCOffloadConfig) -> Bool {
        // Match ONLY this model's daemon (fbt_daemon vs gvhmr_daemon) so an NLF probe
        // never reads a stray GVHMR daemon as alive, and vice-versa.
        // Sent via -EncodedCommand: the PC's default SSH shell is PowerShell, which would
        // otherwise consume the pipeline `$_` in a raw -Command string (turning the filter
        // into a `.CommandLine` error that matches nothing).
        let script = "$ProgressPreference='SilentlyContinue'; (Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match '\(c.daemonMatch)' } | Measure-Object).Count"
        let enc = Data(script.utf16.flatMap { [UInt8($0 & 0xff), UInt8($0 >> 8)] }).base64EncodedString()
        guard let (st, out) = try? ssh(c, "powershell -NoProfile -EncodedCommand \(enc)", timeout: 12), st == 0 else {
            return true
        }
        return (Int(out.trimmingCharacters(in: .whitespacesAndNewlines)) ?? 1) > 0
    }

    /// Kill our daemon on the PC. Returns true only if the SSH command actually ran
    /// (the PC was reachable); false if we couldn't reach it — so the caller can warn
    /// instead of falsely reporting a clean stop while the daemon keeps running.
    @discardableResult
    static func stopDaemon(_ c: PCOffloadConfig) -> Bool {
        // Kill EITHER Fever daemon (gvhmr_daemon OR fbt_daemon). There is only ever one
        // session, so a Stop should FULLY clean up the PC — including the other model's
        // daemon left over from a mid-session model switch (which the old per-model match
        // would orphan). Sent via -EncodedCommand because the PC's default SSH shell is
        // PowerShell and would otherwise expand the pipeline `$_` itself, breaking the
        // Where-Object filter so nothing got killed (the "Stop didn't stop it" bug).
        // Also kill the decoder subprocess (frame_source --decode) the daemon spawns — on
        // Windows it isn't auto-killed with its parent. (It would self-exit within ~5s via
        // the stale-heartbeat guard, but killing it now frees UDP :5000 for an instant restart.)
        let script = "$ProgressPreference='SilentlyContinue'; Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'gvhmr_daemon|fbt_daemon|frame_source' } "
            + "| ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"
        let enc = Data(script.utf16.flatMap { [UInt8($0 & 0xff), UInt8($0 >> 8)] }).base64EncodedString()
        guard let (st, _) = try? ssh(c, "powershell -NoProfile -EncodedCommand \(enc)", timeout: 15) else { return false }
        return st == 0
    }

}
