import Foundation

/// "RGB frame in -> SMPL-24 pose out" — the NLF inference seam (fake-able for tests).
public protocol NLFInferenceService: AnyObject, Sendable {
    func infer(rgb: Data, width: Int, height: Int, timestamp: Double) async -> SMPLPose?
    /// No-op: the sidecar self-tracks and self-resets internally (re-detects when
    /// `has_tracked<=0.5`, re-learns vertical orientation after sustained misses).
    func reset()
}

/// Resolves the external NLF runtime: a venv python, the onnxruntime sidecar
/// script, and the model. ALL stay outside the repo (the model is non-distributable
/// study material). Base dir = env `FEVER_NLF_ROOT`, else the default dev tree.
public struct NLFPaths: Sendable {
    public let python: String
    public let script: String
    public let model: String

    public static func resolve(env: [String: String] = ProcessInfo.processInfo.environment) -> NLFPaths? {
        let base = (env["FEVER_NLF_ROOT"].map { ($0 as NSString).expandingTildeInPath })
            ?? (("~/Dev/BodyPose3DDemo" as NSString).expandingTildeInPath)
        let py = base + "/sidecar/.venv/bin/python3"
        let sc = base + "/sidecar/pinofbt_sidecar.py"
        let md = base + "/models/pino_pose_v4.onnx"
        let fm = FileManager.default
        guard fm.fileExists(atPath: py), fm.fileExists(atPath: sc), fm.fileExists(atPath: md) else { return nil }
        return NLFPaths(python: py, script: sc, model: md)
    }
}

/// Runs the NLF onnxruntime sidecar as a child process and exchanges frames over a
/// serial queue (blocking IO off the concurrency pool, mutual exclusion without an
/// async-unsafe lock). Restarts on crash. Mirrors the old MediaPipe harness shape.
public final class NLFSidecar: NLFInferenceService, @unchecked Sendable {
    private let paths: NLFPaths
    private let q = DispatchQueue(label: "com.fir4s.fever.nlf-sidecar")
    private var process: Process?
    private var toChild: FileHandle?
    private var fromChild: FileHandle?
    private var errHandle: FileHandle?    // child stderr; its readabilityHandler must be cleared to release the source
    private var lineBuf = Data()
    /// Hard, permanent: the child couldn't even spawn (bad python/script path) — no
    /// point retrying until the config changes.
    private var spawnFailed = false
    /// Soft cooldown: a ready-gate timeout/EOF (e.g. a slow first-run CoreML compile or
    /// a startup hiccup) may be transient, so we allow a RETRY — but not before this
    /// time, so a sidecar that keeps dying on startup can't become a respawn storm.
    private var retryNotBefore = Date.distantPast
    private static let retryCooldown: TimeInterval = 5

    public init(paths: NLFPaths) { self.paths = paths }

    public func infer(rgb: Data, width: Int, height: Int, timestamp: Double) async -> SMPLPose? {
        await withCheckedContinuation { cont in
            q.async { cont.resume(returning: self.inferSync(rgb: rgb, width: width, height: height, timestamp: timestamp)) }
        }
    }

    public func reset() {}

    // MARK: - Queue-confined implementation

    private func inferSync(rgb: Data, width: Int, height: Int, timestamp: Double) -> SMPLPose? {
        guard ensureRunning(), let w = toChild else { return nil }
        let frame = NLFProtocol.encodeRequest(width: width, height: height, rgb: rgb)
        do { try w.write(contentsOf: frame) } catch { teardown(); return nil }
        guard let line = readLine() else { teardown(); return nil }   // EOF == sidecar died
        return NLFProtocol.decodeReply(line, timestamp: timestamp)
    }

    private func ensureRunning() -> Bool {
        if let p = process, p.isRunning { return true }
        if spawnFailed { return false }
        if Date() < retryNotBefore { return false }    // cooling down after a transient failure
        let p = Process()
        p.executableURL = URL(fileURLWithPath: paths.python)
        p.arguments = [paths.script, paths.model]      // positional sys.argv[1] = model
        let inPipe = Pipe(); let outPipe = Pipe(); let errPipe = Pipe()
        p.standardInput = inPipe; p.standardOutput = outPipe; p.standardError = errPipe
        do { try p.run() } catch {
            FileHandle.standardError.write(Data("[NLFSidecar] spawn FAILED for \(paths.python): \(error)\n".utf8))
            spawnFailed = true; return false
        }
        FileHandle.standardError.write(Data("[NLFSidecar] launched \(paths.script)\n".utf8))

        toChild = inPipe.fileHandleForWriting
        fromChild = outPipe.fileHandleForReading
        // forward sidecar stderr (CoreML/onnxruntime logs + errors) so failures are visible.
        // CRITICAL: handle EOF (empty availableData) by nil-ing the handler — that cancels
        // the underlying dispatch source. Without it, when the child dies the source fires
        // forever on a now-EOF fd → a 100% CPU spin on a background thread + a leaked
        // FileHandle/source per sidecar death/restart.
        let eh = errPipe.fileHandleForReading
        errHandle = eh
        eh.readabilityHandler = { h in
            let d = h.availableData
            if d.isEmpty { h.readabilityHandler = nil; return }
            FileHandle.standardError.write(Data("[sidecar] ".utf8) + d)
        }

        // gate on the sidecar's `{"ready":true}` stdout line (model + CoreML compile
        // can take a few seconds the first time, then it's cached)
        let deadline = Date().addingTimeInterval(60)
        while Date() < deadline {
            guard let line = readLine() else {
                FileHandle.standardError.write(Data("[NLFSidecar] EOF before ready (sidecar died on startup)\n".utf8))
                break
            }
            if NLFProtocol.isReady(line) {
                FileHandle.standardError.write(Data("[NLFSidecar] ready\n".utf8))
                process = p; return true
            }
        }
        // Ready-gate failed (timeout or EOF). Treat as transient: clean up and arm a
        // cooldown so the NEXT frame retries the launch (a permanent flag here would
        // disable on-device tracking until app restart after one slow first compile).
        p.terminate(); retryNotBefore = Date().addingTimeInterval(Self.retryCooldown)
        errHandle?.readabilityHandler = nil; errHandle = nil   // deliberate stop before EOF → cancel the source ourselves
        toChild = nil; fromChild = nil; lineBuf.removeAll()
        return false
    }

    /// Read one newline-terminated UTF-8 line from the child (nil on EOF).
    private func readLine() -> String? {
        guard let h = fromChild else { return nil }
        while true {
            if let nl = lineBuf.firstIndex(of: 0x0a) {
                let line = lineBuf.subdata(in: lineBuf.startIndex..<nl)
                lineBuf.removeSubrange(lineBuf.startIndex...nl)
                return String(data: line, encoding: .utf8)
            }
            let chunk = h.availableData
            if chunk.isEmpty { return nil }   // EOF
            lineBuf.append(chunk)
        }
    }

    private func teardown() {
        process?.terminate()
        errHandle?.readabilityHandler = nil; errHandle = nil
        process = nil; toChild = nil; fromChild = nil; lineBuf.removeAll()
    }

    deinit { errHandle?.readabilityHandler = nil; process?.terminate() }
}
