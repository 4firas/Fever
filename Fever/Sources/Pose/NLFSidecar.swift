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
    private var lineBuf = Data()
    private var launchFailed = false

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
        if launchFailed { return false }
        let p = Process()
        p.executableURL = URL(fileURLWithPath: paths.python)
        p.arguments = [paths.script, paths.model]      // positional sys.argv[1] = model
        let inPipe = Pipe(); let outPipe = Pipe(); let errPipe = Pipe()
        p.standardInput = inPipe; p.standardOutput = outPipe; p.standardError = errPipe
        do { try p.run() } catch { launchFailed = true; return false }

        toChild = inPipe.fileHandleForWriting
        fromChild = outPipe.fileHandleForReading
        // drain stderr (CoreML compile logs) so the child never blocks on a full pipe
        errPipe.fileHandleForReading.readabilityHandler = { _ = $0.availableData }

        // gate on the sidecar's `{"ready":true}` stdout line (model + CoreML compile
        // can take a few seconds the first time, then it's cached)
        let deadline = Date().addingTimeInterval(60)
        while Date() < deadline {
            guard let line = readLine() else { break }
            if NLFProtocol.isReady(line) { process = p; return true }
        }
        p.terminate(); launchFailed = true
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
        process = nil; toChild = nil; fromChild = nil; lineBuf.removeAll()
    }

    deinit { process?.terminate() }
}
