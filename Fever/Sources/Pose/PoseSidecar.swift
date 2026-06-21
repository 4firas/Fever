import Foundation

/// Abstraction over "RGB frame in -> 33 landmarks out" so the MediaPipe backend
/// can be unit-tested with a fake. `PoseSidecar` is the real implementation.
public protocol PoseInferenceService: AnyObject, Sendable {
    func infer(rgb: Data, width: Int, height: Int, tMicros: UInt64) async -> SidecarReply?
    func reset()
}

/// Resolves the sidecar interpreter, script, and model. Embedded `.app` Resources
/// first (Resources/sidecar/... + Resources/Models/...), else the dev tree.
public struct SidecarPaths: Sendable {
    public let python: String
    public let script: String
    public let model: String

    public init(python: String, script: String, model: String) {
        self.python = python; self.script = script; self.model = model
    }

    public static func resolve(bundle: Bundle, projectRoot: String?) -> SidecarPaths? {
        let fm = FileManager.default
        if let res = bundle.resourceURL?.appendingPathComponent("sidecar") {
            let py = res.appendingPathComponent("python/bin/python3").path
            let sc = res.appendingPathComponent("pose_server.py").path
            let md = bundle.resourceURL!.appendingPathComponent("Models/pose_landmarker_full.task").path
            if fm.isExecutableFile(atPath: py), fm.fileExists(atPath: sc), fm.fileExists(atPath: md) {
                return SidecarPaths(python: py, script: sc, model: md)
            }
        }
        if let root = projectRoot ?? Self.devRoot() {
            let py = root + "/Sidecar/.venv/bin/python3"
            let sc = root + "/Sidecar/pose_server.py"
            let md = root + "/Models/pose_landmarker_full.task"
            if fm.fileExists(atPath: py), fm.fileExists(atPath: sc), fm.fileExists(atPath: md) {
                return SidecarPaths(python: py, script: sc, model: md)
            }
        }
        return nil
    }

    /// Walk up from the executable looking for a dir that contains Sidecar/pose_server.py.
    static func devRoot() -> String? {
        var dir = URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent()
        for _ in 0..<10 {
            if FileManager.default.fileExists(atPath: dir.appendingPathComponent("Sidecar/pose_server.py").path) {
                return dir.path
            }
            dir.deleteLastPathComponent()
        }
        return nil
    }
}

/// Runs `Sidecar/pose_server.py` as a child process and exchanges binary frames.
/// All process state + the blocking write/read round-trip are confined to one
/// serial `DispatchQueue` (keeping blocking IO off the Swift concurrency pool and
/// giving mutual exclusion without an async-unsafe lock). Restarts on crash.
public final class PoseSidecar: PoseInferenceService, @unchecked Sendable {
    private let paths: SidecarPaths
    private let q = DispatchQueue(label: "com.fir4s.fever.sidecar")
    private var process: Process?
    private var toChild: FileHandle?
    private var fromChild: FileHandle?
    private var seq: UInt32 = 0
    private var launchFailed = false

    /// Hard cap on a reply frame's declared length. The real reply body is a
    /// fixed 933 bytes; anything past this signals a framing desync, not a frame.
    private static let maxReplyBytes: UInt32 = 4096

    public init(paths: SidecarPaths) { self.paths = paths }

    public func infer(rgb: Data, width: Int, height: Int, tMicros: UInt64) async -> SidecarReply? {
        await withCheckedContinuation { cont in
            q.async {
                cont.resume(returning: self.inferSync(rgb: rgb, width: width, height: height, tMicros: tMicros))
            }
        }
    }

    /// MediaPipe holds its own VIDEO-mode temporal state; nothing to reset here.
    public func reset() {}

    // MARK: - Queue-confined implementation

    private func inferSync(rgb: Data, width: Int, height: Int, tMicros: UInt64) -> SidecarReply? {
        guard ensureRunning(), let w = toChild else { return nil }
        seq &+= 1
        let mySeq = seq
        let frame = SidecarProtocol.encodeRequest(seq: mySeq, tMicros: tMicros,
                                                  width: width, height: height, rgb: rgb)
        do { try w.write(contentsOf: frame) } catch { teardown(); return nil }
        guard let body = readFrame(),
              let (rseq, reply) = SidecarProtocol.decodeReply(body),
              rseq == mySeq else { teardown(); return nil }
        return reply
    }

    private func ensureRunning() -> Bool {
        if let p = process, p.isRunning { return true }
        if launchFailed { return false }
        let p = Process()
        p.executableURL = URL(fileURLWithPath: paths.python)
        p.arguments = [paths.script, "--model", paths.model]
        let inPipe = Pipe(); let outPipe = Pipe(); let errPipe = Pipe()
        p.standardInput = inPipe; p.standardOutput = outPipe; p.standardError = errPipe
        do { try p.run() } catch { launchFailed = true; return false }

        // Wait for "READY" on stderr (emitted after the model loads). MediaPipe
        // prints glog init lines first, so accumulate and substring-match.
        let errh = errPipe.fileHandleForReading
        var ready = false
        var acc = Data()
        let deadline = Date().addingTimeInterval(30)
        while Date() < deadline {
            let chunk = errh.availableData
            if chunk.isEmpty { if !p.isRunning { break }; continue }
            acc.append(chunk)
            if let s = String(data: acc, encoding: .utf8), s.contains("READY") { ready = true; break }
        }
        if !ready { p.terminate(); launchFailed = true; return false }
        // Keep draining stderr so the child never blocks on a full stderr pipe.
        errh.readabilityHandler = { _ = $0.availableData }

        process = p
        toChild = inPipe.fileHandleForWriting
        fromChild = outPipe.fileHandleForReading
        return true
    }

    private func readFrame() -> Data? {
        guard let h = fromChild else { return nil }
        func readExact(_ n: Int) -> Data? {
            var buf = Data()
            while buf.count < n {
                let chunk = h.readData(ofLength: n - buf.count)
                if chunk.isEmpty { return nil }   // EOF
                buf.append(chunk)
            }
            return buf
        }
        guard let lenData = readExact(4) else { return nil }
        let len = lenData.withUnsafeBytes { $0.loadUnaligned(as: UInt32.self).littleEndian }
        // A desynced/corrupt frame can carry a bogus length; cap it so a framing
        // glitch becomes a clean restart instead of an OOM-scale alloc or a hang.
        guard len <= PoseSidecar.maxReplyBytes else { teardown(); return nil }
        return readExact(Int(len))
    }

    private func teardown() {
        process?.terminate()
        process = nil; toChild = nil; fromChild = nil
    }

    deinit { process?.terminate() }
}
