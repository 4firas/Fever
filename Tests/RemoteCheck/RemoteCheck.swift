import Foundation
import FeverCore

/// Thread-safe counter for poses received from the PC (bumped per RemoteNLFSource.onPreview).
final class Counter: @unchecked Sendable {
    private let lock = NSLock(); private var n = 0
    func inc() { lock.withLock { n += 1 } }
    var value: Int { lock.withLock { n } }
}

/// Headless end-to-end harness for PC REMOTE-inference mode. Drives the REAL Mac pipeline —
/// `RemoteNLFSource` (receives the PC's raw pose) → `TrackingPipeline` (mirror→OneEuro→IK→120 Hz
/// upsampler) → `OSCSender` — using a `StubFrameSource` to clock the worker, with OSC pointed at
/// 127.0.0.1 so a listener can capture + verify the bundles. Run a `fbt_daemon --raw` feeding poses
/// to this process's pose port while this runs. Usage: `RemoteCheck <posePort> <oscPort> <seconds>`.
@main
struct RemoteCheck {
    static func main() async {
        let a = CommandLine.arguments
        let posePort = UInt16(a.count > 1 ? (Int(a[1]) ?? 5099) : 5099)
        let oscPort = a.count > 2 ? (Int(a[2]) ?? 9100) : 9100
        let seconds = a.count > 3 ? (Double(a[3]) ?? 18) : 18
        await run(posePort: posePort, oscPort: oscPort, seconds: seconds)
    }

    @MainActor
    static func run(posePort: UInt16, oscPort: Int, seconds: Double) async {
        let poses = Counter()
        guard let remote = RemoteNLFSource(port: posePort, flipX: true, onPreview: { _ in poses.inc() }) else {
            print("FAIL: couldn't bind RemoteNLFSource on \(posePort)"); exit(1)
        }
        let stub = StubFrameSource(width: 256, height: 256)
        let config = TrackingConfig()
        // The exact on-device pipeline, fed by the PC's pose, OSC → loopback for capture.
        let pipeline = TrackingPipeline(config: config, source: stub, landmarker: remote,
                                        oscHostOverride: "127.0.0.1", oscPortOverride: oscPort)
        pipeline.start()
        print("RemoteCheck: pipeline up — poses on :\(posePort), OSC → 127.0.0.1:\(oscPort). running \(seconds)s…")
        try? await Task.sleep(for: .seconds(seconds))
        pipeline.stop()
        remote.stop()
        let n = poses.value
        print(String(format: "RemoteCheck DONE: poses_received=%d pose_fps=%.1f", n, Double(n) / max(seconds, 0.001)))
        exit(0)
    }
}
