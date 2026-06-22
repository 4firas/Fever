import FeverCore
import Foundation
import AppKit

/// Process entry point and CLI dispatcher.
///
/// This is the SOLE `@main` in the target (the executable is built with
/// `-parse-as-library`, so there is no implicit `main.swift` top-level code and
/// no competing `@main App`). `FeverApp` is a plain `App` value type that we
/// launch explicitly via `FeverApp.main()` for the `--ui` path.
///
/// Usage:
///   Fever                         headless: camera + MediaPipe pose (Python sidecar), OSC -> 127.0.0.1:9000
///   Fever --ui                    launch the SwiftUI (Liquid Glass) window app
///   Fever --stub                  headless, hardware-free: synthetic frames + synthetic pose
///   Fever --no-osc                run the pipeline + telemetry, transmit nothing
///   Fever --host 192.168.1.50     OSC destination host (default 127.0.0.1)
///   Fever --port 9000             OSC destination UDP port (default 9000)
///   Fever -h | --help             print usage and exit
///
/// Flags may be combined (e.g. `--stub --no-osc --port 9000`). `--ui` ignores
/// `--stub`/`--no-osc` for source/landmarker selection (the SwiftUI app builds
/// its own live pipeline) but still honors `--host`/`--port` via persisted
/// settings.
@main
enum FeverMain {

    @MainActor
    static func main() {
        let args = Array(CommandLine.arguments.dropFirst())

        if args.contains("-h") || args.contains("--help") {
            printUsage()
            return
        }

        let flags = Set(args)
        let useStub = flags.contains("--stub")
        let noOSC = flags.contains("--no-osc")
        let forceHeadless = useStub || noOSC || flags.contains("--headless")
        // A double-clicked .app (Finder/launchd) passes no args and MUST open the
        // GUI — otherwise it just bounces in the Dock with no window. A bare CLI
        // invocation defaults to headless. `--ui` forces the window; `--stub` /
        // `--no-osc` / `--headless` force headless even inside the app bundle.
        let inAppBundle = (Bundle.main.bundleURL.pathExtension == "app")
        let useUI = flags.contains("--ui") || (inAppBundle && !forceHeadless)

        // SINGLE-INSTANCE GUARD (GUI/bundle only): if another Fever is
        // already running, focus it and exit. Stacked instances each spin up
        // their own camera session + MediaPipe sidecar pipeline, fighting for
        // the same hardware and collapsing frame rate — the cause of the earlier
        // "ghost process / 3fps". CLI/headless runs are exempt (you may want
        // several).
        if inAppBundle, let bundleID = Bundle.main.bundleIdentifier {
            let myPID = ProcessInfo.processInfo.processIdentifier
            let others = NSRunningApplication
                .runningApplications(withBundleIdentifier: bundleID)
                .filter { $0.processIdentifier != myPID }
            if let existing = others.first {
                existing.activate(options: [.activateAllWindows])
                return
            }
        }

        // Shared, persisted configuration. CLI flags override persisted values
        // for this run (and persist back through TrackingConfig's didSet).
        let config = TrackingConfig()
        applyArgs(args, to: config, noOSC: noOSC)

        if useUI {
            // The SwiftUI scene owns its own TrackingConfig + TrackingPipeline
            // (live camera + MediaPipe sidecar). It reads the same persisted host/port we
            // just wrote above. This call does not return until the app quits.
            FeverApp.main()
            return
        }

        // Headless. --stub => fully hardware-free (synthetic frames + pose);
        // otherwise live camera + MediaPipe body pose (sidecar). The sidecar
        // backend falls back to the stub if it isn't installed so the app runs.
        let source: FrameSource = useStub ? StubFrameSource() : CameraCapture()
        let landmarker: PoseLandmarker = useStub ? StubPoseLandmarker() : makeLivePoseLandmarker()

        // In synthetic stub mode, fire a one-shot Recenter shortly after start so
        // the streamed `/rotation` is the CALIBRATED, rest-relative euler (≈ 0 at
        // the rest pose, bounded under motion) rather than the uncalibrated
        // absolute orientation. Live tracking leaves calibration to the user's
        // Recenter (auto-recentering a real session mid-pose would be surprising).
        let runner = HeadlessRunner(config: config,
                                    source: source,
                                    landmarker: landmarker,
                                    autoCalibrateAfter: useStub ? 1.0 : nil)
        runner.run()   // blocks on the main run loop until SIGINT/SIGTERM
    }

    // MARK: - Argument parsing

    /// Applies `--host`/`--port`/`--no-osc` to `config`. `--no-osc` redirects the
    /// OSC destination to an unroutable sink so the full pipeline still runs and
    /// emits telemetry while nothing reaches VRChat.
    private static func applyArgs(_ args: [String],
                                  to config: TrackingConfig,
                                  noOSC: Bool) {
        if let host = value(of: "--host", in: args) {
            config.oscHost = host
        }
        if let portString = value(of: "--port", in: args) {
            if let port = Int(portString), (1...65535).contains(port) {
                config.oscPort = port
            } else {
                FileHandle.standardError.write(
                    Data("[Fever] ignoring invalid --port '\(portString)' (keeping \(config.oscPort))\n".utf8)
                )
            }
        }

        // Headless runs send trackers regardless of the persisted GUI toggle.
        config.enableTracker = true

        if noOSC {
            // Black-hole the transmit path (TEST-NET-1 / discard sink) without
            // disabling the rest of the pipeline.
            config.oscHost = "192.0.2.0"
        }
    }

    /// Returns the argument immediately following `flag`, or `nil` if `flag` is
    /// absent or has no following value.
    private static func value(of flag: String, in args: [String]) -> String? {
        guard let i = args.firstIndex(of: flag), i + 1 < args.count else { return nil }
        return args[i + 1]
    }

    private static func printUsage() {
        let usage = """
        Fever — webcam full-body VR tracker (MediaPipe pose (Python sidecar) -> VRChat OSC)

        USAGE:
          Fever [options]

        OPTIONS:
          --ui                Launch the SwiftUI (Liquid Glass) window app.
          --headless          Force headless live tracking (no window).
          --stub              Headless, hardware-free: synthetic frames + synthetic pose.
          --no-osc            Run the pipeline and print telemetry without transmitting.
          --host <address>    OSC destination host (default 127.0.0.1).
          --port <port>       OSC destination UDP port (default 9000).
          -h, --help          Show this help and exit.

        DEFAULT (no options):
          Launched as Fever.app (Finder/Dock) -> opens the GUI window.
          Run as a bare CLI binary -> headless live tracking (built-in camera +
          MediaPipe sidecar pose, streaming VRChat trackers to 127.0.0.1:9000).
        """
        print(usage)
    }
}