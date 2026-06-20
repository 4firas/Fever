import FeverCore
import Foundation

/// Runs the full tracking pipeline with no GUI: camera (or stub) -> Vision (or
/// stub) -> One-Euro -> JointSolver -> SLERP -> CoordinateMapper ->
/// TrackerAssembler -> OSCSender. Prints periodic telemetry (FPS, dropped
/// frames, a few live joints) to stdout and shuts down cleanly on SIGINT /
/// SIGTERM.
///
/// The pipeline is a `@MainActor @Observable` object, so telemetry is sampled on
/// the main actor from a repeating timer rather than via Combine publishers.
///
/// `HeadlessRunner` is `@MainActor`-isolated: it is constructed and `run()` is
/// called from `FeverMain.main()` (itself `@MainActor`). Because the runner,
/// the pipeline, the frame source and the landmarker all live on the main actor,
/// the non-`Sendable` source/landmarker never cross an isolation boundary — the
/// only cross-actor traffic is the `Sendable` `OSCTracker` / telemetry handoff
/// inside the pipeline itself. `run()` parks the main run loop with
/// `dispatchMain()`, on which the pipeline's async frame handling, the telemetry
/// timer and the signal sources all deliver.
@MainActor
final class HeadlessRunner {

    private let config: TrackingConfig
    private let source: FrameSource
    private let landmarker: PoseLandmarker
    private let autoCalibrateAfter: TimeInterval?

    /// Telemetry sample period (seconds).
    private let telemetryInterval: TimeInterval = 1.0

    /// - Parameter autoCalibrateAfter: when non-nil, fire a one-shot Recenter
    ///   (rest-pose capture for the rest-relative rotation rebaser + scale/floor
    ///   re-latch) this many seconds after start. Used by the synthetic `--stub`
    ///   run so the streamed `/rotation` is the CALIBRATED, rest-relative euler the
    ///   user gets after Recenter (≈ 0 at the rest pose, bounded under motion),
    ///   rather than the uncalibrated absolute orientation (which legitimately
    ///   reads ±180 for limbs hanging straight down — a true 180° Y-flip, not a
    ///   defect). nil (live tracking) leaves calibration to the user's Recenter.
    init(config: TrackingConfig,
         source: FrameSource,
         landmarker: PoseLandmarker,
         autoCalibrateAfter: TimeInterval? = nil) {
        self.config = config
        self.source = source
        self.landmarker = landmarker
        self.autoCalibrateAfter = autoCalibrateAfter
    }

    func run() {
        setbuf(stdout, nil)   // unbuffered: telemetry survives an abrupt stop

        print("[Fever] headless start -> \(config.oscHost):\(config.oscPort)")
        print("[Fever] frame source: \(typeName(source))")
        print("[Fever] landmarker:   \(typeName(landmarker))")
        print("[Fever] Ctrl+C (SIGINT) or SIGTERM to stop")

        // Build + start the pipeline and the telemetry/quit machinery. Everything
        // here is already on the main actor (so is the `@MainActor` pipeline),
        // so there is no isolation crossing and the non-Sendable source /
        // landmarker are simply handed to the main-actor `Driver`.
        let driver = Driver(config: config,
                            source: source,
                            landmarker: landmarker,
                            telemetryInterval: telemetryInterval,
                            autoCalibrateAfter: autoCalibrateAfter)
        driver.begin()

        // Keep `driver` alive for the lifetime of the process: it owns the
        // pipeline, the telemetry timer and the signal sources, and tears them
        // down (then `exit`s) from its signal handler.
        Self.retainedDriver = driver

        dispatchMain()   // never returns; the Driver calls exit() on shutdown
    }

    private func typeName(_ value: Any) -> String {
        String(describing: type(of: value))
    }

    /// Keeps the live `Driver` (pipeline + timers + signal sources) alive for the
    /// whole process lifetime, since `run()` hands control to `dispatchMain()`
    /// and never returns.
    private static var retainedDriver: Driver?
}

/// Owns the live pipeline plus the telemetry timer and signal handlers. Confined
/// to the main actor because `TrackingPipeline` is `@MainActor`.
@MainActor
private final class Driver {

    private let pipeline: TrackingPipeline
    private let telemetryInterval: TimeInterval
    private let autoCalibrateAfter: TimeInterval?

    private var telemetryTimer: DispatchSourceTimer?
    private var calibrateTimer: DispatchSourceTimer?
    private var signalSources: [DispatchSourceSignal] = []

    /// Last printed values, to suppress unchanged spam.
    private var lastFPS: Double = -1
    private var lastDropped: Int = -1
    private var didShutDown = false

    init(config: TrackingConfig,
         source: FrameSource,
         landmarker: PoseLandmarker,
         telemetryInterval: TimeInterval,
         autoCalibrateAfter: TimeInterval?) {
        self.telemetryInterval = telemetryInterval
        self.autoCalibrateAfter = autoCalibrateAfter
        self.pipeline = TrackingPipeline(config: config,
                                         source: source,
                                         landmarker: landmarker)
    }

    func begin() {
        installSignalHandlers()
        pipeline.start()
        startTelemetry()
        scheduleAutoCalibrate()
    }

    /// One-shot Recenter `autoCalibrateAfter` seconds after start (stub/headless
    /// verification only). Captures the rest pose for the rest-relative rotation
    /// rebaser so the streamed `/rotation` is the calibrated, zero-centered euler.
    private func scheduleAutoCalibrate() {
        guard let delay = autoCalibrateAfter else { return }
        let timer = DispatchSource.makeTimerSource(queue: .main)
        timer.schedule(deadline: .now() + delay)
        timer.setEventHandler { [weak self] in
            MainActor.assumeIsolated {
                self?.pipeline.calibrate()
                print("[Fever] auto-calibrate (rest capture) fired")
            }
        }
        timer.resume()
        calibrateTimer = timer
    }

    // MARK: - Telemetry

    private func startTelemetry() {
        let timer = DispatchSource.makeTimerSource(queue: .main)
        timer.schedule(deadline: .now() + telemetryInterval,
                       repeating: telemetryInterval)
        timer.setEventHandler { [weak self] in
            // The handler is dispatched on the main queue, matching the actor.
            MainActor.assumeIsolated { self?.sampleAndPrint() }
        }
        timer.resume()
        telemetryTimer = timer
    }

    private func sampleAndPrint() {
        let fps = pipeline.fps
        let dropped = pipeline.droppedFrames
        let joints = pipeline.lastFrameJoints

        // Only print when something meaningful changed.
        let fpsChanged = abs(fps - lastFPS) >= 0.5
        let dropChanged = dropped != lastDropped
        guard fpsChanged || dropChanged || !joints.isEmpty else { return }

        lastFPS = fps
        lastDropped = dropped

        var line = String(format: "[Fever] %5.1f fps  drops %d", fps, dropped)
        if joints.isEmpty {
            line += "  (no pose)"
        } else {
            for j in joints.prefix(3) {
                let p = j.position
                line += String(format: "  %@(%.2f,%.2f,%.2f)",
                               j.type.rawValue, p.x, p.y, p.z)
            }
        }
        print(line)
    }

    // MARK: - Signal handling

    /// Installs SIGINT/SIGTERM handlers via GCD signal sources so shutdown runs
    /// on the main actor (a raw C `signal()` handler may not safely touch
    /// actor-isolated state). We ignore the default disposition so the process
    /// is not killed before our handler runs.
    private func installSignalHandlers() {
        for sig in [SIGINT, SIGTERM] {
            signal(sig, SIG_IGN)   // disable default terminate; GCD source handles it
            let source = DispatchSource.makeSignalSource(signal: sig, queue: .main)
            source.setEventHandler { [weak self] in
                MainActor.assumeIsolated { self?.shutdown(signal: sig) }
            }
            source.resume()
            signalSources.append(source)
        }
    }

    private func shutdown(signal sig: Int32) {
        guard !didShutDown else { return }
        didShutDown = true

        let name = (sig == SIGINT) ? "SIGINT" : (sig == SIGTERM ? "SIGTERM" : "signal \(sig)")
        print("\n[Fever] \(name) received — stopping pipeline")

        telemetryTimer?.cancel()
        telemetryTimer = nil
        calibrateTimer?.cancel()
        calibrateTimer = nil
        for s in signalSources { s.cancel() }
        signalSources.removeAll()

        pipeline.stop()

        print("[Fever] stopped cleanly")
        exit(0)
    }
}