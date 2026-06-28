import FeverCore
import SwiftUI
import AVFoundation

/// SwiftUI entry for the windowed Fever experience.
///
/// `FeverMain` (the CLI dispatcher) calls `FeverApp.main()` for the
/// `--ui` path; there is no `@main` here because the executable's `@main` lives
/// in `App/FeverMain.swift`.
///
/// Builds the `@Observable` `TrackingConfig` and the camera-backed
/// `TrackingPipeline`, holds the live `CameraCapture` so the same session can be
/// shown in the full-bleed preview, and exposes a Settings scene (Cmd+,).
struct FeverApp: App {

    // NOTE: `@State` is normally a macro that synthesizes `_config` storage and
    // the `$config` projection. The Command-Line-Tools toolchain does not ship
    // the `SwiftUIMacros` plugin, so the macro is expanded by hand here: the
    // `State<T>` value is the stored property `_config`, and a computed
    // accessor exposes its `wrappedValue`. SwiftUI's dynamic-property machinery
    // discovers the `State` storage by type via reflection exactly as it would
    // for the macro-generated form.
    private var _config = State(initialValue: TrackingConfig())
    private var config: TrackingConfig {
        get { _config.wrappedValue }
        nonmutating set { _config.wrappedValue = newValue }
    }

    private var _pipeline: State<TrackingPipeline>
    private var pipeline: TrackingPipeline {
        get { _pipeline.wrappedValue }
        nonmutating set { _pipeline.wrappedValue = newValue }
    }

    /// Orchestrates "Inference on PC" mode (WoL + SSH daemon + camera stream).
    private var _controller = State(initialValue: PCOffloadController())
    private var controller: PCOffloadController {
        get { _controller.wrappedValue }
        nonmutating set { _controller.wrappedValue = newValue }
    }

    /// Installs an AppKit delegate so quitting Fever can kill the remote PC daemon
    /// before the process exits (SwiftUI's `App` has no terminate hook that can defer
    /// exit). Hand-expanded like `@State` — the CLT toolchain lacks the macro plugin.
    private var _appDelegate = NSApplicationDelegateAdaptor(FeverAppDelegate.self)

    /// The live camera source. Held here so its `AVCaptureSession` can be handed
    /// to both the pipeline (frame inference) and the preview layer (display).
    private let camera: CameraCapture

    init() {
        let config = _config.wrappedValue
        let camera = CameraCapture()
        let pipeline = TrackingPipeline(config: config,
                                        source: camera,
                                        landmarker: makeLiveNLFLandmarker())
        _pipeline = State(initialValue: pipeline)
        self.camera = camera
        camera.preferredDeviceID = config.cameraDeviceID.isEmpty ? nil : config.cameraDeviceID
        // Cap the capture frame rate (default 30) before the session ever starts, so
        // both the on-device pipeline and PC-mode streaming share the same clamp.
        camera.maxFPS = CMTimeScale(config.cameraMaxFPS)
        // Hand the controller to the AppKit termination hook (see FeverAppDelegate).
        AppShutdown.controller = _controller.wrappedValue
        Self.installTerminationSignalHandlers()
    }

    /// `applicationShouldTerminate` covers Cmd-Q, but SIGTERM/SIGINT (killall, logout,
    /// a parent process exiting) bypass it and would orphan the remote PC daemon. Catch
    /// them via dispatch signal sources (safe to do real work in, unlike raw handlers):
    /// kill the daemon, wait briefly, then exit cleanly.
    nonisolated(unsafe) private static var signalSources: [DispatchSourceSignal] = []
    private static func installTerminationSignalHandlers() {
        for sig in [SIGINT, SIGTERM] {
            signal(sig, SIG_IGN)   // disable the default action; the dispatch source handles it
            let src = DispatchSource.makeSignalSource(signal: sig, queue: .main)
            src.setEventHandler {
                let done = DispatchSemaphore(value: 0)
                let stopping = MainActor.assumeIsolated { AppShutdown.controller?.shutdownForQuit { done.signal() } ?? false }
                if stopping { _ = done.wait(timeout: .now() + 18) }   // let the remote daemon kill land
                exit(0)
            }
            src.resume()
            signalSources.append(src)   // retain for the process lifetime
        }
    }

    var body: some Scene {
        WindowGroup {
            ContentView(config: config, pipeline: pipeline, controller: controller, camera: camera)
                // Adapt to the window: a comfortable minimum that still shows
                // both sidebars + preview, and a well-proportioned default size.
                .frame(minWidth: 820, idealWidth: 1200,
                       minHeight: 520, idealHeight: 760)
        }
        .windowStyle(.titleBar)
        .windowToolbarStyle(.unified)
        .defaultSize(width: 1200, height: 760)
        .commands {
            CommandGroup(replacing: .appInfo) {
                Button("About Fever") {
                    NSApplication.shared.orderFrontStandardAboutPanel(nil)
                }
            }
        }

        Settings {
            SettingsView(config: config, camera: camera, controller: controller, pipeline: pipeline)
        }
    }
}

/// Bridges the SwiftUI `App`'s PC-offload controller to AppKit's termination
/// callback. SwiftUI's `App` exposes no "will terminate" that can DEFER exit, but a
/// clean quit must kill the remote daemon first — otherwise it keeps running on the
/// PC (GPU busy, PC never sleeps), breaking the "invisible on the PC" promise.
@MainActor
enum AppShutdown {
    static weak var controller: PCOffloadController?
}

final class FeverAppDelegate: NSObject, NSApplicationDelegate {
    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        MainActor.assumeIsolated {
            guard let controller = AppShutdown.controller else { return .terminateNow }
            // If PC mode is running, hold termination until the remote daemon is killed.
            let stopping = controller.shutdownForQuit {
                DispatchQueue.main.async { NSApp.reply(toApplicationShouldTerminate: true) }
            }
            return stopping ? .terminateLater : .terminateNow
        }
    }
}

/// The Settings scene (Cmd+,). The 1:1 PinoFBT solve is fixed, so the only knobs
/// are: OSC destination, camera, real-world height, left/right mirror, the FPS
/// multiplier, and the 6-/8-point elbow toggle.
struct SettingsView: View {
    @Bindable var config: TrackingConfig
    let camera: CameraCapture
    let controller: PCOffloadController
    let pipeline: TrackingPipeline

    /// A session is live in either mode — used to lock the mode switch (you can't
    /// change where inference runs mid-session; the two modes share one camera).
    private var sessionActive: Bool { controller.isActive || pipeline.isRunning }

    var body: some View {
        VStack(spacing: 0) {
            // Shared mode switch above the tabs, so the active mode is always visible
            // and changeable from either tab (matches the main window's switch).
            Picker("Mode", selection: $config.inferenceOnPC) {
                Text("On Device").tag(false)
                Text("Inference on PC").tag(true)
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .disabled(sessionActive)
            .padding(.horizontal, 16)
            .padding(.top, 14)
            .padding(.bottom, 4)
            .help(sessionActive ? "Stop the current session to switch modes." : "")

            TabView {
                OnDeviceSettings(config: config, camera: camera)
                    .tabItem { Label("On Device", systemImage: "laptopcomputer") }
                PCSettings(config: config, camera: camera, controller: controller, running: sessionActive)
                    .tabItem { Label("Inference on PC", systemImage: "server.rack") }
            }
        }
        .frame(width: 520, height: 680)
        .tint(Theme.crimsonBright)
    }
}

/// On-device (local Apple-silicon inference) tracking settings.
struct OnDeviceSettings: View {
    @Bindable var config: TrackingConfig
    let camera: CameraCapture

    var body: some View {
        Form {
            Section("Output") {
                // Explicit get/set bindings that write through on EVERY edit (and a
                // trim), so the value can't be lost if the field never "commits"
                // before you switch to VRChat — the cause of "I set the Quest IP but
                // it kept sending to 127.0.0.1". Persistence is handled by the
                // config's didSet (regression-tested in ConfigPersistenceTests).
                TextField("VRChat IP", text: Binding(
                    get: { config.oscHost },
                    set: { config.oscHost = $0.trimmingCharacters(in: .whitespacesAndNewlines) }))
                TextField("Port", text: Binding(
                    get: { String(config.oscPort) },
                    set: { if let p = Int($0.trimmingCharacters(in: .whitespacesAndNewlines)) { config.oscPort = p } }))
                Text("Where Fever sends the trackers (UDP 9000 by default). For a standalone Quest, enter the headset's Wi-Fi IP (Quest ▸ Settings ▸ Wi-Fi ▸ your network ▸ Advanced — reserve it in your router so it doesn't change). Use 127.0.0.1 only if VRChat runs on this Mac.")
                    .font(.caption)
                    .foregroundStyle(Theme.textSecondary)
                if config.oscHost == "127.0.0.1" || config.oscHost == "localhost" {
                    Label("Sending to this Mac (loopback). A Quest needs its own Wi-Fi IP here.",
                          systemImage: "exclamationmark.triangle.fill")
                        .font(.caption)
                        .foregroundStyle(Theme.warning)
                }
            }

            Section("Camera") {
                Picker("Camera", selection: Binding(
                    get: { config.cameraDeviceID },
                    set: { config.cameraDeviceID = $0; camera.selectCamera($0.isEmpty ? nil : $0) })) {
                    Text("Auto (built-in)").tag("")
                    ForEach(CameraCapture.availableCameras(), id: \.uniqueID) { cam in
                        Text(cam.localizedName).tag(cam.uniqueID)
                    }
                }
                Picker("Capture FPS", selection: Binding(
                    get: { config.cameraMaxFPS },
                    set: { config.cameraMaxFPS = $0; camera.setMaxFPS($0) })) {
                    Text("24").tag(24)
                    Text("30").tag(30)
                    Text("60").tag(60)
                }
                Text("Pick your GoPro/USB webcam. Takes effect immediately; Auto prefers an external camera over the built-in. Capture FPS caps the camera frame rate for BOTH modes (default 30 — smooth and light); PC streaming never exceeds it.")
                    .font(.caption)
                    .foregroundStyle(Theme.textSecondary)
            }

            Section {
                LabeledContent("Height") {
                    Text(String(format: "%.2f m", config.userHeightMeters))
                        .font(Theme.valueFont)
                        .foregroundStyle(Theme.textPrimary)
                }
                Slider(value: $config.userHeightMeters, in: 1.2...2.2, step: 0.01) {
                    Text("Real-world height")
                }
            } header: {
                Text("Body Scale")
            } footer: {
                Text("Your real-world height — scales the whole skeleton. Shared with the Inference-on-PC tab (it's the same body, either mode).")
                    .foregroundStyle(Theme.textSecondary)
            }

            Section {
                Toggle("Swap tracker handedness (L/R)", isOn: $config.mirrorTracking)
            } footer: {
                Text("Flips the tracker handedness if your avatar appears left-right reversed in VRChat. Affects only the OSC trackers, not the live preview. This is the SAME setting used by PC inference mode, so on-device and PC track identically.")
                    .foregroundStyle(Theme.textSecondary)
            }

            Section {
                HStack {
                    Text("FPS multiplier")
                    Spacer()
                    Text("\(config.fpsMultiplier)×")
                        .monospacedDigit()
                        .foregroundStyle(Theme.textSecondary)
                }
                Slider(value: Binding(get: { Double(config.fpsMultiplier) },
                                      set: { config.fpsMultiplier = Int($0.rounded()) }),
                       in: 1...10, step: 1)
            } footer: {
                Text("Output OSC rate = this × the tracking FPS (capped 120 Hz). The upsampler forward-predicts each sub-frame, so a higher multiplier means a smoother AND lower-latency stream. 1× = raw inference rate; 7× is a good default. The live rate shows as OUT in the preview.")
                    .foregroundStyle(Theme.textSecondary)
            }

            Section {
                HStack {
                    Text("Latency prediction (ms)")
                    Spacer()
                    Text("\(config.predictionLeadMs)")
                        .monospacedDigit()
                        .foregroundStyle(Theme.textSecondary)
                }
                Slider(value: Binding(get: { Double(config.predictionLeadMs) },
                                      set: { config.predictionLeadMs = Int($0.rounded()) }),
                       in: 0...150, step: 5)
            } footer: {
                Text("Forward-predicts your motion to cancel pipeline latency — higher feels lower-latency but overshoots more on fast direction changes. Applies to both On Device and PC modes. ~50ms is a safe start; push higher to chase minimum delay.")
                    .foregroundStyle(Theme.textSecondary)
            }

            Section {
                Toggle("Send elbow trackers (8-point)", isOn: $config.sendElbows)
            } footer: {
                Text("ON = full 8-point (adds the two elbow trackers). OFF = robust 6-point (hip, chest, knees, ankles). A single webcam can't see arm depth well, so the elbows are the least reliable trackers — feet/hips/chest are rock-solid.")
                    .foregroundStyle(Theme.textSecondary)
            }
        }
        .formStyle(.grouped)
    }
}

/// The dedicated "Inference on PC" tab — the full GPU-offload config. Defaults to
/// the 1:1 PinoFBT 2.0 desktop setup: 8-point (elbows on), mirrored camera, a PCVR
/// OSC target, and the byte-exact model + IK running on the RTX GPU.
struct PCSettings: View {
    @Bindable var config: TrackingConfig
    let camera: CameraCapture
    let controller: PCOffloadController
    /// True while any session is live — stream/camera settings here are snapshotted
    /// at Start, so we hint that edits apply next Start.
    var running: Bool = false

    private var statusColor: Color { controller.phase.dotColor }

    /// The OSC host is the PC's own loopback but the PC is a remote LAN box — so a
    /// standalone-Quest user would be sending trackers to the PC, not the headset.
    private var oscTargetsLoopback: Bool {
        let h = config.pcOscHost.trimmingCharacters(in: .whitespaces)
        return h == "127.0.0.1" || h == "localhost"
    }

    var body: some View {
        Form {
            // -- Live status (the mode switch is in the shared header above the tabs) --
            Section {
                HStack(spacing: 8) {
                    StatusDot(statusColor, size: 8)
                    Text(controller.status)
                        .font(.caption)
                        .foregroundStyle(Theme.textSecondary)
                        .lineLimit(2)
                }
            } footer: {
                Text("The Mac becomes the camera + controller; the RTX PC runs the pose model + IK and emits the VRChat OSC. Start/Stop is the main window's button.")
                    .foregroundStyle(Theme.textSecondary)
            }

            // -- Connection --
            Section("PC Connection") {
                TextField("PC address", text: Binding(
                    get: { config.pcHost },
                    set: { config.pcHost = $0.trimmingCharacters(in: .whitespacesAndNewlines) }))
                TextField("SSH user", text: Binding(
                    get: { config.pcUser },
                    set: { config.pcUser = $0.trimmingCharacters(in: .whitespacesAndNewlines) }))
                TextField("Wake-on-LAN MAC", text: Binding(
                    get: { config.pcMAC },
                    set: { config.pcMAC = $0.trimmingCharacters(in: .whitespacesAndNewlines) }))
                Text("Fever wakes the PC (Wake-on-LAN) and connects over SSH with your key. The PC's wired NIC must have WoL armed.")
                    .font(.caption)
                    .foregroundStyle(Theme.textSecondary)
            }

            // -- VRChat OSC target (separate from the On Device tab) --
            Section("VRChat OSC Target") {
                TextField("OSC host", text: Binding(
                    get: { config.pcOscHost },
                    set: { config.pcOscHost = $0.trimmingCharacters(in: .whitespacesAndNewlines) }))
                TextField("OSC port", text: Binding(
                    get: { String(config.pcOscPort) },
                    set: { if let p = Int($0.trimmingCharacters(in: .whitespacesAndNewlines)) { config.pcOscPort = p } }))
                Text("Where the PC sends the trackers. For standalone Quest, enter the headset's Wi-Fi IP (Quest ▸ Settings ▸ Wi-Fi ▸ your network ▸ Advanced — reserve it in your router so it doesn't change). 127.0.0.1 only if VRChat runs on the PC via Quest Link.")
                    .font(.caption)
                    .foregroundStyle(Theme.textSecondary)
                if oscTargetsLoopback {
                    Label("Sending OSC to the PC's own loopback. For a standalone Quest, enter the headset's Wi-Fi IP — otherwise the trackers never reach VRChat.",
                          systemImage: "exclamationmark.triangle.fill")
                        .font(.caption)
                        .foregroundStyle(Theme.warning)
                }
            }

            // -- Camera & stream --
            Section("Camera & Stream") {
                Picker("Camera", selection: Binding(
                    get: { config.cameraDeviceID },
                    set: { config.cameraDeviceID = $0; camera.selectCamera($0.isEmpty ? nil : $0) })) {
                    Text("Auto (built-in)").tag("")
                    ForEach(CameraCapture.availableCameras(), id: \.uniqueID) { cam in
                        Text(cam.localizedName).tag(cam.uniqueID)
                    }
                }
                // NLF uses the SAME handedness setting as on-device (mirrorTracking) so PC
                // and on-device track 1:1 — this is the same control as the General tab's
                // "Swap tracker handedness", surfaced here for convenience.
                Toggle("Swap tracker handedness (L/R)", isOn: $config.mirrorTracking)
                Picker("Resolution", selection: Binding(
                    get: { "\(config.pcStreamWidth)x\(config.pcStreamHeight)" },
                    set: { sel in
                        let p = sel.split(separator: "x").compactMap { Int($0) }
                        if p.count == 2 { config.pcStreamWidth = p[0]; config.pcStreamHeight = p[1] }
                    })) {
                    Text("960 × 540").tag("960x540")
                    Text("1280 × 720").tag("1280x720")
                    Text("1920 × 1080").tag("1920x1080")
                }
                Picker("Stream FPS", selection: Binding(
                    get: { config.pcStreamFPS },
                    set: { config.pcStreamFPS = $0 })) {
                    Text("30").tag(30)
                    Text("60").tag(60)
                }
                HStack {
                    Text("Bitrate")
                    Spacer()
                    Text("\(config.pcBitrateMbps) Mbps")
                        .monospacedDigit()
                        .foregroundStyle(Theme.textSecondary)
                }
                Slider(value: Binding(get: { Double(config.pcBitrateMbps) },
                                      set: { config.pcBitrateMbps = Int($0.rounded()) }),
                       in: 2...25, step: 1)
                Text("Hardware H.264 (VideoToolbox) low-delay over Wi-Fi. Higher resolution/bitrate is sharper but adds a little latency. Mirror reflects the skeleton on the PC (selfie handedness) — the same proper L/R mirror as On Device, not an image flip.")
                    .font(.caption)
                    .foregroundStyle(Theme.textSecondary)
                if running {
                    Text("Stream settings are captured at Start — changes apply the next time you Start.")
                        .font(.caption)
                        .foregroundStyle(Theme.textMuted)
                }
            }

            // -- Preview --
            Section("Preview") {
                Toggle("Show skeleton overlay", isOn: $config.pcShowSkeleton)
                Text("Draw the PC's tracked skeleton over the live camera preview. OFF = camera only. (The skeleton lights up once the PC is streaming back.)")
                    .font(.caption)
                    .foregroundStyle(Theme.textSecondary)
            }

            // -- Tracking: the 1:1 PinoFBT setup --
            Section("Tracking — 1:1 PinoFBT 2.0") {
                Toggle("8-point (elbows on)", isOn: $config.pcSendElbows)
                LabeledContent("Height") {
                    Text(String(format: "%.2f m", config.userHeightMeters))
                        .font(Theme.valueFont)
                        .foregroundStyle(Theme.textPrimary)
                }
                Slider(value: $config.userHeightMeters, in: 1.2...2.2, step: 0.01) {
                    Text("Real-world height")
                }
                Text("The GPU runs the real fast_kinematics solver, so the arms are byte-exact here — 8-point is the PinoFBT desktop default. OneEuro smoothing + IK are fixed to the original; height scales as cm/175 (shared with the On Device tab).")
                    .font(.caption)
                    .foregroundStyle(Theme.textSecondary)
            }

            // -- Latency prediction (shared with the On Device tab) --
            Section {
                HStack {
                    Text("Latency prediction (ms)")
                    Spacer()
                    Text("\(config.predictionLeadMs)")
                        .monospacedDigit()
                        .foregroundStyle(Theme.textSecondary)
                }
                Slider(value: Binding(get: { Double(config.predictionLeadMs) },
                                      set: { config.predictionLeadMs = Int($0.rounded()) }),
                       in: 0...150, step: 5)
            } footer: {
                Text("Forward-predicts your motion to cancel pipeline latency — higher feels lower-latency but overshoots more on fast direction changes. Applies to both On Device and PC modes. ~50ms is a safe start; push higher to chase minimum delay.")
                    .foregroundStyle(Theme.textSecondary)
            }

            // -- Performance / sharing the PC --
            Section("Performance") {
                Toggle("Polite mode (share the PC)", isOn: $config.pcPoliteMode)
                Picker("Max FPS", selection: Binding(
                    get: { config.pcFpsCap },
                    set: { config.pcFpsCap = $0 })) {
                    Text("Unlimited").tag(0)
                    Text("30").tag(30)
                    Text("60").tag(60)
                    Text("90").tag(90)
                }
                Text("Polite mode runs the PC daemon below-normal priority so someone using the PC keeps the GPU. Cap the FPS to lighten the GPU load further (the model can do ~90).")
                    .font(.caption)
                    .foregroundStyle(Theme.textSecondary)
            }

        }
        .formStyle(.grouped)
    }
}
