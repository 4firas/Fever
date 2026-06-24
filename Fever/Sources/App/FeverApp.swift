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
    }

    var body: some Scene {
        WindowGroup {
            ContentView(config: config, pipeline: pipeline)
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
            SettingsView(config: config)
        }
    }
}

/// The Settings scene (Cmd+,). OSC host/port, real-world height, per-tracker
/// enable toggles with address preview, One-Euro + rotation smoothing sliders,
/// body tweak sliders, and the head-reference toggle.
struct SettingsView: View {
    @Bindable var config: TrackingConfig

    var body: some View {
        TabView {
            oscTab
                .tabItem { Label("OSC", systemImage: "network") }
            trackersTab
                .tabItem { Label("Trackers", systemImage: "figure.walk") }
        }
        .frame(width: 460, height: 460)
        .tint(Theme.crimsonBright)
    }

    // MARK: OSC

    private var oscTab: some View {
        Form {
            Section("Output") {
                // Explicit get/set bindings that write through on EVERY edit (and a
                // trim), so the value can't be lost if the field never "commits"
                // before you switch to VRChat — the cause of "I set the Quest IP but
                // it kept sending to 127.0.0.1". Persistence itself is handled by the
                // config's didSet (regression-tested in ConfigPersistenceTests).
                TextField("Host", text: Binding(
                    get: { config.oscHost },
                    set: { config.oscHost = $0.trimmingCharacters(in: .whitespacesAndNewlines) }))
                    .onSubmit { /* commit-on-Return is already covered by the set above */ }
                TextField("Port", text: Binding(
                    get: { String(config.oscPort) },
                    set: { if let p = Int($0.trimmingCharacters(in: .whitespacesAndNewlines)) { config.oscPort = p } }))
                Text("VRChat receives OSC on UDP 9000 by default. The Quest's IP changes on reconnect — set a DHCP reservation to keep it fixed.")
                    .font(.caption)
                    .foregroundStyle(Theme.textSecondary)
            }
            Section("Body Scale") {
                LabeledContent("Height") {
                    Text(String(format: "%.2f m", config.userHeightMeters))
                        .font(Theme.valueFont)
                        .foregroundStyle(Theme.textPrimary)
                }
                Slider(value: $config.userHeightMeters, in: 1.2...2.2, step: 0.01) {
                    Text("Real-world height")
                }
            }
            Section {
                Toggle("Send head reference (PC/no HMD only)", isOn: $config.sendHeadReference)
            } footer: {
                Text("OFF by default. Keep OFF with a Quest/HMD — the headset is already the authoritative head and enabling this creates a conflicting second head that folds the neck in VRChat. Enable only for PC setups without a head-mounted display.")
                    .foregroundStyle(Theme.textSecondary)
            }
            Section {
                Toggle("Mirror left/right", isOn: $config.mirrorTracking)
            } footer: {
                Text("Flips the tracker handedness if your avatar appears left-right reversed in VRChat. Affects only the OSC trackers, not the live preview.")
                    .foregroundStyle(Theme.textSecondary)
            }
            Section {
                Toggle("Send rotation", isOn: $config.sendRotation)
            } footer: {
                Text("Streams /rotation for all body trackers (hip, chest, feet, knees, elbows) — PinoFBT parity. The per-bone solver is fixed: chest follows the spine, feet follow heel→toe with roll locked. The head is always position-only.")
                    .foregroundStyle(Theme.textSecondary)
            }
        }
        .formStyle(.grouped)
    }

    // MARK: Trackers

    private var trackersTab: some View {
        Form {
            Section("Enabled Trackers") {
                ForEach(JointType.allCases, id: \.self) { joint in
                    Toggle(isOn: binding(for: joint)) {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(label(for: joint))
                            if let slot = config.slotMap[joint] {
                                Text("/tracking/trackers/\(slot)")
                                    .font(Theme.monoSmall)
                                    .foregroundStyle(Theme.textSecondary)
                            }
                        }
                    }
                }
            }
            Section {
                Toggle("Foot trackers at heel", isOn: $config.footTrackersAtAnkle)
            } footer: {
                Text("Places the left/right foot trackers at the detected heel (PinoFBT's foot point — lower and steadier for floor contact). Off uses the toe position instead.")
                    .foregroundStyle(Theme.textSecondary)
            }
        }
        .formStyle(.grouped)
    }


    private func binding(for joint: JointType) -> Binding<Bool> {
        Binding(
            get: { config.enabledJoints.contains(joint) },
            set: { on in
                if on { config.enabledJoints.insert(joint) }
                else { config.enabledJoints.remove(joint) }
            }
        )
    }

    private func label(for joint: JointType) -> String {
        switch joint {
        case .head:       return "Head"
        case .chest:      return "Chest"
        case .hip:        return "Hip"
        case .leftElbow:  return "Left Elbow"
        case .rightElbow: return "Right Elbow"
        case .leftKnee:   return "Left Knee"
        case .rightKnee:  return "Right Knee"
        case .leftFoot:   return "Left Foot"
        case .rightFoot:  return "Right Foot"
        }
    }
}