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
                                        landmarker: makeLivePoseLandmarker())
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
            smoothingTab
                .tabItem { Label("Tuning", systemImage: "slider.horizontal.3") }
        }
        .frame(width: 460, height: 460)
        .tint(Theme.crimsonBright)
    }

    // MARK: OSC

    private var oscTab: some View {
        Form {
            Section("Output") {
                TextField("Host", text: $config.oscHost)
                TextField("Port", value: $config.oscPort, format: .number.grouping(.never))
                Text("VRChat receives OSC on UDP 9000 by default.")
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
                Toggle("Send head reference", isOn: $config.sendHeadReference)
            } footer: {
                Text("Streams /tracking/trackers/head to align the tracking space to your avatar's head.")
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
                Text("Off (recommended) = position-only trackers — the working config that matches PinoFBT. VRChat's IK solves limb rotation from positions. Our rotation still needs rework before it helps, so only turn this on to experiment. The head is always position-only regardless.")
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
                Toggle("Foot trackers at ankle", isOn: $config.footTrackersAtAnkle)
            } footer: {
                Text("Places the left/right foot trackers at the ankle (standard VRChat FBT). Off uses the toe position instead.")
                    .foregroundStyle(Theme.textSecondary)
            }
        }
        .formStyle(.grouped)
    }

    // MARK: Smoothing / body tweaks

    private var smoothingTab: some View {
        Form {
            Section("One-Euro Filter") {
                slider("Min cutoff", $config.stabilizerMinCutoff, 0.1...5, "%.2f")
                slider("Beta", $config.stabilizerBeta, 0...0.1, "%.3f")
                slider("Rotation smoothing", $config.rotationSmoothing, 0...1, "%.2f")
            }
            Section("Body Tweaks") {
                slider("Joint size", $config.jointSize, 0.1...3, "%.2f")
                slider("Hip sway", $config.hipExaggerateCoefficient, 1...3, "%.2f")
                slider("Hip lean", $config.hipTwistCoefficient, 1...2, "%.2f")
                slider("Hip length", $config.hipLength, -0.3...0.3, "%.2f")
                slider("Knee position", $config.kneePosition, -0.3...0.3, "%.2f")
            }
        }
        .formStyle(.grouped)
    }

    private func slider(_ title: String,
                        _ value: Binding<Double>,
                        _ range: ClosedRange<Double>,
                        _ fmt: String) -> some View {
        LabeledContent {
            VStack(alignment: .trailing, spacing: 2) {
                Slider(value: value, in: range)
                    .frame(width: 200)
                Text(String(format: fmt, value.wrappedValue))
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(Theme.textSecondary)
            }
        } label: {
            Text(title)
        }
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