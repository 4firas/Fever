import FeverCore
import SwiftUI
import AVFoundation
import simd

/// The main Fever window.
///
/// Liquid Glass layout (macOS 26.0+):
///  - CONTENT layer: full-bleed `CameraPreview` with a `SkeletonOverlay` on top.
///    No glass is ever applied to this layer.
///  - CONTROLS layer: a `NavigationSplitView` (sidebar lists tracker groups +
///    a session panel with the mode switch; detail is the preview) plus an
///    `.inspector` for the selected group, and two `GlassEffectContainer`
///    clusters floating over the preview — one bottom-center control bar
///    (Start/Stop `.glassProminent` + a live readout: FPS/dropped on device, the
///    offload status in PC mode) and one top-center tracker status strip of
///    evenly-spaced tinted pills. (PinoFBT has no Recenter — VRChat calibrates.)
struct ContentView: View {

    @Bindable var config: TrackingConfig
    let pipeline: TrackingPipeline
    let controller: PCOffloadController
    let camera: CameraCapture

    /// Preview video source switches with the mode: in PC mode Fever owns the camera
    /// (shown live while streaming); in local mode it's the pipeline's session.
    private var previewSession: AVCaptureSession? {
        if config.inferenceOnPC { return controller.phase == .streaming ? camera.session : nil }
        // Only show the live layer while running; idle falls through to the
        // "press Start" placeholder (a stopped session would just render blank).
        return pipeline.isRunning ? pipeline.previewSession : nil
    }
    private var previewAuthorized: Bool {
        config.inferenceOnPC ? (camera.authorization == .authorized) : pipeline.cameraAuthorized
    }
    /// Skeleton overlay points: in PC mode the PC's returned skeleton (when the
    /// toggle is on); in local mode the local inference's points — mirrored in x to
    /// match the always-mirrored preview (the local points are in raw-frame space,
    /// the preview layer flips horizontally, so without this the skeleton is reversed).
    private var skeletonPoints: [SIMD2<Float>] {
        if config.inferenceOnPC { return config.pcShowSkeleton ? controller.previewPoints : [] }
        return pipeline.previewPoints.map { SIMD2<Float>(1 - $0.x, $0.y) }
    }

    /// Mode-aware status for the main-window chrome (so PC mode shows PC state, not
    /// the stopped on-device pipeline).
    private var liveStatus: LiveStatus {
        LiveStatus(config: config, pipeline: pipeline, controller: controller)
    }

    // Manual `@State` expansion (CLT lacks the SwiftUIMacros plugin): the
    // `State<T>` value is stored as `_x`, a computed accessor exposes
    // `wrappedValue`, and call sites use `_x.projectedValue` where the macro
    // would have synthesized `$x`.
    private var _selectedGroup = State<TrackerGroup?>(initialValue: .feet)
    private var selectedGroup: TrackerGroup? {
        get { _selectedGroup.wrappedValue }
        nonmutating set { _selectedGroup.wrappedValue = newValue }
    }
    private var _inspectorPresented = State(initialValue: true)
    private var inspectorPresented: Bool {
        get { _inspectorPresented.wrappedValue }
        nonmutating set { _inspectorPresented.wrappedValue = newValue }
    }

    var body: some View {
        NavigationSplitView {
            SidebarView(config: config,
                        pipeline: pipeline,
                        status: liveStatus,
                        selectedGroup: _selectedGroup.projectedValue)
                .navigationTitle("Fever")
                // Adaptive sidebar: grows/shrinks with the window between a
                // readable minimum and a cap that keeps the preview dominant.
                // No fixed `.frame` so the column width drives the layout.
                .navigationSplitViewColumnWidth(min: 230, ideal: 280, max: 360)
        } detail: {
            previewContent
                // Exactly ONE title source: the sidebar's `.navigationTitle`
                // drives the window title. No custom `.principal` toolbar item
                // here, so the toolbar shows the title only once.
                .toolbar {
                    ToolbarItem(placement: .primaryAction) {
                        Button {
                            inspectorPresented.toggle()
                        } label: {
                            Label("Inspector", systemImage: "sidebar.trailing")
                        }
                    }
                }
                .inspector(isPresented: _inspectorPresented.projectedValue) {
                    TrackerInspector(config: config,
                                     pipeline: pipeline,
                                     status: liveStatus,
                                     group: selectedGroup)
                        // Adaptive inspector: proportional, never clipping the
                        // tracker cards' monospaced position/rotation rows.
                        .inspectorColumnWidth(min: 260, ideal: 320, max: 420)
                }
        }
        .onChange(of: config.inferenceOnPC) { _, _ in
            // The two modes share ONE camera. Switching mid-session must stop whatever
            // is running so the camera is released cleanly (otherwise local capture +
            // the PC streamer would fight over it).
            pipeline.stop()
            controller.stop()
        }
    }

    // MARK: - Content layer (camera + skeleton) with the floating Glass controls

    private var previewContent: some View {
        ZStack {
            // Kurokula fill for the WHOLE detail area so the aspect-fit camera
            // frame is letterboxed in the theme background, never a black void.
            Theme.background

            // CONTENT: the camera, never glass. With `.resizeAspect` the WHOLE
            // 16:9 frame is always visible (no cropping), letterboxed inside the
            // detail column. We deliberately do NOT `.ignoresSafeArea()` here so
            // the frame stays fully within the visible detail area and is never
            // pushed under the sidebar or inspector. The overlay shares the exact
            // same bounds, so the skeleton lands on the body. The preview shows a
            // clear placeholder (not a black void) when there is no live feed.
            CameraPreview(session: previewSession,
                          camera: camera,
                          authorized: previewAuthorized,
                          running: liveStatus.running,
                          startHint: config.inferenceOnPC
                              ? "Press Start to wake the PC and stream the camera to it."
                              : "Press Start to begin tracking.",
                          inferredFrame: config.inferenceOnPC ? nil : pipeline.previewImage)
                .overlay {
                    SkeletonOverlay(points: skeletonPoints)
                }

            // CONTROLS: floating chrome, sized to the window. Using a
            // GeometryReader so both the top status strip and the bottom
            // control bar stay centered and degrade gracefully on a narrow
            // window instead of overflowing the preview.
            GeometryReader { geo in
                VStack(spacing: 0) {
                    // Status strip anchored top-center. It scrolls horizontally
                    // (centered when it fits, scrollable when it does not) so it
                    // never overflows a narrow window or crowds the subject.
                    ScrollView(.horizontal, showsIndicators: false) {
                        TrackerStatusStrip(status: liveStatus)
                            // Inset the centered content so the end pills are not
                            // flush against the window edges when it overflows.
                            .frame(minWidth: geo.size.width, alignment: .center)
                            .padding(.horizontal, 16)
                    }
                    .frame(maxWidth: geo.size.width)
                    // Sit clearly BELOW the toolbar / title bar with breathing
                    // room; respect the safe area so it never tucks under the
                    // inspector toggle button.
                    .padding(.top, 20)

                    Spacer()

                    // The single tidy control bar, bottom-center. Allowed to
                    // shrink within the available width rather than clip.
                    ControlBar(config: config,
                               pipeline: pipeline,
                               controller: controller,
                               camera: camera)
                        .frame(maxWidth: geo.size.width)
                        .padding(.bottom, 22)
                }
                .frame(width: geo.size.width, height: geo.size.height)
            }
        }
    }
}

// MARK: - Bottom-center control bar (one GlassEffectContainer)

/// Exactly ONE `GlassEffectContainer` for the control cluster (glass cannot
/// sample glass, so there is one container per cluster). Holds the primary
/// Start/Stop button (`.glassProminent`, crimson tint) and a compact live readout
/// (FPS/dropped on device, the offload status in PC mode) on a single backing
/// capsule with small monospaced numerals. (PinoFBT has no Recenter — VRChat calibrates.)
private struct ControlBar: View {
    @Bindable var config: TrackingConfig
    let pipeline: TrackingPipeline
    let controller: PCOffloadController
    let camera: CameraCapture

    /// Active state for the CURRENT mode: PC-offload (controller) vs local (pipeline).
    private var running: Bool {
        config.inferenceOnPC ? controller.isActive : pipeline.isRunning
    }

    var body: some View {
        GlassEffectContainer(spacing: 14) {
            // On a comfortably wide window everything sits on one row; when
            // space is tight the readout wraps below the buttons rather than
            // clipping, so its labels are never lost.
            ViewThatFits(in: .horizontal) {
                HStack(spacing: 14) {
                    buttons
                    readoutCapsule
                }
                .padding(.horizontal, 6)

                VStack(spacing: 12) {
                    HStack(spacing: 14) { buttons }
                    readoutCapsule
                }
                .padding(.horizontal, 6)
            }
        }
    }

    // PinoFBT has no in-app Recenter or Body Stabilizer: rotations are absolute and
    // VRChat's own T-pose calibration handles alignment. Just Start/Stop here.
    @ViewBuilder private var buttons: some View {
        Button {
            toggleTracking()
        } label: {
            Label(running ? "Stop" : "Start",
                  systemImage: running ? "stop.fill" : "play.fill")
                .font(.system(size: 13, weight: .semibold))
                .frame(minWidth: 74)
        }
        .buttonStyle(.glassProminent)
        .tint(running ? Theme.crimsonBright : Theme.crimson)
        // Block a no-op Start in PC mode with a blank PC address/MAC (it would just
        // spin "Connecting…" for ~a minute before failing). Stop is always allowed.
        .disabled(!canToggle)
        .help(canToggle ? "" : "Enter the PC address and Wake-on-LAN MAC in Settings ▸ Inference on PC.")
    }

    /// Whether Start/Stop can act right now. Stop is always allowed; Start in PC mode
    /// needs at least a PC address and a MAC to attempt a wake.
    private var canToggle: Bool {
        if running { return true }
        if config.inferenceOnPC {
            return !config.pcHost.trimmingCharacters(in: .whitespaces).isEmpty
                && !config.pcMAC.trimmingCharacters(in: .whitespaces).isEmpty
        }
        return true
    }

    // Compact live readout on a single backing capsule so the small monospaced
    // numerals + SF Symbols sit on a legible surface over arbitrary live video.
    // Within the container this merges with the sibling button glass without
    // glass-on-glass conflict. The capsule has a sensible min width and its
    // labels never truncate.
    @ViewBuilder private var readoutCapsule: some View {
        if config.inferenceOnPC {
            pcStatusCapsule
        } else {
            localReadoutCapsule
        }
    }

    // In PC mode the model runs remotely, so the local FPS/dropped readout is
    // meaningless — show the offload connection status instead.
    private var pcStatusCapsule: some View {
        HStack(spacing: 9) {
            StatusDot(pcDotColor, size: 7)
            Text(controller.status)
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(Theme.textPrimary)
                .lineLimit(1)
                // Truncate the middle of a long "Streaming → host · OSC → …" string
                // instead of forcing the whole control bar wider than a narrow window.
                .truncationMode(.middle)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 9)
        .frame(minWidth: 150, maxWidth: 360)
        .glassEffect(.regular, in: .capsule)
    }

    private var pcDotColor: Color { controller.phase.dotColor }

    private var localReadoutCapsule: some View {
        HStack(spacing: 18) {
            readout(value: String(format: "%.0f", pipeline.fps),
                    label: "FPS",
                    systemImage: "speedometer")
            readout(value: "\(Int(pipeline.outputFPS))",
                    label: "OUT \(pipeline.fpsMultiplier)×",
                    systemImage: "wave.3.forward")
            readout(value: "\(pipeline.droppedFrames)",
                    label: "DROP",
                    systemImage: "arrow.down.to.line")
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
        .frame(minWidth: 150)
        .fixedSize(horizontal: true, vertical: false)
        .glassEffect(.regular, in: .capsule)
    }

    private func readout(value: String, label: String, systemImage: String) -> some View {
        HStack(spacing: 7) {
            Image(systemName: systemImage)
                .font(.system(size: 12))
                .foregroundStyle(Theme.textSecondary)
            VStack(alignment: .leading, spacing: 1) {
                Text(value)
                    .font(.system(size: 14, weight: .semibold).monospacedDigit())
                    .foregroundStyle(Theme.textPrimary)
                    .lineLimit(1)
                    .fixedSize()
                Text(label)
                    .font(.system(size: 9, weight: .semibold))
                    .tracking(0.6)
                    .foregroundStyle(Theme.textMuted)
                    // Labels must always render in full — never truncate away.
                    .lineLimit(1)
                    .fixedSize()
            }
        }
        .fixedSize(horizontal: true, vertical: false)
    }

    private func toggleTracking() {
        if config.inferenceOnPC {
            if controller.isActive { controller.stop() } else { controller.start(makePCConfig(), camera: camera) }
        } else {
            if pipeline.isRunning { pipeline.stop() } else { pipeline.start() }
        }
    }

    /// Snapshot the live settings into a PC-offload config, resolving the selected
    /// camera's `uniqueID` to the avfoundation device NAME ffmpeg captures by.
    private func makePCConfig() -> PCOffloadConfig {
        let camName: String? = config.cameraDeviceID.isEmpty ? nil
            : CameraCapture.availableCameras().first { $0.uniqueID == config.cameraDeviceID }?.localizedName
        // Never stream faster than we capture: the camera is capped at cameraMaxFPS
        // (default 30), so encoding at a higher pcStreamFPS would only duplicate frames
        // and add latency. Clamp the effective stream rate to the capture cap.
        let streamFPS = min(config.pcStreamFPS, config.cameraMaxFPS)
        return PCOffloadConfig.make(
            host: config.pcHost, user: config.pcUser, mac: config.pcMAC,
            model: config.pcModel == "gvhmr" ? .gvhmr : .nlf,
            oscIP: config.pcOscHost, oscPort: config.pcOscPort,
            relayViaMac: config.pcOscRelayViaMac, relayPort: 9001,
            heightCm: Int((config.userHeightMeters * 100).rounded()),
            // ONE handedness setting for both modes: PC NLF applies the SAME proper L/R
            // skeleton mirror as on-device, so they track 1:1 (was a separate pcFlipCamera).
            sendElbows: config.pcSendElbows, mirror: config.mirrorTracking,
            predictionLeadMs: config.predictionLeadMs,
            gvhmrK: config.gvhmrK, gvhmrMirror: config.gvhmrMirror,
            gvhmrFlipX: config.gvhmrFlipX, gvhmrMoving: config.gvhmrMoving,
            gvhmrFootContact: config.gvhmrFootContact, gvhmrNativeRot: config.gvhmrNativeRot,
            streamW: config.pcStreamWidth, streamH: config.pcStreamHeight,
            streamFPS: streamFPS, bitrateMbps: config.pcBitrateMbps,
            politeMode: config.pcPoliteMode, fpsCap: config.pcFpsCap,
            // Pinned: the daemon listens on 5000 and only 5000 is opened in the PC
            // firewall, so the stream target must stay 5000 (no user-facing port knob).
            // The OSC relay port (9001) is local to the Mac, only used in Relay route.
            streamPort: 5000, cameraName: camName)
    }
}

// MARK: - Mode-aware status snapshot

/// A small value the main-window chrome reads instead of the raw pipeline, so that
/// in PC-offload mode it reflects the PC session (running, status, OSC target, the
/// configured tracker set) instead of the idle on-device pipeline. In local mode it
/// passes through the pipeline as before.
struct LiveStatus {
    let onPC: Bool
    let running: Bool
    let elbows: Bool
    let statusText: String    // the verbose controller status (PC) — for tooltips/detail
    let stateLabel: String    // short word for the panel: Running / Connecting… / Error / Stopped
    let stateColor: Color
    let oscTarget: String
    let fpsText: String?      // nil in PC mode (the PC owns inference, no Mac fps)
    let dropped: Int?
    private let onPCStreaming: Bool
    // Snapshot of which joints are live, taken at init on the MainActor. Holding it
    // as plain data lets `isLive(_:)` be a cheap nonisolated read (the views query it
    // from non-actor-isolated computed properties).
    private let liveJoints: Set<JointType>

    @MainActor init(config: TrackingConfig, pipeline: TrackingPipeline, controller: PCOffloadController) {
        onPC = config.inferenceOnPC
        onPCStreaming = controller.phase == .streaming
        liveJoints = onPC ? [] : Set(JointType.active(sendElbows: true).filter { pipeline.isLive($0) })
        running = onPC ? controller.isActive : pipeline.isRunning
        elbows = onPC ? config.pcSendElbows : config.sendElbows
        // On device, the detail line carries a health warning (demo pose / no frames)
        // when present — empty otherwise, so the SessionPanel shows it only on a problem.
        statusText = onPC ? controller.status : (pipeline.healthNote ?? "")
        oscTarget = onPC ? "\(config.pcOscHost):\(config.pcOscPort)" : "\(config.oscHost):\(config.oscPort)"
        fpsText = onPC ? nil : String(format: "%.1f", pipeline.fps)
        dropped = onPC ? nil : pipeline.droppedFrames
        if onPC {
            stateColor = controller.phase.dotColor        // shared Phase→Color mapping
            switch controller.phase {
            case .streaming:           stateLabel = "Running on PC"
            case .waking, .starting:   stateLabel = "Connecting…"
            case .stopping:            stateLabel = "Stopping…"
            case .error:               stateLabel = "Error"
            case .idle:                stateLabel = "Stopped"
            }
        } else if pipeline.isRunning {
            // Running, but amber if there's a health warning (demo pose / no frames) so
            // the dot doesn't read "all good" over fake or absent tracking.
            stateLabel = "Running"
            stateColor = pipeline.healthNote == nil ? Theme.good : Theme.warning
        } else {
            (stateLabel, stateColor) = ("Stopped", Theme.textMuted)
        }
    }

    /// Per-tracker liveness. The Mac doesn't see per-joint detail in PC mode (the PC
    /// computes it), so while streaming the enabled trackers read as live.
    func isLive(_ j: JointType) -> Bool {
        onPC ? onPCStreaming : liveJoints.contains(j)
    }
}

// MARK: - Top-center tracker status strip (one GlassEffectContainer)

/// A neat, evenly-spaced single-line strip of small pills (one
/// `GlassEffectContainer`), one per enabled tracker. Each pill is tinted by its
/// live state (green tracking / yellow acquiring / red lost) and morphs via
/// `glassEffectID` as trackers acquire or lose. Kept narrow and top-anchored so
/// it never crowds the subject or the bottom control bar.
private struct TrackerStatusStrip: View {
    let status: LiveStatus

    @Namespace private var stripNamespace

    var body: some View {
        let joints = JointType.active(sendElbows: status.elbows)
        GlassEffectContainer(spacing: 8) {
            HStack(spacing: 8) {
                ForEach(joints, id: \.self) { joint in
                    TrackerStatusPill(
                        joint: joint,
                        slot: joint.pinoSlot,
                        state: TrackerState.resolve(
                            enabled: true,
                            live: status.isLive(joint))
                    )
                    .glassEffectID(joint, in: stripNamespace)
                }
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 6)
        }
        .fixedSize()
    }
}

/// A single per-tracker status pill rendered on tinted regular glass. Tint is
/// driven by the resolved tracker state.
private struct TrackerStatusPill: View {
    let joint: JointType
    let slot: String?
    let state: TrackerState

    var body: some View {
        HStack(spacing: 5) {
            StatusDot(state: state, size: 6)
            Text(slot ?? "–")
                .font(.system(size: 11, weight: .semibold).monospacedDigit())
                .foregroundStyle(Theme.textPrimary)
            Text(joint.shortLabel)
                .font(.system(size: 10, weight: .regular))
                .foregroundStyle(Theme.textSecondary)
        }
        .lineLimit(1)
        .fixedSize(horizontal: true, vertical: false)
        .padding(.horizontal, 10)
        .padding(.vertical, 5)
        .glassEffect(.regular.tint(state.color.opacity(0.28)), in: .capsule)
        // State is color-coded — spell it out so it's not lost for VoiceOver / color-blind users.
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("\(joint.longLabel) tracker")
        .accessibilityValue(state.label)
    }
}

// MARK: - Sidebar (tracker groups + session panel)

/// The five tracker groups surfaced in the sidebar, each mapping to one or more
/// of the 9 solved joint types.
enum TrackerGroup: String, CaseIterable, Identifiable, Hashable {
    case head, hips, feet, knees, elbows

    var id: String { rawValue }

    var title: String {
        switch self {
        case .head:   return "Head"
        case .hips:   return "Hips"
        case .feet:   return "Feet"
        case .knees:  return "Knees"
        case .elbows: return "Elbows"
        }
    }

    var systemImage: String {
        switch self {
        case .head:   return "person.crop.circle"
        case .hips:   return "figure.walk"
        case .feet:   return "shoeprints.fill"
        case .knees:  return "figure.flexibility"
        case .elbows: return "figure.arms.open"
        }
    }

    /// Joint types belonging to this group.
    var joints: [JointType] {
        switch self {
        case .head:   return [.head, .chest]
        case .hips:   return [.hip]
        case .feet:   return [.leftFoot, .rightFoot]
        case .knees:  return [.leftKnee, .rightKnee]
        case .elbows: return [.leftElbow, .rightElbow]
        }
    }
}

private struct SidebarView: View {
    @Bindable var config: TrackingConfig
    let pipeline: TrackingPipeline
    let status: LiveStatus
    @Binding var selectedGroup: TrackerGroup?

    var body: some View {
        List(selection: $selectedGroup) {
            Section {
                ForEach(TrackerGroup.allCases) { group in
                    NavigationLink(value: group) {
                        GroupRow(group: group, status: status)
                    }
                }
            } header: {
                SectionHeader("Trackers")
            }

            Section {
                SessionPanel(config: config, status: status)
                    .listRowInsets(EdgeInsets(top: 8, leading: 8, bottom: 8, trailing: 8))
            } header: {
                SectionHeader("Session")
            }
        }
        .listStyle(.sidebar)
        .tint(Theme.crimsonBright)
    }
}

/// A scannable sidebar row: icon + name on the left, an acquired/total count
/// and a colored state dot on the right (green tracking / red lost / beige idle).
private struct GroupRow: View {
    let group: TrackerGroup
    let status: LiveStatus

    var body: some View {
        HStack(spacing: 11) {
            Image(systemName: group.systemImage)
                .font(.system(size: 14))
                .foregroundStyle(Theme.dustyRose)
                .frame(width: 22, alignment: .center)

            Text(group.title)
                .font(Theme.labelFont)
                .foregroundStyle(Theme.textPrimary)
                .lineLimit(1)

            Spacer(minLength: 8)

            Text("\(acquiredCount)/\(streamedJoints.count)")
                .font(.system(size: 12, weight: .medium).monospacedDigit())
                .foregroundStyle(Theme.textSecondary)

            StatusDot(state: aggregateState, size: 8)
        }
        .padding(.vertical, 3)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(group.title)
        .accessibilityValue("\(acquiredCount) of \(streamedJoints.count) trackers, \(aggregateState.label)")
    }

    /// This group's joints that Fever actually streams: the active tracker set PLUS
    /// the head (the always-on position anchor, excluded from `active()` but always
    /// sent). Without head the Head group could never read better than "1/2".
    private var streamedJoints: [JointType] {
        let active = JointType.active(sendElbows: status.elbows)
        return group.joints.filter { active.contains($0) || $0 == .head }
    }

    /// Liveness of one streamed joint — head is live whenever the session is running
    /// (it has no per-joint detection gate), others use the snapshot.
    private func jointLive(_ j: JointType) -> Bool {
        j == .head ? status.running : status.isLive(j)
    }

    /// Number of this group's streamed joints currently live.
    private var acquiredCount: Int {
        streamedJoints.filter(jointLive).count
    }

    /// Aggregate dot: green if any streamed joint is live, red if streamed-but-none-live,
    /// beige idle if this group streams nothing in the current mode.
    private var aggregateState: TrackerState {
        guard !streamedJoints.isEmpty else { return .idle }
        return streamedJoints.contains(where: jointLive) ? .tracking : .lost
    }
}

/// A compact session panel: the mode switch (On Device / Inference on PC), a clear
/// state line with a colored dot, FPS + dropped frames (local mode only), and the
/// resolved OSC target host:port (monospaced). In PC mode the PC owns inference, so
/// FPS/Dropped are hidden and the state line reflects the offload phase.
private struct SessionPanel: View {
    @Bindable var config: TrackingConfig
    let status: LiveStatus

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Mode switch, always visible in the main window (no need to open
            // Settings). Disabled while a session runs — ContentView's onChange
            // releases the shared camera, so flipping mid-run would yank it.
            Picker("Mode", selection: $config.inferenceOnPC) {
                Text("On Device").tag(false)
                Text("Inference on PC").tag(true)
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .disabled(status.running)

            HStack(spacing: 8) {
                StatusDot(status.stateColor, size: 9)
                Text(status.stateLabel)
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundStyle(Theme.textPrimary)
                    .lineLimit(1)
            }

            // Surface the detail right here where the user is looking — the PC offload
            // status/errors, or an on-device health warning (demo pose / no frames) —
            // not only in the floating control-bar capsule. Wraps so it's fully readable.
            if !status.statusText.isEmpty {
                Text(status.statusText)
                    .font(.caption)
                    .foregroundStyle(status.stateColor == Theme.good ? Theme.textSecondary : status.stateColor)
                    .lineLimit(3)
                    .fixedSize(horizontal: false, vertical: true)
            }

            VStack(spacing: 7) {
                if let fps = status.fpsText {
                    MetricRow(label: "FPS", value: fps)
                }
                if let dropped = status.dropped {
                    MetricRow(label: "Dropped",
                              value: "\(dropped)",
                              valueColor: dropped > 0 ? Theme.warning : Theme.textPrimary)
                }
                MetricRow(label: "OSC",
                          value: status.oscTarget,
                          valueColor: Theme.textSecondary)
            }
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Theme.surfaceRaised, in: RoundedRectangle(cornerRadius: 10, style: .continuous))
    }
}

// MARK: - Inspector for the selected tracker group

private struct TrackerInspector: View {
    @Bindable var config: TrackingConfig
    let pipeline: TrackingPipeline
    let status: LiveStatus
    let group: TrackerGroup?

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                if let group {
                    HStack(spacing: 9) {
                        Image(systemName: group.systemImage)
                            .font(.system(size: 15))
                            .foregroundStyle(Theme.dustyRose)
                        Text(group.title)
                            .font(.system(size: 17, weight: .semibold))
                            .foregroundStyle(Theme.textPrimary)
                    }
                    .padding(.bottom, 2)

                    ForEach(group.joints, id: \.self) { joint in
                        TrackerCard(config: config, pipeline: pipeline, status: status, joint: joint)
                    }
                } else {
                    ContentUnavailableView("No Selection",
                                           systemImage: "hand.point.up.left",
                                           description: Text("Select a tracker group in the sidebar."))
                        .padding(.top, 40)
                }
            }
            .padding(16)
        }
    }
}

/// A clean detail card for one tracker: title + enable toggle, the OSC address
/// (monospaced, muted), Position / Rotation as aligned 3-column monospaced rows,
/// and Confidence as a small green→red bar.
private struct TrackerCard: View {
    @Bindable var config: TrackingConfig
    let pipeline: TrackingPipeline
    let status: LiveStatus
    let joint: JointType

    private var live: LiveTracker? {
        pipeline.liveTrackers.first { $0.joint == joint }
    }
    private var isActive: Bool {
        JointType.active(sendElbows: status.elbows).contains(joint)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 8) {
                StatusDot(state: TrackerState.resolve(
                    enabled: isActive, live: status.isLive(joint)), size: 8)
                Text(joint.longLabel)
                    .font(Theme.titleFont)
                    .foregroundStyle(Theme.textPrimary)
                Spacer()
            }

            HStack {
                Text("Address")
                    .font(Theme.captionFont)
                    .foregroundStyle(Theme.textMuted)
                Spacer()
                Text("/tracking/trackers/\(joint.pinoSlot)")
                    .font(Theme.monoSmall)
                    .foregroundStyle(Theme.textSecondary)
                    .textSelection(.enabled)
            }

            if let live {
                // Per-axis readout only exists for on-device inference (the Mac solves
                // it). In PC mode the GPU solves it, so there's no local detail to show.
                vectorRow(title: "Position", v: live.position)
                vectorRow(title: "Rotation", v: live.eulerDegrees)
            } else {
                Text(statusLine)
                    .font(Theme.captionFont)
                    .foregroundStyle(Theme.textMuted)
            }
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Theme.surface, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .stroke(Theme.textMuted.opacity(0.18), lineWidth: 1)
        )
    }

    /// The line shown when there's no local per-axis readout, worded for the mode and
    /// for whether this tracker is in the active set (so an off elbow doesn't read as
    /// "no data" but as deliberately disabled).
    private var statusLine: String {
        if !isActive {
            return "Disabled — turn on 8-point (elbows) in Settings"
        }
        if status.onPC {
            return status.running ? "Solved on the PC (no per-joint readout on the Mac)"
                                  : "Press Start to stream to the PC"
        }
        return "No live data — press Start"
    }

    /// A label + aligned 3-column monospaced (x y z) row.
    private func vectorRow(title: String, v: SIMD3<Float>) -> some View {
        HStack(alignment: .firstTextBaseline) {
            Text(title)
                .font(Theme.captionFont)
                .foregroundStyle(Theme.textMuted)
            Spacer()
            HStack(spacing: 0) {
                component(v.x)
                component(v.y)
                component(v.z)
            }
        }
    }

    private func component(_ value: Float) -> some View {
        Text(String(format: "%6.2f", value))
            .font(.system(size: 12, weight: .regular, design: .monospaced))
            .foregroundStyle(Theme.textSecondary)
            .frame(width: 56, alignment: .trailing)
    }
}

// MARK: - Joint display labels

extension JointType {
    /// The body trackers Fever streams, in wire-slot order: 6-point always
    /// (chest, hip, knees, ankles) plus the two elbows when `sendElbows` is on.
    /// The head is the always-on position anchor, handled separately.
    static func active(sendElbows: Bool) -> [JointType] {
        var js: [JointType] = [.chest, .hip, .leftElbow, .rightElbow,
                               .leftKnee, .rightKnee, .leftFoot, .rightFoot]
        if !sendElbows { js.removeAll { $0 == .leftElbow || $0 == .rightElbow } }
        return js
    }

    var shortLabel: String {
        switch self {
        case .head:       return "Head"
        case .chest:      return "Chest"
        case .hip:        return "Hip"
        case .leftElbow:  return "L.Elb"
        case .rightElbow: return "R.Elb"
        case .leftKnee:   return "L.Knee"
        case .rightKnee:  return "R.Knee"
        case .leftFoot:   return "L.Foot"
        case .rightFoot:  return "R.Foot"
        }
    }

    var longLabel: String {
        switch self {
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