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
///    a session panel; detail is the preview) plus an `.inspector` for the
///    selected group, and two `GlassEffectContainer` clusters floating over the
///    preview — one bottom-center control bar (Start/Stop `.glassProminent`,
///    Recenter `.glass`, live FPS/dropped readout) and one top-center tracker
///    status strip of evenly-spaced tinted pills.
struct ContentView: View {

    @Bindable var config: TrackingConfig
    let pipeline: TrackingPipeline

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
    private var _calibrationPresented = State(initialValue: false)
    private var calibrationPresented: Bool {
        get { _calibrationPresented.wrappedValue }
        nonmutating set { _calibrationPresented.wrappedValue = newValue }
    }

    var body: some View {
        NavigationSplitView {
            SidebarView(config: config,
                        pipeline: pipeline,
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
                                     group: selectedGroup)
                        // Adaptive inspector: proportional, never clipping the
                        // tracker cards' monospaced position/rotation rows.
                        .inspectorColumnWidth(min: 260, ideal: 320, max: 420)
                }
        }
        .sheet(isPresented: _calibrationPresented.projectedValue) {
            CalibrationSheet(pipeline: pipeline)
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
            CameraPreview(session: pipeline.previewSession,
                          authorized: pipeline.cameraAuthorized)
                .overlay {
                    SkeletonOverlay(points: pipeline.previewPoints)
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
                        TrackerStatusStrip(config: config, pipeline: pipeline)
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
                               onCalibrate: { calibrationPresented = true })
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
/// Start/Stop button (`.glassProminent`, crimson tint), a secondary Recenter
/// (`.glass`), and a compact live readout (FPS, dropped) on a single backing
/// capsule with small monospaced numerals.
private struct ControlBar: View {
    @Bindable var config: TrackingConfig
    let pipeline: TrackingPipeline
    let onCalibrate: () -> Void

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

    @ViewBuilder private var buttons: some View {
        Button {
            toggleTracking()
        } label: {
            Label(pipeline.isRunning ? "Stop" : "Start",
                  systemImage: pipeline.isRunning ? "stop.fill" : "play.fill")
                .font(.system(size: 13, weight: .semibold))
                .frame(minWidth: 74)
        }
        .buttonStyle(.glassProminent)
        .tint(pipeline.isRunning ? Theme.crimsonBright : Theme.crimson)

        Button {
            onCalibrate()
        } label: {
            Label("Recenter", systemImage: "scope")
                .font(.system(size: 13, weight: .medium))
        }
        .buttonStyle(.glass)
        .tint(Theme.dustyRose)
        .disabled(!pipeline.isRunning)
    }

    // Compact live readout on a single backing capsule so the small monospaced
    // numerals + SF Symbols sit on a legible surface over arbitrary live video.
    // Within the container this merges with the sibling button glass without
    // glass-on-glass conflict. The capsule has a sensible min width and its
    // labels never truncate.
    private var readoutCapsule: some View {
        HStack(spacing: 18) {
            readout(value: String(format: "%.0f", pipeline.fps),
                    label: "FPS",
                    systemImage: "speedometer")
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
        if pipeline.isRunning {
            pipeline.stop()
        } else {
            config.enableTracker = true
            pipeline.start()
        }
    }
}

// MARK: - Top-center tracker status strip (one GlassEffectContainer)

/// A neat, evenly-spaced single-line strip of small pills (one
/// `GlassEffectContainer`), one per enabled tracker. Each pill is tinted by its
/// live state (green tracking / yellow acquiring / red lost) and morphs via
/// `glassEffectID` as trackers acquire or lose. Kept narrow and top-anchored so
/// it never crowds the subject or the bottom control bar.
private struct TrackerStatusStrip: View {
    @Bindable var config: TrackingConfig
    let pipeline: TrackingPipeline

    @Namespace private var stripNamespace

    var body: some View {
        if !config.enabledJoints.isEmpty {
            GlassEffectContainer(spacing: 8) {
                HStack(spacing: 8) {
                    ForEach(orderedEnabledJoints, id: \.self) { joint in
                        TrackerStatusPill(
                            joint: joint,
                            slot: config.slotMap[joint],
                            state: TrackerState.resolve(
                                live: pipeline.lastFrameJoints.first { $0.type == joint },
                                enabled: true)
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

    private var orderedEnabledJoints: [JointType] {
        JointType.allCases.filter { config.enabledJoints.contains($0) }
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
    @Binding var selectedGroup: TrackerGroup?

    var body: some View {
        List(selection: $selectedGroup) {
            Section {
                ForEach(TrackerGroup.allCases) { group in
                    NavigationLink(value: group) {
                        GroupRow(group: group, config: config, pipeline: pipeline)
                    }
                }
            } header: {
                SectionHeader("Trackers")
            }

            Section {
                SessionPanel(config: config, pipeline: pipeline)
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
    let config: TrackingConfig
    let pipeline: TrackingPipeline

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

            Text("\(acquiredCount)/\(group.joints.count)")
                .font(.system(size: 12, weight: .medium).monospacedDigit())
                .foregroundStyle(Theme.textSecondary)

            StatusDot(state: aggregateState, size: 8)
        }
        .padding(.vertical, 3)
    }

    /// Number of this group's enabled joints currently tracking with live data.
    private var acquiredCount: Int {
        group.joints.filter { joint in
            guard config.enabledJoints.contains(joint) else { return false }
            return pipeline.lastFrameJoints.contains { $0.type == joint && $0.confidence >= 0.4 }
        }.count
    }

    /// Aggregate dot: green if any joint tracks, red if enabled-but-none-live,
    /// beige idle if nothing in the group is enabled.
    private var aggregateState: TrackerState {
        let enabled = group.joints.filter { config.enabledJoints.contains($0) }
        guard !enabled.isEmpty else { return .idle }
        let tracking = enabled.contains { joint in
            pipeline.lastFrameJoints.contains { $0.type == joint && $0.confidence >= 0.4 }
        }
        return tracking ? .tracking : .lost
    }
}

/// A compact session panel: a clear Running/Stopped state with a colored dot,
/// FPS (monospaced), dropped frames, and the OSC target host:port (monospaced).
private struct SessionPanel: View {
    let config: TrackingConfig
    let pipeline: TrackingPipeline

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 8) {
                StatusDot(pipeline.isRunning ? Theme.good : Theme.textMuted, size: 9)
                Text(pipeline.isRunning ? "Running" : "Stopped")
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundStyle(Theme.textPrimary)
            }

            VStack(spacing: 7) {
                MetricRow(label: "FPS",
                          value: String(format: "%.1f", pipeline.fps))
                MetricRow(label: "Dropped",
                          value: "\(pipeline.droppedFrames)",
                          valueColor: pipeline.droppedFrames > 0 ? Theme.warning : Theme.textPrimary)
                MetricRow(label: "OSC",
                          value: "\(config.oscHost):\(config.oscPort)",
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
                        TrackerCard(config: config, pipeline: pipeline, joint: joint)
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
    let joint: JointType

    private var live: VRJoint? {
        pipeline.lastFrameJoints.first { $0.type == joint }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 8) {
                StatusDot(state: TrackerState.resolve(
                    live: live, enabled: config.enabledJoints.contains(joint)), size: 8)
                Text(joint.longLabel)
                    .font(Theme.titleFont)
                    .foregroundStyle(Theme.textPrimary)
                Spacer()
                Toggle("", isOn: enabledBinding)
                    .labelsHidden()
                    .tint(Theme.crimsonBright)
            }

            if let slot = config.slotMap[joint] {
                HStack {
                    Text("Address")
                        .font(Theme.captionFont)
                        .foregroundStyle(Theme.textMuted)
                    Spacer()
                    Text("/tracking/trackers/\(slot)")
                        .font(Theme.monoSmall)
                        .foregroundStyle(Theme.textSecondary)
                        .textSelection(.enabled)
                }
            }

            if let live {
                vectorRow(title: "Position", v: live.position)
                vectorRow(title: "Rotation", v: eulerDegrees(live.rotation))

                VStack(alignment: .leading, spacing: 5) {
                    HStack {
                        Text("Confidence")
                            .font(Theme.captionFont)
                            .foregroundStyle(Theme.textMuted)
                        Spacer()
                        Text(String(format: "%.0f%%", live.confidence * 100))
                            .font(.system(size: 12, weight: .semibold).monospacedDigit())
                            .foregroundStyle(Theme.textSecondary)
                    }
                    ConfidenceBar(confidence: live.confidence)
                }
            } else {
                Text("No live data")
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

    private var enabledBinding: Binding<Bool> {
        Binding(
            get: { config.enabledJoints.contains(joint) },
            set: { on in
                if on { config.enabledJoints.insert(joint) }
                else { config.enabledJoints.remove(joint) }
            }
        )
    }

    /// Display-only ZXY euler degrees from the solver-frame quaternion.
    private func eulerDegrees(_ q: simd_quatf) -> SIMD3<Float> {
        let x = q.imag.x, y = q.imag.y, z = q.imag.z, w = q.real
        // ZXY extraction.
        let sinX = 2 * (w * x - y * z)
        let ex = abs(sinX) >= 1 ? Float(copysign(.pi / 2, sinX)) : asin(sinX)
        let ey = atan2(2 * (w * y + x * z), 1 - 2 * (x * x + y * y))
        let ez = atan2(2 * (w * z + x * y), 1 - 2 * (x * x + z * z))
        let k: Float = 180 / .pi
        return SIMD3<Float>(ex * k, ey * k, ez * k)
    }
}

// MARK: - Joint display labels

extension JointType {
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