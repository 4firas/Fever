import FeverCore
import SwiftUI

/// Guided recenter sheet. Walks the user through facing the camera and standing
/// in a neutral pose, runs a short countdown, then fires the single head snap
/// pulse via `TrackingPipeline.calibrate()` (a >300ms-isolated head message that
/// instant-snaps VRChat's tracking-space alignment to the avatar head).
struct CalibrationSheet: View {

    let pipeline: TrackingPipeline

    @Environment(\.dismiss) private var dismiss

    // Manual `@State` expansion — the CLT toolchain lacks the SwiftUIMacros
    // plugin, so the macro-generated `_x` storage / `x` accessor are written by
    // hand. None of these need a `$x` projection (no bindings handed out).
    private var _countdown = State<Int?>(initialValue: nil)
    private var countdown: Int? {
        get { _countdown.wrappedValue }
        nonmutating set { _countdown.wrappedValue = newValue }
    }
    private var _didCalibrate = State(initialValue: false)
    private var didCalibrate: Bool {
        get { _didCalibrate.wrappedValue }
        nonmutating set { _didCalibrate.wrappedValue = newValue }
    }
    private var _countdownTask = State<Task<Void, Never>?>(initialValue: nil)
    private var countdownTask: Task<Void, Never>? {
        get { _countdownTask.wrappedValue }
        nonmutating set { _countdownTask.wrappedValue = newValue }
    }

    var body: some View {
        VStack(spacing: 24) {
            VStack(spacing: 8) {
                Image(systemName: didCalibrate ? "checkmark.circle.fill" : "figure.stand")
                    .font(.system(size: 56, weight: .regular))
                    .foregroundStyle(didCalibrate ? .green : .primary)
                    .contentTransition(.symbolEffect(.replace))

                Text(didCalibrate ? "Recentered" : "Recenter / Face Forward")
                    .font(.title2.weight(.semibold))
            }

            Text(instructions)
                .font(.body)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 360)

            // Live preview of the running skeleton so the user can confirm
            // framing before snapping.
            ZStack {
                RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .fill(.black.opacity(0.25))
                SkeletonOverlay(points: pipeline.previewPoints)
            }
            .frame(width: 320, height: 240)
            .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
            .overlay(alignment: .center) {
                if let countdown {
                    Text("\(countdown)")
                        .font(.system(size: 88, weight: .bold, design: .rounded))
                        .foregroundStyle(.white)
                        .shadow(radius: 8)
                        .transition(.scale.combined(with: .opacity))
                }
            }

            HStack(spacing: 12) {
                Button("Cancel") {
                    countdownTask?.cancel()
                    dismiss()
                }
                .buttonStyle(.glass)
                .keyboardShortcut(.cancelAction)

                Button(countdown == nil ? "Recenter" : "Recentering…") {
                    startCountdown()
                }
                .buttonStyle(.glassProminent)
                .disabled(countdown != nil || !pipeline.isRunning)
                .keyboardShortcut(.defaultAction)
            }
        }
        .padding(32)
        .frame(width: 440)
        .onDisappear { countdownTask?.cancel() }
    }

    private var instructions: String {
        if didCalibrate {
            return "Tracking space is aligned to your head. Close this window to continue."
        }
        if !pipeline.isRunning {
            return "Start tracking first, then recenter so your whole body is in frame."
        }
        return "Stand upright facing the camera with your whole body visible. "
             + "When the countdown ends, hold still — Fever snaps VRChat's "
             + "tracking space to your head."
    }

    private func startCountdown() {
        countdownTask?.cancel()
        didCalibrate = false
        countdownTask = Task { @MainActor in
            for n in stride(from: 3, through: 1, by: -1) {
                withAnimation { countdown = n }
                try? await Task.sleep(for: .seconds(1))
                if Task.isCancelled { withAnimation { countdown = nil }; return }
            }
            withAnimation { countdown = nil }
            pipeline.calibrate()
            withAnimation { didCalibrate = true }
        }
    }
}