import FeverCore
import SwiftUI
import AVFoundation

/// Full-bleed live camera preview. Wraps an `AVCaptureVideoPreviewLayer` bound
/// to the running `AVCaptureSession` exposed by the pipeline.
///
/// This is the CONTENT layer of the Liquid Glass design: it sits at the very
/// bottom of the z-stack and is NEVER given a glass effect (glass belongs to
/// the controls layer above content). All floating chrome (the HUD, the
/// navigation split view, the inspector) renders above it.
///
/// Per the locked spec verdicts:
///  - `videoGravity = .resizeAspect` so the WHOLE camera frame is always visible
///    (no cropping), letterboxed in the kurokula background inside the detail
///    area. This also makes the skeleton overlay align exactly.
///  - The connection is horizontally mirrored (front/built-in webcam), matching
///    the horizontal mirror the pose pipeline applies, so the on-screen skeleton
///    lines up with the mirrored video the way a user expects from a "mirror".
///
/// When there is no session yet (stub source, denied access, or the camera has
/// not produced its first frame), the view shows an explicit SwiftUI
/// placeholder INSTEAD of a black void, so the operator always knows what state
/// the camera is in.
struct CameraPreview: View {

    /// Live capture session to display, passed through from the pipeline. `nil`
    /// while no live camera source is active (synthetic source, denied access,
    /// or before the session is wired up).
    let session: AVCaptureSession?
    /// Whether the OS has granted camera access. Drives the placeholder copy
    /// between "grant access" and "starting up".
    let authorized: Bool
    /// The frame inference last ran on. When non-nil (running) it's drawn over the
    /// live layer, so the visible preview advances at the inference rate — preview
    /// fps == inference fps. nil (stopped) falls through to the live camera layer.
    var inferredFrame: CGImage? = nil

    var body: some View {
        ZStack {
            // The content backdrop / letterbox fill. Never glass. Kurokula so the
            // bars around the aspect-fit (.resizeAspect) frame read as the theme
            // background rather than a flat black void.
            Theme.background

            if let session {
                PreviewLayerView(session: session, inferredFrame: inferredFrame)
            } else {
                placeholder
            }
        }
    }

    @ViewBuilder
    private var placeholder: some View {
        if !authorized {
            ContentUnavailableView {
                Label("Camera Access Needed", systemImage: "video.slash")
            } description: {
                Text("Grant access in System Settings ▸ Privacy & Security ▸ Camera, then restart Fever.")
            }
            .foregroundStyle(.white)
        } else {
            ContentUnavailableView {
                Label("Starting Camera…", systemImage: "video")
            } description: {
                Text("Waiting for the first frame from the camera.")
            }
            .foregroundStyle(.white)
        }
    }
}

/// The actual `NSViewRepresentable` that hosts the `AVCaptureVideoPreviewLayer`.
/// Split out so the SwiftUI placeholder above can be plain SwiftUI.
private struct PreviewLayerView: NSViewRepresentable {

    let session: AVCaptureSession
    let inferredFrame: CGImage?

    func makeNSView(context: Context) -> PreviewNSView {
        let view = PreviewNSView()
        view.attach(session: session)
        view.setInferredFrame(inferredFrame)
        return view
    }

    func updateNSView(_ nsView: PreviewNSView, context: Context) {
        nsView.attach(session: session)
        nsView.setInferredFrame(inferredFrame)
    }

    /// Layer-backed `NSView` that owns an `AVCaptureVideoPreviewLayer` sublayer
    /// kept full-bleed via `layout()`.
    final class PreviewNSView: NSView {

        private var previewLayer: AVCaptureVideoPreviewLayer?
        /// Drawn ABOVE the live layer; holds the exact inferred frame so the visible
        /// preview only advances when inference does. Mirrored + aspect-fit to match
        /// the live layer exactly so the skeleton overlay still lines up.
        private var inferredLayer: CALayer?

        override init(frame frameRect: NSRect) {
            super.init(frame: frameRect)
            wantsLayer = true
            layer = CALayer()
            // Kurokula (#131515) so the aspect-fit letterbox bars match the theme.
            layer?.backgroundColor = NSColor(red: 0x13 / 255.0,
                                             green: 0x15 / 255.0,
                                             blue: 0x15 / 255.0,
                                             alpha: 1).cgColor

            // Inferred-frame layer: aspect-fit + horizontally mirrored to overlay the
            // live preview exactly; contents animation disabled so frames swap crisply.
            let inf = CALayer()
            inf.contentsGravity = .resizeAspect
            inf.transform = CATransform3DMakeScale(-1, 1, 1)
            inf.actions = ["contents": NSNull()]
            inf.frame = bounds
            layer?.addSublayer(inf)
            inferredLayer = inf
        }

        @available(*, unavailable)
        required init?(coder: NSCoder) { fatalError("init(coder:) is not used") }

        func attach(session: AVCaptureSession) {
            // Reuse the existing preview layer if it already points at this
            // session; only rebuild when the session identity changes.
            if let existing = previewLayer, existing.session === session {
                return
            }

            previewLayer?.removeFromSuperlayer()

            let preview = AVCaptureVideoPreviewLayer(session: session)
            // Aspect-FIT: show the whole frame, letterboxed (never cropped).
            preview.videoGravity = .resizeAspect
            preview.frame = bounds

            // Horizontally mirror the front/built-in webcam preview so it reads
            // like a mirror and matches the pose pipeline's horizontal mirror.
            if let connection = preview.connection, connection.isVideoMirroringSupported {
                connection.automaticallyAdjustsVideoMirroring = false
                connection.isVideoMirrored = true
            }

            // Below the inferred-frame layer so, while running, the inferred frame
            // (inference rate) covers the live feed; when stopped the live feed shows.
            layer?.insertSublayer(preview, at: 0)
            previewLayer = preview
        }

        /// Swap in the latest inferred frame (nil = show the live layer underneath).
        func setInferredFrame(_ image: CGImage?) {
            inferredLayer?.contents = image
        }

        override func layout() {
            super.layout()
            previewLayer?.frame = bounds
            inferredLayer?.frame = bounds
        }
    }
}