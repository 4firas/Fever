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

    var body: some View {
        ZStack {
            // The content backdrop / letterbox fill. Never glass. Kurokula so the
            // bars around the aspect-fit (.resizeAspect) frame read as the theme
            // background rather than a flat black void.
            Theme.background

            if let session {
                PreviewLayerView(session: session)
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

    func makeNSView(context: Context) -> PreviewNSView {
        let view = PreviewNSView()
        view.attach(session: session)
        return view
    }

    func updateNSView(_ nsView: PreviewNSView, context: Context) {
        nsView.attach(session: session)
    }

    /// Layer-backed `NSView` that owns an `AVCaptureVideoPreviewLayer` sublayer
    /// kept full-bleed via `layout()`.
    final class PreviewNSView: NSView {

        private var previewLayer: AVCaptureVideoPreviewLayer?

        override init(frame frameRect: NSRect) {
            super.init(frame: frameRect)
            wantsLayer = true
            layer = CALayer()
            // Kurokula (#131515) so the aspect-fit letterbox bars match the theme.
            layer?.backgroundColor = NSColor(red: 0x13 / 255.0,
                                             green: 0x15 / 255.0,
                                             blue: 0x15 / 255.0,
                                             alpha: 1).cgColor
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

            layer?.addSublayer(preview)
            previewLayer = preview
        }

        override func layout() {
            super.layout()
            previewLayer?.frame = bounds
        }
    }
}