import FeverCore
import SwiftUI
import simd

/// Draws the live RAW 2D detected skeleton over the camera preview using a
/// `Canvas`, VISO-style: the landmarks are drawn at their TRUE image positions
/// (no cloud re-normalization / centering), UNSMOOTHED, at the full inference
/// rate — so the overlay tracks instantly instead of floating and lagging.
///
/// Part of the CONTENT layer (it overlays the camera, not glass) so it carries
/// NO glass effect. It consumes the pipeline's `previewPoints`: a 33-slot,
/// SCREEN-normalized point array (x ∈ [0,1] from left, y ∈ [0,1] from top),
/// indexed by `BlazePose.Landmark.rawValue`, with `SIMD2(.nan,.nan)` = absent.
///
/// Mapping normalized → view must reproduce the SAME aspect-FIT letterbox the
/// `AVCaptureVideoPreviewLayer` applies (`videoGravity = .resizeAspect`),
/// otherwise the skeleton floats / offsets from the body. The camera is locked
/// to 1280x720 (16:9); we compute the displayed-image rect (the largest 16:9 box
/// that FITS inside the view, centered, with letterbox offsets) and place the
/// normalized points inside it.
/// X is NOT mirrored: Vision landmarks are in the non-mirrored data-output image
/// space (the data buffer is unmirrored even though the preview layer is), so
///   sx = offX + x * drawnW,  sy = offY + y * drawnH.
struct SkeletonOverlay: View {

    /// 33-slot, screen-normalized RAW landmark points; NaN entries are absent.
    let points: [SIMD2<Float>]

    /// Locked camera aspect ratio (1280x720, 16:9). Used to reproduce the
    /// preview layer's aspect-fit letterbox so the overlay lands on the body.
    private let cameraAspect: CGFloat = 1280.0 / 720.0

    /// Bone connections by BlazePose landmark. Drawn only when BOTH endpoints
    /// are present (non-NaN).
    private static let bones: [(BlazePose.Landmark, BlazePose.Landmark)] = [
        (.leftShoulder, .rightShoulder),
        (.leftShoulder, .leftElbow),
        (.leftElbow, .leftWrist),
        (.rightShoulder, .rightElbow),
        (.rightElbow, .rightWrist),
        (.leftShoulder, .leftHip),
        (.rightShoulder, .rightHip),
        (.leftHip, .rightHip),
        (.leftHip, .leftKnee),
        (.leftKnee, .leftAnkle),
        (.rightHip, .rightKnee),
        (.rightKnee, .rightAnkle),
        (.nose, .leftEar),
        (.nose, .rightEar)
    ]

    var body: some View {
        Canvas { context, size in
            guard points.count >= 33 else { return }

            // Reproduce the AVCaptureVideoPreviewLayer's `.resizeAspect`
            // letterbox: the displayed image is the LARGEST 16:9 box that FITS
            // inside the view (fitting one axis, leaving letterbox bars on the
            // other), centered. We map normalized points into that same drawn
            // rect so they land exactly on the body at any window size.
            let viewAspect = size.width / size.height
            var drawnW = size.width
            var drawnH = size.height
            if viewAspect > cameraAspect {
                // View is wider than the camera: fit height, pillarbox L/R.
                drawnH = size.height
                drawnW = size.height * cameraAspect
            } else {
                // View is taller/narrower: fit width, letterbox top/bottom.
                drawnW = size.width
                drawnH = size.width / cameraAspect
            }
            let offX = (size.width - drawnW) / 2
            let offY = (size.height - drawnH) / 2

            // Map a screen-normalized point into the aspect-fit drawn rect.
            // Vision landmarks live in the NON-mirrored data-output image space;
            // although the preview layer is mirrored, its data-output buffer is
            // not, so the overlay must NOT double-mirror X.
            // Returns nil if absent (NaN).
            func project(_ l: BlazePose.Landmark) -> CGPoint? {
                let p = points[l.rawValue]
                guard p.x.isFinite, p.y.isFinite else { return nil }
                let sx = offX + CGFloat(p.x) * drawnW
                let sy = offY + CGFloat(p.y) * drawnH
                return CGPoint(x: sx, y: sy)
            }

            // Live detection bounding box (PinoFBT-style): the padded min/max of
            // all present joints — the region the body currently occupies. It
            // follows you because it IS the detected-body extent; purely visual.
            var minX = CGFloat.greatestFiniteMagnitude, minY = CGFloat.greatestFiniteMagnitude
            var maxX = -CGFloat.greatestFiniteMagnitude, maxY = -CGFloat.greatestFiniteMagnitude
            var hasAny = false
            for l in BlazePose.Landmark.allCases {
                guard let p = project(l) else { continue }
                hasAny = true
                minX = min(minX, p.x); minY = min(minY, p.y)
                maxX = max(maxX, p.x); maxY = max(maxY, p.y)
            }
            if hasAny {
                let pad: CGFloat = 18
                let box = CGRect(x: minX - pad, y: minY - pad,
                                 width: (maxX - minX) + 2 * pad,
                                 height: (maxY - minY) + 2 * pad)
                context.stroke(Path(roundedRect: box, cornerRadius: 6),
                               with: .color(Theme.good.opacity(0.7)),
                               style: StrokeStyle(lineWidth: 2))
            }

            // Bones first, so joint dots draw on top.
            var bonePath = Path()
            for (a, b) in Self.bones {
                guard let pa = project(a), let pb = project(b) else { continue }
                bonePath.move(to: pa)
                bonePath.addLine(to: pb)
            }
            // Faint dark underlay for separation over bright video, then warm
            // beige bones at ~2 pt.
            context.stroke(bonePath,
                           with: .color(.black.opacity(0.35)),
                           style: StrokeStyle(lineWidth: 3.5, lineCap: .round, lineJoin: .round))
            context.stroke(bonePath,
                           with: .color(Theme.textPrimary.opacity(0.9)),
                           style: StrokeStyle(lineWidth: 2, lineCap: .round, lineJoin: .round))

            // Small green joint dots at every present landmark.
            let r: CGFloat = 3
            for l in BlazePose.Landmark.allCases {
                guard let p = project(l) else { continue }
                let rect = CGRect(x: p.x - r, y: p.y - r, width: 2 * r, height: 2 * r)
                context.fill(Path(ellipseIn: rect), with: .color(Theme.good))
            }
        }
        .allowsHitTesting(false)
    }
}