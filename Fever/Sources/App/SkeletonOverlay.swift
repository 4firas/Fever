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

    /// The leveled reference box (PinoQuest-style). Drawn only when `valid`; its
    /// orientation tracks the torso tilt (level on a turn, angled on a bend).
    var box: LeveledBox = .invalid

    /// Animated 0→1 perimeter draw-in, replayed when the box (re)acquires and run
    /// back to 0 (vanish) when the reference is lost. Manual `State` storage — this
    /// CLT-only toolchain can't load the `@State` macro plugin (same hand-expansion
    /// as `FeverApp._config`): the `State<T>` is the stored property, a computed
    /// accessor exposes its `wrappedValue`.
    private var _drawProgress = State(initialValue: 0.0)
    private var drawProgress: Double {
        get { _drawProgress.wrappedValue }
        nonmutating set { _drawProgress.wrappedValue = newValue }
    }

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
        (.nose, .rightEar),
        // Face mesh
        (.leftEar, .leftEye), (.leftEye, .nose),
        (.rightEar, .rightEye), (.rightEye, .nose),
        (.mouthLeft, .mouthRight),
        // Finger fans (wrist → thumb / index / pinky, plus the index–pinky web)
        (.leftWrist, .leftThumb), (.leftWrist, .leftIndex),
        (.leftWrist, .leftPinky), (.leftIndex, .leftPinky),
        (.rightWrist, .rightThumb), (.rightWrist, .rightIndex),
        (.rightWrist, .rightPinky), (.rightIndex, .rightPinky),
        // Feet
        (.leftAnkle, .leftHeel), (.leftHeel, .leftFootIndex), (.leftAnkle, .leftFootIndex),
        (.rightAnkle, .rightHeel), (.rightHeel, .rightFootIndex), (.rightAnkle, .rightFootIndex)
    ]

    /// Body side of a landmark, for PinoQuest-style joint coloring.
    private enum Side { case left, right, center }
    private static func side(_ l: BlazePose.Landmark) -> Side {
        switch l {
        case .nose: return .center
        case .leftEyeInner, .leftEye, .leftEyeOuter, .leftEar, .mouthLeft,
             .leftShoulder, .leftElbow, .leftWrist, .leftPinky, .leftIndex, .leftThumb,
             .leftHip, .leftKnee, .leftAnkle, .leftHeel, .leftFootIndex:
            return .left
        default:
            return .right
        }
    }
    private static func dotColor(_ l: BlazePose.Landmark) -> Color {
        switch side(l) {
        case .left:   return Theme.trackerLeft
        case .right:  return Theme.trackerRight
        case .center: return .white
        }
    }
    /// Face-cluster landmarks (nose…mouth) get smaller dots for a denser mesh look.
    private static func isFace(_ l: BlazePose.Landmark) -> Bool {
        l.rawValue <= BlazePose.Landmark.mouthRight.rawValue   // indices 0...10
    }

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

            // PinoQuest-style leveled reference box: a square anchored on the torso,
            // oriented by the leveled spine — LEVEL when you stand/turn, ANGLED when
            // you bend. It draws in edge-by-edge (top edge first) on (re)acquire and
            // vanishes on a crouch. Drawn first so the skeleton sits on top.
            if box.valid, box.corners.count == 4 {
                func boxPt(_ i: Int) -> CGPoint {
                    let c = box.corners[i]
                    return CGPoint(x: offX + CGFloat(c.x) * drawnW, y: offY + CGFloat(c.y) * drawnH)
                }
                var quad = Path()
                quad.move(to: boxPt(0))      // TL
                quad.addLine(to: boxPt(1))   // TR  (top edge drawn first)
                quad.addLine(to: boxPt(2))   // BR
                quad.addLine(to: boxPt(3))   // BL
                quad.closeSubpath()
                let shown = quad.trimmedPath(from: 0, to: CGFloat(drawProgress))
                context.stroke(shown, with: .color(Theme.good),
                               style: StrokeStyle(lineWidth: 2.5, lineCap: .round, lineJoin: .round))
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
                           with: .color(.white.opacity(0.95)),
                           style: StrokeStyle(lineWidth: 2, lineCap: .round, lineJoin: .round))

            // Joint dots colored by body side (PinoQuest-style): cyan = left,
            // orange = right, white = center (nose). Face-cluster dots are smaller.
            for l in BlazePose.Landmark.allCases {
                guard let p = project(l) else { continue }
                let r: CGFloat = Self.isFace(l) ? 2 : 3
                let rect = CGRect(x: p.x - r, y: p.y - r, width: 2 * r, height: 2 * r)
                context.fill(Path(ellipseIn: rect), with: .color(Self.dotColor(l)))
            }
        }
        .allowsHitTesting(false)
        // Edge-by-edge draw-in when the box (re)acquires; run back to 0 on vanish.
        .onChange(of: box.valid) { _, valid in
            withAnimation(.easeOut(duration: 0.45)) { drawProgress = valid ? 1 : 0 }
        }
    }
}