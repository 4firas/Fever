import FeverCore
import SwiftUI
import simd

/// Draws the live 24-joint SMPL skeleton over the camera preview from the
/// pipeline's `previewPoints` (screen-normalized x,y ∈ [0,1], top-left; NaN =
/// absent), reproducing the `AVCaptureVideoPreviewLayer` `.resizeAspect` letterbox
/// so the skeleton lands on the body at any window size.
struct SkeletonOverlay: View {

    /// 24-slot, screen-normalized points (SMPLJoint order); NaN entries are absent.
    let points: [SIMD2<Float>]

    /// Locked camera aspect (1280x720) to reproduce the preview aspect-fit.
    private let cameraAspect: CGFloat = 1280.0 / 720.0

    private static let leftSet: Set<Int>  = [1, 4, 7, 10, 13, 16, 18, 20, 22]
    private static let rightSet: Set<Int> = [2, 5, 8, 11, 14, 17, 19, 21, 23]

    var body: some View {
        Canvas { context, size in
            guard points.count >= SMPLJoint.count else { return }

            let viewAspect = size.width / size.height
            var drawnW = size.width
            var drawnH = size.height
            if viewAspect > cameraAspect { drawnH = size.height; drawnW = size.height * cameraAspect }
            else { drawnW = size.width; drawnH = size.width / cameraAspect }
            let offX = (size.width - drawnW) / 2
            let offY = (size.height - drawnH) / 2

            func project(_ i: Int) -> CGPoint? {
                let p = points[i]
                guard p.x.isFinite, p.y.isFinite else { return nil }
                return CGPoint(x: offX + CGFloat(p.x) * drawnW, y: offY + CGFloat(p.y) * drawnH)
            }

            // Bones via the SMPL kinematic parent tree (drawn first; dots on top).
            var bonePath = Path()
            for i in 1..<SMPLJoint.count {
                let parent = SMPLJoint.parentIndex[i]
                guard parent >= 0, let a = project(i), let b = project(parent) else { continue }
                bonePath.move(to: a)
                bonePath.addLine(to: b)
            }
            context.stroke(bonePath, with: .color(.black.opacity(0.35)),
                           style: StrokeStyle(lineWidth: 3.5, lineCap: .round, lineJoin: .round))
            context.stroke(bonePath, with: .color(.white.opacity(0.95)),
                           style: StrokeStyle(lineWidth: 2, lineCap: .round, lineJoin: .round))

            // Joint dots, colored by SMPL body side (left/right/center).
            for i in 0..<SMPLJoint.count {
                guard let p = project(i) else { continue }
                let color: Color = Self.leftSet.contains(i) ? Theme.trackerLeft
                                 : Self.rightSet.contains(i) ? Theme.trackerRight : .white
                let r: CGFloat = 3
                context.fill(Path(ellipseIn: CGRect(x: p.x - r, y: p.y - r, width: 2 * r, height: 2 * r)),
                             with: .color(color))
            }
        }
        .allowsHitTesting(false)
    }
}
