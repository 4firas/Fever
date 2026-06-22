import Foundation
import simd
import FeverCore

/// Coverage for `LeveledBoxBuilder` — the green reference box geometry. Pins the
/// behavior the user described from the app: a pure TURN (vertical spine) keeps a
/// LEVEL square, a SIDE-BEND rolls it, a forward toe-reach skews/squishes it, and a
/// crouch makes it vanish.
enum LeveledBoxTests {

    /// Build a frame: `spine` is the shoulder-mid offset above the hip (in solver
    /// space, +Y up); `noseY` lets a crouch drop the head below the hips. Image
    /// points are a simple upright body so the box has a center + size.
    static func frame(spine: SIMD3<Float>, noseY: Float = 0.7)
        -> (lm: [NormalizedLandmark], img: [SIMD2<Float>]) {
        var lm = [NormalizedLandmark](repeating: NormalizedLandmark(position: .zero), count: 33)
        func put(_ l: BlazePose.Landmark, _ p: SIMD3<Float>) { lm[l.rawValue] = NormalizedLandmark(position: p) }
        let shMid = spine                                  // hips at origin → shoulder mid = spine
        put(.leftHip, SIMD3(-0.1, 0, 0));  put(.rightHip, SIMD3(0.1, 0, 0))
        put(.leftShoulder, shMid + SIMD3(-0.2, 0, 0)); put(.rightShoulder, shMid + SIMD3(0.2, 0, 0))
        put(.nose, SIMD3(shMid.x, noseY, shMid.z))         // upright: above hips; crouch: below
        put(.leftAnkle, SIMD3(-0.1, -0.9, 0)); put(.rightAnkle, SIMD3(0.1, -0.9, 0))

        var img = [SIMD2<Float>](repeating: SIMD2(.nan, .nan), count: 33)
        func iput(_ l: BlazePose.Landmark, _ p: SIMD2<Float>) { img[l.rawValue] = p }
        iput(.nose, SIMD2(0.5, 0.25))
        iput(.leftShoulder, SIMD2(0.45, 0.40)); iput(.rightShoulder, SIMD2(0.55, 0.40))
        iput(.leftHip, SIMD2(0.45, 0.60)); iput(.rightHip, SIMD2(0.55, 0.60))
        iput(.leftAnkle, SIMD2(0.45, 0.95)); iput(.rightAnkle, SIMD2(0.55, 0.95))
        return (lm, img)
    }

    static func run(_ t: TestRunner) {
        // Edge lengths / orientation helpers.
        func topTilt(_ b: LeveledBox) -> Float { abs(b.corners[0].y - b.corners[1].y) }   // |TL.y - TR.y|
        func sideTilt(_ b: LeveledBox) -> Float { abs(b.corners[0].x - b.corners[3].x) }   // |TL.x - BL.x|
        func width(_ b: LeveledBox) -> Float { simd_length(b.corners[1] - b.corners[0]) }  // TL→TR
        func height(_ b: LeveledBox) -> Float { simd_length(b.corners[3] - b.corners[0]) } // TL→BL

        t.test("BOX: upright → a valid LEVEL square") {
            let f = frame(spine: SIMD3(0, 0.5, 0))
            let b = LeveledBoxBuilder.build(landmarks: f.lm, imagePoints: f.img)
            t.check(b.valid, "upright box is valid")
            t.check(b.corners.count == 4, "four corners")
            t.close(topTilt(b), 0, tol: 1e-4, "top edge is horizontal (level)")
            t.close(sideTilt(b), 0, tol: 1e-4, "side edge is vertical (level)")
            t.close(width(b), height(b), tol: 1e-3, "square (width ≈ height)")
        }

        t.test("BOX: pure TURN (vertical spine, yaw) stays a level square") {
            // A turned user keeps shoulder-mid directly above hip-mid (spine = +Y),
            // only the shoulders' X/Z swap — the box must NOT tilt.
            var f = frame(spine: SIMD3(0, 0.5, 0))
            // Simulate a 90° turn: shoulders along Z instead of X (mid unchanged).
            f.lm[BlazePose.Landmark.leftShoulder.rawValue]  = NormalizedLandmark(position: SIMD3(0, 0.5, -0.2))
            f.lm[BlazePose.Landmark.rightShoulder.rawValue] = NormalizedLandmark(position: SIMD3(0, 0.5,  0.2))
            let b = LeveledBoxBuilder.build(landmarks: f.lm, imagePoints: f.img)
            t.check(b.valid, "turned box still valid")
            t.close(topTilt(b), 0, tol: 1e-4, "turn does NOT tilt the box (stays level)")
            t.close(sideTilt(b), 0, tol: 1e-4, "turn keeps sides vertical")
        }

        t.test("BOX: SIDE-BEND rolls the box (top edge no longer horizontal)") {
            let f = frame(spine: SIMD3(0.4, 0.5, 0))   // spine leans in +X (lateral)
            let b = LeveledBoxBuilder.build(landmarks: f.lm, imagePoints: f.img)
            t.check(b.valid, "side-bend box valid")
            t.check(topTilt(b) > 0.05, "side-bend rolls the box (top edge tilted, got \(topTilt(b)))")
        }

        t.test("BOX: forward toe-reach foreshortens the box (squishes vertically)") {
            let level = LeveledBoxBuilder.build(landmarks: frame(spine: SIMD3(0, 0.5, 0)).lm,
                                                imagePoints: frame(spine: SIMD3(0, 0.5, 0)).img)
            let f = frame(spine: SIMD3(0, 0.5, 0.45))  // spine tilts in Z (toward camera)
            let b = LeveledBoxBuilder.build(landmarks: f.lm, imagePoints: f.img)
            t.check(b.valid, "forward-bend box valid")
            t.close(topTilt(b), 0, tol: 1e-4, "forward bend keeps the top edge level…")
            t.check(height(b) < height(level) - 0.02, "…but squishes the box vertically (got \(height(b)) vs \(height(level)))")
        }

        t.test("BOX: a head-down crouch makes the box vanish (valid == false)") {
            let f = frame(spine: SIMD3(0, 0.5, 0), noseY: -0.2)   // nose dropped below hips
            let b = LeveledBoxBuilder.build(landmarks: f.lm, imagePoints: f.img)
            t.check(!b.valid, "crouch → box invalid (vanishes)")
        }

        t.test("BOX: missing torso image points → invalid") {
            var f = frame(spine: SIMD3(0, 0.5, 0))
            f.img[BlazePose.Landmark.leftHip.rawValue] = SIMD2(.nan, .nan)
            let b = LeveledBoxBuilder.build(landmarks: f.lm, imagePoints: f.img)
            t.check(!b.valid && b.corners.isEmpty, "missing hip image point → invalid box")
        }
    }
}
