import simd

/// The PinoQuest-style green reference box, as screen-space corners ready for the
/// overlay. Its orientation tracks the **torso's tilt away from vertical, with yaw
/// excluded**, exactly as observed in the app:
///   • turning / a full 360° keeps the spine vertical → a LEVEL square,
///   • a side-bend rolls the torso → the box ROLLS (a tilted square / diamond),
///   • facing sideways and reaching for the toes pitches the torso → the box SKEWS,
///   • a close crouch fails the upright gate → `valid == false` (the box vanishes).
///
/// Camera tilt is already removed upstream (the leveling datum), so an upright user
/// yields a level box even on a tilted webcam. Corners are in the SAME screen-
/// normalized space as `PoseResult.imagePoints` (x∈[0,1] from left, y∈[0,1] from
/// top), so `SkeletonOverlay` projects them through the identical aspect-fit
/// letterbox as the skeleton.
public struct LeveledBox: Sendable, Equatable {
    /// Four corners, clockwise from top-left: [TL, TR, BR, BL]. Screen-normalized.
    public var corners: [SIMD2<Float>]
    /// False when the leveled reference is lost (crouch / missing torso) → the
    /// overlay should not draw the box (it "vanishes").
    public var valid: Bool

    public init(corners: [SIMD2<Float>], valid: Bool) {
        self.corners = corners
        self.valid = valid
    }

    public static let invalid = LeveledBox(corners: [], valid: false)
}

/// Builds a `LeveledBox` from one frame's leveled solver landmarks + image points.
/// Pure and deterministic, so it is unit-testable without the camera/overlay.
public enum LeveledBoxBuilder {
    /// Padding multiplier on the body's image-space reach (margin around the body).
    public static let sizePad: Float = 1.12

    public static func build(landmarks: [NormalizedLandmark],
                             imagePoints: [SIMD2<Float>]) -> LeveledBox {
        guard landmarks.count == 33, imagePoints.count == 33 else { return .invalid }

        func img(_ l: BlazePose.Landmark) -> SIMD2<Float> { imagePoints[l.rawValue] }
        func solver(_ l: BlazePose.Landmark) -> SIMD3<Float> { landmarks[l.rawValue].position }
        func finite2(_ v: SIMD2<Float>) -> Bool { v.x.isFinite && v.y.isFinite }

        let lHipI = img(.leftHip), rHipI = img(.rightHip)
        let lShI = img(.leftShoulder), rShI = img(.rightShoulder)
        guard finite2(lHipI), finite2(rHipI), finite2(lShI), finite2(rShI) else { return .invalid }
        let centerI = (lHipI + rHipI) * 0.5

        // Spine in the LEVELED solver frame (camera tilt already removed). Upright →
        // +Y; side-bend tilts it in X; forward-bend tilts it in Z (depth).
        let neckS = (solver(.leftShoulder) + solver(.rightShoulder)) * 0.5
        let midHipS = (solver(.leftHip) + solver(.rightHip)) * 0.5
        let spine = neckS - midHipS
        guard simd_length(spine) > 0.05 else { return .invalid }
        let u = simd_normalize(spine)                       // torso up (solver)
        let rRaw = simd_cross(u, SIMD3<Float>(0, 0, 1))     // torso right = up × (toward camera)
        guard simd_length(rRaw) > 1e-3 else { return .invalid }  // spine ∥ view axis (degenerate)
        let r = simd_normalize(rRaw)

        // Solver axes → image-space directions: image x = +X (un-mirrored preview),
        // image y is DOWN, so the up axis flips sign. NOT re-normalized in 2D — the
        // projected magnitude (≤1) carries the foreshortening, so a forward toe-reach
        // (spine tilts in Z) squishes the box while a side-bend (spine tilts in X)
        // rolls it. Both read as "angled", a pure turn (vertical spine) stays level.
        let uImg = SIMD2<Float>(u.x, -u.y)
        let rImg = SIMD2<Float>(r.x, -r.y)
        guard finite2(uImg), finite2(rImg) else { return .invalid }

        // Square half-extent = the body's image-space reach from the hip center
        // (out to the farthest present joint) plus margin, so the box frames the body.
        var reach: Float = 0
        for l in BlazePose.Landmark.allCases {
            let p = img(l)
            guard finite2(p) else { continue }
            reach = Swift.max(reach, simd_length(p - centerI))
        }
        guard reach > 1e-3 else { return .invalid }
        let h = reach * sizePad

        let corners = [
            centerI + h * (-rImg + uImg),   // TL
            centerI + h * ( rImg + uImg),   // TR
            centerI + h * ( rImg - uImg),   // BR
            centerI + h * (-rImg - uImg),   // BL
        ]
        guard corners.allSatisfy(finite2) else { return .invalid }

        // The box exists only while the user is upright (PinoQuest vanish-on-crouch).
        let upright = LevelEstimator.uprightSanity(nose: solver(.nose), midHip: midHipS,
                                                   leftAnkle: solver(.leftAnkle),
                                                   rightAnkle: solver(.rightAnkle))
        return LeveledBox(corners: corners, valid: upright)
    }
}
