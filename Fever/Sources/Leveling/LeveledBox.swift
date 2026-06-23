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
///
/// The box is sized from METRIC world landmarks (body proportions in metres,
/// constant at any camera distance) not from the 2D pixel silhouette. This gives
/// a roomy asymmetric volume — taller below the hip where the legs are, generous
/// fixed-metre margins — that floats around the user without sticking to the skin,
/// matching PinoFBT's green reference box behaviour.
public enum LeveledBoxBuilder {

    public static func build(landmarks: [NormalizedLandmark],
                             imagePoints: [SIMD2<Float>]) -> LeveledBox {
        guard landmarks.count == 33, imagePoints.count == 33 else { return .invalid }

        func world(_ l: BlazePose.Landmark) -> SIMD3<Float> { landmarks[l.rawValue].position }
        func img(_ l: BlazePose.Landmark) -> SIMD2<Float> { imagePoints[l.rawValue] }
        func finite2(_ v: SIMD2<Float>) -> Bool { v.x.isFinite && v.y.isFinite }

        let lHipI = img(.leftHip), rHipI = img(.rightHip)
        let lShI = img(.leftShoulder), rShI = img(.rightShoulder)
        guard finite2(lHipI), finite2(rHipI), finite2(lShI), finite2(rShI) else { return .invalid }

        let hipMidI = (lHipI + rHipI) * 0.5
        let shMidI  = (lShI + rShI) * 0.5

        // Metric body axes in the leveled solver frame.
        let midHipW = (world(.leftHip) + world(.rightHip)) * 0.5
        let neckW   = (world(.leftShoulder) + world(.rightShoulder)) * 0.5
        let spine   = neckW - midHipW
        guard simd_length(spine) > 0.05 else { return .invalid }
        let u    = simd_normalize(spine)               // torso up (solver)
        let rRaw = simd_cross(u, SIMD3<Float>(0, 0, 1))
        guard simd_length(rRaw) > 1e-3 else { return .invalid }
        let r = simd_normalize(rRaw)

        // Projection scale (screen-units / metre) from the spine: when the torso
        // tilts in Z (forward bend) the 2D spine projection shortens → scale drops
        // → box squishes, matching real foreshortening.
        let metricSpineLen = simd_length(spine)
        let screenSpineLen = simd_length(shMidI - hipMidI)
        guard screenSpineLen > 1e-4 else { return .invalid }
        let scale = screenSpineLen / metricSpineLen

        // Screen-space axes (image y is DOWN, so up-axis flips sign), scaled so
        // 1 m of body maps to `scale` screen-normalised units.
        let uImg = SIMD2<Float>(u.x, -u.y) * scale
        let rImg = SIMD2<Float>(r.x, -r.y) * scale
        guard finite2(uImg), finite2(rImg) else { return .invalid }

        // Metric half-extents from actual body proportions, with fixed-metre margins
        // so the box is always off the skin regardless of camera distance.
        let noseW   = world(.nose)
        let lAnkleW = world(.leftAnkle)
        let rAnkleW = world(.rightAnkle)

        let halfUpM    = max(simd_length(noseW - midHipW), 0.10) + 0.25   // hip→nose + head room
        let lowestFootY = min(lAnkleW.y, rAnkleW.y)
        let halfDownM  = max(abs(lowestFootY - midHipW.y), 0.30) + 0.15   // hip→foot + floor gap
        let shHalfWidth = max(abs(world(.leftShoulder).x  - neckW.x),
                              abs(world(.rightShoulder).x - neckW.x))
        let halfRightM = shHalfWidth + 0.25                                // shoulder half + arm room

        let center  = hipMidI
        let corners = [
            center + (-halfRightM * rImg + halfUpM   * uImg),   // TL
            center + ( halfRightM * rImg + halfUpM   * uImg),   // TR
            center + ( halfRightM * rImg - halfDownM * uImg),   // BR
            center + (-halfRightM * rImg - halfDownM * uImg),   // BL
        ]
        guard corners.allSatisfy(finite2) else { return .invalid }

        let upright = LevelEstimator.uprightSanity(nose: noseW, midHip: midHipW,
                                                   leftAnkle: lAnkleW, rightAnkle: rAnkleW)
        return LeveledBox(corners: corners, valid: upright)
    }
}
