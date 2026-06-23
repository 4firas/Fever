import simd

/// Converts NLF camera-space joints (right-handed, **+Y down**, meters) into the
/// VRChat world frame (**+Y up**, meters). Positions are kept ABSOLUTE — no
/// hip-relative re-centering, no floor latch — because VRChat re-origins the whole
/// tracker space via the continuous `head/position` stream (findings §6).
///
/// The IK solver runs on these WORLD positions, so the hip rotation it derives is
/// already in VRChat space (no separate rotation conjugation needed). The exact
/// axis signs / front-camera mirror are the documented [UNK] (findings §6/§12):
/// they are TUNABLES, defaulted to a reasonable left-handed + mirrored guess and
/// pinned by a known-pose regression + live VRChat validation in stage 9.
public struct CameraToWorldTransform: Sendable {
    /// Front-camera horizontal mirror (PinoFBT flips the frame internally, §10).
    public var mirrorX: Bool
    /// Camera +Y-down → world +Y-up. Almost certainly true; exposed for completeness.
    public var flipY: Bool
    /// Right-handed → VRChat left-handed depth flip.
    public var flipZ: Bool
    /// NLF emits absolute meters → default 1 (no scaling). Optional user override.
    public var heightScale: Float

    public init(mirrorX: Bool = true, flipY: Bool = true, flipZ: Bool = true, heightScale: Float = 1) {
        self.mirrorX = mirrorX; self.flipY = flipY; self.flipZ = flipZ; self.heightScale = heightScale
    }

    /// Per-axis sign vector applied to every joint.
    public var signs: SIMD3<Float> {
        SIMD3<Float>(mirrorX ? -1 : 1, flipY ? -1 : 1, flipZ ? -1 : 1)
    }

    public func point(_ p: SIMD3<Float>) -> SIMD3<Float> { signs * p * heightScale }

    /// Transform all 24 joints into world space.
    public func apply(_ joints3D: [SIMD3<Float>]) -> [SIMD3<Float>] {
        joints3D.map { point($0) }
    }
}
