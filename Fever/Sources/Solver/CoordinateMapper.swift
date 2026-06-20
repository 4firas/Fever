import simd

/// CoordinateMapper — the single authoritative conversion from the solver's
/// (Apple Vision–derived) coordinate frame into VRChat / Unity world space.
///
/// This module is the **make-or-break correctness step** of the whole pipeline
/// (per the locked spec verdicts). Every position and every orientation that
/// Fever puts on the OSC wire passes through here exactly once, *after* the
/// JointSolver and QuaternionStabilizer have produced joints in a single
/// consistent solver frame and *before* the TrackerAssembler packs them into
/// `/tracking/trackers/{slot}/position` and `/rotation` messages.
///
/// It is intentionally a pure, deterministic value type with no I/O, no clock,
/// and no hidden state, so it is trivially unit-testable.
///
/// ── Frames ───────────────────────────────────────────────────────────────
/// Solver / Vision frame:  RIGHT-handed, hip-root-relative METERS,
///                         +X = right, +Y = up, +Z = toward the camera.
///   (Apple `VNHumanBodyPose3DObservation` point.position is metric and
///    right-handed; positions are relative to the hip "root" joint.)
///
/// VRChat / Unity frame:   LEFT-handed, world-space METERS,
///                         +X = right, +Y = up, +Z = forward (away from camera).
///
/// ── Position transform ──────────────────────────────────────────────────
/// 1. Negate Z  → flips right-handed to left-handed (Vision +Z "toward camera"
///    becomes VRChat +Z "forward").
/// 2. Mirror X (front camera) → a front-facing webcam shows a mirror image, so
///    the user's real right hand appears on the left of the frame; negate X so
///    the avatar moves the same side as the user. Gated by `mirrorHorizontally`.
/// 3. Scale uniformly by `userHeightMeters / referenceHeightMeters`. Without a
///    depth sensor Vision assumes a 1.8 m reference height (heightEstimation ==
///    .reference), so we rescale to the user's true height to make 1.0 == 1 m.
///
///   x' = s · (mirror ? -1 : 1) · x
///   y' = s · y
///   z' = s · (-z)                where  s = userHeightMeters / referenceHeightMeters
///
/// ── Rotation transform ──────────────────────────────────────────────────
/// The position transform is, ignoring uniform positive scale, the diagonal
/// linear map  M = diag(s_x, 1, -1)  with  s_x = mirrorHorizontally ? -1 : 1.
/// M is its own inverse (entries are ±1), and a world-space rotation must be
/// re-expressed in the new basis by the similarity  R' = M · R · M.
///
/// For a diagonal axis-flip M = diag(s_x, s_y, s_z) (s_i ∈ {±1}), the conjugate
/// R' = M·R·M of a rotation given by quaternion q = (w, x, y, z) is again a
/// proper rotation whose quaternion is obtained by scaling each vector
/// component by the product of the *other two* axis signs (the scalar part is
/// unchanged):
///
///   w' = w
///   x' = (s_y · s_z) · x
///   y' = (s_x · s_z) · y
///   z' = (s_x · s_y) · z
///
/// With s_y = 1, s_z = -1 this gives:
///   • no mirror (s_x = +1):  (w, -x, -y,  z)
///   • mirror    (s_x = -1):  (w, -x,  y, -z)
///
/// The re-expressed quaternion is then handed to `quaternionToEulerZXYDegrees`
/// (in Math.swift), which decomposes it into the THREE Euler angles, in
/// DEGREES, that VRChat reconstructs internally in Z·X·Y order for the
/// `/rotation` endpoint. (VRChat's `/rotation` is three floats of euler degrees,
/// NOT a quaternion — see the OSC contract in the spec.)
///
/// ─────────────────────────────────────────────────────────────────────────
/// WORKED NUMERIC EXAMPLE (proves both transforms):
///
/// Settings: userHeightMeters = 1.5, referenceHeightMeters = 1.8,
///           mirrorHorizontally = true  →  s = 1.5/1.8 = 0.8333…, s_x = -1.
///
/// POSITION — a solver-frame point 0.20 m to the user's right, 0.90 m up,
/// 0.30 m toward the camera:  p = (0.20, 0.90, 0.30).
///   x' = s · (-1) · 0.20 = 0.8333 · -0.20 = -0.16667
///   y' = s ·        0.90 = 0.8333 ·  0.90 =  0.75000
///   z' = s · (-1) · 0.30 = 0.8333 · -0.30 = -0.25000
///   →  toVRChatPosition(p) ≈ (-0.16667, 0.75000, -0.25000)
///   Sanity: the right-side point (+x) mirrors to the avatar's left (−x'), the
///   toward-camera point (+z) becomes behind/forward-flipped (−z'), height (y)
///   only rescales — exactly the expected left-handed, mirrored, metric result.
///
/// ROTATION — a +90° yaw about the world up axis Y in the solver frame:
///   q = (w, x, y, z) = (cos45°, 0, sin45°, 0) = (0.70711, 0, 0.70711, 0).
///   mirror case ⇒ q' = (w, -x, y, -z) = (0.70711, 0, 0.70711, 0)  (unchanged,
///   since x = z = 0): a yaw flips sign under an *odd* reflection only via the
///   Z-flip+X-flip cancelling on the Y axis — here it stays a +90° Y rotation
///   as a quaternion, but the surrounding handedness flip means that same
///   physical motion now reads correctly in VRChat's left-handed yaw.
///   Decomposed ZXY: x≈0°, y≈+90°, z≈0°  →  toVRChatEulerDegrees(q) ≈ (0, 90, 0).
///
///   A pure +30° roll about Z, no mirror (s_x = +1):
///   q = (cos15°, 0, 0, sin15°) = (0.96593, 0, 0, 0.25882).
///   no-mirror case ⇒ q' = (w, -x, -y, z) = (0.96593, 0, 0, 0.25882) (unchanged,
///   x=y=0). ZXY decomposition → (0, 0, +30) degrees, i.e. roll is preserved.
/// ─────────────────────────────────────────────────────────────────────────
public struct CoordinateMapper: Sendable, Equatable {

    /// The user's real-world standing height in meters (used as the numerator
    /// of the metric rescale; corrects Vision's 1.8 m reference assumption).
    public var userHeightMeters: Float

    /// The reference height Vision assumes without depth (1.8 m). Denominator
    /// of the metric rescale. Guarded against zero / non-finite values.
    public var referenceHeightMeters: Float

    /// Whether to mirror horizontally (negate X) for the front-facing webcam,
    /// so the avatar moves on the same side as the user.
    public var mirrorHorizontally: Bool

    public init(userHeightMeters: Float,
                referenceHeightMeters: Float = 1.8,
                mirrorHorizontally: Bool = false) {
        self.userHeightMeters = userHeightMeters
        self.referenceHeightMeters = referenceHeightMeters
        self.mirrorHorizontally = mirrorHorizontally
    }

    /// Uniform metric scale (userHeight / referenceHeight), defended against a
    /// zero / negative / non-finite reference so we never emit NaN positions.
    @inlinable
    public var scale: Float {
        let ref = referenceHeightMeters
        guard ref.isFinite, ref > 1e-6, userHeightMeters.isFinite else {
            return 1
        }
        return userHeightMeters / ref
    }

    /// Sign applied to X for the horizontal mirror (−1 mirrored, +1 otherwise).
    @inlinable
    public var mirrorSignX: Float { mirrorHorizontally ? -1 : 1 }

    /// Convert a solver-frame position (right-handed, hip-root-relative meters)
    /// into VRChat / Unity world-space meters: mirror X for the front camera,
    /// negate Z for right→left handedness, and rescale by userHeight/reference.
    @inlinable
    public func toVRChatPosition(_ p: SIMD3<Float>) -> SIMD3<Float> {
        let s = scale
        return SIMD3<Float>(
            s * mirrorSignX * p.x,   // mirror + scale
            s *               p.y,   // scale only
            s * -1          * p.z    // handedness flip + scale
        )
    }

    /// Convert a solver-frame orientation quaternion into the THREE VRChat
    /// `/rotation` Euler angles, in DEGREES, applied internally ZXY by VRChat.
    ///
    /// The quaternion is first re-expressed in the VRChat basis via the
    /// axis-flip conjugation R' = M·R·M (see the class doc), then decomposed by
    /// `quaternionToEulerZXYDegrees`.
    @inlinable
    public func toVRChatEulerDegrees(_ q: simd_quatf) -> SIMD3<Float> {
        // Defend against a denormalized / zero quaternion before reflecting it.
        let len = simd_length(q.vector)
        let qn: simd_quatf = (len > 1e-6) ? simd_quatf(vector: q.vector / len) : q

        // Axis signs of the diagonal reflection M = diag(s_x, 1, -1).
        let sX = mirrorSignX          // ±1
        let sY: Float = 1
        let sZ: Float = -1

        // R' = M·R·M  ⇒  per-component scale by the product of the OTHER axes.
        let w =  qn.real
        let v =  qn.imag
        let reflected = simd_quatf(
            ix: (sY * sZ) * v.x,
            iy: (sX * sZ) * v.y,
            iz: (sX * sY) * v.z,
            r:  w
        )

        return quaternionToEulerZXYDegrees(reflected)
    }
}
