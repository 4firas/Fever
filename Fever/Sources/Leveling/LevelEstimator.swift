import simd

/// IMU-free gravity/up estimation — the core of the leveling ("Body Stabilizer")
/// system that makes tracking correct regardless of camera pitch/roll, the way
/// PinoQuest does it on a Mac webcam (no IMU available).
///
/// The user is assumed to stand straight, so the **spine** (hip-midpoint →
/// shoulder-midpoint) IS gravity-up. The leveling rotation is the one that brings
/// the spine onto +Y using ONLY camera **pitch** (about +X) and optional **roll**
/// (about +Z). Yaw is *never* applied: a vertical spine is yaw-invariant, so a
/// user who is merely turned yields identity and their body yaw flows untouched
/// through to the hip frame, the green reference box, and the OSC wire. That is
/// exactly the dissociation seen in the PinoQuest teardown (the box rotates with
/// the body while the camera stays level).
///
/// All inputs are in the SOLVER FRAME (+X right, +Y up, +Z toward the camera),
/// read AFTER `MediaPipeFrame`'s Y-negate and BEFORE the origin/floor latch (a
/// pure difference like the spine is independent of that later translation).
/// Pure value type, no state — the per-session datum and smoothing live in
/// `BodyStabilizer`.
public enum LevelEstimator {

    /// A spine shorter than this (meters, in the ~1.8 m reference stature frame)
    /// is too ill-conditioned to level from; callers get identity rather than a
    /// rotation amplified by noise.
    public static let minSpineLength: Float = 0.2

    /// The no-rotation quaternion (w = 1).
    public static let identity = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)

    /// The leveling rotation that brings the spine vector (`neck − midHip`) onto
    /// vertical (+Y).
    ///
    /// - pitch: rotate about +X to zero the spine's Z component (removes the
    ///   camera's up/down tilt — the primary IMU-free correction).
    /// - roll (only when `includeRoll`): rotate about +Z to zero the spine's X
    ///   component on the already-pitched spine (removes camera roll).
    ///
    /// The result is a PROPER rotation (det +1) composed solely of axis rotations,
    /// so it must never be folded into `CoordinateMapper`'s reflection. Returns
    /// identity for an ill-conditioned (too short) spine.
    public static func levelingQuaternion(neck: SIMD3<Float>,
                                          midHip: SIMD3<Float>,
                                          includeRoll: Bool) -> simd_quatf {
        let spine = neck - midHip
        let len = simd_length(spine)
        guard len > minSpineLength else { return identity }
        let s = spine / len

        // Pitch about +X so the spine lies in the X-Y plane (z' = 0).
        // Rx(a): z' = y·sin a + z·cos a = 0  ⇒  a = atan2(-z, y).
        let pitch = atan2(-s.z, s.y)
        let qPitch = simd_quatf(angle: pitch, axis: SIMD3<Float>(1, 0, 0))
        guard includeRoll else { return simd_normalize(qPitch) }

        // Roll about +Z so the already-pitched spine is vertical in X too (x' = 0).
        // Rz(b): x' = x·cos b − y·sin b = 0  ⇒  b = atan2(x, y), on the pitched spine.
        let s1 = qPitch.act(s)
        let roll = atan2(s1.x, s1.y)
        let qRoll = simd_quatf(angle: roll, axis: SIMD3<Float>(0, 0, 1))
        return simd_normalize(qRoll * qPitch)
    }

    /// Upright sanity gate: the leveled reference is only valid when the user is
    /// actually standing — head above the hips AND at least one foot below the
    /// hips. When false the caller treats the reference as LOST (the green box
    /// vanishes and the datum is held) rather than fabricating leveling from a
    /// crouch close to the lens — the box-vanish behavior seen in the teardown.
    public static func uprightSanity(nose: SIMD3<Float>,
                                     midHip: SIMD3<Float>,
                                     leftAnkle: SIMD3<Float>,
                                     rightAnkle: SIMD3<Float>) -> Bool {
        let headAboveHips = nose.y > midHip.y
        let aFootBelowHips = leftAnkle.y < midHip.y || rightAnkle.y < midHip.y
        return headAboveHips && aFootBelowHips
    }
}
