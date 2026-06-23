import simd

// MARK: - SIMD quaternion / matrix helpers used across the pipeline

/// Build a right-handed rotation quaternion from an orthonormal frame.
/// `forward` = bone direction (local +Y), `right` = local +X, `up` = local +Z.
@inlinable
public func quaternionFromFrame(forward: SIMD3<Float>,
                                right: SIMD3<Float>,
                                up: SIMD3<Float>) -> simd_quatf {
    // Columns: right, forward, up  → 3x3 rotation
    let m = simd_float3x3(columns: (right, forward, up))
    return simd_quatf(m)
}

/// Quaternion from a single bone direction, using `worldUp` as a reference
/// to derive a stable, **right-handed** orthonormal frame. Falls back to an
/// alternate reference axis when the bone is parallel to `worldUp` (which
/// would otherwise produce a reflection matrix → NaN from `simd_quatf`).
@inlinable
public func quaternionFromBone(direction: SIMD3<Float>,
                               worldUp: SIMD3<Float> = SIMD3<Float>(0, 1, 0)) -> simd_quatf {
    var forward = simd_normalize(direction)
    if !(forward.x.isFinite && forward.y.isFinite && forward.z.isFinite) {
        forward = SIMD3<Float>(0, 0, 1)
    }
    var upRef = worldUp
    if abs(simd_dot(forward, upRef)) > 0.999 {
        upRef = SIMD3<Float>(0, 0, 1)   // bone ∥ worldUp → swap reference
    }
    let right = simd_normalize(simd_cross(forward, upRef))   // forward × upRef
    let up    = simd_cross(right, forward)                    // right × forward
    // columns (right, forward, up) with right×forward = up → right-handed ✓
    return quaternionFromFrame(forward: forward, right: right, up: up)
}

/// Build a right-handed rotation quaternion from TWO in-body direction vectors
/// (the PinoFBT / ju1ce Mediapipe-VR-Fullbody-Tracking recipe), with NO world-up
/// reference and NO fabricated roll. `primary` becomes the frame's local +Y (the
/// bone / spine / shank direction); `secondary` is an in-body lateral hint that
/// only DISAMBIGUATES the remaining two axes (it need not be perpendicular).
///
/// Construction (orthonormal, right-handed columns x,y,z):
///   y = normalize(primary)
///   z = cross(normalize(secondary), y)
///   if |z| < 1e-4  → primary ∥ secondary (degenerate): return `holdLast` — we
///                    do NOT fabricate a roll from world-up (the old
///                    `quaternionFromBone` singularity that pinned hip roll and
///                    snapped 90° when a limb stood vertical). Holding the last
///                    good orientation is bounded and continuous.
///   z = normalize(z);  x = cross(y, z)
///   quat = simd_quatf(columns: (x, y, z))
///
/// Because both inputs are IN-BODY vectors, the resulting orientation has no
/// constant world-up gauge offset — at rest it is a fixed body pose, so the
/// rest-relative delta downstream is ≈ identity (bounded, zero-centered euler).
@inlinable
public func frameFromTwoAxes(primary: SIMD3<Float>,
                             secondary: SIMD3<Float>,
                             holdLast: simd_quatf) -> simd_quatf {
    let pl = simd_length(primary)
    let sl = simd_length(secondary)
    guard pl > 1e-5, sl > 1e-5 else { return holdLast }
    let y = primary / pl
    let zRaw = simd_cross(secondary / sl, y)
    let zl = simd_length(zRaw)
    guard zl > 1e-4 else { return holdLast }   // primary ∥ secondary — do NOT fabricate
    let z = zRaw / zl
    let x = simd_cross(y, z)                    // already unit (y⊥z, both unit)
    // Columns (x, y, z) form a right-handed orthonormal basis → proper rotation.
    let q = simd_quatf(simd_float3x3(columns: (x, y, z)))
    if !(q.vector.x.isFinite && q.vector.y.isFinite &&
         q.vector.z.isFinite && q.vector.w.isFinite) {
        return holdLast
    }
    return q
}

/// Build a rotation from a bone direction plus a reference perpendicular,
/// but fall back to `quaternionFromBone(direction:)` when the two vectors are
/// (near-)parallel and the cross product is degenerate. Prevents NaN quats on
/// straight limbs (collinear shoulder→elbow→wrist, etc.). Frame is built
/// right-handed so `simd_quatf` never gets a reflection matrix.
@inlinable
public func quaternionFromBoneSafe(direction: SIMD3<Float>,
                                   reference: SIMD3<Float>,
                                   worldUp: SIMD3<Float> = SIMD3<Float>(0, 1, 0)) -> simd_quatf {
    let boneDir = simd_normalize(direction)
    let refDir  = simd_normalize(reference)
    let normal  = simd_cross(boneDir, refDir)
    guard simd_length(normal) > 1e-4 else {
        return quaternionFromBone(direction: boneDir, worldUp: worldUp)
    }
    let up    = simd_normalize(normal)
    let right = simd_cross(boneDir, up)          // right-handed: right×forward = up
    return quaternionFromFrame(forward: boneDir, right: right, up: up)
}

/// Swing–twist decomposition of `q` about the unit axis `a`:
///   q == swing * twist,  where `twist` is the rotation purely about `a`
///   and `swing` carries the remaining (off-axis) rotation.
/// Returns the swing quaternion plus the SIGNED twist angle (radians) about `a`.
/// Reconstruct with `swing * simd_quatf(angle: newAngle, axis: a)` to replace the
/// twist while preserving the swing — the yaw stabilizer uses this to smooth ONLY
/// the body's yaw (twist about world-up) without touching pitch/roll.
@inlinable
public func swingTwist(_ q: simd_quatf, axis a: SIMD3<Float>) -> (swing: simd_quatf, twistAngle: Float) {
    let qn: simd_quatf = {
        let l = simd_length(q.vector)
        return l > 1e-6 ? simd_quatf(vector: q.vector / l) : simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
    }()
    let axis = simd_normalize(a)
    let proj = axis * simd_dot(qn.imag, axis)          // imag component along a
    var twist = simd_quatf(ix: proj.x, iy: proj.y, iz: proj.z, r: qn.real)
    let tl = simd_length(twist.vector)
    twist = tl > 1e-6 ? simd_quatf(vector: twist.vector / tl) : simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
    let swing = qn * twist.inverse
    // Signed angle about a: twist = (cos θ/2, sin θ/2 · a) ⇒ θ = 2·atan2(sin, cos).
    let s = simd_dot(twist.imag, axis)
    let angle = 2 * atan2(s, twist.real)
    return (swing, angle)
}

/// Shortest signed delta (radians) from angle `a` to `b`, wrapped to (−π, π], so
/// low-passing a yaw across the ±π seam never takes the long way around.
@inlinable
public func shortestAngleDelta(from a: Float, to b: Float) -> Float {
    var d = (b - a).truncatingRemainder(dividingBy: 2 * .pi)
    if d > .pi { d -= 2 * .pi }
    if d < -.pi { d += 2 * .pi }
    return d
}

/// Slerp wrapper that guards against NaN / zero-length quats.
@inlinable
public func safeSlerp(_ a: simd_quatf, _ b: simd_quatf, _ t: Float) -> simd_quatf {
    if simd_length(a.vector) < 1e-6 { return b }
    if simd_length(b.vector) < 1e-6 { return a }
    return simd_slerp(a, b, t)
}

/// Angle (radians) between two vectors, clamped for numerical safety.
@inlinable
public func angleBetween(_ a: SIMD3<Float>, _ b: SIMD3<Float>) -> Float {
    let d = simd_clamp(simd_dot(simd_normalize(a), simd_normalize(b)), -1, 1)
    return acos(d)
}

// MARK: - VRChat ZXY Euler conversion

/// Convert a (left-handed, VRChat-space) rotation quaternion into the THREE
/// Euler angles, **in degrees**, that VRChat's OSC tracker `/rotation` endpoint
/// expects. VRChat takes the three floats as euler angles and reconstructs a
/// quaternion by applying them internally in **Z, X, Y** order — i.e. the
/// composed rotation is `R = Ry(y) * Rx(x) * Rz(z)` (intrinsic ZXY: first roll
/// about Z, then pitch about X, then yaw about Y). This function is the exact
/// inverse of that composition, so feeding its output back through VRChat's
/// ZXY reconstruction yields the original quaternion (up to sign).
///
/// Derivation (Unity/VRChat convention, column-vector, left-handed +Y up):
///   Rx(x) = [[1,0,0],[0,cx,-sx],[0,sx,cx]]
///   Ry(y) = [[cy,0,sy],[0,1,0],[-sy,0,cy]]
///   Rz(z) = [[cz,-sz,0],[sz,cz,0],[0,0,1]]
///   R = Ry * Rx * Rz
/// Expanding R and reading back the closed-form angles (m[i][j] = row i, col j):
///   m[1][2] = -cos(x)·sin(x?)…  →  the (1,2) entry reduces to  -sin(x),
///     so  x = asin(-m[1][2]).
///   With x known and cos(x) ≠ 0:
///     z = atan2( m[1][0], m[1][1] )    (from the Z-roll terms in row 1)
///     y = atan2( m[0][2], m[2][2] )    (from the Y-yaw terms in column 2)
/// Includes a gimbal-lock guard (|sin x| ≈ 1, x ≈ ±90°) that pins z = 0 and
/// recovers y from the remaining well-conditioned entries.
@inlinable
public func quaternionToEulerZXYDegrees(_ q: simd_quatf) -> SIMD3<Float> {
    // Normalize defensively; a denormalized quaternion yields a non-orthonormal
    // matrix and garbage angles.
    let len = simd_length(q.vector)
    if len < 1e-6 {
        return SIMD3<Float>(0, 0, 0)
    }
    let quat = simd_quatf(vector: q.vector / len)

    // simd is column-major: m.columns.j is column j, so m.columns.j[i] == R[i][j]
    // (row i, column j of the rotation matrix).
    let m = simd_matrix3x3(quat)

    let m12 = m.columns.2[1]   // R[1][2] = -sin(x)
    let m02 = m.columns.2[0]   // R[0][2]
    let m22 = m.columns.2[2]   // R[2][2]
    let m10 = m.columns.0[1]   // R[1][0]
    let m11 = m.columns.1[1]   // R[1][1]

    let sinX = simd_clamp(-m12, -1, 1)
    let x = asin(sinX)

    let y: Float
    let z: Float
    if abs(sinX) > 0.99999 {
        // Gimbal lock: cos(x) ≈ 0 (x ≈ ±90°). Z and Y are degenerate → pin z=0
        // and recover y from the (2,0)/(0,0) entries.
        z = 0
        let m20 = m.columns.0[2]   // R[2][0]
        let m00 = m.columns.0[0]   // R[0][0]
        y = atan2(-m20, m00)
    } else {
        z = atan2(m10, m11)
        y = atan2(m02, m22)
    }

    let radToDeg = Float(180.0 / Double.pi)
    return SIMD3<Float>(x * radToDeg, y * radToDeg, z * radToDeg)
}

/// Build a quaternion from ZXY euler angles in DEGREES — the exact inverse of
/// `quaternionToEulerZXYDegrees` (Unity/VRChat convention R = Ry·Rx·Rz, where the
/// vector components are (pitch X, yaw Y, roll Z)). Used to recompose a rotation
/// after per-axis weighting/clamping (e.g. the spine bend).
public func quatFromEulerZXYDegrees(_ e: SIMD3<Float>) -> simd_quatf {
    let d = Float.pi / 180
    let qx = simd_quatf(angle: e.x * d, axis: SIMD3<Float>(1, 0, 0))
    let qy = simd_quatf(angle: e.y * d, axis: SIMD3<Float>(0, 1, 0))
    let qz = simd_quatf(angle: e.z * d, axis: SIMD3<Float>(0, 0, 1))
    return qy * qx * qz
}

/// Convert normalized BlazePose coords (x∈[0,1] y↓, z toward camera) to
/// VRChat world space (meters, y↑, z forward). RETAINED for the JointSolver's
/// internal frame; the authoritative VRChat-space conversion now lives in
/// `CoordinateMapper`.
@inlinable
public func normalizedToVRChat(_ p: SIMD3<Float>, scale: Float) -> SIMD3<Float> {
    SIMD3<Float>(p.x * scale, (1.0 - p.y) * scale, p.z * scale)
}
