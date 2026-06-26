import simd

// MARK: - SIMD quaternion / matrix helpers used across the pipeline

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
///                    do NOT fabricate a roll from a world-up reference (which would
///                    pin hip roll and snap 90° when a limb stood vertical). Holding
///                    the last good orientation is bounded and continuous.
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

// MARK: - PinoFBT fast_kinematics (exact, reverse-engineered)

/// `get_rotation(u, v)` from PinoFBT's `fast_kinematics` — the shortest-arc
/// quaternion rotating `u` onto `v`. xyz = cross(u,v), w = dot(u,v)+1, normalize.
/// **NO `d>0.999999` early-out** (it drops the tiny residual that propagates
/// through the IK composition — that omission was the old spine's ~10° error).
/// True-antiparallel (|q|<1e-12) → 180° about a perpendicular axis. Byte-exact
/// vs the compiled Numba core. Delegates to `PinoKinematics.getRotation`.
public func shortestArc(_ u: SIMD3<Float>, _ v: SIMD3<Float>) -> simd_quatf {
    let lu = simd_length(u), lv = simd_length(v)
    guard lu > 1e-6, lv > 1e-6 else { return simd_quatf(ix: 0, iy: 0, iz: 0, r: 1) }
    let q = PinoKinematics.getRotation(u, v)   // scalar-last (x,y,z,w)
    return simd_quatf(ix: q.x, iy: q.y, iz: q.z, r: q.w)
}

/// Convert a scalar-last `SIMD4<Float>` quaternion to `simd_quatf`.
@inline(__always)
public func quatfFromScalarLast(_ q: SIMD4<Float>) -> simd_quatf {
    simd_quatf(ix: q.x, iy: q.y, iz: q.z, r: q.w)
}

/// PinoFBT `calc_root_rotation` — pelvis/hip orientation, the BYTE-EXACT dual-prim
/// (spine1 AND spine2) 3-quaternion composition. Uses SMPL joints leftHip(1),
/// rightHip(2), spine1(3), spine2(6) on the PREPROCESSED joints. `a1` = forward
/// ref (0,0,1), `a2` = up ref (0,1,0). Delegates to `PinoKinematics`.
public func calcRootRotation(_ w: [SIMD3<Float>],
                             a1: SIMD3<Float> = SIMD3<Float>(0, 0, 1),
                             a2: SIMD3<Float> = SIMD3<Float>(0, 1, 0)) -> simd_quatf {
    quatfFromScalarLast(PinoKinematics.calcRootRotation(w, a1: a1, a2: a2))
}

/// PinoFBT `calc_chest_rotation` — chest orientation, the BYTE-EXACT frame-based
/// 3-quaternion composition. Uses SMPL joints leftCollar(13), rightCollar(14),
/// spine3(9) on the PREPROCESSED joints. `a1` = forward (0,0,1); `a2` = X-axis
/// (1,0,0). Returns only the quat; use `PinoKinematics.calcChestRotation` for the
/// residual (the arm solver's `in0`). Delegates to `PinoKinematics`.
public func calcChestRotation(_ w: [SIMD3<Float>],
                              a1: SIMD3<Float> = SIMD3<Float>(0, 0, 1),
                              a2: SIMD3<Float> = SIMD3<Float>(1, 0, 0)) -> simd_quatf {
    quatfFromScalarLast(PinoKinematics.calcChestRotation(w, a1: a1, a2: a2).quat)
}
