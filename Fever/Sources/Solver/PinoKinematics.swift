import simd

/// PinoFBT 2.0 `fast_kinematics` — the BYTE-EXACT 1:1 port of the compiled
/// `fast_kinematics.pyd` solver (preprocess + spine + limb IK), reverse-engineered
/// from the Frida live capture and validated against 4229 real desktop frames.
///
/// Quaternions are SCALAR-LAST `(x, y, z, w)` everywhere — represented as
/// `SIMD4<Float>` (NOT `simd_quatf`, whose initializer reorders to (ix,iy,iz,r))
/// so the wire/quat math is unambiguous. Joints are SMPL-24, in the model camera
/// frame (+Y down, meters) on input; `preprocessJoints` produces the normalized,
/// pelvis-centred IK-space joints every downstream solver consumes.
///
/// Validation (see `PinoKinematicsTests`, fixtures captured off the real binary):
///   • preprocess upper body  : 8.4e-8   (byte-exact, float32 floor)
///   • preprocess legs        : 5.9e-3   (documented float-chain accumulation)
///   • calc_root / calc_chest : ~2e-7    (byte-exact)
///   • chest residual         : 5.4e-9   (byte-exact; == arm in0)
///   • knee                   : 1.5e-5   (byte-exact)
///   • ankle                  : median 1.5e-3, max ~1.5e-2 (NEAR — foot-pitch)
///   • arm                    : 7.7e-7   (byte-exact away from the wrist pole)
public enum PinoKinematics {

    // MARK: - Constants (live-confirmed)

    /// Fixed canonical thigh length in preprocess IK space (`THIGH`, std 1.9e-8).
    public static let thighLength: Float = 0.4
    /// Fixed canonical shin length (`SHIN`, std 2.8e-8).
    public static let shinLength: Float = 0.5
    /// Torso-normalization scale numerator (`TORSO`): head sits at 0.56 from pelvis.
    public static let torsoNorm: Float = 0.56
    /// Leg joint indices rebuilt to fixed proportions (`[4,5,7,8,10,11]`).
    public static let legIndices: [Int] = [4, 5, 7, 8, 10, 11]

    /// Constant +3°-about-X camera-tilt correction matrix passed every tick
    /// (rows): the live `rotation_matrix`. Leg bone DIRECTIONS pass through its
    /// TRANSPOSE (Mᵀ, i.e. −3°); the torso/arms/head do NOT get it at all.
    public static let cameraTilt = simd_float3x3(rows: [
        SIMD3<Float>(1, 0,          0),
        SIMD3<Float>(0, 0.99862951, -0.05233596),
        SIMD3<Float>(0, 0.05233596,  0.99862951),
    ])

    // MARK: - Low-level vector / quaternion primitives (scalar-last)

    @inline(__always) static func zeroY(_ v: SIMD3<Float>) -> SIMD3<Float> {
        SIMD3<Float>(v.x, 0, v.z)
    }

    /// `get_rotation(u, v)` — shortest-arc quaternion rotating `u` onto `v`.
    /// xyz = cross(u,v), w = dot(u,v)+1, normalize. **NO `d>0.999999` early-out**
    /// (the early-out drops the tiny residual that propagates through the
    /// composition — that omission was the ~10° error in the old Fever spine).
    /// True-antiparallel (|q|<1e-12) → 180° about a perpendicular axis.
    @inline(__always)
    public static func getRotation(_ u0: SIMD3<Float>, _ v0: SIMD3<Float>) -> SIMD4<Float> {
        let u = normalizeSafe(u0), v = normalizeSafe(v0)
        let c = cross(u, v)
        let d = dot(u, v)
        let q = SIMD4<Float>(c.x, c.y, c.z, d + 1)
        let l = length(q)
        if l < 1e-12 {
            var ax = cross(SIMD3<Float>(1, 0, 0), u)
            if length(ax) < 1e-9 { ax = cross(SIMD3<Float>(0, 1, 0), u) }
            ax = normalizeSafe(ax)
            return SIMD4<Float>(ax.x, ax.y, ax.z, 0)
        }
        return q / l
    }

    /// Standard scalar-last Hamilton product `a ⊗ b`.
    @inline(__always)
    public static func quatMul(_ a: SIMD4<Float>, _ b: SIMD4<Float>) -> SIMD4<Float> {
        SIMD4<Float>(
            a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
            a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
            a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
            a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z
        )
    }

    @inline(__always) static func qConj(_ q: SIMD4<Float>) -> SIMD4<Float> {
        SIMD4<Float>(-q.x, -q.y, -q.z, q.w)
    }

    /// Apply rotation `q` to `v` (forward). `out = v + 2w(a×v) + 2(a×(a×v))`.
    @inline(__always)
    public static func quatApply(_ q: SIMD4<Float>, _ v: SIMD3<Float>) -> SIMD3<Float> {
        let a = SIMD3<Float>(q.x, q.y, q.z)
        let t = 2 * cross(a, v)
        return v + q.w * t + cross(a, t)
    }

    /// `quat_apply` as inlined in the binary @0x180011570 applies the INVERSE
    /// rotation (it conjugates `q` first). All `frame()` uses go through this.
    @inline(__always)
    public static func quatApplyInv(_ q: SIMD4<Float>, _ v: SIMD3<Float>) -> SIMD3<Float> {
        quatApply(qConj(q), v)
    }

    @inline(__always) static func normalizeSafe(_ v: SIMD3<Float>) -> SIMD3<Float> {
        let l = length(v)
        return l > 1e-20 ? v / l : SIMD3<Float>(0, 0, 0)
    }

    /// The iterative two-axis alignment primitive shared by spine, knee, ankle, arm
    /// (NOT a `q2*q1` template). Returns `q1 ⊗ q2 ⊗ q3`.
    @inline(__always)
    public static func frame(_ a1: SIMD3<Float>, _ a2: SIMD3<Float>,
                             prim: SIMD3<Float>, sec: SIMD3<Float>) -> SIMD4<Float> {
        let q1 = getRotation(a1, zeroY(prim))
        let q2 = getRotation(a1, quatApplyInv(q1, prim))
        let s2 = quatApplyInv(q2, quatApplyInv(q1, sec))
        let q3 = getRotation(a2, s2)
        return quatMul(quatMul(q1, q2), q3)
    }

    /// Same as `frame` but also exposes the intermediate quats, so callers that
    /// need the per-stage residual (chest) can reuse the exact computation.
    @inline(__always)
    static func frameParts(_ a1: SIMD3<Float>, _ a2: SIMD3<Float>,
                           prim: SIMD3<Float>, sec: SIMD3<Float>)
        -> (q: SIMD4<Float>, q1: SIMD4<Float>, q2: SIMD4<Float>, q3: SIMD4<Float>, primLen: Float) {
        let q1 = getRotation(a1, zeroY(prim))
        let q2 = getRotation(a1, quatApplyInv(q1, prim))
        let s2 = quatApplyInv(q2, quatApplyInv(q1, sec))
        let q3 = getRotation(a2, s2)
        return (quatMul(quatMul(q1, q2), q3), q1, q2, q3, length(prim))
    }

    // MARK: - preprocess_joints (TRUE path, every tick)

    /// `preprocess_joints(joints, M, is_preprocess=true)` — center on pelvis, flip
    /// `diag(1,-1,-1)` (180° about X), uniform torso scale `s = 0.56/|J15-J0|` on all
    /// non-leg bones, rebuild legs to fixed thigh/shin lengths with directions
    /// through `Mᵀ·FY`, and reconstruct the head to 0.56 from pelvis.
    ///
    /// - Parameter joints: 24 SMPL joints in the model camera frame (after the
    ///   OneEuro filter), index 0 = pelvis.
    /// - Returns: 24 normalized IK-space joints (`O`); `O[0] == (0,0,0)`.
    public static func preprocessJoints(_ joints: [SIMD3<Float>],
                                        rotationMatrix M: simd_float3x3 = cameraTilt) -> [SIMD3<Float>] {
        precondition(joints.count == 24, "preprocessJoints expects 24 SMPL joints")
        let j0 = joints[0]
        let headLen = length(joints[15] - j0)
        let s: Float = headLen > 1e-9 ? torsoNorm / headLen : 1
        let Mt = M.transpose

        // FY = diag(1,-1,-1); upper-body / arm / head are s·FY(J-J0).
        func fy(_ v: SIMD3<Float>) -> SIMD3<Float> { SIMD3<Float>(v.x, -v.y, -v.z) }
        var O = [SIMD3<Float>](repeating: .zero, count: 24)
        for j in 0..<24 { O[j] = s * fy(joints[j] - j0) }

        // Legs: rebuild to fixed thigh/shin lengths; directions through Mᵀ·FY of
        // the ORIGINAL bone vectors. Foot keeps its scaled-base direction.
        for (hip, knee, ankle, foot) in [(1, 4, 7, 10), (2, 5, 8, 11)] {
            let thighDir = normalizeSafe(Mt * fy(joints[knee]  - joints[hip]))
            O[knee]  = O[hip]  + thighLength * thighDir
            let shinDir  = normalizeSafe(Mt * fy(joints[ankle] - joints[knee]))
            O[ankle] = O[knee] + shinLength  * shinDir
            O[foot]  = O[ankle] + s * (Mt * fy(joints[foot] - joints[ankle]))
        }

        // Head (j15): dedicated reconstruction to 0.56 from pelvis. The exact
        // binary source is an OPEN item (PORT_SPEC §10 #4) — best-effort is the
        // 0.56-normalized direction of the s·FY head vector. Only feeds the head
        // /position anchor, which VRChat re-origins to.
        O[15] = torsoNorm * normalizeSafe(O[15])
        O[0] = .zero
        return O
    }

    // MARK: - Spine IK

    /// `calc_root_rotation` — pelvis/hip orientation. DUAL-PRIM (spine1 AND spine2).
    /// Joints: Lhip=1, Rhip=2, spine1=3, spine2=6. Consumes preprocess output.
    public static func calcRootRotation(_ O: [SIMD3<Float>],
                                        a1: SIMD3<Float> = SIMD3<Float>(0, 0, 1),
                                        a2: SIMD3<Float> = SIMD3<Float>(0, 1, 0)) -> SIMD4<Float> {
        let lh = O[1], rh = O[2], s1 = O[3], s2 = O[6]
        let hipmid = (lh + rh) * 0.5
        let hl = rh - lh                       // right - left hip
        let up1 = s1 - hipmid
        let up2 = s2 - hipmid
        let prim1 = cross(up1, hl)             // spine1 pass
        let prim2 = cross(up2, hl)             // spine2 pass
        // take ONLY q3 from the spine1 pass, q1*q2 from the spine2 pass.
        let p1 = frameParts(a1, a2, prim: prim1, sec: up1)
        let p2 = frameParts(a1, a2, prim: prim2, sec: up2)
        return quatMul(quatMul(p2.q1, p2.q2), p1.q3)
    }

    /// `calc_chest_rotation` — chest orientation AND the chest residual vec3.
    /// Joints: Lcollar=13, Rcollar=14, spine3=9. `a2` is the X-AXIS (1,0,0).
    /// Residual = `qapp(chestQuat, a1 · |prim|)` (byte-exact 5.4e-9; == arm in0).
    public static func calcChestRotation(_ O: [SIMD3<Float>],
                                         a1: SIMD3<Float> = SIMD3<Float>(0, 0, 1),
                                         a2: SIMD3<Float> = SIMD3<Float>(1, 0, 0))
        -> (quat: SIMD4<Float>, residual: SIMD3<Float>) {
        let lc = O[13], rc = O[14], s3 = O[9]
        let collar = lc - rc
        let up = (lc + rc) * 0.5 - s3
        let prim = cross(collar, up)
        let p = frameParts(a1, a2, prim: prim, sec: collar)
        let residual = quatApply(p.q, a1 * p.primLen)
        return (p.q, residual)
    }

    // MARK: - Limb IK

    /// `calc_paired_knee_rotations` — one knee quat per side (same formula L & R).
    /// `prim = cross(cross(shin,foot), thigh)`, `sec = -thigh`.
    public static func kneeRotation(hip: SIMD3<Float>, knee: SIMD3<Float>,
                                    ankle: SIMD3<Float>, toe: SIMD3<Float>,
                                    a1: SIMD3<Float> = SIMD3<Float>(0, 0, 1),
                                    a2: SIMD3<Float> = SIMD3<Float>(0, 1, 0)) -> SIMD4<Float> {
        let thigh = knee - hip, shin = ankle - knee, foot = toe - ankle
        let prim = cross(cross(shin, foot), thigh)
        return frame(a1, a2, prim: prim, sec: -thigh)
    }

    /// `calc_paired_ankle_rotations` — shin alignment + foot pitch.
    /// `prim = cross(cross(shin,foot), shin)`, `sec = -shin`, then foot-pitch quat.
    public static func ankleRotation(knee: SIMD3<Float>, ankle: SIMD3<Float>,
                                     toe: SIMD3<Float>,
                                     a1: SIMD3<Float> = SIMD3<Float>(0, 0, 1),
                                     a2: SIMD3<Float> = SIMD3<Float>(0, 1, 0)) -> SIMD4<Float> {
        let shin = ankle - knee, foot = toe - ankle
        let prim = cross(cross(shin, foot), shin)
        let A = frame(a1, a2, prim: prim, sec: -shin)
        let R = getRotation(a1, quatApplyInv(A, foot))   // foot pitch in the shin frame
        return quatMul(A, R)
    }

    /// `calc_paired_arm_rotations`. Per arm `up = elbow-shoulder`, `fore = wrist-elbow`,
    /// `off = 0.10*normalize(chestResidual)`, `B = cross(-up+off, fore+off)`:
    ///   elbow: `frame(Z, +Y, cross(B,-up), -up)`   wrist: `frame(Z, -Y, cross(fore,B), fore)`
    /// Output `(lElbow, lWrist, rElbow, rWrist)` — the L lane is built from the
    /// RIGHT bone block (binary's internal lane swap).
    public static func armElbow(up: SIMD3<Float>, fore: SIMD3<Float>,
                                chestResidual: SIMD3<Float>) -> SIMD4<Float> {
        let off = 0.10 * normalizeSafe(chestResidual)
        let B = cross(-up + off, fore + off)
        return frame(SIMD3<Float>(0, 0, 1), SIMD3<Float>(0, 1, 0), prim: cross(B, -up), sec: -up)
    }

    public static func armWrist(up: SIMD3<Float>, fore: SIMD3<Float>,
                                chestResidual: SIMD3<Float>) -> SIMD4<Float> {
        let off = 0.10 * normalizeSafe(chestResidual)
        let B = cross(-up + off, fore + off)
        return frame(SIMD3<Float>(0, 0, 1), SIMD3<Float>(0, -1, 0), prim: cross(fore, B), sec: fore)
    }

    /// Paired arm solve. `r*`/`l*` are the RIGHT/LEFT shoulder/elbow/wrist joints
    /// (preprocess output). Returns `(lElbow, lWrist, rElbow, rWrist)`.
    public static func calcPairedArmRotations(chestResidual: SIMD3<Float>,
                                              rShoulder: SIMD3<Float>, lShoulder: SIMD3<Float>,
                                              rElbow: SIMD3<Float>, lElbow: SIMD3<Float>,
                                              rWrist: SIMD3<Float>, lWrist: SIMD3<Float>)
        -> (lElbow: SIMD4<Float>, lWrist: SIMD4<Float>, rElbow: SIMD4<Float>, rWrist: SIMD4<Float>) {
        let upR = rElbow - rShoulder, foreR = rWrist - rElbow
        let upL = lElbow - lShoulder, foreL = lWrist - lElbow
        return (armElbow(up: upR, fore: foreR, chestResidual: chestResidual),   // L lane ← RIGHT bones
                armWrist(up: upR, fore: foreR, chestResidual: chestResidual),
                armElbow(up: upL, fore: foreL, chestResidual: chestResidual),   // R lane ← LEFT bones
                armWrist(up: upL, fore: foreL, chestResidual: chestResidual))
    }

    // MARK: - Euler (wire rotation)

    /// Wire `/rotation` euler in DEGREES `(roll=X, pitch=Y, yaw=Z)` of a scalar-last
    /// quat, computed exactly as scipy `as_euler('zxy', degrees=True)[[1,2,0]]`.
    /// scipy LOWERCASE 'zxy' = **EXTRINSIC** z-x-y: `R = Y(c)·X(b)·Z(a)`, yielding
    /// `(rotZ=a, rotX=b, rotY=c)`, then reindex `[[1,2,0]]` → `(rotX, rotY, rotZ)`.
    /// (Intrinsic z-x-y would be `Z·X·Y` — that was a bug; it diverges up to 180°
    /// at large poses.) Verified to 4e-5° vs the live wire, 1.7e-13° vs scipy.
    public static func eulerZXY121Degrees(_ q: SIMD4<Float>) -> SIMD3<Float> {
        let n = length(q)
        let u: SIMD4<Float> = n > 1e-9 ? q / n : SIMD4<Float>(0, 0, 0, 1)
        let x = u.x, y = u.y, z = u.z, w = u.w
        // Active rotation matrix entries R[i][j] (column-vector convention).
        let r00 = 1 - 2 * (y * y + z * z)
        let r02 = 2 * (x * z + y * w)
        let r10 = 2 * (x * y + z * w)
        let r11 = 1 - 2 * (x * x + z * z)
        let r12 = 2 * (y * z - x * w)
        let r20 = 2 * (x * z - y * w)
        let r22 = 1 - 2 * (x * x + y * y)
        // EXTRINSIC z-x-y decomposition: middle axis X from R[1][2] = -sin(b).
        let rotX = asin(simd_clamp(-r12, -1, 1))
        let rotZ: Float, rotY: Float
        if abs(r12) > 0.999999 {
            // Gimbal lock (rotX ≈ ±90°): rotZ folds into rotY.
            rotZ = 0
            rotY = atan2(-r20, r00)
        } else {
            rotZ = atan2(r10, r11)
            rotY = atan2(r02, r22)
        }
        let k = Float(180.0 / Double.pi)
        // reindex zxy (rotZ, rotX, rotY) -> (rotX, rotY, rotZ)
        return SIMD3<Float>(rotX * k, rotY * k, rotZ * k)
    }
}
