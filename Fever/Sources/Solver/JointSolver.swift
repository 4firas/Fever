import simd

/// Converts 33 stabilized BlazePose landmarks → 9 `VRJoint` values.
///
/// IMPORTANT — single coordinate frame: this solver works **entirely in the
/// solver frame** and emits joints in that SAME frame. The input landmark
/// positions are metric, hip-root-relative meters (the MediaPipe sidecar's 33
/// world landmarks, mapped into the solver frame by `MediaPipeFrame`: X kept,
/// Y negated down→up, Z scaled by `zSign`). This solver does NOT flip
/// handedness, does NOT flip Y, does NOT scale to real-world meters, and does
/// NOT convert to VRChat space. `CoordinateMapper` performs the single
/// authoritative VRChat conversion exactly once downstream (Z-flip, user-height
/// scale, horizontal mirror, quaternion → Unity ZXY euler degrees).
///
/// Each joint's `rotation` is a world-space (solver-frame) quaternion derived
/// from an orthonormal frame of landmark triplets, with degenerate/collinear
/// fallbacks (cross-product ≈ 0 on straight or occluded limbs) so feet/limbs
/// never spin or produce NaN quaternions.
///
/// The `jointSize` tweak is a uniform body-scale multiplier applied
/// consistently to every position in the solver frame; because it is uniform
/// and pre-conversion, the downstream mapper still maps once and stays correct.
public struct JointSolver {

    public let settings: TrackingConfig

    /// Per-joint hold-last store for the two-axis OSC rotation frames. A degenerate
    /// frame (the two in-body axes parallel) returns the last good orientation
    /// instead of fabricating a roll. Reference type so it survives across frames
    /// even though `JointSolver` is a value type rebuilt each run; injected by the
    /// FrameProcessor and confined to its single serial worker. `nil` in
    /// position-only / test contexts that never read rotation (then degenerate
    /// frames fall back to identity).
    public let rotationState: RotationState?

    /// Cross-frame per-foot EMA state for the step/stride exaggeration (reference
    /// type, owned by FrameProcessor, confined to the serial worker). `nil` in
    /// position-only / test contexts that don't exercise step exaggeration → the
    /// foot exaggeration is then a no-op (literal foot position).
    public let footMotionState: FootMotionState?

    public init(settings: TrackingConfig,
                rotationState: RotationState? = nil,
                footMotionState: FootMotionState? = nil) {
        self.settings = settings
        self.rotationState = rotationState
        self.footMotionState = footMotionState
    }

    /// Hold-last for `joint` from the injected `rotationState` (identity if none).
    @inline(__always)
    private func held(_ joint: JointType) -> simd_quatf {
        rotationState?.holdLast(joint) ?? simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
    }

    /// Record this frame's good live rotation for `joint` (no-op if no state).
    @inline(__always)
    private func record(_ joint: JointType, _ q: simd_quatf) {
        rotationState?.store(joint, q)
    }

    public func solve(_ pose: PoseResult) -> [VRJoint] {
        guard pose.landmarks.count == 33 else { return [] }

        let l = pose.landmarks
        let s = settings.jointSizeF
        var joints: [VRJoint] = []

        // --- Head ---
        let head = solveHead(l, scale: s)
        joints.append(head)

        // --- Chest ---
        let chest = solveChest(l, scale: s, head: head)
        joints.append(chest)

        // --- Hip ---
        // Stance center = floor-projected midpoint of the two ankles (the base of
        // support). The POSITION-space hip sway amplifies the hip's horizontal
        // deviation from THIS reference (not the head), per the weight-shift /
        // contrapposto model. Computed in the SAME unscaled landmark frame the
        // hip is, then scaled together by solverPosition downstream.
        let stanceCenterRaw = (l[.leftAnkle].position + l[.rightAnkle].position) * 0.5
        let stanceWidthRaw = simd_length(l[.rightAnkle].position - l[.leftAnkle].position)
        // Weight-shift sway is only meaningful when BOTH feet form the base of
        // support. When one foot lifts (raising a leg forward/sideways), the ankle
        // midpoint is no longer the stance center, and amplifying hip-vs-stance
        // shoves the hip the wrong way — the reported "raise a leg → hip pushed
        // back". Ramp the exaggeration off as a foot leaves the floor (ankle Y gap).
        let ankleYGap = abs(l[.leftAnkle].position.y - l[.rightAnkle].position.y)
        let stancePlanted = stancePlantedFactor(ankleYGap)
        let hip = solveHip(l, scale: s, chest: chest)
        joints.append(applyHipAdjustments(hip,
                                          chest: chest,
                                          stanceCenter: solverPosition(stanceCenterRaw, scale: s),
                                          stanceWidth: stanceWidthRaw * s,
                                          stancePlanted: stancePlanted))

        // --- Elbows ---
        joints.append(solveElbow(.leftElbow,
                                 shoulder: l[.leftShoulder],
                                 elbow:    l[.leftElbow],
                                 wrist:    l[.leftWrist],
                                 scale: s))
        joints.append(solveElbow(.rightElbow,
                                 shoulder: l[.rightShoulder],
                                 elbow:    l[.rightElbow],
                                 wrist:    l[.rightWrist],
                                 scale: s))

        // --- Knees ---
        joints.append(applyKneeAdjustments(
            solveKnee(.leftKnee,
                      hip: l[.leftHip], knee: l[.leftKnee], ankle: l[.leftAnkle], scale: s)))
        joints.append(applyKneeAdjustments(
            solveKnee(.rightKnee,
                      hip: l[.rightHip], knee: l[.rightKnee], ankle: l[.rightAnkle], scale: s)))

        // --- Feet --- (solve, then step/stride exaggeration via FootMotionState)
        joints.append(applyFootAdjustments(
            solveFoot(.leftFoot, ankle: l[.leftAnkle], knee: l[.leftKnee],
                      heel: l[.leftHeel], toe: l[.leftFootIndex], scale: s),
            type: .leftFoot, rawAnkle: l[.leftAnkle].position, scale: s))
        joints.append(applyFootAdjustments(
            solveFoot(.rightFoot, ankle: l[.rightAnkle], knee: l[.rightKnee],
                      heel: l[.rightHeel], toe: l[.rightFootIndex], scale: s),
            type: .rightFoot, rawAnkle: l[.rightAnkle].position, scale: s))

        return joints
    }

    // MARK: Position helper

    /// Uniform body-scale applied in the solver frame. Positions are already in
    /// Vision-derived meters; this only applies the user's `jointSize` tweak and
    /// leaves the frame (handedness, axes, units-up-to-scale) untouched so
    /// `CoordinateMapper` can do the VRChat conversion exactly once.
    @inline(__always)
    private func solverPosition(_ p: SIMD3<Float>, scale: Float) -> SIMD3<Float> {
        p * scale
    }

    // MARK: Head

    /// Neck segment (shoulder line → base of skull = head-bone ROOT) as a
    /// fraction of the 1.8 m reference stature the solver frame is built in.
    /// VRChat aligns the head BONE ROOT, not the crown/eye line, so placing the
    /// head reference here (NOT at the ear midpoint) removes the one-segment-too-
    /// high error that dragged the whole constellation down when head was sent.
    private static let referenceNeckFraction: Float = 0.052

    private func solveHead(_ l: [NormalizedLandmark], scale: Float) -> VRJoint {
        let earMid = (l[.leftEar].position + l[.rightEar].position) * 0.5
        // POSITION: the head-bone ROOT, derived from the SAME fixed
        // anthropometric skeleton — shoulder midpoint raised by a fixed neck
        // segment along the spine (shoulder→ear) direction. This keeps the head
        // reference height correct (no one-segment shift) so a single head snap
        // pulse aligns the OSC space without dragging the body down.
        let shMid = (l[.leftShoulder].position + l[.rightShoulder].position) * 0.5
        let neckLen = Self.referenceNeckFraction * 1.8
        let upRawHead = earMid - shMid
        let headRoot: SIMD3<Float>
        if simd_length(upRawHead) > 1e-5 {
            headRoot = shMid + simd_normalize(upRawHead) * neckLen
        } else {
            headRoot = shMid + SIMD3<Float>(0, neckLen, 0)
        }
        let pos = solverPosition(headRoot, scale: scale)
        // Face frame, all in the solver frame: forward = ear-midpoint → nose,
        // right = leftEar → rightEar, up = right × forward.
        let nose = l[.nose].position
        let forwardRaw = nose - earMid
        let rightRaw = l[.rightEar].position - l[.leftEar].position
        let rot: simd_quatf
        let upRaw = simd_cross(rightRaw, forwardRaw)
        if simd_length(forwardRaw) > 1e-5, simd_length(rightRaw) > 1e-5, simd_length(upRaw) > 1e-5 {
            let f = simd_normalize(forwardRaw)
            let r = simd_normalize(rightRaw)
            let u = simd_normalize(upRaw)
            rot = quaternionFromFrame(forward: f, right: r, up: u)
        } else {
            // Degenerate (collinear ears/nose or occluded): fall back to a bone
            // frame from the face-forward direction.
            rot = quaternionFromBone(direction: forwardRaw)
        }
        let conf = min(l[.leftEar].visibility, l[.rightEar].visibility, l[.nose].visibility)
        return VRJoint(type: .head, position: pos, rotation: rot, confidence: conf)
    }

    // MARK: Chest

    private func solveChest(_ l: [NormalizedLandmark], scale: Float, head: VRJoint) -> VRJoint {
        let shMid = (l[.leftShoulder].position + l[.rightShoulder].position) * 0.5
        let pos = solverPosition(shMid, scale: scale)
        // OSC ROTATION (two in-body axes, no world-up gauge): primary = chest→neck
        // (here the head-root above the shoulders ≈ neck), secondary = the shoulder
        // line (leftShoulder→rightShoulder). Degenerate frame holds last.
        let right = l[.rightShoulder].position - l[.leftShoulder].position
        let up = (head.position / max(scale, 1e-5)) - shMid   // chest → neck
        let rot = frameFromTwoAxes(primary: up, secondary: right, holdLast: held(.chest))
        record(.chest, rot)
        let conf = min(l[.leftShoulder].visibility, l[.rightShoulder].visibility)
        return VRJoint(type: .chest, position: pos, rotation: rot, confidence: conf)
    }

    // MARK: Hip

    private func solveHip(_ l: [NormalizedLandmark], scale: Float, chest: VRJoint) -> VRJoint {
        let hipMid = (l[.leftHip].position + l[.rightHip].position) * 0.5
        let pos = solverPosition(hipMid, scale: scale)
        // OSC ROTATION (two in-body axes, no world-up gauge): primary = hip→chest
        // (pelvis up the spine), secondary = the hip line (leftHip→rightHip).
        // Degenerate frame holds last (no fabricated roll → no hip-X pinning).
        let right = l[.rightHip].position - l[.leftHip].position
        let up = (chest.position / max(scale, 1e-5)) - hipMid   // hip → chest
        let rot = frameFromTwoAxes(primary: up, secondary: right, holdLast: held(.hip))
        record(.hip, rot)
        let conf = min(l[.leftHip].visibility, l[.rightHip].visibility)
        return VRJoint(type: .hip, position: pos, rotation: rot, confidence: conf)
    }

    // MARK: Elbow / Knee / Foot

    private func solveElbow(_ type: JointType,
                            shoulder: NormalizedLandmark,
                            elbow: NormalizedLandmark,
                            wrist: NormalizedLandmark,
                            scale: Float) -> VRJoint {
        let pos = solverPosition(elbow.position, scale: scale)
        // OSC ROTATION (two in-body axes): primary = elbow→wrist (forearm = local
        // +Y), secondary = elbow→shoulder (in-plane upper-arm hint). A straight arm
        // (forearm ∥ upper arm) is degenerate → holds last.
        let primary = wrist.position - elbow.position
        let secondary = shoulder.position - elbow.position
        let rot = frameFromTwoAxes(primary: primary, secondary: secondary, holdLast: held(type))
        record(type, rot)
        return VRJoint(type: type, position: pos, rotation: rot, confidence: elbow.visibility)
    }

    private func solveKnee(_ type: JointType,
                           hip: NormalizedLandmark,
                           knee: NormalizedLandmark,
                           ankle: NormalizedLandmark,
                           scale: Float) -> VRJoint {
        let pos = solverPosition(knee.position, scale: scale)
        // OSC ROTATION (two in-body axes): primary = knee→ankle (shank = local +Y),
        // secondary = knee→hip (in-plane thigh hint). A locked (straight) leg is
        // degenerate → holds last.
        let primary = ankle.position - knee.position
        let secondary = hip.position - knee.position
        let rot = frameFromTwoAxes(primary: primary, secondary: secondary, holdLast: held(type))
        record(type, rot)
        return VRJoint(type: type, position: pos, rotation: rot, confidence: knee.visibility)
    }

    private func solveFoot(_ type: JointType,
                           ankle: NormalizedLandmark,
                           knee: NormalizedLandmark,
                           heel: NormalizedLandmark,
                           toe: NormalizedLandmark,
                           scale: Float) -> VRJoint {
        // POSITION: VRChat FBT treats the "foot" tracker as an ANKLE tracker, so
        // when `footTrackersAtAnkle` is set (default) we place it at the ankle
        // landmark; otherwise we fall back to the synthesized toe (foot-index).
        let positionSource = settings.footTrackersAtAnkle ? ankle.position : toe.position
        let pos = solverPosition(positionSource, scale: scale)

        // OSC ROTATION — REAL FOOT FRAME from the DETECTED heel→toe vector.
        // MediaPipe actually detects the heel (29/30) and foot-index (31/32) — the
        // old Vision path SYNTHESIZED them as constants, which is why the foot
        // could never follow the real ankle and instead spun off the shank. The
        // foot's true pointing direction now drives orientation:
        //   forward = heel → toe   → real foot YAW (turn your foot) and PITCH
        //             (point your toes); it responds to ANKLE motion, not the leg.
        //   up      = world-up with the forward component removed (Gram-Schmidt),
        //             so pitch follows the toe while ROLL stays locked (a single
        //             camera has no reliable foot roll).
        // Falls back to the shank's ground projection if the foot landmarks
        // collapse/occlude; a fully degenerate frame holds last (never fabricates).
        let worldUp = SIMD3<Float>(0, 1, 0)
        var forward = toe.position - heel.position
        if simd_length(forward) < 1e-4 {
            let shank = ankle.position - knee.position
            forward = shank - worldUp * simd_dot(shank, worldUp)   // ground projection
        }
        var up = worldUp
        let fl = simd_length(forward)
        if fl > 1e-4 {
            let f = forward / fl
            let perp = worldUp - f * simd_dot(f, worldUp)
            if simd_length(perp) > 1e-4 { up = simd_normalize(perp) }
        }
        let rot = frameFromTwoAxes(primary: up, secondary: forward, holdLast: held(type))
        record(type, rot)
        let conf = settings.footTrackersAtAnkle ? ankle.visibility : toe.visibility
        return VRJoint(type: type, position: pos, rotation: rot, confidence: conf)
    }

    // MARK: Adjustments (all in the solver frame)

    /// Pre-gain deadband (meters, solver/real scale) on the horizontal hip
    /// deviation from stance center. Deviations below this are landmark/monocular
    /// noise near center; not amplifying them stops a constant micro-wobble.
    private static let hipSwayDeadband: Float = 0.015   // 1.5 cm

    /// Floor on the clamp of the AMPLIFIED horizontal deviation so a tracking
    /// spike can never teleport the hip far outside the feet, even with a narrow
    /// stance. The clamp is max(this, halfStanceWidth + margin).
    private static let hipSwayMinClamp: Float = 0.12    // 12 cm
    private static let hipSwayClampMargin: Float = 0.06 // +6 cm past half-stance

    /// Stance is a valid weight-shift base only with both feet planted. As one
    /// ankle lifts above the other (a leg raise / big step), ramp the hip-sway
    /// exaggeration off so leg motion never drags the hip. `ankleYGap` is in the
    /// solver frame (≈ real metres under the MediaPipe world frame).
    private static let stanceLiftFull: Float = 0.10   // gap ≤ this → fully planted
    private static let stanceLiftNone: Float = 0.28   // gap ≥ this → stance broken
    private func stancePlantedFactor(_ ankleYGap: Float) -> Float {
        if ankleYGap <= Self.stanceLiftFull { return 1 }
        if ankleYGap >= Self.stanceLiftNone { return 0 }
        return 1 - (ankleYGap - Self.stanceLiftFull) / (Self.stanceLiftNone - Self.stanceLiftFull)
    }

    private func applyHipAdjustments(_ hip: VRJoint,
                                     chest: VRJoint,
                                     stanceCenter: SIMD3<Float>,
                                     stanceWidth: Float,
                                     stancePlanted: Float) -> VRJoint {
        var j = hip

        // ── POSITION-SPACE HIP SWAY (PinoFBT-style dynamic exaggeration) ──────
        // Amplify the hip's HORIZONTAL deviation from stance center so real
        // weight-shifts read clearly in VR. True pelvic sway is only ~3-6 cm —
        // below the monocular noise/smoothing floor — so a gain > 1 is what makes
        // weight transfer legible. We operate on the SMOOTHED hip (landmarks are
        // already One-Euro filtered upstream), so this amplifies real motion, not
        // jitter. X (lateral) is the hero axis; Z (forward/back lean / hip-lead)
        // is a lower-gain accent; Y (vertical) is NEVER exaggerated (no bounce).
        // Reference frame: stance center = floor-projected ankle midpoint, in the
        // same solver frame as the hip — so this captures contrapposto, not the
        // whole-body translation the HMD head already carries.
        let gainX = settings.hipExaggerateCoefficientF   // lateral sway gain
        let gainZ = settings.hipTwistCoefficientF         // forward/back lean gain

        // Horizontal deviation from stance center (ignore Y: sway is horizontal).
        var dX = j.position.x - stanceCenter.x
        var dZ = j.position.z - stanceCenter.z

        // Pre-gain deadband: zero out sub-noise deviations near center.
        dX = applyDeadband(dX, Self.hipSwayDeadband)
        dZ = applyDeadband(dZ, Self.hipSwayDeadband)

        // Amplified deviation (extra displacement ADDED to the true hip), gated by
        // how planted the stance is — a lifted foot zeroes the exaggeration so a
        // leg raise never drags the hip.
        var exX = dX * (gainX - 1) * stancePlanted
        var exZ = dZ * (gainZ - 1) * stancePlanted

        // Clamp the amplified deviation so the hip never leaves the body, even on
        // a spike or narrow stance. Bound = half stance width + margin (with a
        // sane floor so a feet-together stance still allows a little sway).
        let bound = max(Self.hipSwayMinClamp, stanceWidth * 0.5 + Self.hipSwayClampMargin)
        exX = max(-bound, min(bound, exX))
        exZ = max(-bound, min(bound, exZ))

        if exX.isFinite { j.position.x += exX }
        if exZ.isFinite { j.position.z += exZ }
        // Y is intentionally left at the true value.

        // ── Length offset along chest→hip (spine) direction. Guard the
        // degenerate hip==chest case to avoid a NaN normalize. ────────────────
        let spineRaw = j.position - chest.position
        if simd_length(spineRaw) > 1e-5 {
            let spine = simd_normalize(spineRaw)
            j.position += spine * settings.hipLengthF
        }

        // ── NO fabricated rotation swing/twist here. The hip's OSC rotation now
        // comes solely from the clean two-axis frame in `solveHip`; the
        // `hipExaggerate`/`hipTwist` coefficients are POSITION gains only (applied
        // above). Bending the rotation by them would re-introduce a gauge offset
        // and break the rest-relative zero-centering downstream. ────────────────
        return j
    }

    /// Symmetric deadband: returns 0 inside [-band, band], else shifts toward 0
    /// by `band` so the response is continuous (no jump) at the threshold.
    @inline(__always)
    private func applyDeadband(_ v: Float, _ band: Float) -> Float {
        if v > band { return v - band }
        if v < -band { return v + band }
        return 0
    }

    private func applyKneeAdjustments(_ knee: VRJoint) -> VRJoint {
        var j = knee
        j.position.y += settings.kneePositionF
        return j
    }

    // MARK: Step / stride exaggeration

    private static let footStepDeadband: Float = 0.02   // m, pre-gain
    private static let footStrideClamp: Float = 0.22    // m, horizontal cap
    private static let footLiftClamp: Float = 0.08      // m, vertical (up-only) cap

    /// Amplify a SWINGING foot's displacement from its slow-EMA neutral so steps /
    /// walking / dynamic leg movement read bigger, WITHOUT dragging a PLANTED foot
    /// (swing≈0 → ~no change, stays glued to the floor) and WITHOUT pushing it
    /// through the floor (vertical lift is up-only). Position-only — never bends
    /// the foot rotation. No-op without a FootMotionState (test/position contexts)
    /// or with literal gains (1.0). Operates in the solver frame BEFORE the mapper,
    /// exactly like the hip sway, so the single downstream X-negate/Z-flip neither
    /// double-applies nor cancels it.
    private func applyFootAdjustments(_ foot: VRJoint, type: JointType,
                                      rawAnkle: SIMD3<Float>, scale: Float) -> VRJoint {
        guard let fms = footMotionState else { return foot }
        let (neutralRaw, swing) = fms.update(type, rawAnkle: rawAnkle)
        let neutral = solverPosition(neutralRaw, scale: scale)   // same frame as foot.position
        var j = foot

        let strideGain = settings.stepStrideCoefficientF
        let liftGain = settings.stepLiftCoefficientF

        // Horizontal stride (X lateral, Z fore/aft): deadband, then (gain-1)·swing.
        let dX = applyDeadband(j.position.x - neutral.x, Self.footStepDeadband)
        let dZ = applyDeadband(j.position.z - neutral.z, Self.footStepDeadband)
        var exX = dX * (strideGain - 1) * swing
        var exZ = dZ * (strideGain - 1) * swing
        exX = max(-Self.footStrideClamp, min(Self.footStrideClamp, exX))
        exZ = max(-Self.footStrideClamp, min(Self.footStrideClamp, exZ))

        // Vertical lift — UP ONLY (never push a planted foot down through the floor).
        let dY = max(0, j.position.y - neutral.y)
        let exY = min(dY * (liftGain - 1), Self.footLiftClamp) * swing

        // Smooth the delta so gain / swing transitions ease in (no foot pop).
        let ex = fms.smoothExaggeration(type, SIMD3<Float>(exX, exY, exZ))
        if ex.x.isFinite { j.position.x += ex.x }
        if ex.y.isFinite { j.position.y += ex.y }
        if ex.z.isFinite { j.position.z += ex.z }
        return j
    }
}
