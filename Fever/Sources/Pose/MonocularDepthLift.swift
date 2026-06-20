import simd
import Foundation

/// Anthropometric bone-length model + monocular 2D→3D depth lift.
///
/// This is the OSC-path fix for the two coordinate-model failures that made
/// VRChat FBT collapse while the 2D preview stayed perfect:
///
///   • FIXED ANTHROPOMETRIC SCALE — the old lift derived the image→meter factor
///     by summing the on-screen 2D pixel lengths of the shoulder width, hip
///     width and torso span and dividing the anthropometric lengths by that sum.
///     Those exact bones FORESHORTEN the moment the torso yaws/leans, so the sum
///     shrank, the factor grew, and the WHOLE constellation pulsed in size
///     ("spazz" / growing skeleton). The 0.05 EMA only slowed a systematic bias.
///     We now make scale mathematically INVARIANT to pose/rotation/distance by
///     replicating how BlazePose GHUM worldLandmarks stay fixed-scale: a metric
///     body PRIOR owns the size, the image owns only the configuration. Two
///     pieces do this:
///       (1) a single global image→meter factor seeded from the LEAST-
///           foreshortened quantity available — the MAX observed full-body
///           vertical extent (ankle→head Y span), which is the longest, least
///           foreshortened projection of an upright body — never instantaneous
///           widths, then heavily smoothed / latched (`stableScale`);
///       (2) `retarget()` — a final pass that walks each kinematic chain from
///           the hip and REPLACES every bone's length with its fixed Drillis &
///           Contini anthropometric length while preserving the (3D) direction
///           the image gave it (VNect-style bone-length retargeting). After this
///           pass every emitted bone length is constant every frame, so the
///           skeleton's overall scale cannot change regardless of how the body
///           rotates, leans, or moves toward/away from the camera; a residual
///           error in the global factor only nudges absolute placement, never
///           inter-bone proportions.
///
///   • PLANAR Z=0 — every joint used to be lifted onto a single Z plane, so a
///     knee swung toward the camera projected nearly on top of the ankle and the
///     thigh+shank became near-vertical; VRChat read knee≈foot and the legs
///     collapsed. We synthesize a real per-joint depth from BONE-LENGTH
///     FORESHORTENING (C. J. Taylor, CVIU 2000): a bone of true length `L`
///     projecting to 2D length `p` implies an out-of-plane component
///     `dz = sqrt(L² − p²)`, propagated along the kinematic chains from the hip
///     root. The depth SIGN (the monocular flip) is resolved with cheap
///     kinematic + temporal priors and hysteresis so it never flickers.
///
/// Everything here works in the SAME metric units the rest of the lift uses
/// (`referenceHeight` = 1.8 m units; `CoordinateMapper` does the final
/// true-height rescale downstream), so bone lengths are built against
/// `referenceHeight`, NOT the user's true height — building them against a
/// different height than the XY scale would double-count the rescale.
///
/// Stateful (latched scale + per-segment sign hysteresis), so one instance is
/// owned by the single, strictly-serial inference worker.
public final class MonocularDepthLift {

    /// Drillis & Contini (1966) segment lengths as a fraction of stature.
    /// Used both to size the skeleton (fixed scale / retargeting) and as the
    /// known 3D bone lengths `L` for the foreshortening depth solve.
    public enum Proportion {
        public static let head: Float       = 0.130   // crown→chin-ish head height
        public static let torso: Float      = 0.288   // shoulder → hip (acromion→hip)
        public static let upperArm: Float   = 0.186
        public static let foreArm: Float    = 0.146
        public static let thigh: Float      = 0.245   // hip → knee
        public static let shank: Float      = 0.246   // knee → ankle
        public static let foot: Float       = 0.152
        public static let shoulderWidth: Float = 0.259 // biacromial
        public static let hipWidth: Float   = 0.191   // bi-iliac
        /// Fraction of stature from the FLOOR (ankle) up to the EYE/EAR line —
        /// i.e. the full upright standing extent the image sees as ankle→head.
        /// Drillis & Contini place the ankle at ~0.039 H above the floor and the
        /// ear/eye line at ~0.936 H; ankle→ear ≈ 0.897 H. This is the least-
        /// foreshortened quantity used to seed the global image→meter factor.
        public static let ankleToEye: Float = 0.897
    }

    /// Metric bone lengths (in `referenceHeight` units) computed once. The
    /// `legScale` multiplier scales ONLY the leg bones (thigh/shank) so the
    /// user's slightly-longer-legged avatar is matched without disturbing the
    /// rest of the prior — the Drillis & Contini fractions are a population
    /// prior, not ground truth, so per-user leg tuning is expected.
    public struct BoneLengths {
        public let torso, upperArm, foreArm, thigh, shank, foot, shoulderWidth, hipWidth: Float
        /// Anthropometric ankle→eye standing extent (reference units) — the
        /// metric target the global image→meter factor is solved against.
        public let standingExtent: Float
        init(height H: Float, legScale: Float = 1) {
            torso = H * Proportion.torso
            upperArm = H * Proportion.upperArm
            foreArm = H * Proportion.foreArm
            thigh = H * Proportion.thigh * legScale
            shank = H * Proportion.shank * legScale
            foot = H * Proportion.foot
            shoulderWidth = H * Proportion.shoulderWidth
            hipWidth = H * Proportion.hipWidth
            standingExtent = H * Proportion.ankleToEye
        }
    }

    private let referenceHeight: Float
    private let bones: BoneLengths

    /// LATCHED global metric scale (image-units → reference meters). Seeded once
    /// from the maximum observed full-body vertical extent (the least-
    /// foreshortened projection of an upright body) so it is robust to rotation,
    /// then held with a very long time constant. nil until first seeded.
    ///
    /// NOTE: with `retarget()` enforcing fixed bone lengths downstream, this
    /// factor only affects ABSOLUTE placement magnitude, never inter-bone
    /// proportions — so even a residual error never makes the skeleton grow or
    /// shrink; it just nudges overall position, which the HMD anchor absorbs.
    private var smoothedScale: Float?
    /// Best (largest) observed standing extent in image-units, for the min-
    /// foreshortening estimator. Decays very slowly so a one-off over-long crop
    /// does not permanently bias the latch.
    private var maxExtent: Float = 0
    /// Tiny correction weight pulling the latched scale toward a NEW max extent
    /// (the only direction the min-foreshortening estimator trusts). Deliberately
    /// slow: scale is meant to be effectively frozen after seeding.
    private let scaleAlpha: Float = 0.05

    /// Per-segment last depth SIGN (+1 / −1), keyed by a small segment id, used
    /// for temporal continuity + flip hysteresis so depth never flickers.
    private var lastSign: [Int: Float] = [:]

    /// Per-segment COUNT of consecutive frames the kinematic bias has pointed
    /// AWAY from the latched sign while also clearing the (bone-scaled) deadband.
    /// A flip is only committed once this reaches `flipPersistFrames`, so a
    /// transient foreshortening spike during a body yaw (which momentarily inflates
    /// the bias and can transiently invert its sign) is absorbed instead of
    /// flipping the joint's depth and glitching the avatar. Reset to 0 whenever the
    /// evidence is weak or agrees with the latched sign. FIX 3.
    private var flipStreak: [Int: Int] = [:]

    /// How many CONSECUTIVE frames of strong opposite-sign evidence are required
    /// before the latched depth sign is allowed to flip. Sized for Vision's ~12 fps
    /// worker: ~4 frames ≈ a third of a second of sustained, deadband-clearing
    /// disagreement — long enough that a turn's transient projection spike never
    /// trips it, short enough that a genuine, held limb reversal still resolves.
    private static let flipPersistFrames = 4

    /// Fraction of a segment's anthropometric bone length the kinematic bias must
    /// EXCEED (in addition to disagreeing in sign for `flipPersistFrames`) before a
    /// flip is allowed. The deadband now SCALES with the bone (replacing the old
    /// fixed 0.3 m, which for a 1.74 m user's ~0.25 m thigh meant >100 %
    /// foreshortening and tripped on momentary projection changes during yaw).
    private static let flipDeadbandFraction: Float = 0.6

    /// LATCHED floor plane (reference-meter Y of the lowest foot in the stable
    /// camera/world frame), captured once and then FROZEN. Subtracting a FIXED
    /// floor reference (rather than the current-frame lowest foot) is what lets
    /// the hip translate vertically: re-zeroing the floor every frame would pin
    /// the body to the floor and destroy vertical hip motion (FIX 2). nil until
    /// the first trustworthy standing frame seeds it.
    private var floorRef: Float?

    /// LATCHED horizontal world origin (the hip's metric XZ in the stable camera
    /// frame at the FIRST valid frame), captured once and then FROZEN. The hip's
    /// absolute world translation sits a couple of metres off-origin in +X (the
    /// camera frame is not centred on the user). PinoFBT's trackers sit near X≈0;
    /// ours landed at ~+2.1 m. That offset is harmless on its own (VRChat re-origins
    /// to the head anchor) BUT it amplifies any single-joint dropout into a ~2 m
    /// teleport. Subtracting this FIXED origin from every joint — including the head
    /// — centres the frame near 0 while leaving (tracker − head) head-relative
    /// geometry byte-identical (the same constant is removed from both). Like the
    /// floor, it is latched once and re-seeded only on reset()/Recenter. nil until
    /// the first frame seeds it.
    private var originXZ: SIMD2<Float>?

    public init(referenceHeight: Float, legScale: Float = 1) {
        self.referenceHeight = referenceHeight
        self.bones = BoneLengths(height: referenceHeight, legScale: legScale)
    }

    public func reset() {
        smoothedScale = nil
        maxExtent = 0
        floorRef = nil
        originXZ = nil
        lastSign.removeAll(keepingCapacity: true)
        flipStreak.removeAll(keepingCapacity: true)
    }

    /// Seed the FIXED floor reference once (the lowest-foot Y of the first
    /// trustworthy standing frame, in the stable camera/world frame), then keep
    /// it frozen. Returns the latched value. Once latched, later calls are no-ops
    /// so the floor never re-zeros and the hip keeps its vertical translation.
    public func latchFloor(_ y: Float) -> Float {
        if let f = floorRef { return f }
        floorRef = y
        return y
    }

    /// The currently latched floor reference, or nil if not yet seeded.
    public var floorReference: Float? { floorRef }

    /// Seed the FIXED horizontal world origin once (the hip's metric XZ of the
    /// first frame), then keep it frozen. Returns the latched value. Subsequent
    /// calls are no-ops so the origin never re-zeros (which would pin the hip
    /// horizontally and kill lateral/forward translation), exactly mirroring
    /// `latchFloor`. Re-seeded only by `reset()` (Recenter).
    public func latchOriginXZ(_ xz: SIMD2<Float>) -> SIMD2<Float> {
        if let o = originXZ { return o }
        originXZ = xz
        return xz
    }

    /// The currently latched horizontal origin, or nil if not yet seeded.
    public var originReferenceXZ: SIMD2<Float>? { originXZ }

    // MARK: - Fixed global scale (FIX 1, part 1)

    /// Derive the single global metric scale (image-units → reference meters)
    /// and latch it. The estimator is deliberately NOT a sum of on-screen bone
    /// widths (those foreshorten the instant the body rotates/leans and were the
    /// entire cause of the growing/pulsing skeleton). Instead it uses the
    /// MIN-FORESHORTENING quantity: the MAXIMUM observed full-body vertical
    /// extent (ankle→head Y span). The longest projection an upright body ever
    /// produces is the one with the least foreshortening, so it is the most
    /// rotation-robust seed for the true anthropometric standing extent.
    ///
    /// The latch only ever follows the extent UPWARD (toward less foreshortening)
    /// and is otherwise effectively frozen, so the global factor does not swing
    /// when the user yaws/leans/steps toward or away from the camera. Combined
    /// with `retarget()` (which fixes every bone length), a residual error here
    /// can only nudge absolute placement, never the skeleton's size.
    ///
    /// `xy` are aspect-corrected normalized points (origin lower-left), `present`
    /// the per-landmark presence mask, both 33-slot.
    public func stableScale(xy: [SIMD2<Float>], present: [Bool]) -> Float? {
        @inline(__always) func y(_ l: BlazePose.Landmark) -> Float? {
            present[l.rawValue] ? xy[l.rawValue].y : nil
        }

        // Vertical extent from the highest head/face point down to the lowest
        // ankle — the full upright standing span, the least foreshortened body
        // measure available. Knees are a fallback bottom if ankles are cropped.
        let topY = [y(.nose), y(.leftEye), y(.rightEye), y(.leftEar), y(.rightEar)]
            .compactMap { $0 }.max()
        let botY = [y(.leftAnkle), y(.rightAnkle)]
            .compactMap { $0 }.min()
            ?? [y(.leftKnee), y(.rightKnee)].compactMap { $0 }.min()

        guard let t = topY, let b = botY, t > b else { return smoothedScale }
        let extent = t - b                       // image-units, ankle→head
        guard extent.isFinite, extent > 1e-4 else { return smoothedScale }

        // Track the largest extent ever seen (min-foreshortening), with a very
        // slow decay so a momentary over-tall crop cannot permanently inflate it.
        maxExtent = max(maxExtent * 0.9995, extent)

        // scale = anthropometric standing extent / observed standing extent.
        let measured = bones.standingExtent / maxExtent
        guard measured.isFinite, measured > 0 else { return smoothedScale }

        // Latch: seed once, then ease only gently toward the min-foreshortening
        // estimate (it can change only when a NEW max extent is observed, i.e.
        // the body straightened up), so the global factor stays effectively
        // frozen frame to frame.
        if let s = smoothedScale {
            smoothedScale = s + scaleAlpha * (measured - s)
        } else {
            smoothedScale = measured
        }
        return smoothedScale
    }

    // MARK: - Depth lift

    /// Foreshortening depth magnitude for a bone: `sqrt(L² − p²)` with `p`
    /// clamped to `L` (a noisy over-long limb gives 0, never NaN).
    @inline(__always)
    private static func dzMagnitude(_ parentXY: SIMD2<Float>,
                                    _ childXY: SIMD2<Float>,
                                    _ L: Float) -> Float {
        let p = simd_length(childXY - parentXY)
        let pc = min(p, L)
        return (L * L - pc * pc).squareRoot()
    }

    /// Resolve the depth sign for one segment combining: (a) a kinematic prior
    /// (joints bend toward the camera, so distal joints sit at +Z), and (b) the
    /// previous frame's sign for temporal continuity.
    ///
    /// Hysteresis (FIX 3 — stiffer, so a body yaw can't flip joint depth signs):
    /// once a sign is latched, it flips ONLY when the kinematic bias both (1)
    /// disagrees in sign AND (2) clears a deadband SCALED to this segment's
    /// anthropometric bone length (`flipDeadbandFraction · boneLength`, replacing
    /// the old fixed 0.3 m that was >100 % of a 0.25 m thigh and tripped on turns),
    /// for `flipPersistFrames` CONSECUTIVE frames. A transient foreshortening spike
    /// during a turn lasts a frame or two and so never accumulates enough evidence;
    /// a genuine, sustained limb reversal still resolves within a few frames. The
    /// bent-limb kinematic prior is unchanged — it still seeds the first sign and
    /// still supplies the disagreement signal; only the flip GATE is stricter.
    private func resolveSign(segment id: Int,
                             kinematicBias: Float,
                             boneLength: Float) -> Float {
        let biasSign: Float = kinematicBias >= 0 ? 1 : -1
        guard let prior = lastSign[id] else {
            // First observation for this segment: trust the kinematic prior.
            lastSign[id] = biasSign
            flipStreak[id] = 0
            return biasSign
        }

        let deadband = max(Self.flipDeadbandFraction * boneLength, 1e-4)
        let stronglyOpposite = (biasSign != prior) && (abs(kinematicBias) > deadband)

        if stronglyOpposite {
            // Accumulate evidence; commit the flip only after it has persisted.
            let streak = (flipStreak[id] ?? 0) + 1
            if streak >= Self.flipPersistFrames {
                lastSign[id] = biasSign
                flipStreak[id] = 0
                return biasSign
            }
            flipStreak[id] = streak
            return prior   // hold the latched sign while evidence builds
        }

        // Evidence weak or agrees with the latch → reset the streak, keep the sign.
        flipStreak[id] = 0
        return prior
    }

    /// Synthesize a real per-joint Z (in reference meters) for the 33-slot
    /// landmark array, leaving the accurate XY untouched. `metricXY[i]` is the
    /// already-scaled hip-root-relative XY in reference meters; `present[i]` the
    /// mask. Returns a Z value per slot (0 where not computed).
    ///
    /// Chains, all rooted at hip depth 0:
    ///   spine: hipMid → shoulderMid (→ head via face, kept shallow)
    ///   arms:  shoulder → elbow → wrist   (L/R)
    ///   legs:  hip → knee → ankle → foot  (L/R)
    ///
    /// Kinematic priors: knees/elbows bend the distal joint TOWARD the camera
    /// (+Z) in the solver frame; we bias accordingly and disambiguate with the
    /// temporal sign so a bent knee resolves forward of the hip→ankle line (the
    /// regression that un-collapses the legs).
    public func depths(metricXY: [SIMD2<Float>], present: [Bool]) -> [Float] {
        var z = [Float](repeating: 0, count: 33)

        @inline(__always) func xy(_ l: BlazePose.Landmark) -> SIMD2<Float>? {
            present[l.rawValue] ? metricXY[l.rawValue] : nil
        }
        @inline(__always) func set(_ l: BlazePose.Landmark, _ v: Float) {
            z[l.rawValue] = v
        }

        // Torso reference points (hip root is depth 0 by construction).
        let lHip = xy(.leftHip), rHip = xy(.rightHip)
        let hipMid: SIMD2<Float>
        if let l = lHip, let r = rHip { hipMid = (l + r) * 0.5 }
        else if let l = lHip { hipMid = l }
        else if let r = rHip { hipMid = r }
        else { hipMid = .zero }

        let lSh = xy(.leftShoulder), rSh = xy(.rightShoulder)
        let shMid: SIMD2<Float>? = {
            if let l = lSh, let r = rSh { return (l + r) * 0.5 }
            return lSh ?? rSh
        }()

        // SPINE: shoulders sit at ~hip depth (torso is upright/in-plane); keep
        // shallow and signed forward only slightly so chest doesn't fight legs.
        // (We deliberately do NOT push the chest far in Z — its foreshortening is
        // tiny when standing, and a big chest Z destabilizes the head anchor.)
        var shoulderZ: Float = 0
        if let sh = shMid {
            let mag = Self.dzMagnitude(hipMid, sh, bones.torso)
            shoulderZ = 0.25 * mag   // gentle: torso barely leaves the plane
            set(.leftShoulder, shoulderZ)
            set(.rightShoulder, shoulderZ)
        }

        // ARMS: shoulder(z) → elbow → wrist. Elbows/wrists come forward (+Z).
        func solveArm(shoulder: BlazePose.Landmark,
                      elbow: BlazePose.Landmark,
                      wrist: BlazePose.Landmark,
                      idBase: Int) {
            guard let s = xy(shoulder) else { return }
            let zS = z[shoulder.rawValue]
            if let e = xy(elbow) {
                let mag = Self.dzMagnitude(s, e, bones.upperArm)
                // Elbow tends forward of the shoulder line when the arm bends.
                let sign = resolveSign(segment: idBase, kinematicBias: mag,
                                       boneLength: bones.upperArm)
                let zE = zS + sign * mag
                set(elbow, zE)
                if let w = xy(wrist) {
                    let magW = Self.dzMagnitude(e, w, bones.foreArm)
                    let signW = resolveSign(segment: idBase + 1, kinematicBias: magW,
                                            boneLength: bones.foreArm)
                    set(wrist, zE + signW * magW)
                }
            }
        }
        solveArm(shoulder: .leftShoulder, elbow: .leftElbow, wrist: .leftWrist, idBase: 10)
        solveArm(shoulder: .rightShoulder, elbow: .rightElbow, wrist: .rightWrist, idBase: 20)

        // LEGS: hip(z=0) → knee → ankle → foot. Knee comes FORWARD (+Z); ankle
        // returns back toward/under the hip. This is the chain that un-collapses
        // the leg IK (knee distinctly forward of the hip→ankle line).
        func solveLeg(hip: BlazePose.Landmark,
                      knee: BlazePose.Landmark,
                      ankle: BlazePose.Landmark,
                      idBase: Int) {
            guard let h = xy(hip) else { return }
            let zH = z[hip.rawValue]   // legs root at the hip joints (≈0)
            if let k = xy(knee) {
                let mag = Self.dzMagnitude(h, k, bones.thigh)
                // Knee bends forward of the hip→ankle line → bias +Z.
                let sign = resolveSign(segment: idBase, kinematicBias: mag,
                                       boneLength: bones.thigh)
                let zK = zH + sign * mag
                set(knee, zK)
                if let a = xy(ankle) {
                    let magA = Self.dzMagnitude(k, a, bones.shank)
                    // Shank returns the ankle back under the body → bias −Z
                    // (opposite the knee) so the ankle is not also pushed forward.
                    let signA = resolveSign(segment: idBase + 1, kinematicBias: -magA,
                                            boneLength: bones.shank)
                    set(ankle, zK + signA * magA)
                }
            }
        }
        solveLeg(hip: .leftHip, knee: .leftKnee, ankle: .leftAnkle, idBase: 30)
        solveLeg(hip: .rightHip, knee: .rightKnee, ankle: .rightAnkle, idBase: 40)

        return z
    }

    // MARK: - Bone-length retargeting (FIX 1, part 2)

    /// FINAL geometric step before the skeleton leaves the lift: walk every
    /// kinematic chain from the hip root and REPLACE each bone's length with its
    /// fixed anthropometric length while PRESERVING the 3D direction the lift
    /// produced (VNect / Mehta et al. 2017). After this pass every emitted bone
    /// length is a session constant, so the skeleton's overall metric size is
    /// mathematically invariant to pose, rotation, distance and 2D noise — the
    /// 2D image now contributes ONLY direction. This is the deterministic
    /// analogue of how BlazePose GHUM worldLandmarks stay fixed-scale.
    ///
    /// `pos[i]` is the 3D (metric-XY + foreshortening-Z) hip-rooted position per
    /// 33-slot landmark; `present[i]` the mask. Returns the retargeted 3D array.
    /// The hip MID is the chain root and is left where it is (its world
    /// translation is applied downstream — see FIX 2), so retargeting never
    /// destroys the hip's own motion.
    public func retarget(_ pos: [SIMD3<Float>], present: [Bool]) -> [SIMD3<Float>] {
        var out = pos

        @inline(__always) func have(_ l: BlazePose.Landmark) -> Bool { present[l.rawValue] }
        @inline(__always) func P(_ l: BlazePose.Landmark) -> SIMD3<Float> { out[l.rawValue] }
        @inline(__always) func set(_ l: BlazePose.Landmark, _ v: SIMD3<Float>) {
            out[l.rawValue] = v
        }

        // Hip mid (chain root). Shoulder mid is reconstructed at a fixed torso
        // length above it; left/right shoulders keep the reconstructed mid's
        // depth and a fixed half shoulder-width offset along the measured lateral
        // direction so the upper body never changes size.
        let lHip = have(.leftHip), rHip = have(.rightHip)
        let hipMid: SIMD3<Float>
        if lHip && rHip { hipMid = (P(.leftHip) + P(.rightHip)) * 0.5 }
        else if lHip { hipMid = P(.leftHip) }
        else if rHip { hipMid = P(.rightHip) }
        else { return out }   // no torso root → nothing to retarget

        /// Re-place `child` at a fixed distance `L` from `parent` along the
        /// child−parent direction; degenerate (coincident) → straight up.
        @inline(__always)
        func place(_ parent: SIMD3<Float>, _ child: BlazePose.Landmark, _ L: Float) {
            guard have(child) else { return }
            let dir = out[child.rawValue] - parent
            let len = simd_length(dir)
            let unit = len > 1e-6 ? dir / len : SIMD3<Float>(0, 1, 0)
            set(child, parent + unit * L)
        }

        // SPINE: hipMid → shoulderMid (fixed torso). Then re-hang the two
        // shoulders off the reconstructed mid at fixed half-width, preserving
        // their measured lateral direction.
        if have(.leftShoulder) || have(.rightShoulder) {
            let shMidRaw: SIMD3<Float> = {
                if have(.leftShoulder) && have(.rightShoulder) {
                    return (P(.leftShoulder) + P(.rightShoulder)) * 0.5
                }
                return have(.leftShoulder) ? P(.leftShoulder) : P(.rightShoulder)
            }()
            let spineDir = shMidRaw - hipMid
            let spineLen = simd_length(spineDir)
            let spineUnit = spineLen > 1e-6 ? spineDir / spineLen : SIMD3<Float>(0, 1, 0)
            let shMid = hipMid + spineUnit * bones.torso

            // Lateral axis from the measured shoulders (falls back to +X).
            var lateral = SIMD3<Float>(1, 0, 0)
            if have(.leftShoulder) && have(.rightShoulder) {
                let d = P(.rightShoulder) - P(.leftShoulder)
                if simd_length(d) > 1e-6 { lateral = simd_normalize(d) }
            }
            let half = bones.shoulderWidth * 0.5
            if have(.leftShoulder)  { set(.leftShoulder,  shMid - lateral * half) }
            if have(.rightShoulder) { set(.rightShoulder, shMid + lateral * half) }
        }

        // ARMS: shoulder → elbow → wrist (fixed upperArm, foreArm).
        if have(.leftShoulder) {
            place(P(.leftShoulder), .leftElbow, bones.upperArm)
            if have(.leftElbow) { place(P(.leftElbow), .leftWrist, bones.foreArm) }
        }
        if have(.rightShoulder) {
            place(P(.rightShoulder), .rightElbow, bones.upperArm)
            if have(.rightElbow) { place(P(.rightElbow), .rightWrist, bones.foreArm) }
        }

        // HIPS: re-hang the two hip joints off the hip mid at fixed half width
        // along the measured lateral direction (keeps pelvis width constant).
        if lHip && rHip {
            var lateral = P(.rightHip) - P(.leftHip)
            let len = simd_length(lateral)
            lateral = len > 1e-6 ? lateral / len : SIMD3<Float>(1, 0, 0)
            let half = bones.hipWidth * 0.5
            set(.leftHip,  hipMid - lateral * half)
            set(.rightHip, hipMid + lateral * half)
        }

        // LEGS: hip → knee → ankle (fixed thigh, shank, with the per-user leg
        // scale already baked into `bones`). Feet (heel/toe) are synthesized
        // downstream from the retargeted ankle.
        if have(.leftHip) {
            place(P(.leftHip), .leftKnee, bones.thigh)
            if have(.leftKnee) { place(P(.leftKnee), .leftAnkle, bones.shank) }
        }
        if have(.rightHip) {
            place(P(.rightHip), .rightKnee, bones.thigh)
            if have(.rightKnee) { place(P(.rightKnee), .rightAnkle, bones.shank) }
        }

        return out
    }
}
