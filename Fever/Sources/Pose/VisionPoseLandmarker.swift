import CoreVideo
import Vision
import Foundation
import simd

/// Native pose backend built on Apple's Vision body-pose estimator.
///
/// IMPORTANT — 2D, not 3D: we use `VNDetectHumanBodyPoseRequest` (the 2D
/// detector), NOT `VNDetectHumanBodyPose3DRequest`. Benchmarked on this M4:
/// the 3D request is ~81 ms/frame (~12 fps ceiling) and its monocular depth is
/// only a 1.8 m-reference assumption anyway, while the 2D request is ~5 ms/frame
/// (~180 fps) — so 2D is the only path that tracks at real camera frame rate.
///
/// ── OSC-tracker rework (this file) ───────────────────────────────────────────
/// The OSC tracker positions (NOT the 2D preview, which reads `imagePoints` and
/// is perfect) are built here in the solver's metric frame. Two failures that
/// made VRChat FBT collapse used to live here and are now fixed:
///
///   1. STABLE HEIGHT-BASED SCALE — the old lift scaled the whole skeleton by a
///      per-frame body-span ratio (`k = referenceHeight / span`), a noisy 2D
///      measurement that pulsed ±15-25% frame to frame (the "spazz"). We now use
///      a STABLE anthropometric scale (`MonocularDepthLift.stableScale`): the
///      metric size comes from anthropometric bone proportions (shoulder/hip
///      width + torso) and is heavily exponentially smoothed, so the overall
///      size does not jump frame to frame.
///   2. MONOCULAR DEPTH — joints were lifted onto a single Z=0 plane, so a knee
///      swung toward the camera projected onto the ankle and the legs collapsed
///      (knee≈foot). We now synthesize a real per-joint Z from anthropometric
///      bone-length foreshortening (`MonocularDepthLift.depths`), keeping XY
///      exactly as the accurate 2D lift gives it, with temporal sign hysteresis
///      so depth never flickers.
///   3. FLOOR ANCHOR — instead of hip-at-origin (feet at negative Y, i.e. feet
///      underground), we offset the whole skeleton in Y so the lower foot rests
///      near the floor plane (Y ≈ 0) in the solver frame, consistent with the
///      head reference.
///
/// Vision 2D returns up to 19 joints (`nose, l/rEye, l/rEar, neck, l/rShoulder,
/// l/rElbow, l/rWrist, root, l/rHip, l/rKnee, l/rAnkle`) with a per-joint
/// confidence, as NORMALIZED points (origin lower-left, +Y up, range 0…1). We:
///   1. aspect-correct X (multiply by width/height so equal normalized deltas
///      are equal real distances),
///   2. re-origin to the hip "root" (midpoint of the hips),
///   3. scale by the STABLE anthropometric scale → reference (1.8 m) meters (the
///      final true-height rescale happens once downstream in `CoordinateMapper`),
///   4. synthesize real per-joint Z from bone-length foreshortening,
///   5. floor-anchor in Y so the feet rest near 0.
/// The result is remapped into the existing 33-slot BlazePose array (heels/toes
/// synthesized from the ankles so the foot solver has a forward axis), so the
/// JointSolver / CoordinateMapper / TrackerAssembler chain is reused unchanged.
public final class VisionPoseLandmarker: PoseLandmarker {

    /// Joints below this confidence are treated as absent.
    private let minConfidence: Float = 0.1
    /// The reference body height (meters) the lifted skeleton is scaled to.
    /// CoordinateMapper then rescales 1.8 → the user's true height.
    private let referenceHeight: Float = 1.8

    /// Minimum trusted vertical body span (normalized height-units, Y ∈ 0…1).
    /// Below this we have too little body in frame to derive a reliable measure,
    /// so the frame is rejected rather than seeding garbage. ~1/8 of frame height.
    private static let minTrustedSpan: Float = 0.125

    /// The stable anthropometric scale + monocular foreshortening depth lift.
    /// Stateful (smoothed scale + per-segment sign hysteresis); confined to the
    /// single serial inference worker like the rest of this object.
    private let depthLift: MonocularDepthLift

    /// Per-user leg-length multiplier applied ONLY to the thigh/shank
    /// anthropometric fractions. The Drillis & Contini fractions are a population
    /// prior, not ground truth; the user's avatar has slightly longer legs, so a
    /// modest >1 multiplier matches the leg bones to the avatar without disturbing
    /// the rest of the fixed skeleton.
    private static let legScale: Float = 1.08

    public init() {
        self.depthLift = MonocularDepthLift(referenceHeight: 1.8, legScale: Self.legScale)
    }

    /// Clear the per-run temporal state (smoothed anthropometric scale + the
    /// depth-sign hysteresis) so a fresh run does not inherit a stale scale.
    public func reset() {
        depthLift.reset()
    }

    /// Single reused body-pose request. `VNDetectHumanBodyPoseRequest` is
    /// stateless across `perform` calls (each `VNImageRequestHandler.perform`
    /// fully repopulates `request.results`), so reusing one instance across
    /// frames avoids a per-frame allocation. Safe because `detect` is only ever
    /// called from the single, strictly-serial inference worker (never
    /// concurrently) — confirmed by `TrackingPipeline`'s worker design.
    private let request = VNDetectHumanBodyPoseRequest()

    /// Per-frame scratch, reused across calls (single-threaded worker confines
    /// it). `raw[i]` holds the aspect-corrected normalized point for BlazePose
    /// landmark `i`; `present[i]` marks whether it was detected this frame.
    private var raw = [SIMD2<Float>](repeating: .zero, count: 33)
    private var present = [Bool](repeating: false, count: 33)

    public func detect(_ pixelBuffer: CVPixelBuffer,
                       at time: TimeInterval) async -> PoseResult? {
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer,
                                            orientation: .up,
                                            options: [:])
        do {
            try handler.perform([request])
        } catch {
            return nil
        }
        guard let obs = request.results?.first else { return nil }

        let w = CVPixelBufferGetWidth(pixelBuffer)
        let h = CVPixelBufferGetHeight(pixelBuffer)
        let aspect = h > 0 ? Float(w) / Float(h) : 1   // X scaled into height-units

        // Gather aspect-corrected normalized points (origin lower-left, +Y up).
        guard let points = try? obs.recognizedPoints(.all) else { return nil }

        // Reset the per-frame presence mask (scratch reused across frames).
        for i in 0..<33 { present[i] = false }
        // RAW screen-normalized preview points (UNSMOOTHED, before any metric
        // conversion): x ∈ [0,1] from LEFT, y ∈ [0,1] from TOP. Vision points are
        // origin lower-left (+Y up), so flip Y to top-origin for the overlay.
        var imagePoints = [SIMD2<Float>](repeating: SIMD2<Float>(.nan, .nan), count: 33)
        for (jointName, p) in points {
            guard p.confidence >= minConfidence,
                  let blaze = Self.jointToBlaze[jointName] else { continue }
            let i = blaze.rawValue
            raw[i] = SIMD2<Float>(Float(p.location.x) * aspect, Float(p.location.y))
            present[i] = true
            imagePoints[i] = SIMD2<Float>(Float(p.location.x), 1 - Float(p.location.y))
        }

        // Hip-root origin: midpoint of the hips (fall back to whatever torso
        // points exist) so positions are hip-root-relative like the 3D path.
        let root = Self.rootOrigin(raw, present: present)

        // Require enough body in frame to seed/trust the anthropometric scale.
        // A real full body spans roughly the full normalized height; below ~1/8
        // of the frame there is too little body to derive a reliable measure, so
        // reject the frame rather than scaling garbage onto the wire.
        let span = Self.bodySpan(raw, present: present)
        guard span >= Self.minTrustedSpan else { return nil }

        // STABLE HEIGHT-BASED SCALE — image-units → reference (1.8 m) meters.
        // Derived from anthropometric proportions (shoulder/hip width + torso)
        // and heavily exponentially smoothed inside the lift, so the overall
        // size does NOT jump frame to frame (the old per-frame body-span ratio
        // was the spazz cause). nil only before the very first measurement seeds
        // it — reject those frames so we never emit an unscaled skeleton.
        guard let k = depthLift.stableScale(xy: raw, present: present),
              k.isFinite, k > 0 else { return nil }

        return Self.assemble(raw: raw, present: present, root: root, k: k,
                             depthLift: depthLift,
                             imagePoints: imagePoints, time: time)
    }

    /// Vision 2D `JointName` → BlazePose 33-slot landmark.
    private static let jointToBlaze: [VNHumanBodyPoseObservation.JointName: BlazePose.Landmark] = [
        .nose: .nose,
        .leftEye: .leftEye,
        .rightEye: .rightEye,
        .leftEar: .leftEar,
        .rightEar: .rightEar,
        .leftShoulder: .leftShoulder,
        .rightShoulder: .rightShoulder,
        .leftElbow: .leftElbow,
        .rightElbow: .rightElbow,
        .leftWrist: .leftWrist,
        .rightWrist: .rightWrist,
        .leftHip: .leftHip,
        .rightHip: .rightHip,
        .leftKnee: .leftKnee,
        .rightKnee: .rightKnee,
        .leftAnkle: .leftAnkle,
        .rightAnkle: .rightAnkle,
        // neck / root are torso-internal; the solver derives chest/hip from the
        // shoulders + hips, so they need no distinct BlazePose slot.
    ]

    /// Reads a present landmark from the scratch arrays, or `nil` if absent.
    @inline(__always)
    private static func value(_ raw: [SIMD2<Float>], _ present: [Bool],
                              _ l: BlazePose.Landmark) -> SIMD2<Float>? {
        present[l.rawValue] ? raw[l.rawValue] : nil
    }

    /// Hip-root origin = midpoint of the hips, or the available torso center.
    /// PUBLIC so the geometry-sanity test computes the root exactly as `detect()`.
    public static func rootOrigin(_ raw: [SIMD2<Float>], present: [Bool]) -> SIMD2<Float> {
        let lHip = value(raw, present, .leftHip), rHip = value(raw, present, .rightHip)
        if let l = lHip, let r = rHip { return (l + r) * 0.5 }
        if let l = lHip { return l }
        if let r = rHip { return r }
        if let ls = value(raw, present, .leftShoulder),
           let rs = value(raw, present, .rightShoulder) { return (ls + rs) * 0.5 }
        // Last resort: centroid of everything present.
        var sum = SIMD2<Float>.zero
        var count = 0
        for i in 0..<present.count where present[i] { sum += raw[i]; count += 1 }
        guard count > 0 else { return .zero }
        return sum / Float(count)
    }

    /// Vertical body span (normalized height-units) from the highest available
    /// head point down to the lowest available foot point, for metric scaling.
    private static func bodySpan(_ raw: [SIMD2<Float>], present: [Bool]) -> Float {
        @inline(__always) func y(_ l: BlazePose.Landmark) -> Float? { value(raw, present, l)?.y }
        let top = [y(.nose), y(.leftEye), y(.rightEye), y(.leftEar), y(.rightEar)]
            .compactMap { $0 }.max()
        let bottom = [y(.leftAnkle), y(.rightAnkle), y(.leftKnee), y(.rightKnee)]
            .compactMap { $0 }.min()
        if let t = top, let b = bottom, t > b { return t - b }
        // Fall back to a torso measure (shoulder→hip) scaled up to a full body.
        let sh = [y(.leftShoulder), y(.rightShoulder)].compactMap { $0 }.max()
        let hp = [y(.leftHip), y(.rightHip)].compactMap { $0 }.min()
        if let s = sh, let h = hp, s > h { return (s - h) * 3.0 }
        return 0
    }

    /// Builds the 33-slot BlazePose `PoseResult` from the lifted joints in the
    /// STABLE camera/world frame (reference meters, +X right / +Y up / +Z toward
    /// camera). The pipeline is now:
    ///
    ///   1. HIP-RELATIVE 3D — metric XY = (raw − hipRoot)·k; per-joint Z from
    ///      anthropometric bone-length foreshortening (`depths`). This is a clean
    ///      hip-rooted frame used ONLY to compute limb configuration/direction.
    ///   2. FIXED-LENGTH RETARGET (FIX 1) — `retarget()` walks each chain from
    ///      the hip and replaces every bone length with its fixed anthropometric
    ///      length, preserving the 3D direction. After this the skeleton's size
    ///      is invariant to pose/rotation/distance; the global factor `k` only
    ///      affects placement, never proportions.
    ///   3. HIP WORLD TRANSLATION (FIX 2) — instead of pinning the hip at the
    ///      origin, the hip's REAL metric position in the stable camera frame
    ///      (`hipRoot·k`, never subtracted out) is added back so every emitted
    ///      joint = hipWorld + (joint − hip). The hip's frame-to-frame XY/Z sway
    ///      now survives onto the wire (a hip pinned at 0 can never translate).
    ///   4. FIXED FLOOR — the WHOLE skeleton is shifted so a LATCHED (one-time,
    ///      then frozen) floor plane sits at Y ≈ 0. Because the floor is fixed
    ///      and not re-zeroed every frame, vertical hip motion is preserved while
    ///      the feet still calibrate near the floor.
    ///
    /// Synthesizes heels and toes from the (retargeted) ankles so the foot solver
    /// has a forward axis. Returns `nil` if the torso is not present.
    ///
    /// PUBLIC so the headless geometry-sanity test can drive the EXACT same lift
    /// code path the live `detect()` uses, rather than re-implementing it.
    public static func assemble(raw: [SIMD2<Float>],
                                present: [Bool],
                                root: SIMD2<Float>,
                                k: Float,
                                depthLift: MonocularDepthLift,
                                imagePoints: [SIMD2<Float>],
                                time: TimeInterval) -> PoseResult? {
        // 1. Hip-RELATIVE metric XY (reference meters), hip at origin, used only
        //    to derive limb configuration + foreshortening depth.
        var metricXY = [SIMD2<Float>](repeating: .zero, count: 33)
        for i in 0..<33 where present[i] {
            metricXY[i] = (raw[i] - root) * k
        }
        let z = depthLift.depths(metricXY: metricXY, present: present)

        // Pack into hip-relative 3D for the fixed-length retarget pass.
        var rel = [SIMD3<Float>](repeating: .zero, count: 33)
        for i in 0..<33 where present[i] {
            rel[i] = SIMD3<Float>(metricXY[i].x, metricXY[i].y, z[i])
        }

        // 2. FIXED-LENGTH RETARGET — every bone now a session constant.
        rel = depthLift.retarget(rel, present: present)

        // 3. HIP WORLD TRANSLATION — the hip's actual metric position in the
        //    stable camera frame. Adding it back keeps the hip free to translate
        //    (the core of FIX 2); never subtract the current-frame hip again.
        let hipWorld = SIMD3<Float>(root.x * k, root.y * k, 0)

        // CENTER THE FRAME (FIX 2): the hip's absolute XZ sits a couple of metres
        // off-origin in the camera frame (~+2.1 m in X), which amplified every
        // dropout into a multi-metre teleport. Latch the hip's horizontal world
        // origin ONCE (first frame), then subtract that SAME frozen constant from
        // every joint below — head included. Because the identical constant is
        // removed from head and body, (tracker − head) head-relative geometry is
        // byte-unchanged; only the absolute frame shifts to sit near 0 like PinoFBT.
        // Re-seeded on reset()/Recenter, exactly like the floor latch. We do NOT
        // touch Y here (the floor latch below owns vertical), so latch X/Z only.
        let origin = depthLift.latchOriginXZ(SIMD2<Float>(hipWorld.x, hipWorld.z))
        let centeredWorld = SIMD3<Float>(hipWorld.x - origin.x, hipWorld.y, hipWorld.z - origin.y)

        var lm = [NormalizedLandmark](repeating: NormalizedLandmark(position: .zero,
                                                                    visibility: 0, presence: 0),
                                      count: 33)
        for i in 0..<33 where present[i] {
            lm[i] = NormalizedLandmark(position: rel[i] + centeredWorld, visibility: 1, presence: 1)
        }

        // Require both shoulders and at least one hip so the solver always has a
        // valid torso reference frame.
        let haveShoulders = lm[.leftShoulder].presence > 0 && lm[.rightShoulder].presence > 0
        let haveHip = lm[.leftHip].presence > 0 || lm[.rightHip].presence > 0
        guard haveShoulders, haveHip else { return nil }

        // Synthesize heels (below + behind the ankle) and toes (below + forward)
        // so the foot bone has length and a forward (+Z) axis. Offsets are in the
        // same metric solver units and ADD to the ankle's now-real Z.
        for (ankle, heel) in [(BlazePose.Landmark.leftAnkle, BlazePose.Landmark.leftHeel),
                              (BlazePose.Landmark.rightAnkle, BlazePose.Landmark.rightHeel)] {
            guard lm[ankle].presence > 0 else { continue }
            let a = lm[ankle].position
            lm[heel] = NormalizedLandmark(position: SIMD3<Float>(a.x, a.y - 0.06, a.z - 0.06),
                                          visibility: 1, presence: 1)
        }
        for (ankle, toe) in [(BlazePose.Landmark.leftAnkle, BlazePose.Landmark.leftFootIndex),
                             (BlazePose.Landmark.rightAnkle, BlazePose.Landmark.rightFootIndex)] {
            guard lm[ankle].presence > 0 else { continue }
            let a = lm[ankle].position
            lm[toe] = NormalizedLandmark(position: SIMD3<Float>(a.x, a.y - 0.08, a.z + 0.12),
                                         visibility: 1, presence: 1)
        }

        // 4. FIXED FLOOR — shift the whole skeleton so a LATCHED floor plane sits
        //    at Y ≈ 0. The floor is captured ONCE (lowest foot of the first
        //    standing frame) and then frozen, so subsequent frames subtract a
        //    CONSTANT, not the current-frame lowest foot. That is what preserves
        //    vertical hip motion (re-zeroing every frame would pin the body to
        //    the floor and kill vertical sway — FIX 2) while still seating the
        //    feet near the floor for VRChat auto-center / FBT calibrate.
        let footSlots: [BlazePose.Landmark] = [.leftAnkle, .rightAnkle,
                                               .leftHeel, .rightHeel,
                                               .leftFootIndex, .rightFootIndex]
        var lowestFoot: Float? = nil
        for s in footSlots where lm[s].presence > 0 {
            let y = lm[s].position.y
            lowestFoot = lowestFoot.map { Swift.min($0, y) } ?? y
        }
        if let lf = lowestFoot {
            let floor = depthLift.latchFloor(lf)
            for i in 0..<33 where lm[i].presence > 0 {
                lm[i].position.y -= floor
            }
        }

        return PoseResult(landmarks: lm, timestamp: time, imagePoints: imagePoints)
    }
}
