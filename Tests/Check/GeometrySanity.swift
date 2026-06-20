import Foundation
import simd
import FeverCore

/// GEOMETRY SANITY — the end-to-end OSC-geometry regression guard.
///
/// The 2D preview tracks perfectly (image-space landmarks are accurate); the bug
/// was ONLY in the OSC tracker positions sent to VRChat. These tests feed a
/// synthetic UPRIGHT standing pose through the FULL OSC geometry chain —
///
///     raw 2D (Vision normalized, +Y up)
///       → MonocularDepthLift.stableScale        (stable anthropometric scale)
///       → MonocularDepthLift.depths             (foreshortening per-joint Z)
///       → VisionPoseLandmarker.assemble         (heel/toe synth + FLOOR ANCHOR)
///       → JointSolver.solve                     (9 VRJoints, solver frame)
///       → CoordinateMapper.toVRChatPosition     (single VRChat conversion)
///       → TrackerAssembler.assemble             (numbered slots + head ref)
///
/// — and assert the ASSEMBLED VRChat tracker positions form a COHERENT BODY:
///   • foot.y < knee.y < hip.y < chest.y  (vertical joint ordering, feet lowest)
///   • feet near the floor (y ≈ 0, NOT deeply negative / underground)
///   • the knee has a DIFFERENT Z (depth) than the hip (legs are not coplanar —
///     the regression that made VRChat read knee≈foot and collapse the legs)
///   • overall constellation height ≈ userHeight (1.74 ± tolerance)
///   • running the SAME input twice gives the SAME scale (stable, no per-frame
///     drift / "spazz")
/// plus default `enabledJoints` has all 8 numbered trackers and the head
/// reference is produced.
///
/// This drives the SAME `VisionPoseLandmarker.assemble` lift used by live
/// `detect()` (it is `public` precisely so the test exercises production code,
/// not a re-implementation).
enum GeometrySanity {

    /// A synthetic UPRIGHT standing pose in Vision-style normalized 2D coords:
    /// origin lower-left, +Y up, x∈[0,1] (aspect 1 so x is already height-units).
    /// Standing tall: head high, feet low. Slightly bent knees so the leg has a
    /// real out-of-plane depth (knee forward of the hip→ankle line).
    static func makeUprightRaw() -> (raw: [SIMD2<Float>], present: [Bool], image: [SIMD2<Float>]) {
        var raw = [SIMD2<Float>](repeating: .zero, count: 33)
        var present = [Bool](repeating: false, count: 33)
        var image = [SIMD2<Float>](repeating: SIMD2<Float>(.nan, .nan), count: 33)
        let c: Float = 0.5

        func set(_ l: BlazePose.Landmark, _ x: Float, _ y: Float) {
            raw[l.rawValue] = SIMD2<Float>(x, y)   // Vision: +Y up
            present[l.rawValue] = true
            image[l.rawValue] = SIMD2<Float>(x, 1 - y)  // preview: +Y down (unused here)
        }

        // Head / face (high in frame).
        set(.nose, c, 0.92)
        set(.leftEye, c - 0.03, 0.93); set(.rightEye, c + 0.03, 0.93)
        set(.leftEar, c - 0.05, 0.92); set(.rightEar, c + 0.05, 0.92)
        // Shoulders, elbows, wrists (arms down at the sides).
        set(.leftShoulder, c - 0.13, 0.80); set(.rightShoulder, c + 0.13, 0.80)
        set(.leftElbow, c - 0.15, 0.66);    set(.rightElbow, c + 0.15, 0.66)
        set(.leftWrist, c - 0.16, 0.52);    set(.rightWrist, c + 0.16, 0.52)
        // Hips.
        set(.leftHip, c - 0.08, 0.52); set(.rightHip, c + 0.08, 0.52)
        // Knees — slightly bent: projected thigh is SHORT vs true bone length, so
        // the foreshortening lift gives the knee a real forward (+Z) depth.
        set(.leftKnee, c - 0.08, 0.30); set(.rightKnee, c + 0.08, 0.30)
        // Ankles (low, near the bottom of the frame).
        set(.leftAnkle, c - 0.08, 0.08); set(.rightAnkle, c + 0.08, 0.08)
        return (raw, present, image)
    }

    /// Build the solver-frame `PoseResult` by driving the REAL lift code path.
    /// Mirrors `VisionPoseLandmarker.detect()`: root origin → stable scale →
    /// foreshortening depth + heel/toe synth + floor anchor (all inside
    /// `assemble`). `lift` is passed in so callers can re-use one instance (to
    /// test scale stability across repeated frames).
    static func lift(_ raw: [SIMD2<Float>],
                     _ present: [Bool],
                     _ image: [SIMD2<Float>],
                     using liftEngine: MonocularDepthLift,
                     time: TimeInterval = 0,
                     warmupFrames: Int = 60) -> PoseResult? {
        let root = VisionPoseLandmarker.rootOrigin(raw, present: present)
        // Warm the exponential scale smoother on the steady pose, exactly as a
        // few seconds of live tracking would, so the scale is fully converged.
        var k: Float = 0
        for _ in 0..<warmupFrames {
            k = liftEngine.stableScale(xy: raw, present: present) ?? k
        }
        guard k.isFinite, k > 0 else { return nil }
        return VisionPoseLandmarker.assemble(raw: raw, present: present, root: root,
                                             k: k, depthLift: liftEngine,
                                             imagePoints: image, time: time)
    }

    // MARK: - Tests

    static func run(_ t: TestRunner) {
        let (raw, present, image) = makeUprightRaw()

        // ── Full chain on a converged, floor-anchored upright pose ──────────
        let cfg = TrackingConfig()
        cfg.mirrorTracking = false            // pin handedness for deterministic Y/Z
        cfg.userHeightMeters = 1.74           // pin the expected overall height
        let userHeight: Float = 1.74

        let liftEngine = MonocularDepthLift(referenceHeight: 1.8)
        guard let pose = lift(raw, present, image, using: liftEngine) else {
            t.test("GEOMETRY: lift produced a pose") {
                t.check(false, "lift returned nil (scale never seeded)")
            }
            return
        }

        let solver = JointSolver(settings: cfg)
        let joints = solver.solve(pose)
        let mapper = CoordinateMapper(userHeightMeters: userHeight,
                                      referenceHeightMeters: 1.8,
                                      mirrorHorizontally: cfg.mirrorTracking)
        let assembler = TrackerAssembler(enabled: cfg.enabledJoints,
                                         slotMap: cfg.slotMap)
        let (body, head) = assembler.assemble(joints, mapper: mapper)

        // Index the assembled VRChat trackers by their joint type (via slot map).
        // slot: 1=hip 2=lFoot 3=rFoot 4=chest 5=lKnee 6=rKnee 7=lElbow 8=rElbow.
        var bySlot = [String: OSCTracker]()
        for tr in body { bySlot[tr.slot] = tr }

        func y(_ slot: String) -> Float? { bySlot[slot]?.position.y }
        func z(_ slot: String) -> Float? { bySlot[slot]?.position.z }

        // ── 1. Coherent vertical ordering: foot < knee < hip < chest ────────
        t.test("GEOMETRY: foot.y < knee.y < hip.y < chest.y (coherent body)") {
            guard let footL = y("2"), let footR = y("3"),
                  let kneeL = y("5"), let kneeR = y("6"),
                  let hip = y("1"), let chest = y("4") else {
                t.check(false, "missing assembled trackers: \(bySlot.keys.sorted())")
                return
            }
            let foot = min(footL, footR)
            let knee = min(kneeL, kneeR)
            t.check(foot < knee, "foot.y (\(foot)) must be below knee.y (\(knee))")
            t.check(knee < hip,  "knee.y (\(knee)) must be below hip.y (\(hip))")
            t.check(hip < chest, "hip.y (\(hip)) must be below chest.y (\(chest))")
        }

        // ── 2. Feet near the floor (y ≈ 0, not deeply negative) ─────────────
        t.test("GEOMETRY: feet rest near the floor (y ≈ 0, not underground)") {
            guard let footL = y("2"), let footR = y("3") else {
                t.check(false, "missing foot trackers"); return
            }
            let lowest = min(footL, footR)
            // Floor-anchored: lowest foot sits at/just above 0; never deeply
            // negative (the old "feet underground" hip-at-origin failure).
            t.check(lowest > -0.10, "lowest foot must NOT be deeply negative: \(lowest)")
            t.check(lowest < 0.20,  "lowest foot must be NEAR the floor: \(lowest)")
            t.check(footL.isFinite && footR.isFinite, "foot Y finite")
        }

        // ── 3. Knee has a DIFFERENT Z (depth) than the hip — not coplanar ───
        t.test("GEOMETRY: knee Z differs from hip Z (legs not coplanar)") {
            guard let hipZ = z("1"), let kneeLZ = z("5"), let kneeRZ = z("6") else {
                t.check(false, "missing hip/knee trackers for Z check"); return
            }
            t.check(abs(kneeLZ - hipZ) > 0.02,
                    "left knee Z must differ from hip Z: knee=\(kneeLZ) hip=\(hipZ)")
            t.check(abs(kneeRZ - hipZ) > 0.02,
                    "right knee Z must differ from hip Z: knee=\(kneeRZ) hip=\(hipZ)")
            t.check(kneeLZ.isFinite && kneeRZ.isFinite && hipZ.isFinite, "Z finite")
        }

        // ── 4. Overall constellation height ≈ userHeight (1.74 ±) ───────────
        t.test("GEOMETRY: overall height ≈ userHeight (1.74 m)") {
            // Span from the head reference (top) down to the lowest foot tracker.
            guard let h = head else { t.check(false, "no head reference"); return }
            let footL = y("2") ?? 0, footR = y("3") ?? 0
            let lowestFoot = min(footL, footR)
            let topY = h.position.y
            let height = topY - lowestFoot
            // The head reference sits at the ear midpoint (a bit below the crown),
            // so the head→foot span is a touch under true stature; allow a wide
            // ± band but pin it to the right METER scale (the spazz bug produced
            // wildly wrong / pulsing scales).
            t.check(height > 1.3 && height < 2.1,
                    "constellation height must be ≈ \(userHeight) m: got \(height)")
            t.check(topY.isFinite, "head Y finite")
        }

        // ── 5. Stable scale: SAME input twice → SAME scale (no drift) ───────
        t.test("GEOMETRY: same input twice gives the same scale (stable)") {
            // Two independent lift engines, identical converged input, must yield
            // byte-identical assembled foot positions (deterministic, no per-frame
            // drift). This pins the stable-scale fix (the old per-frame body-span
            // ratio pulsed and was NOT reproducible).
            let a = MonocularDepthLift(referenceHeight: 1.8)
            let b = MonocularDepthLift(referenceHeight: 1.8)
            guard let poseA = lift(raw, present, image, using: a),
                  let poseB = lift(raw, present, image, using: b) else {
                t.check(false, "lift nil during stability check"); return
            }
            let jA = solver.solve(poseA), jB = solver.solve(poseB)
            let (bodyA, _) = assembler.assemble(jA, mapper: mapper)
            let (bodyB, _) = assembler.assemble(jB, mapper: mapper)
            func footY(_ tk: [OSCTracker]) -> Float? {
                tk.first(where: { $0.slot == "2" })?.position.y
            }
            guard let fa = footY(bodyA), let fb = footY(bodyB) else {
                t.check(false, "no foot tracker in stability check"); return
            }
            t.check(fa == fb, "identical input must give identical scale: \(fa) vs \(fb)")
        }

        // ── 6. Re-running the same engine over the steady pose does not drift
        t.test("GEOMETRY: repeated frames hold the scale (no per-frame drift)") {
            let eng = MonocularDepthLift(referenceHeight: 1.8)
            var scales: [Float] = []
            for _ in 0..<300 {
                if let s = eng.stableScale(xy: raw, present: present) { scales.append(s) }
            }
            // After convergence the last 100 frames must be effectively constant.
            let tail = Array(scales.suffix(100))
            let lo = tail.min() ?? 0, hi = tail.max() ?? 0
            let rel = lo > 0 ? (hi - lo) / lo : 1
            t.check(rel < 1e-3, "converged scale must be steady: span \(rel)")
        }

        // ── 7. Default enabled set has all 8; head reference produced ───────
        t.test("GEOMETRY: default enabledJoints has all 8 + head reference") {
            let expected: Set<JointType> = [.hip, .chest,
                                            .leftElbow, .rightElbow,
                                            .leftKnee, .rightKnee,
                                            .leftFoot, .rightFoot]
            t.check(cfg.enabledJoints == expected,
                    "default enabledJoints must be all 8: \(cfg.enabledJoints)")
            t.check(body.count == 8, "all 8 numbered trackers must assemble: \(body.count)")
            t.check(head != nil, "head reference must be produced")
            t.check(head?.slot == "head", "head slot must be 'head'")
        }

        // ── Diagnostic dump (helps eyeball the assembled constellation) ─────
        t.test("GEOMETRY: assembled tracker positions are all finite") {
            for tr in body + (head.map { [$0] } ?? []) {
                let p = tr.position
                t.check(p.x.isFinite && p.y.isFinite && p.z.isFinite,
                        "slot \(tr.slot) position non-finite: \(p)")
            }
            // Print the constellation for the run log / report.
            func fmt(_ s: String, _ name: String) {
                if let tr = bySlot[s] {
                    let p = tr.position
                    print(String(format: "  [geom] %-9@ slot %@  y=%+.3f  z=%+.3f  x=%+.3f",
                                 name as NSString, s, p.y, p.z, p.x))
                }
            }
            fmt("4", "chest"); fmt("1", "hip")
            fmt("5", "leftKnee"); fmt("6", "rightKnee")
            fmt("2", "leftFoot"); fmt("3", "rightFoot")
            if let h = head {
                print(String(format: "  [geom] %-9@ slot %@  y=%+.3f  z=%+.3f  x=%+.3f",
                             "head" as NSString, h.slot, h.position.y, h.position.z, h.position.x))
            }
        }
    }
}
