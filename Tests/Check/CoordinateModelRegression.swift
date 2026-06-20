import Foundation
import simd
import FeverCore

/// COORDINATE-MODEL REGRESSION GUARDS — the two problem-specific tests that pin
/// the coordinate-model fix so it can never silently regress.
///
/// These exercise the SAME production lift path used by live `detect()`
/// (`VisionLiftGeometry.rootOrigin` → `stableScale` → `assemble`, which runs
/// `depths` + `retarget` + hip-world-translation + floor-latch internally), then
/// the real `JointSolver` → `CoordinateMapper` → `TrackerAssembler`. Nothing is
/// re-implemented.
///
///   PROBLEM 1 — SCALE INVARIANCE: feeding the SAME pose at several simulated
///   body ROTATIONS (the shoulder/hip width shrinks as the torso turns away from
///   the camera, i.e. foreshortening) must NOT change the reconstructed
///   skeleton's overall height or its key bone lengths. The old width-summing
///   scale grew as those widths foreshortened ("growing/pulsing skeleton"); the
///   fixed vertical-extent scale + bone-length retarget make size mathematically
///   invariant to rotation.
///
///   PROBLEM 2 — HIP TRANSLATION: when the whole body shifts horizontally /
///   vertically in frame, the ASSEMBLED hip tracker position must MOVE by the
///   expected amount — it must NOT be pinned at a constant (the old hip-at-origin
///   bug). The vertical ordering (foot<knee<hip<chest, feet near floor) must stay
///   coherent through the shift.
enum CoordinateModelRegression {

    // MARK: - Synthetic pose generator

    /// Build a synthetic UPRIGHT standing pose in Vision-style normalized 2D
    /// coords (origin lower-left, +Y up, aspect 1 so x is already height-units).
    ///
    /// - `center`:      horizontal placement of the body centerline (X shift).
    /// - `vertical`:    vertical placement of the whole body (Y shift up/down).
    /// - `widthShrink`: 0…1 fraction the LATERAL (shoulder/hip/knee/ankle) width
    ///                  is multiplied DOWN by, to mimic the torso yawing away from
    ///                  the camera (lateral foreshortening). 0 = full-on facing,
    ///                  0.6 = strongly turned. The VERTICAL extent (head→ankle) is
    ///                  deliberately held CONSTANT so only the rotation cue changes.
    static func makePose(center c: Float = 0.5,
                         vertical v: Float = 0.0,
                         widthShrink: Float = 0.0)
        -> (raw: [SIMD2<Float>], present: [Bool], image: [SIMD2<Float>]) {
        var raw = [SIMD2<Float>](repeating: .zero, count: 33)
        var present = [Bool](repeating: false, count: 33)
        var image = [SIMD2<Float>](repeating: SIMD2<Float>(.nan, .nan), count: 33)
        let wf = 1.0 - widthShrink   // lateral width factor

        // Place a landmark: dx is the lateral half-offset (scaled by wf to
        // foreshorten width), dy is the vertical offset from the body's vertical
        // origin `v`. Vertical offsets are NEVER scaled by wf — turning the torso
        // does not change the standing height the camera sees.
        func set(_ l: BlazePose.Landmark, _ dx: Float, _ y: Float) {
            let x = c + dx * wf
            raw[l.rawValue] = SIMD2<Float>(x, y + v)
            present[l.rawValue] = true
            image[l.rawValue] = SIMD2<Float>(x, 1 - (y + v))
        }

        // Head / face (high in frame) — face points keep the head extent.
        set(.nose, 0, 0.92)
        set(.leftEye, -0.03, 0.93); set(.rightEye, 0.03, 0.93)
        set(.leftEar, -0.05, 0.92); set(.rightEar, 0.05, 0.92)
        // Shoulders / elbows / wrists (arms down at the sides).
        set(.leftShoulder, -0.13, 0.80); set(.rightShoulder, 0.13, 0.80)
        set(.leftElbow, -0.15, 0.66);    set(.rightElbow, 0.15, 0.66)
        set(.leftWrist, -0.16, 0.52);    set(.rightWrist, 0.16, 0.52)
        // Hips.
        set(.leftHip, -0.08, 0.52); set(.rightHip, 0.08, 0.52)
        // Knees — slightly bent so the leg has a real out-of-plane depth.
        set(.leftKnee, -0.08, 0.30); set(.rightKnee, 0.08, 0.30)
        // Ankles (low, near the bottom of the frame).
        set(.leftAnkle, -0.08, 0.08); set(.rightAnkle, 0.08, 0.08)
        return (raw, present, image)
    }

    /// Drive the production lift on a fresh engine (own latch/sign state) for one
    /// pose, returning the solver-frame `PoseResult`. Warms the scale smoother so
    /// the latch is converged, exactly like a couple seconds of live tracking.
    static func liftPose(_ p: (raw: [SIMD2<Float>], present: [Bool], image: [SIMD2<Float>]),
                         using eng: MonocularDepthLift,
                         time: TimeInterval = 0,
                         warmupFrames: Int = 120) -> PoseResult? {
        let root = VisionLiftGeometry.rootOrigin(p.raw, present: p.present)
        var k: Float = 0
        for _ in 0..<warmupFrames {
            k = eng.stableScale(xy: p.raw, present: p.present) ?? k
        }
        guard k.isFinite, k > 0 else { return nil }
        return VisionLiftGeometry.assemble(raw: p.raw, present: p.present, root: root,
                                             k: k, depthLift: eng,
                                             imagePoints: p.image, time: time)
    }

    // MARK: - Geometry helpers on solved joints

    static func joint(_ joints: [VRJoint], _ type: JointType) -> VRJoint? {
        joints.first { $0.type == type }
    }

    /// Bone length (meters) between two solved joints in the SOLVER frame.
    static func boneLen(_ joints: [VRJoint], _ a: JointType, _ b: JointType) -> Float? {
        guard let ja = joint(joints, a), let jb = joint(joints, b) else { return nil }
        return simd_length(ja.position - jb.position)
    }

    /// TRUE anthropometric bone length between two retargeted BlazePose landmarks
    /// in the lifted `PoseResult` (the fixed-length skeleton `retarget()`
    /// produces). This is the quantity the FIX makes a session constant — it is
    /// measured between the actual chain endpoints (e.g. leftHip→leftKnee), NOT
    /// the hip-MIDPOINT tracker, so it isolates the bone-length invariant from the
    /// (legitimately width-dependent) midpoint geometry.
    static func landmarkBone(_ pose: PoseResult,
                             _ a: BlazePose.Landmark, _ b: BlazePose.Landmark) -> Float? {
        let la = pose.landmarks[a.rawValue], lb = pose.landmarks[b.rawValue]
        guard la.presence > 0, lb.presence > 0 else { return nil }
        return simd_length(la.position - lb.position)
    }

    /// Total reconstructed skeleton height (solver frame): top of head reference
    /// down to the lowest foot.
    static func totalHeight(_ joints: [VRJoint]) -> Float? {
        guard let head = joint(joints, .head) else { return nil }
        let feetY = [joint(joints, .leftFoot)?.position.y,
                     joint(joints, .rightFoot)?.position.y].compactMap { $0 }
        guard let lowest = feetY.min() else { return nil }
        return head.position.y - lowest
    }

    // MARK: - Tests

    static func run(_ t: TestRunner) {
        runScaleInvariance(t)
        runHipTranslation(t)
    }

    // MARK: PROBLEM 1 — scale invariance across rotation / foreshortening

    /// Feed the SAME standing pose at a sweep of simulated torso rotations
    /// (lateral width shrunk 0% → 50%, mimicking the body turning away so the
    /// shoulder/hip widths foreshorten) and assert the reconstructed skeleton's
    /// TOTAL HEIGHT and KEY BONE LENGTHS stay essentially CONSTANT.
    ///
    /// This is the direct regression guard for problem 1: the old width-summing
    /// scale grew as the widths shrank (skeleton "grew"/pulsed). With the fixed
    /// vertical-extent scale + bone-length retarget, every measured length is a
    /// session constant regardless of how far the torso is turned.
    static func runScaleInvariance(_ t: TestRunner) {
        let cfg = TrackingConfig()
        cfg.mirrorTracking = false
        let solver = JointSolver(settings: cfg)

        // Sweep of rotation strengths (lateral foreshortening fractions).
        let shrinks: [Float] = [0.0, 0.10, 0.25, 0.40, 0.50]

        var heights: [Float] = []
        var torsoLens: [Float] = []      // shoulderMid↔hipMid (retargeted spine)
        var thighLens: [Float] = []      // leftHip↔leftKnee  (retargeted bone)
        var shankLens: [Float] = []      // leftKnee↔leftAnkle (retargeted bone)

        for sh in shrinks {
            // Fresh engine per rotation so each is an independent reconstruction
            // (no shared latch carrying scale across the sweep). The point is that
            // each INDEPENDENT solve lands at the same size, not that one latched
            // engine ignores later frames.
            let eng = MonocularDepthLift(referenceHeight: 1.8,
                                         legScale: 1.08)
            guard let pose = liftPose(makePose(widthShrink: sh), using: eng) else {
                t.test("SCALE-INV: lift produced a pose (shrink \(sh))") {
                    t.check(false, "lift returned nil at shrink \(sh)")
                }
                return
            }
            let joints = solver.solve(pose)
            // Total height from the SOLVED tracker constellation (head→lowest
            // foot). Bone lengths are measured between the actual retargeted
            // chain endpoints in the lifted landmarks — that is the fixed-length
            // invariant the fix guarantees, isolated from the hip-midpoint
            // geometry that legitimately depends on shoulder/hip width.
            guard let h = totalHeight(joints),
                  let torso = landmarkBone(pose, .leftShoulder, .leftHip)
                    ?? landmarkBone(pose, .rightShoulder, .rightHip),
                  let thigh = landmarkBone(pose, .leftHip, .leftKnee),
                  let shank = landmarkBone(pose, .leftKnee, .leftAnkle) else {
                t.test("SCALE-INV: all key joints solved (shrink \(sh))") {
                    t.check(false, "missing solved joints at shrink \(sh)")
                }
                return
            }
            heights.append(h); torsoLens.append(torso)
            thighLens.append(thigh); shankLens.append(shank)
        }

        // Relative spread (max-min)/min across the rotation sweep. A scale that
        // GROWS with foreshortening (the bug) would blow this up; the fixed model
        // must hold it to a tiny fraction.
        func relSpread(_ xs: [Float]) -> Float {
            guard let lo = xs.min(), let hi = xs.max(), lo > 1e-6 else { return .infinity }
            return (hi - lo) / lo
        }

        t.test("SCALE-INV: total height constant across rotations (no scale growth)") {
            let rel = relSpread(heights)
            print(String(format: "  [scale-inv] heights across shrink %@ = %@  (rel spread %.5f)",
                         shrinks.map { String(format: "%.2f", $0) }.joined(separator: ",") as NSString,
                         heights.map { String(format: "%.4f", $0) }.joined(separator: ",") as NSString,
                         rel))
            t.check(rel < 0.01,
                    "reconstructed height must stay constant across rotation: rel spread \(rel)")
            // Pin the absolute meter scale too (reference units, pre true-height
            // rescale): a coherent standing skeleton, not a pulsing one.
            if let h0 = heights.first { t.check(h0 > 1.3 && h0 < 2.1, "height in human range: \(h0)") }
        }

        t.test("SCALE-INV: torso bone length constant across rotations") {
            let rel = relSpread(torsoLens)
            print(String(format: "  [scale-inv] torsoLen = %@  (rel %.5f)",
                         torsoLens.map { String(format: "%.4f", $0) }.joined(separator: ",") as NSString, rel))
            t.check(rel < 0.01, "torso (shoulder↔hip) length must be invariant: \(rel)")
        }

        t.test("SCALE-INV: thigh bone length constant across rotations") {
            let rel = relSpread(thighLens)
            print(String(format: "  [scale-inv] thighLen = %@  (rel %.5f)",
                         thighLens.map { String(format: "%.4f", $0) }.joined(separator: ",") as NSString, rel))
            t.check(rel < 0.01, "thigh (hip↔knee) length must be invariant: \(rel)")
        }

        t.test("SCALE-INV: shank bone length constant across rotations") {
            let rel = relSpread(shankLens)
            print(String(format: "  [scale-inv] shankLen = %@  (rel %.5f)",
                         shankLens.map { String(format: "%.4f", $0) }.joined(separator: ",") as NSString, rel))
            t.check(rel < 0.01, "shank (knee↔foot) length must be invariant: \(rel)")
        }
    }

    // MARK: PROBLEM 2 — hip translation (hip must move, not be pinned)

    /// Feed two frames whose body has SHIFTED horizontally (and a third shifted
    /// vertically) and assert the ASSEMBLED hip tracker position MOVES by the
    /// expected metric amount. The old hip-at-origin model pinned the hip at a
    /// constant; this guards that the hip's real world translation survives onto
    /// the wire. Also re-asserts the vertical ordering stays coherent after the
    /// shift (foot<knee<hip<chest, feet near floor).
    static func runHipTranslation(_ t: TestRunner) {
        let cfg = TrackingConfig()
        cfg.mirrorTracking = false      // pin handedness so X maps straight through
        cfg.userHeightMeters = 1.74
        let solver = JointSolver(settings: cfg)
        let mapper = CoordinateMapper(userHeightMeters: 1.74,
                                      referenceHeightMeters: 1.8,
                                      mirrorHorizontally: false)
        let assembler = TrackerAssembler(enabled: cfg.enabledJoints,
                                         slotMap: cfg.slotMap)

        /// Lift+solve+assemble one pose on its OWN engine, returning the assembled
        /// trackers indexed by slot, plus the head reference.
        func assemble(center c: Float, vertical v: Float)
            -> (bySlot: [String: OSCTracker], head: OSCTracker?)? {
            let eng = MonocularDepthLift(referenceHeight: 1.8, legScale: 1.08)
            guard let pose = liftPose(makePose(center: c, vertical: v), using: eng) else {
                return nil
            }
            let joints = solver.solve(pose)
            let (body, head) = assembler.assemble(joints, mapper: mapper)
            var bySlot = [String: OSCTracker]()
            for tr in body { bySlot[tr.slot] = tr }
            return (bySlot, head)
        }

        // The hip X in the solver frame is root.x*k (root = hip midpoint *
        // aspect). Two frames whose centerline differs by Δcenter (normalized,
        // aspect 1) must move the hip X by Δcenter*k in the solver frame, then by
        // the mapper's userHeight/reference scale on the wire. We don't hard-code
        // k (it is anthropometric); instead we assert the hip MOVES in the same
        // direction by a clearly non-zero amount proportional to the input shift.

        // ── 2a. Horizontal hip translation ──────────────────────────────────
        t.test("HIP-TRANS: hip tracker moves horizontally when body shifts (not pinned)") {
            let dC: Float = 0.20   // shift the body 0.20 (normalized) to the right
            // Use ONE shared engine: the XZ-centering origin latches on frame A
            // (and is then FROZEN), so frame B's horizontal shift survives as a
            // real translation off that frozen origin — exactly the centering
            // contract (latch once, motion survives). Two independent engines
            // would each re-center their own first frame to 0 and the translation
            // would vanish; the live pipeline only ever latches once per run.
            let eng = MonocularDepthLift(referenceHeight: 1.8, legScale: 1.08)
            func hipTracker(center c: Float, time: TimeInterval, warmup: Int) -> OSCTracker? {
                guard let pose = liftPose(makePose(center: c, vertical: 0), using: eng,
                                          time: time, warmupFrames: warmup) else { return nil }
                let joints = solver.solve(pose)
                let (body, _) = assembler.assemble(joints, mapper: mapper)
                return body.first { $0.slot == "1" }
            }
            // Seed/latch on A (warm the scale + freeze the XZ origin), then B on
            // the SAME engine with the body shifted right.
            guard let hipA = hipTracker(center: 0.40, time: 0, warmup: 120),
                  let hipB = hipTracker(center: 0.40 + dC, time: 1, warmup: 1) else {
                t.check(false, "lift/assemble returned nil"); return
            }
            let dxHip = hipB.position.x - hipA.position.x
            print(String(format: "  [hip-trans] hipX a=%+.4f b=%+.4f  Δ=%+.4f  (input Δcenter=%.2f)",
                         hipA.position.x, hipB.position.x, dxHip, dC))
            // It MUST move (not pinned), in the SAME direction as the body shift,
            // by a clearly non-trivial amount.
            t.check(dxHip > 0.05,
                    "hip X must move right with the body (not pinned): Δ=\(dxHip)")
            // And it must move by the metric-scaled input shift. The expected
            // solver-frame move is dC*k where k≈ standingExtent/observedExtent.
            // For this pose observed head→ankle ≈ 0.85, standingExtent =
            // 1.8*0.897 ≈ 1.6146 → k ≈ 1.9, then ×(1.74/1.8) on the wire ≈ 1.84.
            // So Δhip ≈ 0.20*1.84 ≈ 0.37; assert it is within a generous band of
            // a proportional move (catches both "pinned" and "wrong scale").
            let expected = dC * 1.84
            t.check(abs(dxHip - expected) < 0.12,
                    "hip X move must match the scaled body shift ~\(expected): Δ=\(dxHip)")
            // Y unchanged by a pure horizontal shift (floor-latched).
            t.close(hipA.position.y, hipB.position.y, tol: 0.05,
                    "hip Y must not change on a pure horizontal shift")
        }

        // ── 2b. Vertical hip translation (hip not floor-pinned) ─────────────
        t.test("HIP-TRANS: hip tracker moves vertically when body rises (floor not re-zeroed)") {
            // Frame A seeds the floor latch on a normal standing frame. Frame B,
            // on the SAME engine, is the body raised by dV — because the floor is
            // latched ONCE and frozen, the hip must rise, NOT be re-pinned to the
            // floor. Use one shared engine so the latch is seeded on A and held.
            let eng = MonocularDepthLift(referenceHeight: 1.8, legScale: 1.08)
            let dV: Float = 0.10
            // Warm + seed on the grounded pose A (this latches the floor).
            guard let poseA = liftPose(makePose(center: 0.5, vertical: 0), using: eng,
                                       time: 0, warmupFrames: 120) else {
                t.check(false, "poseA lift nil"); return
            }
            // Now the SAME engine on the raised pose B (floor stays latched).
            guard let poseB = liftPose(makePose(center: 0.5, vertical: dV), using: eng,
                                       time: 1, warmupFrames: 1) else {
                t.check(false, "poseB lift nil"); return
            }
            func hipY(_ pose: PoseResult) -> Float? {
                let joints = solver.solve(pose)
                let (body, _) = assembler.assemble(joints, mapper: mapper)
                return body.first { $0.slot == "1" }?.position.y
            }
            guard let yA = hipY(poseA), let yB = hipY(poseB) else {
                t.check(false, "no hip tracker for vertical check"); return
            }
            let dyHip = yB - yA
            print(String(format: "  [hip-trans] hipY a=%+.4f b=%+.4f  Δ=%+.4f  (input Δvert=%.2f)",
                         yA, yB, dyHip, dV))
            // Hip must RISE (positive), not be pinned to a constant floor.
            t.check(dyHip > 0.05,
                    "hip Y must rise when the body rises (floor frozen, not re-zeroed): Δ=\(dyHip)")
        }

        // ── 2c. Coherent vertical ordering survives the shift ───────────────
        t.test("HIP-TRANS: vertical ordering stays coherent after a shift (foot<knee<hip<chest)") {
            guard let s = assemble(center: 0.65, vertical: 0), let head = s.head else {
                t.check(false, "shifted assemble nil"); return
            }
            func y(_ slot: String) -> Float? { s.bySlot[slot]?.position.y }
            guard let footL = y("2"), let footR = y("3"),
                  let kneeL = y("5"), let kneeR = y("6"),
                  let hip = y("1"), let chest = y("4") else {
                t.check(false, "missing trackers after shift: \(s.bySlot.keys.sorted())"); return
            }
            let foot = min(footL, footR), knee = min(kneeL, kneeR)
            t.check(foot < knee, "foot.y (\(foot)) < knee.y (\(knee))")
            t.check(knee < hip,  "knee.y (\(knee)) < hip.y (\(hip))")
            t.check(hip < chest, "hip.y (\(hip)) < chest.y (\(chest))")
            t.check(chest < head.position.y,
                    "chest.y (\(chest)) < head.y (\(head.position.y))")
            // Feet near the floor (not underground / not floating).
            t.check(foot > -0.10 && foot < 0.20,
                    "feet must rest near the floor after a shift: \(foot)")
        }
    }
}
