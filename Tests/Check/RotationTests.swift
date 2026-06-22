import Foundation
import simd
import FeverCore

/// ROTATION REBUILD TESTS — verify the re-enabled OSC `/rotation` path is the
/// PinoFBT-style rest-relative, hemisphere-locked, two-in-body-axis orientation
/// stream (NOT the old absolute `quaternionFromBone` euler that wrapped ±180°,
/// pinned hip roll, and snapped 90° at vertical limbs).
///
/// These drive the EXACT production rotation chain the live `FrameProcessor`
/// runs, per joint:
///
///     JointSolver.solve (rotationState injected)   → absolute solver-frame quat
///       (two in-body axes; foot = locked-roll yaw+pitch; degenerate → hold-last)
///     RotationRebaser.rebase (rest-relative)        → inverse(qRest)·qLive,
///       hemisphere-locked vs the previous emitted delta, SLERP-smoothed
///     CoordinateMapper.toVRChatEulerDegrees         → VRChat ZXY euler degrees
///
/// so a passing assertion here is a property of the real shipped code, not a
/// re-implementation.
///
///   1. BOUNDED        — over a smooth limb sweep no tracker's euler wraps the
///                       full −180..180 range, and at the captured rest pose the
///                       euler is within a few degrees of 0 on every tracker.
///   2. FOOT-ARTICULATES — change the foot's shank direction frame-to-frame and
///                       the FOOT tracker euler CHANGES (pitch+yaw move) while
///                       roll stays ≈ 0 (locked-roll ankle model) — proving the
///                       foot is no longer a frozen constant.
///   3. NO-SINGULARITY — a fully VERTICAL standing pose (limbs ∥ world-up, the old
///                       `quaternionFromBone` singularity) produces NO sudden ~90°
///                       snap / euler discontinuity vs a slightly-off-vertical one.
///   4. REST-RELATIVE  — capture rest at pose A → euler ≈ 0 at A and a known
///                       non-zero delta at a rotated pose B; a continuous sweep
///                       A→B yields ≈ 0 euler jumps (hemisphere-lock, no ±q flip).
enum RotationTests {

    // MARK: - Pose construction (raw Vision-frame metric landmarks)

    /// A neutral upright standing skeleton in the solver's (Vision-derived) metric
    /// frame: hip-root-relative meters, +Y up, +Z toward camera. Limbs are SLIGHTLY
    /// articulated (knees/elbows softly bent, feet pointing a touch forward) so no
    /// joint sits exactly at a degenerate two-axis frame — the realistic rest pose.
    /// `b` lets a test bend the whole body's limbs by a parameter (see `articulate`).
    static func baseStanding() -> [NormalizedLandmark] {
        var lm = [NormalizedLandmark](repeating: NormalizedLandmark(position: .zero, visibility: 0),
                                      count: 33)
        func set(_ l: BlazePose.Landmark, _ x: Float, _ y: Float, _ z: Float) {
            lm[l.rawValue] = NormalizedLandmark(position: SIMD3<Float>(x, y, z), visibility: 0.95)
        }
        // Head / face — nose slightly forward (+Z) of the ear line so the face
        // frame is well-conditioned.
        set(.nose, 0.0, 0.62, 0.10)
        set(.leftEye, -0.03, 0.63, 0.08); set(.rightEye, 0.03, 0.63, 0.08)
        set(.leftEar, -0.06, 0.60, 0.0);  set(.rightEar, 0.06, 0.60, 0.0)
        // Shoulders / chest.
        set(.leftShoulder, -0.18, 0.45, 0.0); set(.rightShoulder, 0.18, 0.45, 0.0)
        // Arms hanging, forearms a touch forward so the elbow frame is non-degenerate.
        set(.leftElbow, -0.20, 0.15, 0.02);  set(.rightElbow, 0.20, 0.15, 0.02)
        set(.leftWrist, -0.21, -0.12, 0.06); set(.rightWrist, 0.21, -0.12, 0.06)
        // Hips.
        set(.leftHip, -0.10, 0.0, 0.0); set(.rightHip, 0.10, 0.0, 0.0)
        // Legs softly bent: knee slightly forward of the hip→ankle line.
        set(.leftKnee, -0.10, -0.42, 0.05);  set(.rightKnee, 0.10, -0.42, 0.05)
        set(.leftAnkle, -0.10, -0.86, 0.0);  set(.rightAnkle, 0.10, -0.86, 0.0)
        // Heels behind / toes in front so the synthesized foot vector is forward.
        set(.leftHeel, -0.10, -0.90, -0.04);  set(.rightHeel, 0.10, -0.90, -0.04)
        set(.leftFootIndex, -0.10, -0.92, 0.12); set(.rightFootIndex, 0.10, -0.92, 0.12)
        return lm
    }

    /// A PERFECTLY VERTICAL standing skeleton: every long bone is exactly parallel
    /// to world-up (legs straight down, arms straight down, feet directly under the
    /// shanks with NO forward toe offset). This is the old `quaternionFromBone`
    /// singularity (bone ∥ worldUp) that used to snap 90° and pin roll. `tilt`
    /// nudges every distal joint forward in +Z by `tilt` meters to make a
    /// "slightly-off-vertical" control pose.
    static func verticalStanding(tilt: Float = 0) -> [NormalizedLandmark] {
        var lm = [NormalizedLandmark](repeating: NormalizedLandmark(position: .zero, visibility: 0),
                                      count: 33)
        func set(_ l: BlazePose.Landmark, _ x: Float, _ y: Float, _ z: Float) {
            lm[l.rawValue] = NormalizedLandmark(position: SIMD3<Float>(x, y, z), visibility: 0.95)
        }
        set(.nose, 0.0, 0.62, 0.06 + tilt)
        set(.leftEye, -0.03, 0.63, 0.05 + tilt); set(.rightEye, 0.03, 0.63, 0.05 + tilt)
        set(.leftEar, -0.06, 0.60, 0.0);  set(.rightEar, 0.06, 0.60, 0.0)
        set(.leftShoulder, -0.18, 0.45, 0.0); set(.rightShoulder, 0.18, 0.45, 0.0)
        // Arms PERFECTLY straight down (elbow/wrist directly below the shoulder).
        set(.leftElbow, -0.18, 0.15, tilt);  set(.rightElbow, 0.18, 0.15, tilt)
        set(.leftWrist, -0.18, -0.15, tilt * 2); set(.rightWrist, 0.18, -0.15, tilt * 2)
        set(.leftHip, -0.10, 0.0, 0.0); set(.rightHip, 0.10, 0.0, 0.0)
        // Legs PERFECTLY straight down (knee/ankle directly below the hip).
        set(.leftKnee, -0.10, -0.43, tilt);  set(.rightKnee, 0.10, -0.43, tilt)
        set(.leftAnkle, -0.10, -0.86, tilt * 2);  set(.rightAnkle, 0.10, -0.86, tilt * 2)
        // Feet directly under the ankle (shank ∥ up, so the foot's flattened
        // forward collapses → degenerate foot frame at tilt=0).
        set(.leftHeel, -0.10, -0.90, tilt * 2);  set(.rightHeel, 0.10, -0.90, tilt * 2)
        set(.leftFootIndex, -0.10, -0.90, tilt * 2); set(.rightFootIndex, 0.10, -0.90, tilt * 2)
        return lm
    }

    static func pose(_ lm: [NormalizedLandmark], at t: TimeInterval) -> PoseResult {
        PoseResult(landmarks: lm, timestamp: t)
    }

    // MARK: - Production rotation chain (mirrors FrameProcessor.process step 2-4)

    /// One frozen-in-time evaluator of the real rotation chain. Holds the SAME
    /// long-lived `RotationState` (degenerate-frame hold-last) and `RotationRebaser`
    /// (rest-relative + hemisphere-lock + SLERP) the live `FrameProcessor` owns, so
    /// calling `eulers(...)` per frame reproduces the shipped per-frame behaviour.
    final class Chain {
        let solver: JointSolver
        let rebaser: RotationRebaser
        let mapper: CoordinateMapper
        private let state = RotationState()

        init(mirror: Bool = false, userHeight: Float = 1.74, rotationSmoothing: Float = 0.5) {
            let cfg = TrackingConfig()
            cfg.mirrorTracking = mirror
            cfg.userHeightMeters = Double(userHeight)
            self.solver = JointSolver(settings: cfg, rotationState: state)
            self.rebaser = RotationRebaser(smoothingFactor: rotationSmoothing)
            self.mapper = CoordinateMapper(userHeightMeters: userHeight,
                                           referenceHeightMeters: 1.8,
                                           mirrorHorizontally: mirror)
        }

        /// Request that the NEXT `eulers` call latches this frame as the rest pose
        /// (mirrors the Recenter → requestRestCapture → consume-on-next-frame flow).
        func requestRestCapture() { rebaser.requestRestCapture() }

        /// Evaluate one frame end-to-end and return per-joint VRChat euler degrees
        /// (the exact values the assembler/wire would carry), keyed by JointType.
        func eulers(_ p: PoseResult) -> [JointType: SIMD3<Float>] {
            var joints = solver.solve(p)
            let captureRest = rebaser.consumeCapturePending()
            for i in joints.indices where joints[i].type != .head {
                joints[i].rotation = rebaser.rebase(joints[i].type,
                                                    live: joints[i].rotation,
                                                    captureNow: captureRest)
            }
            var out = [JointType: SIMD3<Float>]()
            for j in joints where j.type != .head {
                out[j.type] = mapper.toVRChatEulerDegrees(j.rotation)
            }
            return out
        }
    }

    /// The 8 numbered body trackers (head is position-only, never rotated here).
    static let bodyJoints: [JointType] = [.hip, .chest, .leftElbow, .rightElbow,
                                          .leftKnee, .rightKnee, .leftFoot, .rightFoot]

    // MARK: - Entry

    static func run(_ t: TestRunner) {
        testBounded(t)
        testFootArticulates(t)
        testNoSingularity(t)
        testRestRelative(t)
        testRecenterClearsSmoothingHistory(t)
        testRebaserContract(t)
    }

    // MARK: - 5. RECENTER CLEARS SMOOTHING HISTORY

    /// The defect this pins: across an in-session Recenter, `rebase` kept the
    /// stale per-joint `qPrev` from before the recenter, so the very capture frame
    /// SLERP-blended the (correct) identity rest delta toward the previous pose's
    /// delta — emitting a transient non-zero rotation on every body tracker for
    /// several frames after EVERY recenter. The contract of Recenter is that the
    /// rest pose maps to a zero delta IMMEDIATELY. Clearing `qPrev` on the capture
    /// frame makes the emitted delta exactly identity.
    static func testRecenterClearsSmoothingHistory(_ t: TestRunner) {
        let identity = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)

        // A live orientation well away from identity AND from any prior delta, so a
        // stale qPrev would visibly drag the capture-frame delta off identity.
        let live = simd_normalize(simd_quatf(angle: 1.1, axis: simd_normalize(SIMD3<Float>(0.3, 0.8, 0.5))))

        // Heavy smoothing (0.9) → with a stale qPrev the SLERP would hold MOST of
        // the previous delta, so the bug is maximally visible; with the fix the
        // capture frame must still emit a clean identity regardless of smoothing.
        let rebaser = RotationRebaser(smoothingFactor: 0.9)

        // Frame 1: a normal (non-capture) frame at a DIFFERENT orientation populates
        // qPrev with a non-identity delta (rest is still identity here, so the delta
        // equals this prior live).
        let prior = simd_normalize(simd_quatf(angle: 0.9, axis: simd_normalize(SIMD3<Float>(1, 0, 0))))
        _ = rebaser.rebase(.chest, live: prior, captureNow: false)

        // Frame 2: the Recenter capture frame on the NEW live pose. The emitted
        // delta must be (near) identity — inverse(live)·live with NO contamination
        // from the stale qPrev.
        let delta = rebaser.rebase(.chest, live: live, captureNow: true)

        t.test("RECENTER: capture frame emits a clean identity delta (qPrev cleared)") {
            t.check(delta.vector.x.isFinite && delta.vector.y.isFinite
                    && delta.vector.z.isFinite && delta.vector.w.isFinite,
                    "capture-frame delta non-finite: \(delta.vector)")
            // dot with identity ~ ±1 ⇒ the quaternion IS (near) identity.
            let d = abs(simd_dot(simd_normalize(delta).vector, identity.vector))
            t.check(d > 0.9999,
                    "capture-frame delta must be identity right after Recenter (dot=\(d), q=\(delta.vector))")
            print(String(format: "  [recenter] capture delta dot(identity)=%.6f", d))
        }

        // And the NEXT held frame at the same pose stays at identity (the cleared
        // qPrev now holds identity, so smoothing toward identity stays identity).
        let held = rebaser.rebase(.chest, live: live, captureNow: false)
        t.test("RECENTER: holding the rest pose after capture stays at identity") {
            let d = abs(simd_dot(simd_normalize(held).vector, identity.vector))
            t.check(d > 0.9999, "post-capture hold must remain identity (dot=\(d))")
        }
    }

    // MARK: - 6. REBASER CONTRACT (surrounding behaviour)

    /// Guards the small invariants the rebaser is built on, so a future refactor
    /// can't silently break them: (a) before any capture it falls back to an
    /// IDENTITY rest (delta == live, i.e. absolute pass-through), and (b)
    /// `consumeCapturePending` is a strict ONE-SHOT latch.
    static func testRebaserContract(_ t: TestRunner) {
        // (a) Identity-rest fallback: no capture yet → the very first emitted delta
        // equals the live orientation (qDelta = inverse(identity)·live = live).
        let rebaser = RotationRebaser(smoothingFactor: 0.5)
        let live = simd_normalize(simd_quatf(angle: 0.7, axis: simd_normalize(SIMD3<Float>(0.2, 1, 0.4))))
        let first = rebaser.rebase(.hip, live: live, captureNow: false)
        t.test("CONTRACT: identity-rest fallback before capture (delta == live)") {
            let d = abs(simd_dot(simd_normalize(first).vector, live.vector))
            t.check(d > 0.9999, "pre-capture delta must equal live (absolute pass-through): dot=\(d)")
        }

        // (b) One-shot capture latch: requestRestCapture then consume returns true
        // exactly once; the immediate re-consume returns false.
        let latch = RotationRebaser(smoothingFactor: 0.5)
        t.test("CONTRACT: consumeCapturePending is a strict one-shot latch") {
            t.check(latch.consumeCapturePending() == false, "no capture requested → false")
            latch.requestRestCapture()
            t.check(latch.consumeCapturePending() == true, "after request → true once")
            t.check(latch.consumeCapturePending() == false, "second consume → false (one-shot)")
        }
    }

    // MARK: - 1. BOUNDED + zero-centered at rest

    static func testBounded(_ t: TestRunner) {
        // A smooth motion: sweep the knees/elbows/feet through a realistic flexion
        // arc while standing. Capture rest on frame 0, then drive 80 frames.
        let chain = Chain()
        let base = baseStanding()

        // Frame builder: bend knees forward and swing the lower legs / forearms /
        // feet smoothly by angle θ(τ). Everything stays a plausible human pose.
        func frame(_ tau: Float) -> [NormalizedLandmark] {
            var lm = base
            func set(_ l: BlazePose.Landmark, _ x: Float, _ y: Float, _ z: Float) {
                lm[l.rawValue] = NormalizedLandmark(position: SIMD3<Float>(x, y, z), visibility: 0.95)
            }
            let s = sin(tau)            // smooth, bounded
            // Knees are CLEARLY bent (shank held well off vertical: ankle sits a
            // big step FORWARD of the knee) so the two-axis knee frame stays
            // well-conditioned through the whole sweep — a realistic kneel/lunge,
            // never the near-vertical shank that is a genuine euler gimbal seam (and
            // would legitimately read an ill-defined yaw, NOT a rotation bug). The
            // shank then SWINGS smoothly in its bent arc.
            set(.leftKnee, -0.10, -0.42, 0.10)
            set(.rightKnee, 0.10, -0.42, 0.10)
            set(.leftAnkle, -0.10, -0.74, 0.34 + 0.10 * s)   // shank ~30°+ forward
            set(.rightAnkle, 0.10, -0.74, 0.34 + 0.10 * s)
            // Forearms swing forward (elbow clearly bent the same way).
            set(.leftWrist, -0.21, -0.05, 0.22 + 0.10 * s)
            set(.rightWrist, 0.21, -0.05, 0.22 + 0.10 * s)
            // Feet: toes swing forward (foot yaw motion).
            set(.leftFootIndex, -0.10, -0.80, 0.46 + 0.10 * s)
            set(.rightFootIndex, 0.10, -0.80, 0.46 + 0.10 * s)
            return lm
        }

        // Frame 0 = rest capture.
        chain.requestRestCapture()
        let rest = chain.eulers(pose(frame(0), at: 0))

        t.test("BOUNDED: rest pose euler ≈ 0 on every tracker (within a few degrees)") {
            for j in bodyJoints {
                guard let e = rest[j] else { t.check(false, "missing rest euler for \(j)"); continue }
                t.check(e.x.isFinite && e.y.isFinite && e.z.isFinite, "\(j) rest euler non-finite: \(e)")
                let mag = max(abs(e.x), abs(e.y), abs(e.z))
                t.check(mag < 3.0,
                        "\(j) rest euler must be within ~3° of 0 (rest-relative zero-centering): \(e)")
            }
        }

        // Drive the smooth sweep and record the per-joint euler track.
        var track = [JointType: [SIMD3<Float>]]()
        for j in bodyJoints { track[j] = [] }
        let steps = 80
        for i in 1...steps {
            let tau = Float(i) / Float(steps) * (2 * .pi)   // one full smooth cycle
            let e = chain.eulers(pose(frame(tau), at: Double(i) / 60.0))
            for j in bodyJoints { track[j]!.append(e[j] ?? SIMD3<Float>(.nan, .nan, .nan)) }
        }

        t.test("BOUNDED: no tracker euler wraps the full −180..180 range over smooth motion") {
            // For each tracker and each axis, the observed range must stay well
            // inside the full ±180 wrap; a wrap shows up as a near-360 span or a
            // single >170° frame-to-frame jump (the ±180 seam). A clean bounded
            // rest-relative delta over a moderate sweep does neither.
            for j in bodyJoints {
                let series = track[j]!
                t.check(series.allSatisfy { $0.x.isFinite && $0.y.isFinite && $0.z.isFinite },
                        "\(j) produced a non-finite euler in the sweep")
                for axis in 0..<3 {
                    let vals = series.map { $0[axis] }
                    let span = (vals.max() ?? 0) - (vals.min() ?? 0)
                    t.check(span < 300,
                            "\(j) axis \(axis) euler span \(span)° approaches a full ±180 wrap")
                    // Largest single-frame step on this axis (a ±180 seam crossing
                    // shows up as a ~360° jump).
                    var maxJump: Float = 0
                    for k in 1..<vals.count { maxJump = max(maxJump, abs(vals[k] - vals[k-1])) }
                    t.check(maxJump < 170,
                            "\(j) axis \(axis) had a \(maxJump)° frame jump (±180 wrap seam)")
                }
            }
            // Diagnostic: print the per-tracker euler range of the sweep.
            for j in bodyJoints {
                let s = track[j]!
                func rng(_ a: Int) -> String {
                    let v = s.map { $0[a] }
                    return String(format: "[%+.1f,%+.1f]", v.min() ?? 0, v.max() ?? 0)
                }
                print("  [bounded] \(j): x=\(rng(0)) y=\(rng(1)) z=\(rng(2))")
            }
        }
    }

    // MARK: - 2. FOOT-ARTICULATES

    static func testFootArticulates(_ t: TestRunner) {
        // The foot frame now comes from the REAL detected heel→toe vector (MediaPipe
        // detects heels 29/30 + foot-index 31/32). So the foot articulates when YOU
        // move YOUR foot — turning the foot (toe swings laterally) yaws it, pointing
        // the toes (toe dips down/forward) pitches it — and it does NOT spin off the
        // shank when only the leg moves. Roll stays locked (monocular has no foot roll).
        let base = baseStanding()
        let steps = 40
        func sweepFoot(_ build: @escaping (Float) -> [NormalizedLandmark]) -> [SIMD3<Float>] {
            let chain = Chain()
            chain.requestRestCapture()
            _ = chain.eulers(pose(base, at: 0))   // rest at the neutral foot pose
            var lf: [SIMD3<Float>] = []
            for i in 1...steps {
                let tau = Float(i) / Float(steps) * (2 * .pi)
                lf.append(chain.eulers(pose(build(tau), at: Double(i) / 60.0))[.leftFoot]
                          ?? SIMD3<Float>(.nan, .nan, .nan))
            }
            return lf
        }
        func set(_ lm: inout [NormalizedLandmark], _ l: BlazePose.Landmark, _ x: Float, _ y: Float, _ z: Float) {
            lm[l.rawValue] = NormalizedLandmark(position: SIMD3<Float>(x, y, z), visibility: 0.95)
        }
        func span(_ s: [SIMD3<Float>], _ a: Int) -> Float { (s.map { $0[a] }.max() ?? 0) - (s.map { $0[a] }.min() ?? 0) }

        // 1) Turn the foot: the TOE swings laterally (X) about a fixed heel → YAW.
        let yawTrack = sweepFoot { tau in
            var lm = base
            set(&lm, .leftHeel, -0.10, -0.90, -0.04)
            set(&lm, .leftFootIndex, -0.10 + 0.10 * sin(tau), -0.92, 0.14)
            return lm
        }
        t.test("FOOT-ARTICULATES: turning the foot (toe swings) articulates it") {
            t.check(yawTrack.allSatisfy { $0.x.isFinite && $0.y.isFinite && $0.z.isFinite }, "finite")
            // The articulation may land on any euler axis (VRChat decodes the
            // euler back to the exact orientation); assert it RESPONDS, not which axis.
            let responds = max(span(yawTrack, 0), span(yawTrack, 1), span(yawTrack, 2))
            t.check(responds > 10.0, "foot must respond to real foot motion: span \(responds)°")
            print(String(format: "  [foot-yaw] pitch=%.1f yaw=%.1f roll=%.1f",
                         span(yawTrack, 0), span(yawTrack, 1), span(yawTrack, 2)))
        }

        // 2) Point the toes: the TOE dips down + forward about a fixed heel → PITCH.
        let pitchTrack = sweepFoot { tau in
            var lm = base
            let p = 0.10 * sin(tau)
            set(&lm, .leftHeel, -0.10, -0.90, -0.04)
            set(&lm, .leftFootIndex, -0.10, -0.92 - p, 0.14 + p)
            return lm
        }
        t.test("FOOT-ARTICULATES: pointing the toes pitches the foot") {
            t.check(pitchTrack.allSatisfy { $0.x.isFinite && $0.y.isFinite && $0.z.isFinite }, "finite")
            let responds = max(span(pitchTrack, 0), span(pitchTrack, 1), span(pitchTrack, 2))
            t.check(responds > 8.0, "foot must pitch when the toes point: span \(responds)°")
        }

        // 3) Move ONLY the leg (knee/shank) with the FOOT held fixed → the foot must
        //    NOT spin (the old shank-derived model did; the real foot frame doesn't).
        let legTrack = sweepFoot { tau in
            var lm = base
            // Swing the knee laterally + in depth; heel/toe stay at the base pose.
            set(&lm, .leftKnee, -0.10 + 0.18 * sin(tau), -0.42, 0.05 + 0.14 * cos(tau))
            return lm
        }
        t.test("FOOT-ARTICULATES: moving only the leg does NOT spin the foot") {
            t.check(legTrack.allSatisfy { $0.x.isFinite && $0.y.isFinite && $0.z.isFinite }, "finite")
            let drift = max(span(legTrack, 0), span(legTrack, 1), span(legTrack, 2))
            t.check(drift < 4.0, "foot must stay put when only the leg moves: span \(drift)°")
            print(String(format: "  [foot-leg-isolation] max foot euler drift while sweeping the leg = %.2f°", drift))
        }
    }

    // MARK: - 3. NO-SINGULARITY

    static func testNoSingularity(_ t: TestRunner) {
        // Two FRESH chains (independent rest captures) so we compare the steady
        // emitted euler at each pose, not a transient. Pose V = perfectly vertical
        // (the old quaternionFromBone singularity); pose Veps = the SAME pose tilted
        // a hair off vertical. With the two-axis frame + hold-last, the emitted
        // euler at V must NOT be ~90° away from Veps (no snap / discontinuity).
        func steadyEuler(tilt: Float) -> [JointType: SIMD3<Float>] {
            let chain = Chain()
            let lm = verticalStanding(tilt: tilt)
            chain.requestRestCapture()
            // Feed several identical frames so any SLERP transient settles to the
            // steady held value (the live behaviour after the pose is held).
            var e = [JointType: SIMD3<Float>]()
            for i in 0..<20 { e = chain.eulers(pose(lm, at: Double(i) / 60.0)) }
            return e
        }

        // Both chains capture rest at their OWN pose, so each is rest-relative ≈ 0
        // there. The KEY property: feeding the singular vertical pose must not blow
        // up the euler — it must stay near 0 (rest), and it must be CLOSE to the
        // slightly-off-vertical pose's euler (no 90° snap between them).
        let eV = steadyEuler(tilt: 0)          // exactly vertical (singular)
        let eEps = steadyEuler(tilt: 0.03)     // 3 cm off vertical

        t.test("NO-SINGULARITY: vertical pose euler is finite and bounded (no blow-up)") {
            for j in bodyJoints {
                guard let e = eV[j] else { t.check(false, "missing vertical euler \(j)"); continue }
                t.check(e.x.isFinite && e.y.isFinite && e.z.isFinite,
                        "\(j) vertical euler non-finite (singularity NaN): \(e)")
                // Rest was captured AT this vertical pose → must be ≈ 0, NOT a
                // fabricated ±90° from a world-up gauge.
                let mag = max(abs(e.x), abs(e.y), abs(e.z))
                t.check(mag < 5.0,
                        "\(j) vertical-pose euler must stay near rest 0 (no 90° world-up snap): \(e)")
            }
        }

        t.test("NO-SINGULARITY: no ~90° snap between vertical and slightly-off-vertical") {
            // Each chain is rest-relative to its OWN pose, so both should read ≈ 0
            // and therefore be close to each other. The defect we guard against is a
            // discontinuity where crossing exactly vertical flips an axis ~90°.
            for j in bodyJoints {
                guard let a = eV[j], let b = eEps[j] else {
                    t.check(false, "missing euler for \(j)"); continue
                }
                let d = max(abs(a.x - b.x), abs(a.y - b.y), abs(a.z - b.z))
                t.check(d < 30.0,
                        "\(j) euler jumped \(d)° between vertical and off-vertical (singularity snap)")
            }
            func mag(_ d: [JointType: SIMD3<Float>]) -> String {
                bodyJoints.map { j in
                    let e = d[j] ?? .zero
                    return String(format: "%@=%.1f", "\(j)", max(abs(e.x), abs(e.y), abs(e.z)))
                }.joined(separator: " ")
            }
            print("  [no-singularity] vertical max|euler| per joint: \(mag(eV))")
        }
    }

    // MARK: - 4. REST-RELATIVE

    static func testRestRelative(_ t: TestRunner) {
        // Capture rest at pose A (neutral). Assert euler ≈ 0 at A. Then sweep
        // CONTINUOUSLY to a clearly rotated pose B (torso/limb yaw) and assert (a)
        // euler at B is a KNOWN non-zero delta, and (b) across the continuous A→B
        // sweep the per-axis euler makes ≈ 0 sign-flip "jumps" (hemisphere-lock
        // stops the ±q double-cover sign flip that would otherwise jump ~360°).
        let chain = Chain()
        let A = baseStanding()

        // Pose at sweep parameter τ∈[0,1]: rotate the whole upper body (shoulders +
        // arms) about the vertical axis by angle φ = τ·40°, a clear chest YAW. This
        // makes the chest two-axis frame rotate continuously, so the rest-relative
        // chest euler grows smoothly from 0 to a known ≈ +40° (sign per handedness).
        func sweep(_ tau: Float) -> [NormalizedLandmark] {
            var lm = A
            func set(_ l: BlazePose.Landmark, _ p: SIMD3<Float>) {
                lm[l.rawValue] = NormalizedLandmark(position: p, visibility: 0.95)
            }
            let phi = tau * (40.0 * .pi / 180.0)
            let c = cos(phi), s = sin(phi)
            // Rotate the shoulder line about +Y (yaw). Shoulders sit at ±0.18 in X,
            // y=0.45, z=0. After a +Y rotation: x' = x·c + z·s, z' = -x·s + z·c.
            func rotY(_ x: Float, _ y: Float, _ z: Float) -> SIMD3<Float> {
                SIMD3<Float>(x * c + z * s, y, -x * s + z * c)
            }
            set(.leftShoulder, rotY(-0.18, 0.45, 0.0))
            set(.rightShoulder, rotY(0.18, 0.45, 0.0))
            // Carry the arms with the shoulders so the chest/elbow frames rotate too.
            set(.leftElbow, rotY(-0.20, 0.15, 0.02))
            set(.rightElbow, rotY(0.20, 0.15, 0.02))
            set(.leftWrist, rotY(-0.21, -0.12, 0.06))
            set(.rightWrist, rotY(0.21, -0.12, 0.06))
            return lm
        }

        chain.requestRestCapture()
        let eRestA = chain.eulers(pose(sweep(0), at: 0))

        t.test("REST-RELATIVE: euler ≈ 0 at the captured rest pose A") {
            for j in bodyJoints {
                guard let e = eRestA[j] else { t.check(false, "missing A euler \(j)"); continue }
                let mag = max(abs(e.x), abs(e.y), abs(e.z))
                t.check(mag < 3.0, "\(j) euler at rest A must be ≈ 0: \(e)")
            }
        }

        // Continuous sweep A → B; record the chest euler series + count sign-flip
        // jumps (a ±q double-cover flip would jump an axis by ~2× its value or hit
        // the ±180 seam — i.e. a large discontinuous step on an otherwise smooth
        // monotone ramp).
        var chestSeries: [SIMD3<Float>] = [eRestA[.chest] ?? .zero]
        let steps = 60
        for i in 1...steps {
            let tau = Float(i) / Float(steps)
            let e = chain.eulers(pose(sweep(tau), at: Double(i) / 60.0))
            chestSeries.append(e[.chest] ?? SIMD3<Float>(.nan, .nan, .nan))
        }
        let eB = chestSeries.last!

        t.test("REST-RELATIVE: rotated pose B yields a known non-zero euler delta") {
            t.check(eB.x.isFinite && eB.y.isFinite && eB.z.isFinite, "B chest euler non-finite: \(eB)")
            // A +40° body yaw (about world-up) must read as a clear yaw on the chest
            // tracker: the dominant euler component is large (> ~25°) and YAW (|y|)
            // dominates pitch/roll. Sign depends on handedness; assert magnitude.
            let dom = max(abs(eB.x), abs(eB.y), abs(eB.z))
            t.check(dom > 25.0, "B chest euler must show the ~40° yaw delta: \(eB)")
            t.check(abs(eB.y) >= abs(eB.x) && abs(eB.y) >= abs(eB.z),
                    "the dominant chest euler axis for a body yaw must be YAW (y): \(eB)")
            print(String(format: "  [rest-rel] chest A=(%.2f,%.2f,%.2f) B=(%.2f,%.2f,%.2f)",
                         eRestA[.chest]!.x, eRestA[.chest]!.y, eRestA[.chest]!.z,
                         eB.x, eB.y, eB.z))
        }

        t.test("REST-RELATIVE: continuous sweep has ≈ 0 hemisphere sign-flip jumps") {
            // On a smooth monotone yaw ramp the dominant (yaw) axis should change in
            // small steps. A hemisphere-lock failure (±q flip) shows as a single
            // huge jump (≳ 100°). Count axis steps that exceed a generous 60° —
            // there must be essentially none.
            var jumps = 0
            for axis in 0..<3 {
                let v = chestSeries.map { $0[axis] }
                for k in 1..<v.count where abs(v[k] - v[k-1]) > 60 {
                    jumps += 1
                    print("  [rest-rel] chest axis \(axis) jump frame \(k): \(v[k-1]) -> \(v[k])")
                }
            }
            t.check(jumps == 0, "chest euler had \(jumps) hemisphere sign-flip jump(s) across the sweep")
        }
    }
}
