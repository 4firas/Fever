import simd

/// FPS-MUX output upsampler — turns the sparse inference rate (~10–18 fps) into a
/// silky high-rate OSC stream that follows every movement without stair-stepping.
///
/// At 10 fps the real solved pose only updates every ~100 ms, so pure extrapolation
/// can't bridge the gap (it predicts partway, then freezes until the next frame →
/// the "still 10 fps" stutter). Instead this CRITICALLY-DAMPS the emitted joints
/// toward the latest pose every output tick and re-runs the byte-exact `PinoSolver`
/// IK on the smoothed joints:
///
///   • critically damped → it CANNOT overshoot → no rubberbanding;
///   • it always converges to the target → every real movement is followed, none
///     dropped (just eased — a smooth glide between the sparse poses);
///   • `smoothTime` scales with the inference period (passed in per tick) so the
///     follower is STILL gliding when the next frame arrives → continuous, never a
///     freeze-then-jump;
///   • damping the JOINTS (then re-solving) means rotations come from the exact IK,
///     never extrapolated euler → no angle-wrap / gimbal artifacts;
///   • an optional small `lead` (default 0) can extrapolate the target forward to
///     trade a little of the follow-delay back for responsiveness; with `lead = 0`
///     it is a pure smooth follower (silky, accepts the inherent delay).
///
/// On the first tick (and after `reset()`) the smoothed pose is SEEDED to the target,
/// so the very first output is byte-identical to the raw solver (1:1 preserved).
public final class PredictiveUpsampler {
    private let solver: PinoSolver
    /// Per-joint displacement clamp (meters) for the optional forward `lead` — a hard
    /// ceiling so a noisy velocity spike can never throw a limb.
    private let maxStep: Float

    // Output-damping state: the smoothed pose we ease toward the target each tick.
    private var dampJoints: [SIMD3<Float>]?
    private var dampVel: [SIMD3<Float>] = []

    public init(heightCm: Float, maxStep: Float = 0.18) {
        self.solver = PinoSolver(heightCm: heightCm)
        self.maxStep = maxStep
    }

    public func reset() { solver.reset(); dampJoints = nil; dampVel = [] }

    /// Unity-style SmoothDamp — a critically-damped spring toward `target` (NEVER
    /// overshoots). `vel` carries momentum across calls.
    @inline(__always)
    private static func smoothDamp(_ cur: Float, _ tgt: Float, _ vel: inout Float,
                                   _ st: Float, _ dt: Float) -> Float {
        let s = max(0.0001, st)
        let omega = 2 / s
        let x = omega * dt
        let expf = 1 / (1 + x + 0.48 * x * x + 0.235 * x * x * x)
        let change = cur - tgt
        let temp = (vel + omega * change) * dt
        vel = (vel - omega * temp) * expf
        return tgt + (change + temp) * expf
    }

    /// One output tick.
    /// - `joints`: the latest OneEuro-smoothed model joints (the follow target).
    /// - `velocity`: their smoothed velocity (only used when `lead > 0`).
    /// - `smoothTime`: damping time-constant (s) — scale to ~the inference period for
    ///   a continuous glide; bigger = silkier + more delay, 0 = no damping (snap).
    /// - `dt`: this output tick's interval (s).
    /// - `lead`: optional forward extrapolation horizon (s); 0 = pure follower.
    /// `heightCm`/`sendElbows` are read per tick so a live Settings change applies now.
    public func step(joints: [SIMD3<Float>], velocity: [SIMD3<Float>],
                     smoothTime: Float, dt: Float, lead: Float,
                     tracked: Bool, heightCm: Float, sendElbows: Bool)
        -> (body: [OSCTracker], head: OSCTracker?) {
        solver.setHeightCm(heightCm)

        // TARGET = latest joints, optionally extrapolated forward by `lead` (clamped).
        var target = joints
        if tracked, lead > 0, velocity.count == joints.count {
            for i in 0..<joints.count {
                var d = velocity[i] * lead
                let m = simd_length(d)
                if m > maxStep { d *= (maxStep / m) }
                target[i] = joints[i] + d
            }
        }

        // Critically-damp the emitted joints toward the target for a smooth glide.
        // Seed on first use / after reset so the first tick emits the target exactly.
        let solveJoints: [SIMD3<Float>]
        if tracked, smoothTime > 0, dt > 0 {
            if dampJoints?.count != target.count {
                dampJoints = target
                dampVel = [SIMD3<Float>](repeating: .zero, count: target.count)
            }
            var dj = dampJoints!
            for i in 0..<target.count {
                var v = dampVel[i]
                dj[i] = SIMD3<Float>(
                    Self.smoothDamp(dj[i].x, target[i].x, &v.x, smoothTime, dt),
                    Self.smoothDamp(dj[i].y, target[i].y, &v.y, smoothTime, dt),
                    Self.smoothDamp(dj[i].z, target[i].z, &v.z, smoothTime, dt))
                dampVel[i] = v
            }
            dampJoints = dj
            solveJoints = dj
        } else {
            dampJoints = target          // keep state coherent for a later re-enable
            dampVel = [SIMD3<Float>](repeating: .zero, count: target.count)
            solveJoints = target
        }

        let solved = solver.solve(joints: solveJoints, tracked: tracked)
        return assemblePinoBundle(solved, sendElbows: sendElbows)
    }
}

/// Build the OSC tracker bundle from a solved frame: numbered slots 1…8 (elbows
/// 3/4 only when `sendElbows`) plus the position-only head. Shared by the live UI
/// snapshot and the upsampler so both stay in lockstep with the slot map.
public func assemblePinoBundle(_ solved: SolvedFrame, sendElbows: Bool)
    -> (body: [OSCTracker], head: OSCTracker) {
    var body: [OSCTracker] = []
    body.reserveCapacity(TrackerMapPino.slots.count)
    for slot in TrackerMapPino.slots {
        if !sendElbows, slot.index == 3 || slot.index == 4 { continue }   // 6-point default
        let pos = solved.slotPositions[slot.index] ?? .zero
        let euler = solved.slotEulers[slot.index] ?? .zero
        body.append(OSCTracker(slot: slot.path, position: pos, eulerDegrees: euler))
    }
    let head = OSCTracker(slot: "head", position: solved.headPosition, eulerDegrees: .zero)
    return (body, head)
}
