import simd

/// Per-joint hold-last store for the OSC ROTATION path.
///
/// `frameFromTwoAxes` returns "hold-last" when the two in-body axes are
/// (near-)parallel and the frame is degenerate (rather than fabricating a roll
/// from world-up, the old `quaternionFromBone` singularity). The JointSolver is
/// a value type rebuilt each run, so the last good per-joint LIVE quaternion is
/// kept here in a reference type that survives across frames. Touched ONLY from
/// the single serial inference worker (it is injected into the solver the worker
/// owns), so it needs no internal locking.
public final class RotationState {
    private var last: [JointType: simd_quatf] = [:]

    public init() {}

    /// The last good live (absolute, solver-frame) quaternion for `joint`, or the
    /// identity if none has been recorded yet (so the very first frame's
    /// degenerate joints sit at neutral instead of NaN).
    public func holdLast(_ joint: JointType) -> simd_quatf {
        last[joint] ?? simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
    }

    /// Record this frame's good live quaternion for `joint`.
    public func store(_ joint: JointType, _ q: simd_quatf) {
        last[joint] = q
    }

    /// Drop all held orientations (Recenter / run reset).
    public func reset() { last.removeAll(keepingCapacity: true) }
}

/// REST-RELATIVE rotation rebaser for the OSC path (PinoFBT behaviour).
///
/// The solver emits ABSOLUTE world (solver-frame) orientations. PinoFBT instead
/// streams the DELTA from a calibration rest pose, which is what keeps its wire
/// values bounded and zero-centered (euler ≈ 0 at rest) instead of large and
/// wrapping through ±180°. This type:
///
///   1. captures `qRest[joint] = qLive[joint]` on calibrate()/Recenter (the same
///      T/I-pose Recenter that re-latches scale + floor), and
///   2. every frame emits `qDelta = inverse(qRest[joint]) * qLive[joint]`.
///
/// Before the captured rest, it falls back to identity rest (so qDelta == qLive,
/// i.e. absolute) — tracking still works before the first Recenter, just not
/// zero-centered. After the delta it does double-cover HEMISPHERE-LOCK against
/// the previous emitted delta (flip sign if dot < 0) and SLERP smoothing toward
/// it, so the stream is continuous and jitter-free.
///
/// Confined to the single serial inference worker (owned by FrameProcessor), so
/// no internal locking is required.
public final class RotationRebaser {
    private var qRest: [JointType: simd_quatf] = [:]
    private var qPrev: [JointType: simd_quatf] = [:]
    /// 1 = raw delta, 0 = frozen on the previous emitted delta.
    public var smoothingFactor: Float
    /// Set on Recenter; consumed on the next processed frame to latch the rest
    /// pose from the live orientations of THAT frame (the user's standing pose).
    private var capturePending = false

    public init(smoothingFactor: Float = 0.5) {
        self.smoothingFactor = smoothingFactor
    }

    /// Request a one-shot rest-pose capture on the next `rebase` call. Driven by
    /// the SAME Recenter that re-latches scale/floor, so one T-pose Recenter does
    /// scale + floor + rest-rotation together.
    public func requestRestCapture() { capturePending = true }

    /// Drop the rest pose + smoothing history (full run reset). After this the
    /// rebaser falls back to identity rest (absolute) until the next capture.
    public func reset() {
        qRest.removeAll(keepingCapacity: true)
        qPrev.removeAll(keepingCapacity: true)
        capturePending = false
    }

    /// Rebase one joint's ABSOLUTE live (solver-frame) quaternion into the
    /// rest-relative, hemisphere-locked, SLERP-smoothed delta to put on the wire.
    /// `captureNow` (true on the Recenter frame) latches the rest pose first.
    public func rebase(_ joint: JointType, live: simd_quatf, captureNow: Bool) -> simd_quatf {
        if captureNow {
            qRest[joint] = live
            // Drop the smoothing history so the capture frame emits a CLEAN
            // identity delta. Otherwise a stale qPrev (from before this Recenter)
            // would SLERP the rest pose toward the previous pose's delta, leaking
            // a transient non-zero rotation onto every body tracker for several
            // frames after every in-session Recenter (the avatar visibly settles
            // instead of snapping to the calibrated rest). With qPrev cleared the
            // `if let prev` branch below is skipped on the capture frame and the
            // emitted delta is exactly inverse(live)·live = identity.
            qPrev[joint] = nil
        }

        // qDelta = inverse(qRest) * qLive. No rest yet → identity rest (absolute).
        let rest = qRest[joint] ?? simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
        var qDelta = simd_normalize(rest.inverse * live)

        // Double-cover hemisphere-lock against the previous emitted delta, then
        // SLERP toward it for temporal continuity (no ±q flip discontinuity).
        if let prev = qPrev[joint] {
            if simd_dot(qDelta.vector, prev.vector) < 0 {
                qDelta = simd_quatf(vector: -qDelta.vector)
            }
            let t = 1.0 - smoothingFactor   // higher smoothing → hold more of prev
            qDelta = safeSlerp(prev, qDelta, t)
        }
        qPrev[joint] = qDelta
        return qDelta
    }

    /// Whether a rest-capture is pending (and clears it). Called once per frame by
    /// the processor so capture happens exactly on the post-Recenter frame.
    public func consumeCapturePending() -> Bool {
        guard capturePending else { return false }
        capturePending = false
        return true
    }
}
