import simd
import Foundation

/// Owns the per-session leveling datum and the smoothed live leveling rotation —
/// the stateful half of the "Body Stabilizer" feature (the pure math lives in
/// `LevelEstimator`). PinoQuest parity, from the video teardown:
///   • the datum is FROZEN on Re-center (and on the first upright frame at start),
///     so a fixed tilted camera tracks correctly without drifting. This baseline
///     leveling is active even with the toggle OFF — exactly what the clip showed
///     (the green box leveled while "Body Stabilizer" was off);
///   • when the Body Stabilizer toggle is ON, the live leveling is continuously
///     re-estimated and low-pass-slerped away from the datum, tracking slow camera
///     or posture drift without snapping;
///   • when the upright sanity gate fails (a close crouch), the reference is LOST:
///     leveling holds the last datum and `isLevelLost()` becomes true, so the green
///     box can vanish rather than fabricating leveling from a non-upright pose.
///
/// Reference type confined to the single serial inference worker (owned by
/// `MediaPipePoseLandmarker`, touched only in `detect`/`reset`), guarded by a lock
/// like `FloorOriginLatch` so it is sound as `@unchecked Sendable`.
public final class BodyStabilizer: @unchecked Sendable {
    /// Time constant (seconds) for continuous re-leveling when the toggle is ON.
    /// Deliberately long, so it tracks slow camera drift, not fast posture changes.
    public static let reLevelTau: Float = 1.0

    private let lock = NSLock()
    private var datum = LevelEstimator.identity     // frozen leveling (the session datum)
    private var live = LevelEstimator.identity      // smoothed current leveling
    private var recapturePending = true             // capture the datum on the next upright frame
    private var haveDatum = false
    private var lost = false
    private var _includeRoll: Bool

    public init(includeRoll: Bool = false) { self._includeRoll = includeRoll }

    /// Compute the leveling rotation to apply to this frame's landmarks, inside
    /// `MediaPipeFrame.toSolverFrame`.
    /// - Parameters:
    ///   - reply: the raw sidecar reply (MediaPipe world landmarks, y-DOWN).
    ///   - zSign: the backend's Z sign, matching `toSolverFrame`.
    ///   - enabled: the Body Stabilizer toggle (continuous re-leveling). When false,
    ///     the frozen datum is held (baseline leveling).
    ///   - dt: seconds since the previous frame (only consulted when `enabled`).
    public func levelRotation(reply: SidecarReply, zSign: Float,
                              enabled: Bool, dt: Float) -> simd_quatf {
        guard reply.found, reply.world.count == 33, reply.visibility.count == 33 else {
            return lock.withLock { !haveDatum ? LevelEstimator.identity : (enabled ? live : datum) }
        }
        // Pure extraction of the few landmarks we level from, axis-fixed to the
        // solver frame (matches toSolverFrame's x, -y, z*zSign).
        func fixed(_ l: BlazePose.Landmark) -> SIMD3<Float> {
            let w = reply.world[l.rawValue]
            return SIMD3(w.x, -w.y, w.z * zSign)
        }
        let midHip = (fixed(.leftHip) + fixed(.rightHip)) * 0.5
        let nose = fixed(.nose)
        let lAnkle = fixed(.leftAnkle)
        let rAnkle = fixed(.rightAnkle)
        // Foot midpoint used as the gravity reference (hip-to-foot = "up").
        // Feet are nearly directly below the hips regardless of torso lean,
        // so this avoids baking natural shoulder-forward posture into the datum.
        let footMid = (lAnkle + rAnkle) * 0.5

        return lock.withLock {
            let upright = LevelEstimator.uprightSanity(nose: nose, midHip: midHip,
                                                       leftAnkle: lAnkle, rightAnkle: rAnkle)
            let raw = LevelEstimator.levelingQuaternion(midHip: midHip, footMid: footMid,
                                                        includeRoll: _includeRoll)
            lost = !upright

            // (Re)establish the datum on the first UPRIGHT frame after start or
            // Re-center. Until that happens there is no leveling (identity).
            if recapturePending {
                if upright {
                    datum = raw
                    live = raw
                    haveDatum = true
                    recapturePending = false
                } else {
                    return LevelEstimator.identity
                }
            }

            guard enabled else {
                // Toggle OFF: hold the frozen datum (baseline leveling).
                live = datum
                return datum
            }
            // Toggle ON: continuous re-leveling. Hold on a lost (crouch) frame so we
            // never fabricate leveling from a non-upright pose.
            if lost { return live }
            let alpha = simd_clamp(1 - exp(-dt / Self.reLevelTau), 0, 1)
            live = safeSlerp(live, raw, alpha)
            return live
        }
    }

    /// Arm a one-shot datum recapture — coupled into the landmarker's `reset()`
    /// so one Re-center re-freezes the leveling datum from the current standing pose.
    public func requestDatumRecapture() { lock.withLock { recapturePending = true } }

    /// True when the most recent frame failed the upright gate (close crouch): the
    /// leveled reference is lost, so the green box should vanish.
    public func isLevelLost() -> Bool { lock.withLock { lost } }

    /// Set the camera-roll correction flag (Body Stabilizer setting). Thread-safe.
    public func setIncludeRoll(_ on: Bool) { lock.withLock { _includeRoll = on } }

    public func reset() {
        lock.withLock {
            datum = LevelEstimator.identity
            live = LevelEstimator.identity
            recapturePending = true
            haveDatum = false
            lost = false
        }
    }
}
