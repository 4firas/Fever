import Foundation
import simd

/// JointPredictor — conservative, real-time predictive gap-filling.
///
/// Sits in the tracking graph RIGHT AFTER `detect()` and BEFORE the One-Euro
/// stabilizer / solver, operating on the raw per-frame `PoseResult`. It fills
/// brief per-landmark dropouts (fast moves, momentary occlusion) so a joint
/// eases through a gap instead of vanishing or snapping. Because it rewrites the
/// `PoseResult` in place it benefits BOTH downstream consumers:
///
///   • the metric `landmarks` (x,y ∈ [0,1] image space + relative z) that feed
///     One-Euro → JointSolver → OSC trackers, and
///   • the `imagePoints` (screen-normalized) that drive the instant preview
///     skeleton.
///
/// ── Design goals (must NOT add lag or overshoot when joints ARE present) ──────
///   • A PRESENT landmark passes through UNCHANGED. We only update bookkeeping
///     (last-good position, EMA velocity, reset its age). There is zero filtering
///     of real samples here — One-Euro downstream owns smoothing.
///   • An ABSENT / low-confidence landmark is synthesized for a SHORT hold
///     window (`maxHoldSeconds`, ~0.3 s) as `lastGood + velocity·dt`, with the
///     velocity DECAYED every missing frame (`velocityDecay`) so it eases to a
///     stop rather than flying off. Image coords are clamped to [0,1].
///   • Beyond the hold window the landmark is marked ABSENT again (NaN
///     imagePoint, presence/visibility 0) so a frozen joint is never
///     hallucinated forever.
///   • On REAPPEARANCE the real sample is blended in from the last prediction
///     over a couple frames (`reappearBlendFrames`) to avoid a visible snap.
///
/// ── Concurrency ──────────────────────────────────────────────────────────────
/// All mutable state lives in this `final class` and is touched ONLY from the
/// single serial inference worker (exactly like `LandmarkStabilizer` /
/// `JointSolver`). It is therefore `@unchecked Sendable` under the same
/// single-worker-confinement contract; it never adds locking of its own.
///
/// ── Determinism / testability ────────────────────────────────────────────────
/// `predict(_:)` is a pure function of the instance's accumulated state and the
/// frames fed to it: given the same constructor tunables and the same sequence
/// of `PoseResult`s it always produces the same outputs. No clocks, no RNG — dt
/// is derived purely from `PoseResult.timestamp`.
public final class JointPredictor: @unchecked Sendable {

    // MARK: - Tunables

    /// Longest a missing landmark is synthesized before it is dropped to absent.
    /// Past this the joint is released (NaN / presence 0) so we never freeze a
    /// stale joint indefinitely. Conservative default ≈ 9 frames at 30 fps.
    public let maxHoldSeconds: TimeInterval

    /// EMA weight for the velocity estimate on each GOOD frame, in [0,1].
    /// `vel = (1-α)·vel + α·instantaneous`. Lower = steadier (less overshoot on
    /// the first missing frame), higher = more responsive to direction changes.
    public let velocityEMA: Float

    /// Per-missing-frame multiplicative decay applied to the carried velocity, in
    /// [0,1]. < 1 makes a held joint coast to a stop instead of drifting away.
    public let velocityDecay: Float

    /// Number of frames over which a reappearing landmark is blended from the
    /// last prediction to the real sample (linear), to avoid a snap. 0 disables.
    public let reappearBlendFrames: Int

    /// Presence/visibility at or below which a landmark is treated as ABSENT.
    /// (Vision marks dropped landmarks with presence 0; a NaN position is also
    /// treated as absent.)
    public let presenceThreshold: Float

    // MARK: - Per-landmark state

    private struct State {
        /// Last GOOD metric position (image x,y ∈ [0,1] + relative z).
        var metric: SIMD3<Float> = .zero
        /// Last GOOD image point (screen-normalized, y from TOP).
        var image: SIMD2<Float> = .zero
        /// EMA velocity of the metric position (units / second).
        var metricVel: SIMD3<Float> = .zero
        /// EMA velocity of the image point (units / second).
        var imageVel: SIMD2<Float> = .zero
        /// Whether we have ever seen a good sample for this landmark.
        var seeded: Bool = false
        /// Wall time (frame timestamp) of the last GOOD sample. Drives the hold
        /// window (`elapsed = t - lastSeenT`).
        var lastSeenT: TimeInterval = 0
        /// Wall time of the last EMITTED frame for this landmark (good or held).
        /// Drives the per-frame integration step so successive held frames each
        /// advance by one real frame's dt (not the whole elapsed gap).
        var lastEmitT: TimeInterval = 0
        /// Number of consecutive synthesized (held) frames since last good.
        var heldFrames: Int = 0
        /// Frames remaining in the reappearance blend (counts down to 0).
        var blendRemaining: Int = 0
        /// Whether the previous emitted frame for this landmark was a prediction
        /// (drives whether a reappearance needs blending).
        var wasPredicting: Bool = false
    }

    private var states: [State]

    /// Reused output buffers (single-worker confinement) — no per-frame alloc.
    private var metricScratch: [NormalizedLandmark]
    private var imageScratch: [SIMD2<Float>]

    // MARK: - Init

    public init(count: Int = 33,
                maxHoldSeconds: TimeInterval = 0.3,
                velocityEMA: Float = 0.35,
                velocityDecay: Float = 0.85,
                reappearBlendFrames: Int = 3,
                presenceThreshold: Float = 0.0) {
        self.maxHoldSeconds = maxHoldSeconds
        self.velocityEMA = velocityEMA
        self.velocityDecay = velocityDecay
        self.reappearBlendFrames = reappearBlendFrames
        self.presenceThreshold = presenceThreshold
        self.states = Array(repeating: State(), count: count)
        self.metricScratch = Array(repeating: NormalizedLandmark(position: .zero,
                                                                 visibility: 0,
                                                                 presence: 0),
                                   count: count)
        self.imageScratch = Array(repeating: SIMD2<Float>(.nan, .nan), count: count)
    }

    /// Clear all accumulated state (call on start/stop while the worker is idle).
    public func reset() {
        for i in states.indices { states[i] = State() }
    }

    // MARK: - Core

    /// Gap-fill one raw inference result, returning a new `PoseResult` with both
    /// the metric `landmarks` and the `imagePoints` predicted where landmarks
    /// were missing. Pure given the instance state + the fed frame sequence.
    public func predict(_ raw: PoseResult) -> PoseResult {
        let lms = raw.landmarks
        let t = raw.timestamp

        // Size the reused buffers / state to the input once (no-op at 33).
        if states.count != lms.count {
            states = Array(repeating: State(), count: lms.count)
            metricScratch = Array(repeating: NormalizedLandmark(position: .zero,
                                                                visibility: 0,
                                                                presence: 0),
                                  count: lms.count)
        }
        // imagePoints may legitimately be empty (some sources omit them); only
        // size the scratch when present.
        let hasImage = raw.imagePoints.count == lms.count
        if hasImage && imageScratch.count != lms.count {
            imageScratch = Array(repeating: SIMD2<Float>(.nan, .nan), count: lms.count)
        }

        for i in lms.indices {
            let lm = lms[i]
            let img = hasImage ? raw.imagePoints[i] : SIMD2<Float>(.nan, .nan)
            let present = isPresent(lm)

            if present {
                emitGood(i, lm: lm, image: img, t: t, hasImage: hasImage)
            } else {
                emitMissing(i, fallback: lm, t: t, hasImage: hasImage)
            }
        }

        return PoseResult(landmarks: metricScratch,
                          timestamp: t,
                          imagePoints: hasImage ? imageScratch : raw.imagePoints)
    }

    // MARK: - Present landmark: pass through + update bookkeeping

    private func emitGood(_ i: Int,
                          lm: NormalizedLandmark,
                          image: SIMD2<Float>,
                          t: TimeInterval,
                          hasImage: Bool) {
        let metric = lm.position
        var outMetric = metric
        var outImage = image

        if states[i].seeded {
            // dt measured from the LAST EMITTED frame (good or held); both
            // `states[i].metric` and `lastEmitT` describe that emitted sample, so
            // the instantaneous velocity is consistent whether or not we were
            // mid-hold.
            let dt = Float(max(t - states[i].lastEmitT, 1e-6))

            // Capture the carried (pre-update) velocity FIRST: the reappearance
            // blend base must continue the PRIOR motion, not the just-measured
            // jump to the real sample (which would make the prediction equal the
            // real value and defeat the blend).
            let priorMetricVel = states[i].metricVel
            let priorImageVel = states[i].imageVel

            // Update EMA velocity from this good sample (so a future hold coasts
            // in the right direction).
            let instMetricVel = (metric - states[i].metric) / dt
            states[i].metricVel = mix(states[i].metricVel, instMetricVel, velocityEMA)
            if hasImage {
                let instImageVel = (image - states[i].image) / dt
                states[i].imageVel = mix(states[i].imageVel, instImageVel, velocityEMA)
            }

            // Reappearance blend: if we were predicting last frame, ease from the
            // last prediction to the real sample over a couple frames so the real
            // value does not snap in. The blend base is the continued prediction
            // from the last emitted (predicted) sample using the PRIOR velocity.
            if states[i].wasPredicting && reappearBlendFrames > 0 {
                states[i].blendRemaining = reappearBlendFrames
            }
            if states[i].blendRemaining > 0 {
                // Fraction toward the real sample: 1/(N+1), 2/(N+1), ... N/(N+1).
                let step = reappearBlendFrames - states[i].blendRemaining + 1
                let frac = Float(step) / Float(reappearBlendFrames + 1)
                let predMetric = states[i].metric + priorMetricVel * dt
                outMetric = mix(predMetric, metric, frac)
                if hasImage {
                    let predImage = states[i].image + priorImageVel * dt
                    outImage = mix(predImage, image, frac)
                }
                states[i].blendRemaining -= 1
            }
        }

        // Record this good sample as the new anchor.
        states[i].metric = metric
        states[i].image = image
        states[i].seeded = true
        states[i].lastSeenT = t
        states[i].lastEmitT = t
        states[i].heldFrames = 0
        states[i].wasPredicting = false

        // Pass the (real, or briefly blended) value through. Confidence is the
        // detector's own — we do not alter it for present landmarks.
        metricScratch[i] = NormalizedLandmark(position: outMetric,
                                              visibility: lm.visibility,
                                              presence: lm.presence)
        if hasImage { imageScratch[i] = clamp01(outImage) }
    }

    // MARK: - Absent landmark: synthesize within the hold window, else drop

    private func emitMissing(_ i: Int,
                             fallback lm: NormalizedLandmark,
                             t: TimeInterval,
                             hasImage: Bool) {
        // Never seen a good value → nothing to predict from; stay absent.
        guard states[i].seeded else {
            states[i].wasPredicting = false
            metricScratch[i] = NormalizedLandmark(position: lm.position,
                                                  visibility: 0, presence: 0)
            if hasImage { imageScratch[i] = SIMD2<Float>(.nan, .nan) }
            return
        }

        // `elapsed` (from last GOOD) drives the hold window; `dt` (from last
        // EMITTED) is the per-frame integration step, so each held frame advances
        // by exactly one real frame's worth of motion.
        let elapsed = t - states[i].lastSeenT
        let dt = Float(max(t - states[i].lastEmitT, 1e-6))

        // Beyond the hold window → release the joint (do NOT freeze forever).
        guard elapsed <= maxHoldSeconds else {
            states[i].wasPredicting = false
            states[i].lastEmitT = t
            metricScratch[i] = NormalizedLandmark(position: states[i].metric,
                                                  visibility: 0, presence: 0)
            if hasImage { imageScratch[i] = SIMD2<Float>(.nan, .nan) }
            return
        }

        // Decay the carried velocity once per held frame so it eases to a stop.
        states[i].heldFrames += 1
        states[i].metricVel *= velocityDecay
        states[i].imageVel *= velocityDecay

        // Synthesize from the previous emitted position + decayed velocity over
        // one frame's dt. `states[i].metric/image` hold the LAST EMITTED sample,
        // so this integrates forward one step at a time.
        let predMetric = states[i].metric + states[i].metricVel * dt
        states[i].metric = predMetric
        var predImage = states[i].image
        if hasImage {
            predImage = clamp01(states[i].image + states[i].imageVel * dt)
            states[i].image = predImage
        }
        states[i].lastEmitT = t

        // Decaying confidence so downstream still uses the joint but knows it is
        // synthesized: scale from the threshold up toward 1 by how fresh it is.
        let freshness = Float(max(0, 1 - elapsed / maxHoldSeconds))
        let conf = max(0.001, freshness)

        states[i].wasPredicting = true
        metricScratch[i] = NormalizedLandmark(position: predMetric,
                                              visibility: conf,
                                              presence: conf)
        if hasImage { imageScratch[i] = predImage }
    }

    // MARK: - Helpers

    private func isPresent(_ lm: NormalizedLandmark) -> Bool {
        // Absent if the metric position is NaN, OR confidence is at/below the
        // threshold. (Vision zeroes presence/visibility for dropped landmarks.)
        let p = lm.position
        if !p.x.isFinite || !p.y.isFinite || !p.z.isFinite { return false }
        if lm.presence <= presenceThreshold && lm.visibility <= presenceThreshold {
            return false
        }
        return true
    }

    private func clamp01(_ v: SIMD2<Float>) -> SIMD2<Float> {
        SIMD2<Float>(min(max(v.x, 0), 1), min(max(v.y, 0), 1))
    }

    private func mix(_ a: SIMD3<Float>, _ b: SIMD3<Float>, _ t: Float) -> SIMD3<Float> {
        a + (b - a) * t
    }
    private func mix(_ a: SIMD2<Float>, _ b: SIMD2<Float>, _ t: Float) -> SIMD2<Float> {
        a + (b - a) * t
    }
}
