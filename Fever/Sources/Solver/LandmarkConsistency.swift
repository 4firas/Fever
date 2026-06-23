import simd
import Foundation

/// Pre-smoothing cleanup of raw MediaPipe landmarks (reference type, confined to
/// the serial inference worker, reset on Recenter / run start-stop). Runs BEFORE
/// the JointPredictor + One-Euro, on the 33 hip-centred world landmarks, and fixes
/// two monocular failure modes that One-Euro can NOT fix (it only smooths a wrong
/// assignment into a smooth wrong path):
///
///  1. LEFT/RIGHT SWAP under self-occlusion (in profile the near leg occludes the
///     far one; MediaPipe transposes the L/R labels frame-to-frame → a one-frame
///     teleport). We re-assign a pair ONLY when it teleports — a large single-frame
///     jump the swap explains far better. A smooth body TURN moves each landmark a
///     little per frame and is left alone, so the anti-swap no longer fights
///     MediaPipe's correct tracking through a rotation (which used to cross the
///     torso into an X and stick).
///  2. OCCLUDED-LIMB HALLUCINATION. MediaPipe emits ALL 33 landmarks every frame,
///     even fully occluded ones, as confidently-placed GUESSES (visibility is the
///     only signal they are fabricated). We gate on visibility with hysteresis and
///     mark a disengaged landmark ABSENT, so the downstream JointPredictor holds
///     its last good position instead of forwarding the hallucinated coordinate.
///
/// (The remaining facing-AWAY "inside-out" depth inversion is a separate, harder
/// fix — a facing classifier + depth re-solve — and is not attempted here.)
public final class LandmarkConsistency: @unchecked Sendable {

    private var prev = [SIMD3<Float>](repeating: .zero, count: 33)
    private var engaged = [Bool](repeating: true, count: 33)
    private var seeded = false

    /// L/R landmark pairs that can transpose: hips, knees, ankles, heels, toes,
    /// shoulders, elbows, wrists. Heel + foot-index are paired alongside the ankle
    /// so a leg swap moves the whole foot (ankle + knee + heel + toe) as one rigid
    /// unit — otherwise a corrected foot solves its orientation from the OPPOSITE
    /// foot's heel→toe vector and points the wrong way.
    private static let pairs: [(Int, Int)] = [
        (BlazePose.Landmark.leftHip.rawValue,       BlazePose.Landmark.rightHip.rawValue),
        (BlazePose.Landmark.leftKnee.rawValue,      BlazePose.Landmark.rightKnee.rawValue),
        (BlazePose.Landmark.leftAnkle.rawValue,     BlazePose.Landmark.rightAnkle.rawValue),
        (BlazePose.Landmark.leftHeel.rawValue,      BlazePose.Landmark.rightHeel.rawValue),
        (BlazePose.Landmark.leftFootIndex.rawValue, BlazePose.Landmark.rightFootIndex.rawValue),
        (BlazePose.Landmark.leftShoulder.rawValue,  BlazePose.Landmark.rightShoulder.rawValue),
        (BlazePose.Landmark.leftElbow.rawValue,     BlazePose.Landmark.rightElbow.rawValue),
        (BlazePose.Landmark.leftWrist.rawValue,     BlazePose.Landmark.rightWrist.rawValue),
    ]

    /// Minimum keep-cost (metres) before a swap is even considered: a real
    /// MediaPipe L/R transposition is a SINGLE-FRAME teleport (the two landmarks
    /// jump ~0.2–0.4 m apart at once), whereas a body TURN moves each landmark only
    /// ~0.01–0.04 m per frame. Gating on this teleport distance is what stops the
    /// anti-swap from mis-firing during a smooth turn (the bug that crossed the
    /// torso into an X and then stuck, because the swap was written back to `prev`).
    private let swapTeleport: Float = 0.15
    /// And the swap must explain that teleport much better than keeping (costSwap
    /// well under half of costKeep), so only a clear transposition flips.
    private let swapRatio: Float = 0.5
    /// Each landmark must also land within this distance of the PARTNER's previous
    /// position (not just have a lower total cost). This prevents false swaps when
    /// both limbs cross simultaneously: crossing wrists move 20–30 cm but each ends
    /// up 5+ cm from the other's old position, whereas a true MediaPipe L/R label
    /// transposition means A lands ≈ exactly where B was and vice-versa (≤ 4 cm).
    private let swapSymmetryTolerance: Float = 0.04
    /// Don't reorder a pair when either landmark is occluded (its position is a
    /// guess) — raise this to match the engage threshold (vOn) so a freshly
    /// re-engaged borderline landmark is never reassigned while still untrustworthy.
    private let minSwapVisibility: Float = 0.5
    /// Visibility hysteresis: engage at vOn, disengage at vOff (kills the noisy
    /// 0.86<->0.92 flicker MediaPipe shows on borderline landmarks).
    private let vOn: Float = 0.5
    private let vOff: Float = 0.35

    public init() {}

    public func reset() {
        prev = [SIMD3<Float>](repeating: .zero, count: 33)
        engaged = [Bool](repeating: true, count: 33)
        seeded = false
    }

    public func process(_ pose: PoseResult) -> PoseResult {
        guard pose.landmarks.count == 33 else { return pose }
        var lm = pose.landmarks
        var img = pose.imagePoints.count == 33
            ? pose.imagePoints
            : [SIMD2<Float>](repeating: SIMD2(.nan, .nan), count: 33)

        // 1) Temporal LEFT/RIGHT anti-swap.
        if seeded {
            for (a, b) in Self.pairs {
                guard lm[a].visibility > minSwapVisibility,
                      lm[b].visibility > minSwapVisibility else { continue }
                let pa = lm[a].position, pb = lm[b].position
                let costKeep = simd_distance(pa, prev[a]) + simd_distance(pb, prev[b])
                let costSwap = simd_distance(pa, prev[b]) + simd_distance(pb, prev[a])
                // Only correct a genuine single-frame transposition: (1) the kept labels
                // must have TELEPORTED (not just drifted), (2) the swap must explain it
                // far better, AND (3) each landmark must land within swapSymmetryTolerance
                // of where the OTHER one was — a true label swap means A lands where B
                // was and B lands where A was, so both individual distances are near zero.
                // Crossing limbs produce a low total-cost swap but each individual distance
                // is 5+ cm → condition (3) rejects them without touching real transpositions.
                if costKeep > swapTeleport, costSwap < costKeep * swapRatio,
                   simd_distance(pa, prev[b]) < swapSymmetryTolerance,
                   simd_distance(pb, prev[a]) < swapSymmetryTolerance {
                    lm.swapAt(a, b)
                    img.swapAt(a, b)
                }
            }
        }

        // 2) VISIBILITY gating (hysteresis) → mark occluded landmarks ABSENT so the
        //    JointPredictor holds-last instead of forwarding the hallucinated guess.
        for i in 0..<33 {
            let v = lm[i].visibility
            if engaged[i] {
                if v < vOff { engaged[i] = false }
            } else {
                if v >= vOn { engaged[i] = true }
            }
            if seeded && !engaged[i] {
                lm[i] = NormalizedLandmark(position: lm[i].position, visibility: 0, presence: 0)
                img[i] = SIMD2<Float>(.nan, .nan)
            }
        }

        // 3) Record accepted positions for next frame's anti-swap (engaged only;
        //    a disengaged landmark keeps its last-good prev so a re-engage compares
        //    against the last trustworthy position, not a stale guess).
        for i in 0..<33 where engaged[i] { prev[i] = lm[i].position }
        seeded = true

        return PoseResult(landmarks: lm, timestamp: pose.timestamp, imagePoints: img)
    }
}
