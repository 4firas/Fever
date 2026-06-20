import Foundation
import Observation
import SwiftUI

/// All persistent, user-tunable tracking settings, observable by SwiftUI.
///
/// REWORKED to the Observation framework (`@Observable`). Persistence is done
/// manually against `UserDefaults` (the macro replaces `@AppStorage`, which is
/// a property wrapper incompatible with `@Observable` stored properties).
///
/// Numeric tunables are stored as `Double` (slider-friendly) and exposed to
/// SIMD code via `Float`-suffixed computed mirrors (e.g. `jointSize` →
/// `jointSizeF`).
///
/// Defaults: OSC host 127.0.0.1, OSC port **9000**, user height **1.74 m**,
/// enabled joints = ALL 8 numbered body trackers (hip, chest, l/r elbow,
/// l/r knee, l/r foot). The head is the always-on position-only REFERENCE point
/// (`sendHeadReference`), handled separately — not a numbered body tracker.
@Observable
public final class TrackingConfig {

    // MARK: - Network / output
    public var oscHost: String {
        didSet { UserDefaults.standard.set(oscHost, forKey: Keys.oscHost) }
    }
    public var oscPort: Int {
        didSet {
            // Clamp to the valid UDP port range so the GUI port field (which
            // accepts any Int) can never push an out-of-range value down to
            // `OSCSender` / `NWEndpoint.Port` (a UInt16 conversion that would
            // trap). Mirrors the CLI validation in FeverMain. Re-assigning
            // here re-enters didSet once with an already-in-range value (no-op).
            let clamped = min(max(oscPort, 1), 65535)
            if clamped != oscPort {
                oscPort = clamped
                return
            }
            UserDefaults.standard.set(oscPort, forKey: Keys.oscPort)
        }
    }
    public var enableTracker: Bool {
        didSet { UserDefaults.standard.set(enableTracker, forKey: Keys.enableTracker) }
    }
    public var sendHeadReference: Bool {
        didSet { UserDefaults.standard.set(sendHeadReference, forKey: Keys.sendHeadReference) }
    }
    /// Whether the OSC/tracker path mirrors left/right (negates X). DEFAULT true —
    /// CORRECTED via the PinoFBT wire diff. Apple Vision's solver frame is right-
    /// handed (+X = camera-right); VRChat is left-handed. We always negate Z for the
    /// handedness flip, so X must ALSO negate to complete the same right→left
    /// handedness change — otherwise the skeleton is reflected and lands MIRRORED.
    /// Ground truth: PinoFBT's LEFT-side trackers sit at −X, ours (un-mirrored) sat
    /// at +X — a pure left↔right reflection. That both (a) showed joints visibly
    /// flipped in calibrate mode standing still and (b) made VRChat IK fight itself
    /// on turns (driving the left leg from a right-side tracker → glitching). With
    /// the X negate ON, our LEFT lands at −X / RIGHT at +X exactly like PinoFBT, and
    /// the HEAD reference flips with the body (the mapper applies the same M to every
    /// joint incl. head), so head-relative X signs match too. Composes cleanly with
    /// hip-sway lateral gain and the XZ-centering latch — both run in solver space
    /// BEFORE the mapper, so the single post-solve X negate neither double-applies
    /// nor cancels them. Affects only the OSC CoordinateMapper, never the live
    /// preview overlay. Still user-toggleable for the rare rear-camera / odd setup.
    public var mirrorTracking: Bool {
        didSet { UserDefaults.standard.set(mirrorTracking, forKey: Keys.mirrorTracking) }
    }
    /// Whether the left/right foot trackers are placed at the ANKLE landmark
    /// (standard VRChat FBT). DEFAULT true. When false they use the synthesized
    /// foot-index (toe) position.
    public var footTrackersAtAnkle: Bool {
        didSet { UserDefaults.standard.set(footTrackersAtAnkle, forKey: Keys.footTrackersAtAnkle) }
    }
    /// Whether to transmit `/rotation` for trackers. DEFAULT true now that the
    /// rotation path is REBUILT (PinoFBT/ju1ce recipe): per-joint orientations are
    /// derived from TWO in-body axes (no fabricated world-up roll / singularity),
    /// feet use a locked-roll yaw+pitch frame, and everything is emitted REST-
    /// RELATIVE (delta from the Recenter T-pose) so the wire is bounded and
    /// zero-centered (euler ≈ 0 at rest) instead of the old large/wrapping values.
    /// The old garbage-rotation reasons for keeping this off are resolved.
    public var sendRotation: Bool {
        didSet { UserDefaults.standard.set(sendRotation, forKey: Keys.sendRotation) }
    }

    // MARK: - Body scale
    public var userHeightMeters: Double {
        didSet { UserDefaults.standard.set(userHeightMeters, forKey: Keys.userHeightMeters) }
    }

    // MARK: - Enabled trackers + fixed slot map
    public var enabledJoints: Set<JointType> {
        didSet {
            UserDefaults.standard.set(enabledJoints.map(\.rawValue).joined(separator: ","),
                              forKey: Keys.enabledJoints)
        }
    }

    /// Fixed VRChat numbered slot map. 1=hip, 2=leftFoot, 3=rightFoot (MVP) plus
    /// optional 4..8. Head is handled separately as the alignment reference.
    /// One body part per index, every frame (NO slot cycling).
    public var slotMap: [JointType: String]

    // MARK: - One-Euro / SLERP smoothing
    public var stabilizerMinCutoff: Double {
        didSet { UserDefaults.standard.set(stabilizerMinCutoff, forKey: Keys.stabilizerMinCutoff) }
    }
    public var stabilizerBeta: Double {
        didSet { UserDefaults.standard.set(stabilizerBeta, forKey: Keys.stabilizerBeta) }
    }
    public var rotationSmoothing: Double {
        didSet { UserDefaults.standard.set(rotationSmoothing, forKey: Keys.rotationSmoothing) }
    }

    // MARK: - Body tweaks
    public var jointSize: Double {
        didSet { UserDefaults.standard.set(jointSize, forKey: Keys.jointSize) }
    }
    /// HIP SWAY gain (lateral / X). Repurposed for the POSITION-only pipeline:
    /// amplifies the hip tracker's side-to-side deviation from stance center
    /// (ankle midpoint) by this factor so weight shifts read clearly in VR.
    /// 1.0 = literal (no exaggeration); default 2.0; >3 looks like a caricature.
    /// (Also still drives the optional rotation swing when `sendRotation` is on.)
    public var hipExaggerateCoefficient: Double {
        didSet { UserDefaults.standard.set(hipExaggerateCoefficient, forKey: Keys.hipExaggerateCoefficient) }
    }
    /// HIP LEAN gain (forward-back / Z). Repurposed for the POSITION-only
    /// pipeline: amplifies the hip's forward/back deviation (lean / hip-lead) by
    /// this factor. Kept LOWER than lateral because depth is Vision's weakest
    /// axis. 1.0 = literal; default 1.4. (Also drives the optional rotation twist
    /// when `sendRotation` is on.)
    public var hipTwistCoefficient: Double {
        didSet { UserDefaults.standard.set(hipTwistCoefficient, forKey: Keys.hipTwistCoefficient) }
    }
    public var hipLength: Double {
        didSet { UserDefaults.standard.set(hipLength, forKey: Keys.hipLength) }
    }
    public var kneePosition: Double {
        didSet { UserDefaults.standard.set(kneePosition, forKey: Keys.kneePosition) }
    }
    /// STEP / STRIDE gain — amplifies a SWINGING foot's horizontal displacement
    /// from its slow-EMA neutral rest position so steps / walking / dynamic leg
    /// movement read bigger in VR. 1.0 = literal (no exaggeration); default 1.6.
    /// Swing-gated (a planted foot is never exaggerated → stays glued to the
    /// floor) and clamped so a spike can't throw the foot away.
    public var stepStrideCoefficient: Double {
        didSet { UserDefaults.standard.set(stepStrideCoefficient, forKey: Keys.stepStrideCoefficient) }
    }
    /// STEP LIFT gain — amplifies a SWINGING foot's UPWARD lift above the floor so
    /// marches / high steps read bigger. Up-only (never pushes a planted foot down
    /// through the floor). 1.0 = literal; default 1.3.
    public var stepLiftCoefficient: Double {
        didSet { UserDefaults.standard.set(stepLiftCoefficient, forKey: Keys.stepLiftCoefficient) }
    }

    // MARK: - Float mirrors for SIMD math
    public var userHeightMetersF: Float { Float(userHeightMeters) }
    public var jointSizeF: Float { Float(jointSize) }
    public var hipExaggerateCoefficientF: Float { Float(hipExaggerateCoefficient) }
    public var hipTwistCoefficientF: Float { Float(hipTwistCoefficient) }
    public var hipLengthF: Float { Float(hipLength) }
    public var kneePositionF: Float { Float(kneePosition) }
    public var stepStrideCoefficientF: Float { Float(stepStrideCoefficient) }
    public var stepLiftCoefficientF: Float { Float(stepLiftCoefficient) }
    public var stabilizerMinCutoffF: Float { Float(stabilizerMinCutoff) }
    public var stabilizerBetaF: Float { Float(stabilizerBeta) }
    public var rotationSmoothingF: Float { Float(rotationSmoothing) }

    // MARK: - Init (loads persisted values; falls back to spec defaults)
    public init() {
        let d = UserDefaults.standard

        oscHost = (d.string(forKey: Keys.oscHost)) ?? "127.0.0.1"
        oscPort = d.object(forKey: Keys.oscPort) as? Int ?? 9000
        enableTracker = d.object(forKey: Keys.enableTracker) as? Bool ?? false
        // DEFAULT true — CORRECTED via PinoFBT wire capture. PinoFBT streams
        // /tracking/trackers/head/POSITION continuously (~30 Hz); that head point
        // is the ANCHOR VRChat uses to re-origin the OSC space, which CANCELS the
        // body trackers' absolute frame offset. With NO head, the body was measured
        // landing ~2 m off in +X (broken placement, no response to motion). Head is
        // POSITION-ONLY on the wire (OSCSender.sendHeadPosition) — never head
        // rotation (that was the earlier slow-yaw-drift; PinoFBT omits it too). The
        // head position is the head-bone root from the fixed anthropometric skeleton.
        sendHeadReference = d.object(forKey: Keys.sendHeadReference) as? Bool ?? true
        // DEFAULT true — see the `mirrorTracking` property doc. X must negate with Z
        // to complete the right→left handedness flip; un-mirrored landed the whole
        // skeleton reflected (LEFT parts at +X vs PinoFBT's −X), flipping joints in
        // calibrate and making IK fight a mirrored skeleton on turns.
        mirrorTracking = d.object(forKey: Keys.mirrorTracking) as? Bool ?? true
        footTrackersAtAnkle = d.object(forKey: Keys.footTrackersAtAnkle) as? Bool ?? true
        // DEFAULT true — the rotation path is now REBUILT (two in-body axes, no
        // world-up gauge/singularity; locked-roll feet; rest-relative delta from
        // the Recenter T-pose → bounded, zero-centered euler like PinoFBT). The old
        // garbage euler (full -180..180 wraps, stuck axis offsets) is gone, so we
        // stream rotation by default; still user-toggleable for position-only.
        sendRotation = d.object(forKey: Keys.sendRotation) as? Bool ?? true

        userHeightMeters = d.object(forKey: Keys.userHeightMeters) as? Double ?? 1.74

        if let raw = d.string(forKey: Keys.enabledJoints) {
            enabledJoints = Set(raw.split(separator: ",")
                .compactMap { JointType(rawValue: String($0)) })
        } else {
            // Default to all 8 numbered body trackers (head is the always-on
            // position-only reference, handled separately).
            enabledJoints = [.hip, .chest,
                             .leftElbow, .rightElbow,
                             .leftKnee, .rightKnee,
                             .leftFoot, .rightFoot]
        }

        slotMap = [
            .hip:        "1",
            .leftFoot:   "2",
            .rightFoot:  "3",
            .chest:      "4",
            .leftKnee:   "5",
            .rightKnee:  "6",
            .leftElbow:  "7",
            .rightElbow: "8"
        ]

        stabilizerMinCutoff = d.object(forKey: Keys.stabilizerMinCutoff) as? Double ?? 1.0
        stabilizerBeta = d.object(forKey: Keys.stabilizerBeta) as? Double ?? 0.007
        rotationSmoothing = d.object(forKey: Keys.rotationSmoothing) as? Double ?? 0.5

        jointSize = d.object(forKey: Keys.jointSize) as? Double ?? 1.0
        // Tasteful out-of-the-box exaggeration: a bit of intentional liveliness,
        // not zero (stiff) and not cartoonish. Lateral sway 2.0x, fwd/back 1.4x.
        // These coefficients changed MEANING from rotation amounts (old range
        // -1...1) to POSITION gains (range 1...3 / 1...2). Any persisted value
        // below 1.0 is a stale rotation-era value (or the old 0.0 default) that
        // would collapse the hip onto stance center, so clamp loads up to the
        // tasteful default rather than honoring an inert old setting.
        let loadedSway = d.object(forKey: Keys.hipExaggerateCoefficient) as? Double ?? 2.0
        hipExaggerateCoefficient = loadedSway >= 1.0 ? loadedSway : 2.0
        let loadedLean = d.object(forKey: Keys.hipTwistCoefficient) as? Double ?? 1.4
        hipTwistCoefficient = loadedLean >= 1.0 ? loadedLean : 1.4
        hipLength = d.object(forKey: Keys.hipLength) as? Double ?? 0.0
        kneePosition = d.object(forKey: Keys.kneePosition) as? Double ?? 0.0
        // Step exaggeration: same >=1.0 load clamp as the hip gains so a stale
        // sub-1.0 persisted value can never collapse a foot onto its neutral.
        let loadedStride = d.object(forKey: Keys.stepStrideCoefficient) as? Double ?? 1.6
        stepStrideCoefficient = loadedStride >= 1.0 ? loadedStride : 1.6
        let loadedLift = d.object(forKey: Keys.stepLiftCoefficient) as? Double ?? 1.3
        stepLiftCoefficient = loadedLift >= 1.0 ? loadedLift : 1.3
    }

    // MARK: - Persistence keys
    private enum Keys {
        static let oscHost = "oscHost"
        static let oscPort = "oscPort"
        static let sendRotation = "sendRotation"
        static let enableTracker = "enableTracker"
        static let sendHeadReference = "sendHeadReference"
        static let mirrorTracking = "mirrorTracking"
        static let footTrackersAtAnkle = "footTrackersAtAnkle"
        static let userHeightMeters = "userHeightMeters"
        static let enabledJoints = "enabledJoints"
        static let stabilizerMinCutoff = "stabilizerMinCutoff"
        static let stabilizerBeta = "stabilizerBeta"
        static let rotationSmoothing = "rotationSmoothing"
        static let jointSize = "jointSize"
        static let hipExaggerateCoefficient = "hipExaggerateCoefficient"
        static let hipTwistCoefficient = "hipTwistCoefficient"
        static let hipLength = "hipLength"
        static let kneePosition = "kneePosition"
        static let stepStrideCoefficient = "stepStrideCoefficient"
        static let stepLiftCoefficient = "stepLiftCoefficient"
    }
}
