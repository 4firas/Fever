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
    /// Whether the OSC/tracker path negates X. DEFAULT **false** — VERIFIED by a
    /// live OSC-wire capture diff against PinoFBT (the 1:1 benchmark).
    ///
    /// The net MediaPipe-world → VRChat position map is `diag(mirrorSignX, −1, −1)`.
    /// PinoFBT outputs the anatomical-LEFT limb at NEGATIVE X (and it tracks
    /// correctly in VRChat), so Fever must match that sign. Captured result:
    ///   • mirror OFF (det +1): Fever left limb → −X  ✓ matches PinoFBT.
    ///   • mirror ON  (det −1): Fever left limb → +X  ✗ mirrored.
    /// The non-obvious reason the signs work out this way: the MediaPipe **Tasks
    /// API** (Fever) emits world-landmark X with the OPPOSITE sign to the legacy
    /// MediaPipe **GPU graph** that PinoFBT bundles — so even though PinoFBT's
    /// binary negates X, Fever must NOT, to land on the same VRChat side. Affects
    /// only the OSC CoordinateMapper, never the preview. User-toggleable safety hatch.
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
    /// Whether `/rotation` is streamed REST-RELATIVE (delta from the Recenter pose)
    /// instead of ABSOLUTE world orientation. DEFAULT **false** (absolute).
    ///
    /// CORRECTED via PinoFBT live OSC capture: PinoFBT streams ABSOLUTE limb
    /// orientations — at a neutral standing pose its elbows/knees carry real
    /// non-zero euler (e.g. elbow ≈ (+30,·,−24)), NOT zeros. The old rest-relative
    /// rebase captured a rest pose on Recenter and emitted `inverse(qRest)·qLive`,
    /// which ZEROED every joint at the calibration pose — erasing real limb
    /// orientation and making turns/limbs read wrong. VRChat does its own
    /// tracker→bone calibration, so it expects absolute poses. Kept as a toggle for
    /// experimentation, but absolute is the shipping default.
    public var rotationRestRelative: Bool {
        didSet { UserDefaults.standard.set(rotationRestRelative, forKey: Keys.rotationRestRelative) }
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

    // MARK: - Leveling / Body Stabilizer (PinoQuest-style gravity leveling)
    /// Continuous re-leveling. When ON, the gravity-leveling datum is continuously
    /// re-estimated and low-pass-filtered, tracking slow camera/posture drift; when
    /// OFF, the datum frozen at the last Re-center is held (baseline leveling still
    /// applies either way). DEFAULT false. Shown to the user as "Body Stabilizer".
    public var bodyStabilizer: Bool {
        didSet { UserDefaults.standard.set(bodyStabilizer, forKey: Keys.bodyStabilizer) }
    }
    /// Whether leveling also corrects camera ROLL (about the view axis), not just
    /// pitch. DEFAULT false — a desk webcam is rarely rolled, and roll estimated from
    /// a possibly-leaning user adds noise. Exposed for tilted / handheld rigs.
    public var levelIncludeRoll: Bool {
        didSet { UserDefaults.standard.set(levelIncludeRoll, forKey: Keys.levelIncludeRoll) }
    }
    /// YAW Body-Stabilizer (PinoFBT-style): derive ONE smoothed body-facing yaw
    /// from the hip and impose it coherently on the torso (hip + chest), so the
    /// body turns as one stable unit instead of each tracker's monocular yaw
    /// jittering/flipping when you face away. DEFAULT false (opt-in) — it changes
    /// the torso yaw, so it's off until validated in-headset; PinoFBT runs its
    /// equivalent ON. See `YawStabilizer`.
    public var yawStabilizer: Bool {
        didSet { UserDefaults.standard.set(yawStabilizer, forKey: Keys.yawStabilizer) }
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
        // DEFAULT false — An HMD user (Quest) already has an authoritative head; Fever's
        // FABRICATED head (derived from shoulder/ear geometry) creates a SECOND head that
        // fights the HMD, causing VRChat to fold the neck to reconcile the conflict
        // (the 90° neck-fold bug). TrackingPipeline.swift:360-365 already documents
        // the correct design: "We deliberately send NO head OSC point — ever. The user
        // wears a Quest HMD, which is the authoritative head."
        // Toggle ON only for room-scale PC setups WITHOUT an HMD where the head anchor
        // is needed to origin the OSC space.
        sendHeadReference = d.object(forKey: Keys.sendHeadReference) as? Bool ?? false
        // DEFAULT false — VERIFIED by live OSC capture diff against PinoFBT.
        // KEY FINDING: the MediaPipe Tasks API (what Fever uses) emits world-landmark
        // X with the OPPOSITE sign to the legacy GPU graph PinoFBT uses. PinoFBT
        // (works in VRChat) outputs the anatomical-LEFT limb at NEGATIVE X. With
        // mirror ON, Fever output left at +X (mirrored/wrong); with mirror OFF it
        // outputs left at −X = matches PinoFBT. So mirror OFF is correct for the
        // Tasks-API frame (negate Y,Z only, det +1). The earlier "mirror ON to match
        // PinoFBT's binary signs" was wrong — it ignored the Tasks-API vs legacy-graph
        // sign difference. (The real "L/R mangle" was broken chest/foot ROTATION, not
        // position chirality — see rotationRestRelative / OSCSender.rotationSlots.)
        mirrorTracking = d.object(forKey: Keys.mirrorTracking) as? Bool ?? false
        footTrackersAtAnkle = d.object(forKey: Keys.footTrackersAtAnkle) as? Bool ?? true
        // DEFAULT true — the rotation path is now REBUILT (two in-body axes, no
        // world-up gauge/singularity; locked-roll feet; rest-relative delta from
        // the Recenter T-pose → bounded, zero-centered euler like PinoFBT). The old
        // garbage euler (full -180..180 wraps, stuck axis offsets) is gone, so we
        // stream rotation by default; still user-toggleable for position-only.
        sendRotation = d.object(forKey: Keys.sendRotation) as? Bool ?? true
        // DEFAULT false = ABSOLUTE world rotation (PinoFBT ground truth, confirmed by
        // live OSC capture). Rest-relative zeroed real limb orientation at the
        // Recenter pose; absolute is what VRChat expects. See the property doc.
        rotationRestRelative = d.object(forKey: Keys.rotationRestRelative) as? Bool ?? false

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

        bodyStabilizer = d.object(forKey: Keys.bodyStabilizer) as? Bool ?? false
        levelIncludeRoll = d.object(forKey: Keys.levelIncludeRoll) as? Bool ?? false
        yawStabilizer = d.object(forKey: Keys.yawStabilizer) as? Bool ?? false

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
        static let rotationRestRelative = "rotationRestRelative"
        static let enableTracker = "enableTracker"
        static let sendHeadReference = "sendHeadReference"
        static let mirrorTracking = "mirrorTracking"
        static let footTrackersAtAnkle = "footTrackersAtAnkle"
        static let userHeightMeters = "userHeightMeters"
        static let enabledJoints = "enabledJoints"
        static let stabilizerMinCutoff = "stabilizerMinCutoff"
        static let stabilizerBeta = "stabilizerBeta"
        static let rotationSmoothing = "rotationSmoothing"
        static let bodyStabilizer = "bodyStabilizer"
        static let levelIncludeRoll = "levelIncludeRoll"
        static let yawStabilizer = "yawStabilizer"
        static let jointSize = "jointSize"
        static let hipExaggerateCoefficient = "hipExaggerateCoefficient"
        static let hipTwistCoefficient = "hipTwistCoefficient"
        static let hipLength = "hipLength"
        static let kneePosition = "kneePosition"
        static let stepStrideCoefficient = "stepStrideCoefficient"
        static let stepLiftCoefficient = "stepLiftCoefficient"
    }
}
