import Foundation
import Observation

/// All persistent, user-tunable tracking settings, observable by SwiftUI.
///
/// REWORKED to the Observation framework (`@Observable`). Persistence is manual
/// against `UserDefaults` (the macro replaces `@AppStorage`, which is a property
/// wrapper incompatible with `@Observable` stored properties).
///
/// This is the BYTE-EXACT PinoFBT 2.0 port: the per-tracker solve is fixed (the
/// reverse-engineered `PinoSolver`/`PinoKinematics`), so the only knobs that affect
/// the live wire are the few below. Everything the old MediaPipe pipeline exposed
/// (leveling, yaw stabilizer, per-joint enables, exaggeration gains, manual smoothing
/// constants) is gone — the model + IK handle all of it.
///
/// Defaults: OSC 127.0.0.1:9000, height 1.74 m, mirror ON (the webcam shows the
/// user mirrored), elbows OFF (clean 6-point), fps multiplier 7×.
@Observable
public final class TrackingConfig {

    // MARK: - Network / output

    public var oscHost: String {
        didSet { UserDefaults.standard.set(oscHost, forKey: Keys.oscHost) }
    }

    public var oscPort: Int {
        didSet {
            // Clamp to the valid UDP port range so the GUI port field (which accepts
            // any Int) can never push an out-of-range value down to `OSCSender` /
            // `NWEndpoint.Port` (a UInt16 conversion that would trap). Re-assigning
            // here re-enters didSet once with an in-range value (no-op).
            let clamped = min(max(oscPort, 1), 65535)
            if clamped != oscPort { oscPort = clamped; return }
            UserDefaults.standard.set(oscPort, forKey: Keys.oscPort)
        }
    }

    /// Selected camera `uniqueID` ("" = auto: prefer an external camera).
    public var cameraDeviceID: String {
        didSet { UserDefaults.standard.set(cameraDeviceID, forKey: Keys.cameraDeviceID) }
    }

    /// Cap the camera capture frame rate (fps) for BOTH the built-in webcam and external
    /// cameras (GoPro/Continuity/UVC). DEFAULT 30 — smooth, light, and plenty for body
    /// pose; raising it only helps if the camera genuinely supports a higher rate. Shared
    /// by on-device and PC modes (the capture session is shared); the PC stream fps is
    /// clamped so it never exceeds this (you can't stream faster than you capture).
    public var cameraMaxFPS: Int {
        didSet {
            let c = min(max(cameraMaxFPS, 1), 240)
            if c != cameraMaxFPS { cameraMaxFPS = c; return }
            UserDefaults.standard.set(cameraMaxFPS, forKey: Keys.cameraMaxFPS)
        }
    }

    // MARK: - Tracking

    /// Real-world user height in meters → `PinoSolver` `user_height_ratio = cm/175`.
    public var userHeightMeters: Double {
        didSet { UserDefaults.standard.set(userHeightMeters, forKey: Keys.userHeightMeters) }
    }

    /// Reflect the SMPL skeleton left↔right (a PROPER reflection: swap L/R joints +
    /// negate X) before the IK, converting the Mac webcam's mirrored handedness to
    /// PinoFBT's capture handedness. DEFAULT true — the webcam shows you mirrored.
    public var mirrorTracking: Bool {
        didSet { UserDefaults.standard.set(mirrorTracking, forKey: Keys.mirrorTracking) }
    }

    /// Send the elbow trackers (slots 3/4). DEFAULT OFF → clean 6-point (chest, hip,
    /// knees, ankles). A single webcam can't see arm depth well, so elbows are the
    /// least reliable trackers (and VRChat recommends fewer trackers for stable IK).
    public var sendElbows: Bool {
        didSet { UserDefaults.standard.set(sendElbows, forKey: Keys.sendElbows) }
    }

    /// FPS-mux output multiplier (1–10×). OSC send rate = clamp(this × inferenceFPS,
    /// inferenceFPS, 120 Hz). Higher = smoother, lower-latency stream (the predictor
    /// fills the gap between inferences with forward-extrapolated sub-frames).
    public var fpsMultiplier: Int {
        didSet { UserDefaults.standard.set(fpsMultiplier, forKey: Keys.fpsMultiplier) }
    }

    /// Forward-lead horizon in MILLISECONDS: the predictive upsampler extrapolates the
    /// joints forward along their velocity by this much to CANCEL pipeline latency.
    /// Higher feels lower-latency but overshoots more on fast direction changes. DEFAULT
    /// 50; clamped 0…150. Applies to BOTH modes — on-device (the upsampler lead) and PC
    /// (the daemon's `--lead-ms`).
    public var predictionLeadMs: Int {
        didSet {
            let c = min(max(predictionLeadMs, 0), 150)
            if c != predictionLeadMs { predictionLeadMs = c; return }
            UserDefaults.standard.set(predictionLeadMs, forKey: Keys.predictionLeadMs)
        }
    }

    // MARK: - Inference on PC (offload)

    /// When ON, Fever does NOT run local inference: it wakes the GPU PC, launches
    /// the headless byte-exact PinoFBT daemon there, and streams this Mac's camera
    /// to it. The PC runs the model + IK and emits the VRChat OSC. `oscHost`/`oscPort`
    /// become the PC's OSC TARGET (127.0.0.1 when VRChat runs on the PC via Quest
    /// Link; the Quest's IP when it runs standalone).
    public var inferenceOnPC: Bool {
        didSet { UserDefaults.standard.set(inferenceOnPC, forKey: Keys.inferenceOnPC) }
    }

    /// GPU PC address (for SSH + the UDP video stream).
    public var pcHost: String {
        didSet { UserDefaults.standard.set(pcHost, forKey: Keys.pcHost) }
    }

    /// SSH user on the GPU PC (passwordless key auth).
    public var pcUser: String {
        didSet { UserDefaults.standard.set(pcUser, forKey: Keys.pcUser) }
    }

    /// GPU PC wired-NIC MAC for Wake-on-LAN (hex, ':'/'-' optional).
    public var pcMAC: String {
        didSet { UserDefaults.standard.set(pcMAC, forKey: Keys.pcMAC) }
    }

    // MARK: - Inference on PC — tracking + transport config (the 1:1 PinoFBT 2.0 defaults)

    /// VRChat OSC target for PC mode — kept SEPARATE from the on-device `oscHost` so
    /// switching modes never clobbers the other's destination. The PC sends here:
    /// 127.0.0.1 = VRChat on the PC (Quest Link); the Quest's IP = standalone.
    public var pcOscHost: String {
        didSet { UserDefaults.standard.set(pcOscHost, forKey: Keys.pcOscHost) }
    }
    public var pcOscPort: Int {
        didSet {
            let c = min(max(pcOscPort, 1), 65535)
            if c != pcOscPort { pcOscPort = c; return }
            UserDefaults.standard.set(pcOscPort, forKey: Keys.pcOscPort)
        }
    }

    /// OSC route in PC mode. false (default) = Direct: the PC daemon sends OSC straight to
    /// `pcOscHost:pcOscPort` (the Quest). true = Relay via Mac: the daemon sends to this
    /// Mac's LAN IP and a Mac-side relay forwards every packet on to the Quest — for when
    /// the wired PC can't reach the Quest's Wi-Fi subnet but the Mac can.
    public var pcOscRelayViaMac: Bool {
        didSet { UserDefaults.standard.set(pcOscRelayViaMac, forKey: Keys.pcOscRelayViaMac) }
    }

    /// 8-point trackers (elbows ON) — the PinoFBT 2.0 desktop default. ON for PC mode
    /// (the GPU runs the full byte-exact arm solver, so elbows are first-class here).
    public var pcSendElbows: Bool {
        didSet { UserDefaults.standard.set(pcSendElbows, forKey: Keys.pcSendElbows) }
    }


    /// Mac→PC H.264 stream geometry / rate / bitrate.
    public var pcStreamWidth: Int {
        didSet { UserDefaults.standard.set(pcStreamWidth, forKey: Keys.pcStreamWidth) }
    }
    public var pcStreamHeight: Int {
        didSet { UserDefaults.standard.set(pcStreamHeight, forKey: Keys.pcStreamHeight) }
    }
    public var pcStreamFPS: Int {
        didSet { UserDefaults.standard.set(pcStreamFPS, forKey: Keys.pcStreamFPS) }
    }
    public var pcBitrateMbps: Int {
        didSet { UserDefaults.standard.set(pcBitrateMbps, forKey: Keys.pcBitrateMbps) }
    }
    /// Politeness for sharing the PC: run the daemon below-normal priority so a
    /// person using the PC always gets the GPU/CPU first.
    public var pcPoliteMode: Bool {
        didSet { UserDefaults.standard.set(pcPoliteMode, forKey: Keys.pcPoliteMode) }
    }
    /// Cap PC-side processing FPS (0 = unlimited; the stream rate is the practical
    /// ceiling). Lower = lighter GPU load when sharing the PC.
    public var pcFpsCap: Int {
        didSet { UserDefaults.standard.set(pcFpsCap, forKey: Keys.pcFpsCap) }
    }

    /// In PC mode, draw the skeleton overlay (sent back from the PC) over the live
    /// camera preview. OFF = camera preview only. DEFAULT on.
    public var pcShowSkeleton: Bool {
        didSet { UserDefaults.standard.set(pcShowSkeleton, forKey: Keys.pcShowSkeleton) }
    }

    // MARK: - Inference on PC — pose model (NLF vs GVHMR)

    /// Which pose model the PC daemon runs in PC mode. "nlf" (default) = the byte-exact
    /// PinoFBT 2.0 NLF model + IK (per-frame, camera-frame). "gvhmr" = the world-grounded,
    /// gravity-aligned GVHMR model. Only affects PC mode; the OSC route + target are reused
    /// unchanged for both. Stored as a string ("nlf" | "gvhmr") so it round-trips trivially.
    public var pcModel: String {
        didSet { UserDefaults.standard.set(pcModel, forKey: Keys.pcModel) }
    }

    /// GVHMR readout lookahead (frames). Trades latency against smoothness: higher reads
    /// further ahead (smoother, more latency), lower is snappier. DEFAULT 5; range 0–15.
    /// GVHMR-only; needs live VRChat tuning.
    public var gvhmrK: Int {
        didSet {
            let c = min(max(gvhmrK, 0), 15)
            if c != gvhmrK { gvhmrK = c; return }
            UserDefaults.standard.set(gvhmrK, forKey: Keys.gvhmrK)
        }
    }

    /// GVHMR facing tune — pass `--mirror` to the GVHMR daemon when ON. GVHMR-only; needs
    /// live VRChat tuning (the world-grounded frame's handedness can't be known beforehand).
    public var gvhmrMirror: Bool {
        didSet { UserDefaults.standard.set(gvhmrMirror, forKey: Keys.gvhmrMirror) }
    }

    /// GVHMR facing tune — pass `--flip-x` to the GVHMR daemon when ON. GVHMR-only; needs
    /// live VRChat tuning, paired with `gvhmrMirror` to dial in the avatar's facing.
    public var gvhmrFlipX: Bool {
        didSet { UserDefaults.standard.set(gvhmrFlipX, forKey: Keys.gvhmrFlipX) }
    }

    /// GVHMR camera mode. false (default) = Static: the fast real-time path (fixed camera).
    /// true = Moving: run GVHMR world-grounded with camera motion ("as it was meant
    /// originally") — heavier/experimental. GVHMR-only; passes `--moving` when true.
    public var gvhmrMoving: Bool {
        didSet { UserDefaults.standard.set(gvhmrMoving, forKey: Keys.gvhmrMoving) }
    }

    /// GVHMR foot-contact damping — pass `--foot-contact` when ON. Uses GVHMR's own
    /// foot-contact prediction to damp planted feet (less foot-slide). DEFAULT off;
    /// GVHMR-only; tune live against VRChat.
    public var gvhmrFootContact: Bool {
        didSet { UserDefaults.standard.set(gvhmrFootContact, forKey: Keys.gvhmrFootContact) }
    }

    /// GVHMR native rotations — pass `--native-rot` when ON. Uses GVHMR's own joint
    /// rotations instead of re-deriving them from positions (richer twist/roll).
    /// DEFAULT off; GVHMR-only; experimental, needs live facing/convention tuning.
    public var gvhmrNativeRot: Bool {
        didSet { UserDefaults.standard.set(gvhmrNativeRot, forKey: Keys.gvhmrNativeRot) }
    }

    // MARK: - Init (loads persisted values; falls back to defaults)

    public init() {
        let d = UserDefaults.standard
        oscHost = d.string(forKey: Keys.oscHost) ?? "127.0.0.1"
        oscPort = d.object(forKey: Keys.oscPort) as? Int ?? 9000
        cameraDeviceID = d.string(forKey: Keys.cameraDeviceID) ?? ""
        cameraMaxFPS = min(240, max(1, d.object(forKey: Keys.cameraMaxFPS) as? Int ?? 30))
        userHeightMeters = d.object(forKey: Keys.userHeightMeters) as? Double ?? 1.74
        mirrorTracking = d.object(forKey: Keys.mirrorTracking) as? Bool ?? true
        sendElbows = d.object(forKey: Keys.sendElbows) as? Bool ?? false
        fpsMultiplier = min(10, max(1, d.object(forKey: Keys.fpsMultiplier) as? Int ?? 7))
        predictionLeadMs = min(150, max(0, d.object(forKey: Keys.predictionLeadMs) as? Int ?? 50))
        inferenceOnPC = d.object(forKey: Keys.inferenceOnPC) as? Bool ?? false
        // No hardcoded host/user/MAC defaults: those are personal/network identifiers
        // (this is a public repo) and the Start button is already gated on a blank
        // host+MAC. A real value the user enters persists here in UserDefaults.
        pcHost = d.string(forKey: Keys.pcHost) ?? ""
        pcUser = d.string(forKey: Keys.pcUser) ?? ""
        pcMAC = d.string(forKey: Keys.pcMAC) ?? ""
        // PC mode = the 1:1 PinoFBT 2.0 config by default (8-point, mirrored, PCVR target).
        pcOscHost = d.string(forKey: Keys.pcOscHost) ?? "127.0.0.1"
        pcOscPort = d.object(forKey: Keys.pcOscPort) as? Int ?? 9000
        pcOscRelayViaMac = d.object(forKey: Keys.pcOscRelayViaMac) as? Bool ?? false
        pcSendElbows = d.object(forKey: Keys.pcSendElbows) as? Bool ?? true
        // 960×540 default: the model crops to a ~256px person box, so 540p is already
        // supersampled for it, while 720p just spends more MJPEG encode + UDP packets
        // (reassembly jitter) + PC decode for pixels the model throws away. The Mac
        // preview is the LOCAL full-res camera, not this stream, so this is invisible
        // on-screen and only cuts transport latency. Migrate installs still sitting on
        // the old 1280×720 default down to 540p (a deliberate higher pick is preserved).
        var sw = d.object(forKey: Keys.pcStreamWidth) as? Int ?? 960
        var sh = d.object(forKey: Keys.pcStreamHeight) as? Int ?? 540
        if sw == 1280 && sh == 720 { sw = 960; sh = 540 }
        pcStreamWidth = sw; pcStreamHeight = sh
        pcStreamFPS = d.object(forKey: Keys.pcStreamFPS) as? Int ?? 60
        pcBitrateMbps = d.object(forKey: Keys.pcBitrateMbps) as? Int ?? 10
        pcPoliteMode = d.object(forKey: Keys.pcPoliteMode) as? Bool ?? false
        pcFpsCap = d.object(forKey: Keys.pcFpsCap) as? Int ?? 0
        pcShowSkeleton = d.object(forKey: Keys.pcShowSkeleton) as? Bool ?? true
        pcModel = d.string(forKey: Keys.pcModel) ?? "nlf"
        // GVHMR reads out the pose k frames in the PAST (the denoiser's bidirectional
        // window uses k FUTURE frames to stabilize that frame). k is therefore pure
        // latency: ~k/fps seconds (k=5 @ 30fps ≈ 167ms, @ 25fps ≈ 200ms). The window's
        // 119 PAST frames already denoise the readout even at k=0, and output-side OneEuro
        // + the predictive upsampler absorb residual jitter — so a low k is a big latency
        // win for marginal quality. Default 2; migrate installs off the old laggy k=5.
        var gk = min(15, max(0, d.object(forKey: Keys.gvhmrK) as? Int ?? 2))
        if d.object(forKey: Keys.gvhmrK) as? Int == 5 { gk = 2 }
        gvhmrK = gk
        gvhmrMirror = d.object(forKey: Keys.gvhmrMirror) as? Bool ?? false
        gvhmrFlipX = d.object(forKey: Keys.gvhmrFlipX) as? Bool ?? false
        gvhmrMoving = d.object(forKey: Keys.gvhmrMoving) as? Bool ?? false
        gvhmrFootContact = d.object(forKey: Keys.gvhmrFootContact) as? Bool ?? false
        gvhmrNativeRot = d.object(forKey: Keys.gvhmrNativeRot) as? Bool ?? false
    }

    // MARK: - Persistence keys

    private enum Keys {
        static let oscHost = "oscHost"
        static let oscPort = "oscPort"
        static let cameraDeviceID = "cameraDeviceID"
        static let cameraMaxFPS = "cameraMaxFPS"
        static let userHeightMeters = "userHeightMeters"
        static let mirrorTracking = "mirrorTracking"
        static let sendElbows = "sendElbows"
        static let fpsMultiplier = "fpsMultiplier"
        static let predictionLeadMs = "predictionLeadMs"
        static let inferenceOnPC = "inferenceOnPC"
        static let pcHost = "pcHost"
        static let pcUser = "pcUser"
        static let pcMAC = "pcMAC"
        static let pcOscHost = "pcOscHost"
        static let pcOscPort = "pcOscPort"
        static let pcOscRelayViaMac = "pcOscRelayViaMac"
        static let pcSendElbows = "pcSendElbows"
        static let pcStreamWidth = "pcStreamWidth"
        static let pcStreamHeight = "pcStreamHeight"
        static let pcStreamFPS = "pcStreamFPS"
        static let pcBitrateMbps = "pcBitrateMbps"
        static let pcPoliteMode = "pcPoliteMode"
        static let pcFpsCap = "pcFpsCap"
        static let pcShowSkeleton = "pcShowSkeleton"
        static let pcModel = "pcModel"
        static let gvhmrK = "gvhmrK"
        static let gvhmrMirror = "gvhmrMirror"
        static let gvhmrFlipX = "gvhmrFlipX"
        static let gvhmrMoving = "gvhmrMoving"
        static let gvhmrFootContact = "gvhmrFootContact"
        static let gvhmrNativeRot = "gvhmrNativeRot"
    }
}
