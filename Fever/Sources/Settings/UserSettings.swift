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

    /// 8-point trackers (elbows ON) — the PinoFBT 2.0 desktop default. ON for PC mode
    /// (the GPU runs the full byte-exact arm solver, so elbows are first-class here).
    public var pcSendElbows: Bool {
        didSet { UserDefaults.standard.set(pcSendElbows, forKey: Keys.pcSendElbows) }
    }

    /// Horizontally mirror the camera before streaming — matches PinoFBT's internal
    /// `cv2.flip` so handedness lands the same as the original desktop app.
    public var pcFlipCamera: Bool {
        didSet { UserDefaults.standard.set(pcFlipCamera, forKey: Keys.pcFlipCamera) }
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

    // MARK: - Init (loads persisted values; falls back to defaults)

    public init() {
        let d = UserDefaults.standard
        oscHost = d.string(forKey: Keys.oscHost) ?? "127.0.0.1"
        oscPort = d.object(forKey: Keys.oscPort) as? Int ?? 9000
        cameraDeviceID = d.string(forKey: Keys.cameraDeviceID) ?? ""
        userHeightMeters = d.object(forKey: Keys.userHeightMeters) as? Double ?? 1.74
        mirrorTracking = d.object(forKey: Keys.mirrorTracking) as? Bool ?? true
        sendElbows = d.object(forKey: Keys.sendElbows) as? Bool ?? false
        fpsMultiplier = min(10, max(1, d.object(forKey: Keys.fpsMultiplier) as? Int ?? 7))
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
        pcSendElbows = d.object(forKey: Keys.pcSendElbows) as? Bool ?? true
        pcFlipCamera = d.object(forKey: Keys.pcFlipCamera) as? Bool ?? true
        pcStreamWidth = d.object(forKey: Keys.pcStreamWidth) as? Int ?? 1280
        pcStreamHeight = d.object(forKey: Keys.pcStreamHeight) as? Int ?? 720
        pcStreamFPS = d.object(forKey: Keys.pcStreamFPS) as? Int ?? 30
        pcBitrateMbps = d.object(forKey: Keys.pcBitrateMbps) as? Int ?? 8
        pcPoliteMode = d.object(forKey: Keys.pcPoliteMode) as? Bool ?? false
        pcFpsCap = d.object(forKey: Keys.pcFpsCap) as? Int ?? 0
        pcShowSkeleton = d.object(forKey: Keys.pcShowSkeleton) as? Bool ?? true
    }

    // MARK: - Persistence keys

    private enum Keys {
        static let oscHost = "oscHost"
        static let oscPort = "oscPort"
        static let cameraDeviceID = "cameraDeviceID"
        static let userHeightMeters = "userHeightMeters"
        static let mirrorTracking = "mirrorTracking"
        static let sendElbows = "sendElbows"
        static let fpsMultiplier = "fpsMultiplier"
        static let inferenceOnPC = "inferenceOnPC"
        static let pcHost = "pcHost"
        static let pcUser = "pcUser"
        static let pcMAC = "pcMAC"
        static let pcOscHost = "pcOscHost"
        static let pcOscPort = "pcOscPort"
        static let pcSendElbows = "pcSendElbows"
        static let pcFlipCamera = "pcFlipCamera"
        static let pcStreamWidth = "pcStreamWidth"
        static let pcStreamHeight = "pcStreamHeight"
        static let pcStreamFPS = "pcStreamFPS"
        static let pcBitrateMbps = "pcBitrateMbps"
        static let pcPoliteMode = "pcPoliteMode"
        static let pcFpsCap = "pcFpsCap"
        static let pcShowSkeleton = "pcShowSkeleton"
    }
}
