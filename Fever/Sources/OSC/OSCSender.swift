import Foundation
import Network
import simd

/// A single VRChat tracker frame ready for the wire: a slot identifier
/// ("1".."8" or "head"), a world-space position in METERS, and an orientation
/// expressed as Unity ZXY EULER ANGLES IN DEGREES.
///
/// The coordinate/handedness/unit conversion has already happened upstream in
/// `CoordinateMapper`; `OSCSender` only serializes and transmits.
public struct OSCTracker: Sendable {
    public let slot: String
    public let position: SIMD3<Float>
    public let eulerDegrees: SIMD3<Float>

    public init(slot: String, position: SIMD3<Float>, eulerDegrees: SIMD3<Float>) {
        self.slot = slot
        self.position = position
        self.eulerDegrees = eulerDegrees
    }
}

/// Actor owning ONE long-lived UDP `NWConnection` to the VRChat OSC receive
/// endpoint (default `127.0.0.1:9000`).
///
/// Wire contract (per verified verdicts):
///   - `/tracking/trackers/<slot>/position` = exactly 3 big-endian float32
///     (type tag `,fff`) in METERS.
///   - `/tracking/trackers/<slot>/rotation` = exactly 3 big-endian float32
///     (type tag `,fff`) EULER ANGLES IN DEGREES, applied by VRChat internally
///     in ZXY order. There is NO 4-float quaternion variant on the wire.
///   - Fixed slot identifiers, one body part per index per frame; no cycling.
///
/// Sends are fire-and-forget (UDP) so the high-frequency tracking path never
/// blocks on network completion.
public actor OSCSender {

    public let host: String
    public let port: Int

    private var connection: NWConnection?
    private let queue = DispatchQueue(label: "fever.osc", qos: .userInitiated)

    /// When false (default), trackers are sent POSITION-ONLY — no `/rotation`
    /// messages at all. Streaming rotation makes VRChat slow-lerp its tracking
    /// space toward our euler yaw, so after an in-game auto-calibrate the avatar
    /// eases back to the wrong orientation. Position-only lets VRChat's IK solve
    /// limb rotation itself, which is the stable behavior for monocular tracking.
    private var rotationEnabled: Bool = false

    /// Slots that carry `/rotation`: hip + feet + knees + elbows. The CHEST (slot 4)
    /// is excluded — wire-confirmed PinoFBT does NOT send chest rotation (capture
    /// 2026-06-24: /tracking/trackers/4/rotation never present); the chest rides
    /// VRChat's IK off positions. Feet/knees/elbows DO rotate (restored — they
    /// tracked well). Head is position-only (the re-origin anchor).
    public static let rotationSlots: Set<String> = ["1", "2", "3", "5", "6", "7", "8"]

    /// Per-slot LAST-VALID position, keyed by slot id ("1".."8", "head"). When a
    /// joint blips out of detection or the solver yields a degenerate value, the
    /// upstream pipeline can hand us (0,0,0)/NaN, which teleports that tracker to
    /// the OSC-space origin (~2 m off head-relative) and spazzes the avatar — the
    /// #1 defect vs PinoFBT (which sent ZERO such samples because it HOLDS the last
    /// good value). We do the same here, at the single wire chokepoint so BOTH the
    /// live send and the steady repeater are covered: if the incoming position is
    /// invalid (NaN or all-zero), emit the last-valid position for that slot
    /// instead. A real (0,0,0) is never emitted. Seeded by the first valid sample;
    /// until then an invalid sample for an unseen slot is dropped (not sent).
    private var lastValidPosition: [String: SIMD3<Float>] = [:]

    public init(host: String, port: Int) {
        self.host = host
        self.port = port
    }

    /// Toggle whether `/rotation` is transmitted (default off = position-only).
    public func setRotationEnabled(_ on: Bool) { rotationEnabled = on }

    /// Seed a hold-last-valid FALLBACK for every slot we will ever send, so the
    /// VERY FIRST frame can already emit every slot even if a joint hasn't produced
    /// a valid sample yet (the ~5% early-session foot dropout: feet had NO prior
    /// `lastValidPosition`, so an invalid first sample was skipped and the slot went
    /// missing — PinoFBT sends every slot every frame). The seed is a NEUTRAL,
    /// non-zero placeholder (never (0,0,0)), overwritten the instant a real valid
    /// sample arrives via `resolveHeldPosition`. Only seeds slots that have no value
    /// yet, so a re-`start()` never clobbers a good held pose. Idempotent.
    ///
    /// We use a tiny per-slot Y nudge (a few mm) rather than a true (0,0,0) so the
    /// placeholder is always `isValidPosition`-valid; the magnitudes are negligible
    /// and are replaced within the first valid frame for that slot.
    public func seedSlots(_ slots: [String]) {
        for slot in slots where lastValidPosition[slot] == nil {
            lastValidPosition[slot] = SIMD3<Float>(0, 0.001, 0)
        }
    }

    /// Open the long-lived UDP connection. Idempotent. No-op (leaves
    /// `connection` nil, so all sends safely no-op) when the port is outside the
    /// valid UDP range — `UInt16(port)` would otherwise TRAP. The GUI clamps the
    /// port at the config layer and the CLI validates it, but guard here too so a
    /// directly-constructed sender can never crash the process.
    public func start() {
        guard connection == nil else { return }
        guard let portValue = UInt16(exactly: port), portValue >= 1 else {
            return
        }
        let endpoint = NWEndpoint.hostPort(
            host: NWEndpoint.Host(host),
            port: NWEndpoint.Port(integerLiteral: portValue)
        )
        let conn = NWConnection(to: endpoint, using: .udp)
        // Observe connection lifecycle. UDP sends are fire-and-forget, so a
        // connection that enters `.failed`/`.cancelled` would otherwise linger as
        // a dead resource while every send silently no-ops forever. On failure,
        // tear it down so a later `start()` can rebuild a healthy one. Hop back
        // into the actor to mutate `connection` safely.
        conn.stateUpdateHandler = { [weak self] state in
            switch state {
            case .failed, .cancelled:
                Task { await self?.handleConnectionDown(conn) }
            default:
                break
            }
        }
        conn.start(queue: queue)
        connection = conn
    }

    /// Tear down a connection that went `.failed`/`.cancelled`, but only if it is
    /// still the live one (avoid clobbering a connection a later `start()` made).
    private func handleConnectionDown(_ conn: NWConnection) {
        guard connection === conn else { return }
        connection?.cancel()
        connection = nil
    }

    /// Tear down the connection. Idempotent.
    public func stop() {
        connection?.stateUpdateHandler = nil
        connection?.cancel()
        connection = nil
    }

    /// Send a full frame of trackers. Each tracker emits two OSC messages:
    /// `/position` (`,fff` meters) and `/rotation` (`,fff` euler degrees).
    public func send(trackers: [OSCTracker]) {
        guard let connection else { return }
        for tracker in trackers {
            // Hold last-valid: only proceed if we have a real position to emit
            // (this frame's, or the held one). If neither, skip the slot entirely
            // rather than emit a (0,0,0) teleport.
            guard let held = resolveHeldPosition(tracker) else { continue }
            sendPosition(slot: tracker.slot, position: held, over: connection)
            // Position-only for the extremities (knees/elbows): only hip/feet/chest
            // carry rotation, so depth-fragile limb rotation never fights the IK.
            if rotationEnabled, Self.rotationSlots.contains(tracker.slot) {
                sendRotation(tracker, over: connection)
            }
        }
    }

    /// Send the head POSITION only — the alignment anchor (matches PinoFBT, which
    /// streams `/tracking/trackers/head/position` continuously). VRChat shifts the
    /// whole OSC space so this aligns to the avatar's head bone, which CANCELS any
    /// absolute frame offset in the body trackers (the reason a head-less stream
    /// lands the body metres off in +X). NEVER send head ROTATION — that makes
    /// VRChat slow-lerp the yaw and the body drifts back wrong (PinoFBT also omits
    /// head rotation: it sends only head/position).
    public func sendHeadPosition(_ head: OSCTracker) {
        guard let connection else { return }
        // Same hold-last-valid policy as the body so a head dropout never sends
        // (0,0,0) (which would yank the whole re-origined OSC space).
        guard let held = resolveHeldPosition(head) else { return }
        sendPosition(slot: head.slot, position: held, over: connection)
    }

    // MARK: - Hold-last-valid

    /// Resolve the position to actually emit for a slot, applying the
    /// hold-last-valid policy: a finite, non-zero incoming position is emitted and
    /// latched; an invalid one (NaN or all-zero) is replaced by the slot's last
    /// valid position; if none has ever been seen, returns nil so the caller skips
    /// the slot (never emits a fabricated (0,0,0)).
    private func resolveHeldPosition(_ t: OSCTracker) -> SIMD3<Float>? {
        if Self.isValidPosition(t.position) {
            lastValidPosition[t.slot] = t.position
            return t.position
        }
        return lastValidPosition[t.slot]
    }

    /// A position is valid when every component is finite AND it is not the exact
    /// origin (a (0,0,0) is the dropout/absent sentinel, never a legitimate pose
    /// for a centred-frame tracker).
    private static func isValidPosition(_ p: SIMD3<Float>) -> Bool {
        guard p.x.isFinite, p.y.isFinite, p.z.isFinite else { return false }
        return p != .zero
    }

    // MARK: - Per-tracker encoding

    private func sendPosition(slot: String, position: SIMD3<Float>, over connection: NWConnection) {
        let msg = OSCMessage(
            address: "/tracking/trackers/\(slot)/position",
            arguments: [.float(position.x),
                        .float(position.y),
                        .float(position.z)]
        )
        send(msg.encoded(), over: connection)
    }

    private func sendRotation(_ t: OSCTracker, over connection: NWConnection) {
        // Three floats = Euler ANGLES IN DEGREES (ZXY), NOT a quaternion.
        let msg = OSCMessage(
            address: "/tracking/trackers/\(t.slot)/rotation",
            arguments: [.float(t.eulerDegrees.x),
                        .float(t.eulerDegrees.y),
                        .float(t.eulerDegrees.z)]
        )
        send(msg.encoded(), over: connection)
    }

    private func send(_ data: Data, over connection: NWConnection) {
        connection.send(content: data, completion: .contentProcessed { _ in })
    }
}
