import simd

/// Maps the 9 solved `VRJoint` values onto FIXED VRChat numbered tracker slots,
/// applying the single authoritative `CoordinateMapper` conversion to each
/// joint's position and orientation.
///
/// Default slot map (one body part per index, every frame — no cycling):
///   1 = hip, 2 = leftFoot, 3 = rightFoot         (MVP defaults)
///   4 = chest, 5 = leftKnee, 6 = rightKnee,
///   7 = leftElbow, 8 = rightElbow                 (optional)
///   head = continuous space-alignment reference   (handled separately)
///
/// Only joints in `enabled` are emitted. The `head` joint, if present and
/// enabled, is returned separately so the pipeline can stream it to
/// `/tracking/trackers/head/...` and drive `sendHeadSnapPulse` on calibrate.
public struct TrackerAssembler {

    /// The set of joint types the user wants transmitted.
    public var enabled: Set<JointType>

    /// JointType -> wire slot identifier ("1".."8"). The head joint is NOT in
    /// this map; it always uses the reserved "head" slot.
    public var slotMap: [JointType: String]

    public init(enabled: Set<JointType>, slotMap: [JointType: String]) {
        self.enabled = enabled
        self.slotMap = slotMap
    }

    /// The default fixed numbered slot mapping per the locked spec.
    public static let defaultSlotMap: [JointType: String] = [
        .hip:        "1",
        .leftFoot:   "2",
        .rightFoot:  "3",
        .chest:      "4",
        .leftKnee:   "5",
        .rightKnee:  "6",
        .leftElbow:  "7",
        .rightElbow: "8",
    ]

    /// Convert + slot the joints for this frame.
    ///
    /// - Returns: `body` = numbered-slot trackers for every enabled, mapped
    ///   joint; `head` = the head reference tracker if the head joint is present
    ///   and enabled, else `nil`.
    public func assemble(_ joints: [VRJoint],
                         mapper: CoordinateMapper) -> (body: [OSCTracker], head: OSCTracker?) {
        var body: [OSCTracker] = []
        body.reserveCapacity(joints.count)
        var head: OSCTracker?

        for joint in joints {
            // The head is the tracking-space alignment REFERENCE, not a numbered
            // body tracker. It is always assembled when solved (independent of
            // the numbered `enabled` set); the pipeline decides whether to
            // transmit it based on `TrackingConfig.sendHeadReference`. All other
            // joints must be explicitly enabled.
            if joint.type == .head {
                let position = mapper.toVRChatPosition(joint.position)
                let euler = mapper.toVRChatEulerDegrees(joint.rotation)
                head = OSCTracker(slot: "head", position: position, eulerDegrees: euler)
                continue
            }

            guard enabled.contains(joint.type) else { continue }

            let position = mapper.toVRChatPosition(joint.position)
            let euler = mapper.toVRChatEulerDegrees(joint.rotation)

            guard let slot = slotMap[joint.type] else { continue }
            body.append(OSCTracker(slot: slot, position: position, eulerDegrees: euler))
        }

        return (body, head)
    }
}
