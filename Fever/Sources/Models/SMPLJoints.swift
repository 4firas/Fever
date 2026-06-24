import Foundation

/// Canonical SMPL-24 body joints — the exact output order of the extracted NLF
/// (Neural Localizer Fields) model. Confirmed by numbered-overlay probing of the
/// ONNX on multiple subjects (see PINOFBT_VRCHAT_OSC_FINDINGS.md §2).
public enum SMPLJoint: Int, CaseIterable, Sendable {
    case pelvis = 0, leftHip, rightHip, spine1, leftKnee, rightKnee, spine2,
         leftAnkle, rightAnkle, spine3, leftFoot, rightFoot, neck,
         leftCollar, rightCollar, head, leftShoulder, rightShoulder,
         leftElbow, rightElbow, leftWrist, rightWrist, leftHand, rightHand

    public static let count = 24

    /// SMPL kinematic-tree parent index per joint (pelvis is the root, -1).
    public static let parentIndex: [Int] =
        [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]

    /// Parent joint, or nil for the pelvis root.
    public var parent: SMPLJoint? {
        let p = Self.parentIndex[rawValue]
        return p >= 0 ? SMPLJoint(rawValue: p) : nil
    }
}

/// One VRChat OSC tracker slot, wired to the SMPL joint that drives it.
public struct VRTrackerSlot: Sendable {
    public let index: Int        // 1...8 (numbered tracker), or 0 for the head reference
    public let path: String      // OSC address component: "1"..."8" or "head"
    public let joint: SMPLJoint
    public let sendsRotation: Bool
    public init(index: Int, path: String, joint: SMPLJoint, sendsRotation: Bool) {
        self.index = index; self.path = path; self.joint = joint; self.sendsRotation = sendsRotation
    }
}

/// Tracker map A — wire-confirmed PinoFBT VRChat mapping (findings §7). Slot 1
/// (hip/pelvis) is the ROOT and the only rotation carrier in the captured build;
/// every other slot is position-only. `head` is a position-only alignment anchor.
public enum TrackerMapA {
    // Standard reliable FBT set: hip + chest + feet + knees. Elbow trackers are
    // dropped — monocular arm tracking is unreliable (the floating dots above the
    // shoulder + the elbow driving the chest dot in move-calibrate), and elbow FBT
    // trackers are uncommon in VRChat (arms ride the controllers/hands). Re-add if
    // upper-arm tracking ever proves stable.
    public static let slots: [VRTrackerSlot] = [
        VRTrackerSlot(index: 1, path: "1", joint: .pelvis,     sendsRotation: true),
        VRTrackerSlot(index: 2, path: "2", joint: .leftAnkle,  sendsRotation: false),
        VRTrackerSlot(index: 3, path: "3", joint: .rightAnkle, sendsRotation: false),
        VRTrackerSlot(index: 4, path: "4", joint: .spine3,     sendsRotation: false),
        VRTrackerSlot(index: 5, path: "5", joint: .leftKnee,   sendsRotation: false),
        VRTrackerSlot(index: 6, path: "6", joint: .rightKnee,  sendsRotation: false),
    ]
}

/// Which model joint feeds the continuous `head/position` anchor. Default head(15);
/// exposed because the exact head-anchor source is an open item (findings §7/§12).
public enum HeadAnchorSource: String, Sendable, CaseIterable {
    case head15, neck12, headNeckMidpoint

    /// Primary joint (midpoint blends head+neck downstream).
    public var primaryJoint: SMPLJoint {
        switch self {
        case .head15:           return .head
        case .neck12:           return .neck
        case .headNeckMidpoint: return .head
        }
    }
}
