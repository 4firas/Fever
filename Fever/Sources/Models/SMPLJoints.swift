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

/// Tracker map PINO — the BYTE-EXACT desktop PinoFBT 2.0 VRChat slot numbering
/// (live-confirmed by correlating each slot's euler magnitude to its solver quat):
///   1=chest 2=hip 3=L_elbow 4=R_elbow 5=L_knee 6=R_knee 7=L_ankle 8=R_ankle.
/// Both `/position` and `/rotation` are emitted for all 8; head is `/position` only.
/// This is the default for the 1:1 port (`PinoSolver` keys its slots by these
/// indices). The `joint` field is informational; positions come straight from the
/// solver by slot index.
public enum TrackerMapPino {
    public static let slots: [VRTrackerSlot] = [
        VRTrackerSlot(index: 1, path: "1", joint: .spine3,     sendsRotation: true),  // chest
        VRTrackerSlot(index: 2, path: "2", joint: .pelvis,     sendsRotation: true),  // hip
        VRTrackerSlot(index: 3, path: "3", joint: .leftElbow,  sendsRotation: true),  // L_elbow
        VRTrackerSlot(index: 4, path: "4", joint: .rightElbow, sendsRotation: true),  // R_elbow
        VRTrackerSlot(index: 5, path: "5", joint: .leftKnee,   sendsRotation: true),  // L_knee
        VRTrackerSlot(index: 6, path: "6", joint: .rightKnee,  sendsRotation: true),  // R_knee
        VRTrackerSlot(index: 7, path: "7", joint: .leftAnkle,  sendsRotation: true),  // L_ankle
        VRTrackerSlot(index: 8, path: "8", joint: .rightAnkle, sendsRotation: true),  // R_ankle
    ]
}

/// The semantic body trackers Fever streams to VRChat: the 8 numbered slots plus
/// the head reference. Purely a UI/labelling type — the wire slot numbering lives
/// in `TrackerMapPino` and the solved values come straight from `PinoSolver` by
/// slot index. (1=chest 2=hip 3/4=elbows 5/6=knees 7/8=ankles, head=anchor.)
public enum JointType: String, CaseIterable, Codable, Sendable {
    case head, chest, hip
    case leftElbow, rightElbow
    case leftKnee, rightKnee
    case leftFoot, rightFoot

    public var isLeft: Bool {
        switch self {
        case .leftElbow, .leftKnee, .leftFoot: return true
        default: return false
        }
    }

    /// Fixed PinoFBT desktop OSC slot path for this tracker ("1"…"8", or "head").
    public var pinoSlot: String {
        switch self {
        case .chest:      "1"
        case .hip:        "2"
        case .leftElbow:  "3"
        case .rightElbow: "4"
        case .leftKnee:   "5"
        case .rightKnee:  "6"
        case .leftFoot:   "7"
        case .rightFoot:  "8"
        case .head:       "head"
        }
    }

    /// Reverse map: the tracker a given PinoFBT slot path drives (nil if unknown).
    public static func forPinoSlot(_ slot: String) -> JointType? {
        switch slot {
        case "1": .chest
        case "2": .hip
        case "3": .leftElbow
        case "4": .rightElbow
        case "5": .leftKnee
        case "6": .rightKnee
        case "7": .leftFoot
        case "8": .rightFoot
        case "head": .head
        default: nil
        }
    }
}
