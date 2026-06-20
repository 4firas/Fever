import simd

/// The 9 VRChat joints Fever derives from 33 BlazePose landmarks.
///
/// NOTE: the wire slot mapping (1=hip, 2=leftFoot, 3=rightFoot, optional 4..8,
/// plus the head reference) lives in `TrackerAssembler`, NOT here. A joint type
/// is purely a semantic label; the assembler decides which numbered VRChat
/// tracker slot carries it (one body part per index, every frame — no cycling).
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
}

/// A solved VR joint ready for coordinate mapping + OSC transmission.
/// Carries position in meters and orientation as a world-space quaternion in
/// the solver's (Vision-derived) frame; `CoordinateMapper` performs the single
/// authoritative conversion into VRChat space (Z-flip, meters scale, ZXY euler).
public struct VRJoint: Sendable {
    public let type: JointType
    public var position: SIMD3<Float>     // meters, solver frame
    public var rotation: simd_quatf       // quaternion, solver frame
    public var confidence: Float

    public init(type: JointType,
                position: SIMD3<Float>,
                rotation: simd_quatf,
                confidence: Float = 1) {
        self.type = type
        self.position = position
        self.rotation = rotation
        self.confidence = confidence
    }
}
