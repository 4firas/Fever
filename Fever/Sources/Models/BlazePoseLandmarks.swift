import simd
import Foundation

/// BlazePose 33-landmark indices (MediaPipe Pose topology).
public enum BlazePose {
    public enum Landmark: Int, CaseIterable, Sendable {
        case nose = 0
        case leftEyeInner = 1, leftEye = 2, leftEyeOuter = 3
        case rightEyeInner = 4, rightEye = 5, rightEyeOuter = 6
        case leftEar = 7, rightEar = 8
        case mouthLeft = 9, mouthRight = 10
        case leftShoulder = 11, rightShoulder = 12
        case leftElbow = 13, rightElbow = 14
        case leftWrist = 15, rightWrist = 16
        case leftPinky = 17, rightPinky = 18
        case leftIndex = 19, rightIndex = 20
        case leftThumb = 21, rightThumb = 22
        case leftHip = 23, rightHip = 24
        case leftKnee = 25, rightKnee = 26
        case leftAnkle = 27, rightAnkle = 28
        case leftHeel = 29, rightHeel = 30
        case leftFootIndex = 31, rightFootIndex = 32
    }
}

/// A normalized 3D landmark (BlazePose image space + visibility).
public struct NormalizedLandmark: Equatable, Sendable {
    public var position: SIMD3<Float>   // x,y ∈ [0,1], z relative depth
    public var visibility: Float         // [0,1]
    public var presence: Float           // [0,1]

    public init(position: SIMD3<Float>, visibility: Float = 1, presence: Float = 1) {
        self.position = position
        self.visibility = visibility
        self.presence = presence
    }
}


/// Result of a single pose inference pass.
public struct PoseResult: Sendable {
    public let landmarks: [NormalizedLandmark]   // 33 entries

    /// RAW 2D detected landmarks for the live preview overlay, in SCREEN-
    /// normalized coordinates: x ∈ [0,1] from the LEFT, y ∈ [0,1] from the TOP.
    /// 33-slot, indexed by `BlazePose.Landmark.rawValue`; absent landmarks are
    /// `SIMD2(.nan, .nan)`. UNSMOOTHED and independent of the metric `landmarks`
    /// (which feed the OSC/tracker path); the preview draws these directly.
    public var imagePoints: [SIMD2<Float>]

    public let timestamp: Double

    public init(landmarks: [NormalizedLandmark],
                timestamp: Double,
                imagePoints: [SIMD2<Float>] = []) {
        self.landmarks = landmarks
        self.timestamp = timestamp
        self.imagePoints = imagePoints
    }

    public subscript(_ l: BlazePose.Landmark) -> NormalizedLandmark {
        landmarks[l.rawValue]
    }
}

public extension Array where Element == NormalizedLandmark {
    subscript(_ l: BlazePose.Landmark) -> NormalizedLandmark {
        get { self[l.rawValue] }
        set { self[l.rawValue] = newValue }
    }
}
