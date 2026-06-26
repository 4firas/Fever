import Foundation
import simd

/// One frame of NLF model output, in the model's NATIVE space.
///
/// `joints3D` are CAMERA-space meters with **+Y down** (head ≈ 0.45, feet ≈ 1.9 on a
/// standing person) and right-handed — NOT the floor-anchored +Y-up VRChat frame.
/// The conversion (pelvis-center, 180°-about-X flip, torso/leg scale) is applied later
/// by `PinoKinematics.preprocessJoints`. `joints2D` are pixels in the fed frame (top-left origin).
public struct SMPLPose: Sendable {
    public var joints3D: [SIMD3<Float>]   // count == SMPLJoint.count (24)
    public var joints2D: [SIMD2<Float>]   // count == 24, pixels in the fed frame
    public var hasTracked: Float          // detection/tracking confidence scalar
    public var timestamp: Double          // capture time (seconds, monotonic)
    public var width: Int                 // fed-frame width  (for normalizing joints2D)
    public var height: Int                // fed-frame height

    public init(joints3D: [SIMD3<Float>], joints2D: [SIMD2<Float>],
                hasTracked: Float, timestamp: Double, width: Int = 0, height: Int = 0) {
        self.joints3D = joints3D
        self.joints2D = joints2D
        self.hasTracked = hasTracked
        self.timestamp = timestamp
        self.width = width
        self.height = height
    }

    /// Screen-normalized 2D points (x,y in [0,1], top-left) for the preview overlay;
    /// absent / not-tracked joints become NaN so the overlay skips them.
    public func normalizedPoints() -> [SIMD2<Float>] {
        guard width > 0, height > 0, isTracked else {
            return Array(repeating: SIMD2<Float>(.nan, .nan), count: SMPLJoint.count)
        }
        return joints2D.map { p in
            (p.x == 0 && p.y == 0) ? SIMD2<Float>(.nan, .nan)
                                   : SIMD2<Float>(p.x / Float(width), p.y / Float(height))
        }
    }

    public var isTracked: Bool { hasTracked > 0.5 }

    public subscript(_ joint: SMPLJoint) -> SIMD3<Float> { joints3D[joint.rawValue] }
    public func px(_ joint: SMPLJoint) -> SIMD2<Float> { joints2D[joint.rawValue] }

    /// A not-detected frame (all-zero joints, hasTracked 0).
    public static func untracked(timestamp: Double = 0) -> SMPLPose {
        SMPLPose(joints3D: Array(repeating: .zero, count: SMPLJoint.count),
                 joints2D: Array(repeating: .zero, count: SMPLJoint.count),
                 hasTracked: 0, timestamp: timestamp)
    }
}
