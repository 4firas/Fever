import simd

/// Temporal quaternion stabilizer via SLERP toward the previous sample.
/// Prevents rotation jitter without the lag of a fixed low-pass.
public final class QuaternionStabilizer {
    private var prev: [JointType: simd_quatf] = [:]
    public var smoothingFactor: Float   // 0 = frozen, 1 = raw

    public init(smoothingFactor: Float = 0.5) {
        self.smoothingFactor = smoothingFactor
    }

    public func stabilize(_ joint: VRJoint) -> VRJoint {
        var j = joint
        if let prevQ = prev[joint.type] {
            // t = 1 - smoothing: higher smoothing → more of the previous (stable) value
            let t = 1.0 - smoothingFactor
            j.rotation = safeSlerp(prevQ, joint.rotation, t)
        }
        prev[joint.type] = j.rotation
        return j
    }

    public func reset() { prev.removeAll() }
}
