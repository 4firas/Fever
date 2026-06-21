import simd
import Foundation

/// Cross-frame state for the step / stride EXAGGERATION (analogue of
/// `RotationState`): a reference type injected into the value-type `JointSolver`,
/// owned by the FrameProcessor, confined to the single serial inference worker,
/// and reset on Recenter / run start-stop.
///
/// Per foot it maintains a SLOW-EMA "neutral" rest position to amplify
/// displacement FROM (so steady standing shows ~zero exaggeration and a real step
/// is a large transient that decays as the EMA catches up — motion without drift),
/// a frozen floor-contact height, a smoothed swing/stance factor, and the previous
/// exaggeration delta for output smoothing. All positions are in the UNSCALED
/// landmark frame (the caller scales with `solverPosition` consistently).
public final class FootMotionState: @unchecked Sendable {

    private struct Foot {
        var neutralXZ = SIMD2<Float>.zero   // slow-EMA lateral/fore-aft rest
        var floorY: Float = 0               // contact height (tracked while planted)
        var swing: Float = 0                // smoothed swing factor [0,1]
        var exPrev = SIMD3<Float>.zero      // last emitted exaggeration delta (scaled)
        var seeded = false
    }

    // Index 0 = left foot, 1 = right foot.
    private var feet = [Foot](repeating: Foot(), count: 2)

    // EMA / ramp constants.
    private let neutralAlpha: Float = 0.02   // slow (~multi-second) rest tracking
    private let floorAlpha: Float = 0.05     // floor tracking while planted
    private let swingAlpha: Float = 0.30     // swing-factor smoothing
    private let liftFull: Float = 0.03       // lift ≤ this → planted (swing 0)
    private let liftNone: Float = 0.10        // lift ≥ this → full swing (1)

    public init() {}

    public func reset() { feet = [Foot](repeating: Foot(), count: 2) }

    private static func index(_ type: JointType) -> Int? {
        switch type { case .leftFoot: return 0; case .rightFoot: return 1; default: return nil }
    }

    /// Update one foot's EMA state from its RAW (unscaled) ankle position and
    /// return the rest neutral (x = floorY, packed as SIMD3) + swing factor.
    /// `neutral` is (neutralXZ.x, floorY, neutralXZ.y).
    public func update(_ type: JointType, rawAnkle p: SIMD3<Float>) -> (neutral: SIMD3<Float>, swing: Float) {
        guard let i = Self.index(type) else { return (p, 0) }
        if !feet[i].seeded {
            feet[i].neutralXZ = SIMD2(p.x, p.z)
            feet[i].floorY = p.y
            feet[i].seeded = true
            return (p, 0)
        }
        // Swing factor from lift above the latched floor.
        let lift = p.y - feet[i].floorY
        let raw = Self.ramp(lift, liftFull, liftNone)
        feet[i].swing += (raw - feet[i].swing) * swingAlpha

        // Slow rest tracking (XZ always; floor only while planted so a swinging
        // foot never drags the contact height up).
        feet[i].neutralXZ += (SIMD2(p.x, p.z) - feet[i].neutralXZ) * neutralAlpha
        if feet[i].swing < 0.2 {
            feet[i].floorY += (p.y - feet[i].floorY) * floorAlpha
        }
        let neutral = SIMD3<Float>(feet[i].neutralXZ.x, feet[i].floorY, feet[i].neutralXZ.y)
        return (neutral, feet[i].swing)
    }

    /// Smooth this frame's exaggeration delta (in the SCALED solver frame) against
    /// the previous one so gain / swing transitions ease in (no foot pop).
    public func smoothExaggeration(_ type: JointType, _ ex: SIMD3<Float>, factor: Float = 0.5) -> SIMD3<Float> {
        guard let i = Self.index(type) else { return ex }
        feet[i].exPrev += (ex - feet[i].exPrev) * factor
        return feet[i].exPrev
    }

    /// Linear ramp: 0 at/below `lo`, 1 at/above `hi`, linear between (lo<hi).
    @inline(__always)
    static func ramp(_ v: Float, _ lo: Float, _ hi: Float) -> Float {
        if v <= lo { return 0 }
        if v >= hi { return 1 }
        return (v - lo) / (hi - lo)
    }
}
