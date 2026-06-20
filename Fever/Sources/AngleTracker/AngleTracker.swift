import Foundation
import simd

#if os(iOS)
import CoreMotion
#endif

/// Single-device IMU fusion. On iOS, `CMMotionManager.deviceMotion.attitude`
/// supplies a stable world-up reference frame that the camera pose is composed
/// against. On macOS (most Macs lack motion sensors) this is a graceful
/// identity no-op and the pipeline falls back to camera-only orientation.
///
/// All CoreMotion usage is wrapped in `#if os(iOS)`, so a macOS build pulls in
/// no CoreMotion symbols whatsoever.
public final class AngleTracker {

    #if os(iOS)
    private let manager = CMMotionManager()
    private let queue = OperationQueue()
    #endif

    /// Latest world-orientation reported by the IMU. Stays identity on macOS.
    private(set) var attitude: simd_quatf = simd_quatf(vector: SIMD4<Float>(0, 0, 0, 1))

    /// Latest gravity direction in device space. Defaults to straight down.
    private(set) var gravity: SIMD3<Float> = SIMD3<Float>(0, -1, 0)

    /// Timestamp (seconds) of the most recent IMU sample.
    private(set) var lastUpdate: Double = 0

    /// Optional offset to align IMU timestamps with the OSC server clock.
    public var serverTimeOffset: Double = 0

    public init(updateRate: Double = 60) {
        #if os(iOS)
        manager.deviceMotionUpdateInterval = 1.0 / updateRate
        #endif
    }

    /// `true` only when a usable motion sensor exists (iOS). Always `false` on macOS.
    public var isAvailable: Bool {
        #if os(iOS)
        return manager.isDeviceMotionAvailable
        #else
        return false
        #endif
    }

    /// Begin streaming device-motion updates. No-op on macOS.
    public func start() {
        #if os(iOS)
        guard isAvailable, manager.isDeviceMotionActive == false else { return }
        manager.startDeviceMotionUpdates(to: queue) { [weak self] motion, _ in
            guard let self, let m = motion else { return }
            let a = m.attitude.quaternion
            self.attitude = simd_quatf(vector: SIMD4<Float>(Float(a.x), Float(a.y), Float(a.z), Float(a.w)))
            let g = m.gravity
            self.gravity = SIMD3<Float>(Float(g.x), Float(g.y), Float(g.z))
            self.lastUpdate = m.timestamp
        }
        #endif
    }

    /// Stop streaming device-motion updates. No-op on macOS.
    public func stop() {
        #if os(iOS)
        manager.stopDeviceMotionUpdates()
        #endif
    }

    /// Compose IMU world rotation with a camera-local rotation. When no IMU
    /// is available (most macOS), `attitude` stays identity and this returns
    /// the camera rotation unchanged.
    public func fuseWorldRotation(_ cameraLocal: simd_quatf) -> simd_quatf {
        guard isAvailable else { return cameraLocal }
        return attitude * cameraLocal
    }

    /// Joint flexion angle (radians) between two bones meeting at `mid`:
    /// the incoming bone `mid - toParent` and the outgoing bone `toChild - mid`.
    public static func flexionAngle(mid: SIMD3<Float>,
                                    toParent: SIMD3<Float>,
                                    toChild: SIMD3<Float>) -> Float {
        angleBetween(mid - toParent, toChild - mid)
    }
}
