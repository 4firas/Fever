import Foundation
import simd
import CoreVideo
#if canImport(Darwin)
import Darwin
#endif

/// PC-offload REMOTE-INFERENCE pose source.
///
/// In remote mode the PC runs ONLY the model (`fbt_daemon --raw`) and sends the RAW pose back
/// over UDP; this conforms to `NLFPoseSource` so the EXACT on-device `TrackingPipeline`
/// (mirror → OneEuro → IK → 120 Hz predictive upsampler → OSC) runs on the Mac, fed by the
/// remote joints instead of the local sidecar. That makes PC mode as smooth as on-device — all
/// the timing-sensitive math + the OSC send happen on the Mac's proven pipeline, not on Windows.
///
/// Wire packet (492 B, little-endian, from the daemon): f32 `ht`, f64 `pts`, 24×3 f32 `joints3D`
/// (RAW model output, +Y-down camera space — exactly what the local sidecar returns), 24×2 f32
/// `joints2D` (normalized [0,1]). `detect()` returns the latest UNCONSUMED pose (nil if none new),
/// stamped with the PC's `pts` so OneEuro sees the clean frame-clock (not Mac arrival jitter).
///
/// `@unchecked Sendable`: the `DispatchSource` handler runs on a background queue and touches only
/// lock-guarded state; the `@Sendable onPreview` hops to the main actor itself. The blocking `recv`
/// lives in a nonisolated context, so a background-queue fire never trips a main-actor assertion.
public final class RemoteNLFSource: NLFPoseSource, @unchecked Sendable {

    private let fd: Int32
    private var source: DispatchSourceRead?
    private let lock = NSLock()
    private var pending: SMPLPose?     // latest received pose, not yet consumed by detect()
    private var seq: UInt64 = 0        // bumped per received pose
    private var consumed: UInt64 = 0   // last seq returned by detect() (latest-wins)

    public var isLive: Bool { true }

    /// - port: UDP port to receive RAW pose packets on (the daemon's `--skeleton-back` port).
    /// - flipX: mirror the overlay points' x to match the always-mirrored preview layer.
    /// - onPreview: called (on a background queue) with the normalized, flipped overlay points;
    ///   the caller hops to the main actor to publish them.
    public init?(port: UInt16, flipX: Bool, onPreview: @escaping @Sendable ([SIMD2<Float>]) -> Void) {
        let s = socket(AF_INET, SOCK_DGRAM, 0)
        guard s >= 0 else { return nil }
        var reuse: Int32 = 1
        setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &reuse, socklen_t(MemoryLayout<Int32>.size))
        setsockopt(s, SOL_SOCKET, SO_REUSEPORT, &reuse, socklen_t(MemoryLayout<Int32>.size))
        var addr = sockaddr_in()
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = port.bigEndian
        addr.sin_addr.s_addr = 0   // INADDR_ANY
        let bound = withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                bind(s, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        guard bound == 0 else { close(s); return nil }
        self.fd = s

        let src = DispatchSource.makeReadSource(fileDescriptor: s, queue: DispatchQueue.global(qos: .userInitiated))
        src.setEventHandler { [weak self] in
            var buf = [UInt8](repeating: 0, count: 600)
            let n = recv(s, &buf, buf.count, 0)
            guard n == 492 else { return }   // ignore anything that isn't a full pose packet
            // Parse little-endian. The daemon packs with struct '<', and Apple Silicon is
            // little-endian, so a direct load matches the wire bytes. loadUnaligned handles the
            // unaligned f64 at offset 4. Layout: [ht f32][pts f64][24×3 j3 f32][24×2 j2 f32].
            let (pose, overlay): (SMPLPose, [SIMD2<Float>]) = buf.withUnsafeBytes { raw in
                let ht = raw.loadUnaligned(fromByteOffset: 0, as: Float32.self)
                let pts = raw.loadUnaligned(fromByteOffset: 4, as: Float64.self)
                var j3 = [SIMD3<Float>](); j3.reserveCapacity(24)
                for i in 0..<24 {
                    let o = 12 + i * 12
                    j3.append(SIMD3<Float>(raw.loadUnaligned(fromByteOffset: o, as: Float32.self),
                                           raw.loadUnaligned(fromByteOffset: o + 4, as: Float32.self),
                                           raw.loadUnaligned(fromByteOffset: o + 8, as: Float32.self)))
                }
                var j2 = [SIMD2<Float>](); j2.reserveCapacity(24)
                var ov = [SIMD2<Float>](); ov.reserveCapacity(24)
                for i in 0..<24 {
                    let o = 300 + i * 8
                    let x = raw.loadUnaligned(fromByteOffset: o, as: Float32.self)
                    let y = raw.loadUnaligned(fromByteOffset: o + 4, as: Float32.self)
                    j2.append(SIMD2<Float>(x, y))
                    // Overlay: NaN out an absent (0,0) joint so the overlay skips it; flip x to
                    // match the always-mirrored preview layer.
                    ov.append((x == 0 && y == 0) ? SIMD2<Float>(.nan, .nan)
                                                 : SIMD2<Float>(flipX ? (1 - x) : x, y))
                }
                // width/height = 1 because joints2D are already normalized; the IK path uses only
                // joints3D + timestamp, so this is solely for the (unused-in-PC) preview build.
                let p = SMPLPose(joints3D: j3, joints2D: j2, hasTracked: ht,
                                 timestamp: pts, width: 1, height: 1)
                return (p, ov)
            }
            self?.store(pose)
            onPreview(overlay)
        }
        src.setCancelHandler { close(s) }
        src.resume()
        self.source = src
    }

    private func store(_ p: SMPLPose) {
        lock.withLock { pending = p; seq &+= 1 }
    }

    /// Returns the latest UNCONSUMED pose (nil if none arrived since the last call). The
    /// `pixelBuffer`/`time` are ignored — the PC already inferred from its own stream; the pose
    /// carries the PC's `pts` as its timestamp so OneEuro clocks on the clean frame-time.
    public func detect(_ pixelBuffer: CVPixelBuffer, at time: TimeInterval) async -> SMPLPose? {
        lock.withLock {
            guard seq != consumed, let p = pending else { return nil }
            consumed = seq
            return p
        }
    }

    public func reset() { lock.withLock { pending = nil; seq = 0; consumed = 0 } }

    public func stop() { source?.cancel(); source = nil }

    deinit { source?.cancel() }
}
