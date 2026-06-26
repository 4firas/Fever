import Foundation
import simd
#if canImport(Darwin)
import Darwin
#endif

/// Receives the PC's returned skeleton (24×2 normalized float32 = 192 bytes/frame)
/// on a UDP port and hands each frame's points to `onPoints`.
///
/// This is a NON-isolated class on purpose: a `DispatchSource` event/cancel handler
/// defined inside a `@MainActor` method would inherit main-actor isolation, and
/// `DispatchSource` fires it on a background queue — which trips the Swift runtime's
/// executor assertion and traps (that was the Stop-crash). Here the handlers live in
/// a non-isolated context and only touch Sendable state, so they run safely on the
/// dispatch queue; the @Sendable `onPoints` hops to the main actor itself.
final class PCSkeletonReceiver: @unchecked Sendable {

    private let fd: Int32
    private var source: DispatchSourceRead?

    init?(port: UInt16, flipX: Bool, onPoints: @escaping @Sendable ([SIMD2<Float>]) -> Void) {
        let s = socket(AF_INET, SOCK_DGRAM, 0)
        guard s >= 0 else { return nil }
        // Allow an immediate re-bind: the DispatchSource cancel-handler closes the fd
        // asynchronously, so a quick Stop→Start (or mode toggle) can otherwise hit
        // EADDRINUSE on the same port and silently lose the skeleton overlay.
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
        src.setEventHandler {
            var buf = [Float32](repeating: 0, count: 48)   // 24 joints × (x,y)
            let n = recv(s, &buf, 48 * MemoryLayout<Float32>.size, 0)
            guard n == 48 * MemoryLayout<Float32>.size else { return }
            var pts = [SIMD2<Float>](); pts.reserveCapacity(24)
            for i in 0..<24 {
                let x = flipX ? (1 - buf[i * 2]) : buf[i * 2]
                pts.append(SIMD2<Float>(x, buf[i * 2 + 1]))
            }
            onPoints(pts)
        }
        src.setCancelHandler { close(s) }
        src.resume()
        self.source = src
    }

    func stop() {
        source?.cancel()
        source = nil
    }

    // Safety net: if a receiver is dropped without stop() (e.g. an init path that
    // replaces it), still cancel the resumed source so its fd doesn't leak. stop()
    // nils `source`, so a prior stop makes this a harmless no-op.
    deinit { source?.cancel() }
}
