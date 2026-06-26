import Foundation
#if canImport(Darwin)
import Darwin
#endif

/// Mac-side OSC relay for PC-offload mode: listens on a local UDP port and forwards
/// every datagram VERBATIM to the Quest 3 (`questIP:questPort`). In "Relay via Mac"
/// route mode the PC daemon sends its OSC bundles to `MacLANIP:relayPort` instead of
/// straight to the Quest, and this relay bounces them on to the headset.
///
/// Why: the PC is on wired ethernet and often CANNOT reach the Quest's Wi-Fi subnet
/// directly, but the Mac (on Wi-Fi) can — so the Mac becomes the bridge. The bytes are
/// forwarded untouched, so the wire VRChat sees is identical to the direct path.
///
/// Non-isolated `@unchecked Sendable` for the same reason as `PCSkeletonReceiver`: a
/// `DispatchSource` handler defined in a `@MainActor` context would inherit main-actor
/// isolation and trap when fired on a background queue. The handlers here touch only
/// Sendable state and run safely on the dispatch queue. Torn down cleanly on stop().
public final class PCOscRelay: @unchecked Sendable {

    private let listenFD: Int32
    private let sendFD: Int32
    private var source: DispatchSourceRead?

    /// - port: local UDP port to listen on (the daemon's `--osc-port` in relay mode).
    /// - forwardHost/forwardPort: the Quest 3 OSC endpoint to forward to (usually :9000).
    /// Returns nil if the listen socket can't be created/bound or the forward host is
    /// unresolvable, so the caller can surface a real error rather than silently drop OSC.
    public init?(port: UInt16, forwardHost: String, forwardPort: UInt16) {
        // -- Listen socket (bind the relay port) --
        let ls = socket(AF_INET, SOCK_DGRAM, 0)
        guard ls >= 0 else { return nil }
        var reuse: Int32 = 1
        setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &reuse, socklen_t(MemoryLayout<Int32>.size))
        setsockopt(ls, SOL_SOCKET, SO_REUSEPORT, &reuse, socklen_t(MemoryLayout<Int32>.size))
        var addr = sockaddr_in()
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = port.bigEndian
        addr.sin_addr.s_addr = 0   // INADDR_ANY
        let bound = withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                bind(ls, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        guard bound == 0 else { close(ls); return nil }

        // -- Forward socket (connected to the Quest so sends are a cheap write()) --
        let fs = socket(AF_INET, SOCK_DGRAM, 0)
        guard fs >= 0 else { close(ls); return nil }
        var fwd = sockaddr_in()
        fwd.sin_family = sa_family_t(AF_INET)
        fwd.sin_port = forwardPort.bigEndian
        if inet_pton(AF_INET, forwardHost, &fwd.sin_addr) != 1 {
            // Resolve a hostname (a numeric IP took the fast path above).
            var hints = addrinfo(ai_flags: 0, ai_family: AF_INET, ai_socktype: SOCK_DGRAM,
                                 ai_protocol: 0, ai_addrlen: 0, ai_canonname: nil, ai_addr: nil, ai_next: nil)
            var res: UnsafeMutablePointer<addrinfo>?
            guard getaddrinfo(forwardHost, nil, &hints, &res) == 0, let info = res else {
                close(ls); close(fs); return nil
            }
            defer { freeaddrinfo(info) }
            guard let sa = info.pointee.ai_addr else { close(ls); close(fs); return nil }
            fwd.sin_addr = sa.withMemoryRebound(to: sockaddr_in.self, capacity: 1) { $0.pointee.sin_addr }
        }
        let connected = withUnsafePointer(to: &fwd) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                connect(fs, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        guard connected == 0 else { close(ls); close(fs); return nil }

        self.listenFD = ls
        self.sendFD = fs

        let src = DispatchSource.makeReadSource(fileDescriptor: ls, queue: DispatchQueue.global(qos: .userInitiated))
        src.setEventHandler {
            // OSC bundles are well under an MTU; a generous buffer covers the 17-message
            // PinoFBT bundle with room to spare. Forward exactly the bytes received.
            var buf = [UInt8](repeating: 0, count: 2048)
            let n = recv(ls, &buf, buf.count, 0)
            guard n > 0 else { return }
            _ = buf.withUnsafeBytes { raw in send(fs, raw.baseAddress, n, 0) }
        }
        src.setCancelHandler { close(ls); close(fs) }
        src.resume()
        self.source = src
    }

    public func stop() {
        source?.cancel()
        source = nil
    }

    // Safety net: a relay dropped without stop() still cancels its source so the fds
    // don't leak. stop() nils `source`, so a prior stop makes this a no-op.
    deinit { source?.cancel() }
}
