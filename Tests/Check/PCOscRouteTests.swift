import Foundation
import FeverCore
#if canImport(Darwin)
import Darwin
#endif

/// Validates the PC-mode OSC routing toggle.
///   • PCOffloadConfig carries the route (relayViaMac/relayPort) + the proper-mirror flag
///     through `.make()` for both Direct and Relay.
///   • PCOscRelay forwards a datagram VERBATIM from its listen port to the forward
///     (Quest) endpoint — the actual relay path, proven over loopback so VRChat would
///     receive byte-identical OSC whether the PC sends Direct or via the Mac.
enum PCOscRouteTests {

    static func run(_ t: TestRunner) {
        t.test("PCOffloadConfig.make carries the OSC route + mirror (Direct)") {
            let c = makeConfig(relay: false, mirror: true)
            t.check(c.relayViaMac == false, "Direct route")
            t.check(c.oscIP == "192.168.1.50" && c.oscPort == 9000, "final Quest target preserved")
            t.check(c.mirror == true, "mirror flag carried")
        }
        t.test("PCOffloadConfig.make carries the OSC route + mirror (Relay)") {
            let c = makeConfig(relay: true, mirror: false)
            t.check(c.relayViaMac == true, "Relay route")
            t.check(c.relayPort == 9001, "relay port carried")
            t.check(c.oscIP == "192.168.1.50" && c.oscPort == 9000, "final Quest target still the Quest")
            t.check(c.mirror == false, "mirror-off flag carried")
        }

        // Real loopback: bind a "Quest" listener, stand up a relay pointing at it, send a
        // datagram to the relay's port, and assert the Quest listener gets the SAME bytes.
        t.test("PCOscRelay forwards a datagram verbatim to the Quest endpoint") {
            let questPort: UInt16 = 19809
            let relayPort: UInt16 = 19808
            guard let quest = boundReceiver(port: questPort) else {
                t.check(false, "couldn't bind the loopback Quest receiver"); return
            }
            defer { close(quest) }
            guard let relay = PCOscRelay(port: relayPort, forwardHost: "127.0.0.1", forwardPort: questPort) else {
                t.check(false, "couldn't create PCOscRelay"); return
            }
            defer { relay.stop() }

            let payload: [UInt8] = Array("#bundle\0".utf8) + [0, 0, 0, 0, 0, 0, 0, 1, 0xDE, 0xAD, 0xBE, 0xEF]
            sendDatagram(payload, toPort: relayPort)

            let got = recvDatagram(quest, timeoutMs: 1500)
            t.check(got == payload, "relayed bytes must equal the sent bytes (got \(got?.count ?? -1) bytes)")
        }
    }

    // MARK: - helpers

    private static func makeConfig(relay: Bool, mirror: Bool) -> PCOffloadConfig {
        PCOffloadConfig.make(
            host: "10.0.0.9", user: "u", mac: "AA:BB:CC:DD:EE:FF",
            oscIP: "192.168.1.50", oscPort: 9000,
            relayViaMac: relay, relayPort: 9001,
            heightCm: 174, sendElbows: true, mirror: mirror,
            streamW: 1280, streamH: 720, streamFPS: 60, bitrateMbps: 10,
            politeMode: false, fpsCap: 0, streamPort: 5000, cameraName: nil)
    }

    private static func boundReceiver(port: UInt16) -> Int32? {
        let s = socket(AF_INET, SOCK_DGRAM, 0)
        guard s >= 0 else { return nil }
        var reuse: Int32 = 1
        setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &reuse, socklen_t(MemoryLayout<Int32>.size))
        var addr = sockaddr_in()
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = port.bigEndian
        addr.sin_addr.s_addr = inet_addr("127.0.0.1")
        let ok = withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                bind(s, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        guard ok == 0 else { close(s); return nil }
        return s
    }

    private static func sendDatagram(_ bytes: [UInt8], toPort port: UInt16) {
        let s = socket(AF_INET, SOCK_DGRAM, 0)
        guard s >= 0 else { return }
        defer { close(s) }
        var addr = sockaddr_in()
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = port.bigEndian
        addr.sin_addr.s_addr = inet_addr("127.0.0.1")
        _ = bytes.withUnsafeBytes { raw in
            withUnsafePointer(to: &addr) { ap in
                ap.withMemoryRebound(to: sockaddr.self, capacity: 1) { sa in
                    sendto(s, raw.baseAddress, raw.count, 0, sa, socklen_t(MemoryLayout<sockaddr_in>.size))
                }
            }
        }
    }

    private static func recvDatagram(_ fd: Int32, timeoutMs: Int) -> [UInt8]? {
        var tv = timeval(tv_sec: timeoutMs / 1000, tv_usec: Int32((timeoutMs % 1000) * 1000))
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, socklen_t(MemoryLayout<timeval>.size))
        var buf = [UInt8](repeating: 0, count: 2048)
        let n = recv(fd, &buf, buf.count, 0)
        guard n > 0 else { return nil }
        return Array(buf[0..<n])
    }
}
