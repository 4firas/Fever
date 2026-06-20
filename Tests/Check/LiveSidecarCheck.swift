import Foundation
import CoreVideo
import FeverCore

/// End-to-end integration probe for the REAL Python sidecar (not a fake):
/// resolves the sidecar, launches it, and times round-trips on real-sized frames.
/// Proves the Swift<->Python IPC (READY handshake, frame send, reply decode) works
/// and reports a per-frame latency ceiling. A blank frame yields found=false, which
/// is still a valid decoded reply — that is what we assert (nil == IPC failure).
/// Run: `swift run FeverCheck --live-sidecar`.
enum LiveSidecarCheck {
    static func run() async -> Bool {
        guard let paths = SidecarPaths.resolve(bundle: .main, projectRoot: nil) else {
            print("LIVE-SIDECAR: sidecar not installed — run Scripts/setup-sidecar.sh")
            return false
        }
        print("LIVE-SIDECAR: python = \(paths.python)")
        print("LIVE-SIDECAR: model  = \(paths.model)")
        let sidecar = PoseSidecar(paths: paths)

        // A real-sized gray frame (RGB888, no body).
        let w = 384, h = 216
        let rgb = Data(repeating: 110, count: w * h * 3)

        // First call includes the model load (warmup).
        let t0 = Date()
        let first = await sidecar.infer(rgb: rgb, width: w, height: h, tMicros: 1_000)
        let warm = Date().timeIntervalSince(t0)
        guard let first else {
            print("LIVE-SIDECAR: FAIL — nil reply (launch/IPC error)")
            return false
        }
        print(String(format: "LIVE-SIDECAR: round-trip OK  found=%@  warmup=%dms (incl. model load)",
                     first.found ? "true" : "false", Int(warm * 1000)))

        // Steady-state timing.
        var times: [Double] = []
        for i in 0..<15 {
            let t = Date()
            let r = await sidecar.infer(rgb: rgb, width: w, height: h,
                                        tMicros: UInt64(2_000 + i * 33_000))
            if r == nil { print("LIVE-SIDECAR: FAIL — nil reply mid-stream"); return false }
            times.append(Date().timeIntervalSince(t))
        }
        let avg = times.reduce(0, +) / Double(times.count)
        let mn = times.min() ?? 0
        print(String(format: "LIVE-SIDECAR: steady round-trip avg=%dms min=%dms (~%d fps ceiling)",
                     Int(avg * 1000), Int(mn * 1000), Int(1.0 / avg)))
        print("LIVE-SIDECAR: NOTE blank frame = detector-only; a real body also runs the "
              + "landmark model, so live fps will be somewhat lower than this ceiling.")
        return true
    }
}
