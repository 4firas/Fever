import Foundation
import FeverCore
#if canImport(Darwin)
import Darwin
#endif

/// Covers the 30 fps capture cap, the latency-prediction knob, and — critically — the
/// EXACT daemon launch command the NLF model builds (the contract the PC daemon's argparse
/// must match). The command builder (`PCOffloadConfig.daemonCommandLine`) is pure/string-only,
/// so the launch arguments are asserted byte-for-byte here without any SSH or live PC.
enum PCModelLaunchTests {

    static func run(_ t: TestRunner) {
        persistence(t)
        launchArgs(t)
    }

    // MARK: - Settings persistence

    private static func persistence(_ t: TestRunner) {
        t.test("cameraMaxFPS defaults to 30 and persists when changed") {
            UserDefaults.standard.removeObject(forKey: "cameraMaxFPS")
            t.check(TrackingConfig().cameraMaxFPS == 30, "capture FPS cap defaults to 30")
            let cfg = TrackingConfig()
            cfg.cameraMaxFPS = 60
            t.check(TrackingConfig().cameraMaxFPS == 60, "cameraMaxFPS must persist to UserDefaults")
            UserDefaults.standard.removeObject(forKey: "cameraMaxFPS")
        }

        t.test("predictionLeadMs defaults to 50, persists, and clamps to 0…150") {
            UserDefaults.standard.removeObject(forKey: "predictionLeadMs")
            t.check(TrackingConfig().predictionLeadMs == 50, "latency prediction defaults to 50 ms")
            let cfg = TrackingConfig()
            cfg.predictionLeadMs = 90
            t.check(TrackingConfig().predictionLeadMs == 90, "predictionLeadMs must persist to UserDefaults")
            cfg.predictionLeadMs = 999
            t.check(TrackingConfig().predictionLeadMs == 150, "predictionLeadMs clamps to the 0…150 max")
            cfg.predictionLeadMs = -10
            t.check(TrackingConfig().predictionLeadMs == 0, "predictionLeadMs clamps to the 0…150 min")
            UserDefaults.standard.removeObject(forKey: "predictionLeadMs")
        }
    }

    // MARK: - Launch-arg construction

    /// Build configs under a KNOWN bridge base so the launch command is fully deterministic
    /// (independent of the operator's real `FEVER_PC_BRIDGE` / SSH user).
    private static func launchArgs(_ t: TestRunner) {
        let prior = ProcessInfo.processInfo.environment["FEVER_PC_BRIDGE"]
        setenv("FEVER_PC_BRIDGE", #"C:\BR"#, 1)
        defer {
            if let prior { setenv("FEVER_PC_BRIDGE", prior, 1) } else { unsetenv("FEVER_PC_BRIDGE") }
        }

        // -- NLF: the ORIGINAL launch + the shared user-tunable --lead-ms tail --
        t.test("NLF launch command (python/script/flags + lead-ms)") {
            let c = config(sendElbows: false)   // 6-point, mirror on, polite off, lead 50
            let cmd = c.daemonCommandLine(oscIP: "192.168.1.50", oscPort: 9000, skeletonBack: "10.0.0.2:5001")
            let expected = #""C:\BR\py311\pythonw.exe" "C:\BR\work\fbt_daemon.py" --osc-ip 192.168.1.50 --osc-port 9000 --height 174 --six-point --skeleton-back 10.0.0.2:5001 --lead-ms 50"#
            t.check(cmd == expected, "NLF command mismatch:\n  got:  \(cmd)\n  want: \(expected)")
            t.check(c.daemonMatch == "fbt_daemon", "NLF daemon match token")
            t.check(!cmd.contains("gvhmr"), "NLF command must not reference gvhmr")
        }

        t.test("NLF 8-point + no-mirror + polite + fps-cap flags (+ lead-ms tail)") {
            var c = config(sendElbows: true)
            c.mirror = false; c.politeMode = true; c.fpsCap = 60
            let cmd = c.daemonCommandLine(oscIP: "1.2.3.4", oscPort: 9100, skeletonBack: nil)
            let expected = #""C:\BR\py311\pythonw.exe" "C:\BR\work\fbt_daemon.py" --osc-ip 1.2.3.4 --osc-port 9100 --height 174 --no-mirror --polite --fps-cap 60 --lead-ms 50"#
            t.check(cmd == expected, "NLF flag mismatch:\n  got:  \(cmd)\n  want: \(expected)")
        }
    }

    /// A deterministic config for the launch tests (real `.make` path construction under
    /// the test's `FEVER_PC_BRIDGE`). Direct OSC route; the caller tweaks flags as needed.
    private static func config(sendElbows: Bool) -> PCOffloadConfig {
        PCOffloadConfig.make(
            host: "10.0.0.9", user: "u", mac: "AA:BB:CC:DD:EE:FF",
            oscIP: "192.168.1.50", oscPort: 9000,
            heightCm: 174, sendElbows: sendElbows, mirror: true, predictionLeadMs: 50,
            streamW: 1280, streamH: 720, streamFPS: 30, bitrateMbps: 10,
            politeMode: false, fpsCap: 0, streamPort: 5000, cameraName: nil)
    }
}
