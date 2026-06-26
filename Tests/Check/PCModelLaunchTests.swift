import Foundation
import FeverCore
#if canImport(Darwin)
import Darwin
#endif

/// Covers the NLF-vs-GVHMR model selector, the static/moving camera toggle, the
/// 30 fps capture cap, and — critically — the EXACT daemon launch command each model
/// builds (the contract the PC daemons' argparse must match). The command builder
/// (`PCOffloadConfig.daemonCommandLine`) is pure/string-only, so the launch arguments
/// are asserted byte-for-byte here without any SSH or live PC.
enum PCModelLaunchTests {

    static func run(_ t: TestRunner) {
        persistence(t)
        launchArgs(t)
    }

    // MARK: - Settings persistence

    private static func persistence(_ t: TestRunner) {
        t.test("pcModel defaults to NLF and persists when set to GVHMR") {
            UserDefaults.standard.removeObject(forKey: "pcModel")
            t.check(TrackingConfig().pcModel == "nlf", "pose model defaults to NLF")
            let cfg = TrackingConfig()
            cfg.pcModel = "gvhmr"
            t.check(TrackingConfig().pcModel == "gvhmr", "pcModel must persist to UserDefaults")
            UserDefaults.standard.removeObject(forKey: "pcModel")
        }

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

        t.test("gvhmrMoving defaults to Static (false) and persists when set to Moving") {
            UserDefaults.standard.removeObject(forKey: "gvhmrMoving")
            t.check(TrackingConfig().gvhmrMoving == false, "camera mode defaults to Static")
            let cfg = TrackingConfig()
            cfg.gvhmrMoving = true
            t.check(TrackingConfig().gvhmrMoving == true, "gvhmrMoving must persist to UserDefaults")
            UserDefaults.standard.removeObject(forKey: "gvhmrMoving")
        }

        t.test("GVHMR facing/lookahead knobs persist (k, mirror, flip-x)") {
            for k in ["gvhmrK", "gvhmrMirror", "gvhmrFlipX"] { UserDefaults.standard.removeObject(forKey: k) }
            let d = TrackingConfig()
            // k defaults to 2 (low latency): k frames of readout lag ≈ k/fps seconds, and the
            // window's past frames already denoise the readout, so a low k is the big PC-mode
            // latency win (was 5 ≈ 167ms@30fps).
            t.check(d.gvhmrK == 2 && d.gvhmrMirror == false && d.gvhmrFlipX == false,
                    "GVHMR defaults: k=2, mirror off, flip-x off (got k=\(d.gvhmrK))")
            let cfg = TrackingConfig()
            cfg.gvhmrK = 9; cfg.gvhmrMirror = true; cfg.gvhmrFlipX = true
            let r = TrackingConfig()
            t.check(r.gvhmrK == 9 && r.gvhmrMirror == true && r.gvhmrFlipX == true,
                    "GVHMR knobs must persist (got k=\(r.gvhmrK), mirror=\(r.gvhmrMirror), flipX=\(r.gvhmrFlipX))")
            t.check(TrackingConfig().gvhmrK == 9, "k clamp keeps an in-range value")
            cfg.gvhmrK = 99
            t.check(TrackingConfig().gvhmrK == 15, "k clamps to the 0…15 range on overflow")
            for k in ["gvhmrK", "gvhmrMirror", "gvhmrFlipX"] { UserDefaults.standard.removeObject(forKey: k) }
        }

        t.test("GVHMR experimental toggles default OFF and persist (foot-contact, native-rot)") {
            for k in ["gvhmrFootContact", "gvhmrNativeRot"] { UserDefaults.standard.removeObject(forKey: k) }
            let d = TrackingConfig()
            t.check(d.gvhmrFootContact == false && d.gvhmrNativeRot == false,
                    "GVHMR defaults: foot-contact off, native-rot off")
            let cfg = TrackingConfig()
            cfg.gvhmrFootContact = true; cfg.gvhmrNativeRot = true
            let r = TrackingConfig()
            t.check(r.gvhmrFootContact == true && r.gvhmrNativeRot == true,
                    "GVHMR experimental toggles must persist (got footContact=\(r.gvhmrFootContact), nativeRot=\(r.gvhmrNativeRot))")
            for k in ["gvhmrFootContact", "gvhmrNativeRot"] { UserDefaults.standard.removeObject(forKey: k) }
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
            let c = config(model: .nlf, sendElbows: false)   // 6-point, mirror on, polite off, lead 50
            let cmd = c.daemonCommandLine(oscIP: "192.168.1.50", oscPort: 9000, skeletonBack: "10.0.0.2:5001")
            let expected = #""C:\BR\py311\pythonw.exe" "C:\BR\work\fbt_daemon.py" --osc-ip 192.168.1.50 --osc-port 9000 --height 174 --six-point --skeleton-back 10.0.0.2:5001 --lead-ms 50"#
            t.check(cmd == expected, "NLF command mismatch:\n  got:  \(cmd)\n  want: \(expected)")
            t.check(c.daemonMatch == "fbt_daemon", "NLF daemon match token")
            t.check(!cmd.contains("gvhmr"), "NLF command must not reference gvhmr")
        }

        t.test("NLF 8-point + no-mirror + polite + fps-cap flags (+ lead-ms tail)") {
            var c = config(model: .nlf, sendElbows: true)
            c.mirror = false; c.politeMode = true; c.fpsCap = 60
            let cmd = c.daemonCommandLine(oscIP: "1.2.3.4", oscPort: 9100, skeletonBack: nil)
            let expected = #""C:\BR\py311\pythonw.exe" "C:\BR\work\fbt_daemon.py" --osc-ip 1.2.3.4 --osc-port 9100 --height 174 --no-mirror --polite --fps-cap 60 --lead-ms 50"#
            t.check(cmd == expected, "NLF flag mismatch:\n  got:  \(cmd)\n  want: \(expected)")
        }

        // -- GVHMR: the new daemon, its own python/.venv + --listen/--k/--out-hz contract --
        t.test("GVHMR launch command: python/.venv, script, listen, k, out-hz/lead-ms") {
            let c = config(model: .gvhmr, sendElbows: false)   // 6-point, k=5, static, no facing flags
            // GVHMR now sends the preview skeleton back too (the daemon projects in-camera joints to 2D).
            let cmd = c.daemonCommandLine(oscIP: "192.168.1.50", oscPort: 9000, skeletonBack: "10.0.0.2:5001")
            let expected = #""C:\BR\gvhmr\.venv\Scripts\python.exe" "C:\BR\gvhmr\gvhmr_daemon.py" --listen udp://0.0.0.0:5000?overrun_nonfatal=1&fifo_size=16384 --osc-ip 192.168.1.50 --osc-port 9000 --height 174 --k 5 --six-point --skeleton-back 10.0.0.2:5001 --out-hz 120 --lead-ms 50"#
            t.check(cmd == expected, "GVHMR command mismatch:\n  got:  \(cmd)\n  want: \(expected)")
            t.check(c.daemonMatch == "gvhmr_daemon", "GVHMR daemon match token")
            t.check(cmd.contains("--skeleton-back 10.0.0.2:5001"), "GVHMR must pass --skeleton-back for the preview overlay")
            t.check(!cmd.contains("fbt_daemon"), "GVHMR command must not reference fbt_daemon")
        }

        t.test("GVHMR 8-point + facing flags + custom k, no --moving when Static") {
            var c = config(model: .gvhmr, sendElbows: true)   // 8-point → no --six-point
            c.gvhmrK = 8; c.gvhmrMirror = true; c.gvhmrFlipX = true; c.gvhmrMoving = false
            let cmd = c.daemonCommandLine(oscIP: "1.2.3.4", oscPort: 9100, skeletonBack: nil)
            let expected = #""C:\BR\gvhmr\.venv\Scripts\python.exe" "C:\BR\gvhmr\gvhmr_daemon.py" --listen udp://0.0.0.0:5000?overrun_nonfatal=1&fifo_size=16384 --osc-ip 1.2.3.4 --osc-port 9100 --height 174 --k 8 --mirror --flip-x --out-hz 120 --lead-ms 50"#
            t.check(cmd == expected, "GVHMR 8-point/facing mismatch:\n  got:  \(cmd)\n  want: \(expected)")
            t.check(!cmd.contains("--six-point"), "8-point must omit --six-point")
            t.check(!cmd.contains("--moving"), "Static must omit --moving")
        }

        t.test("GVHMR Moving toggle adds --moving") {
            var c = config(model: .gvhmr, sendElbows: false)
            c.gvhmrMoving = true
            let cmd = c.daemonCommandLine(oscIP: "1.2.3.4", oscPort: 9100, skeletonBack: nil)
            t.check(cmd.contains(" --moving"), "Moving camera mode must pass --moving (got \(cmd))")
            // Static produces the SAME command minus the flag.
            var s = config(model: .gvhmr, sendElbows: false)
            s.gvhmrMoving = false
            let staticCmd = s.daemonCommandLine(oscIP: "1.2.3.4", oscPort: 9100, skeletonBack: nil)
            t.check(!staticCmd.contains("--moving"), "Static must omit --moving")
            t.check(cmd == staticCmd.replacingOccurrences(of: " --out-hz", with: " --moving --out-hz"),
                    "Moving differs from Static only by the inserted --moving flag")
        }

        t.test("GVHMR Foot contact toggle adds --foot-contact (after --moving, before --out-hz)") {
            var c = config(model: .gvhmr, sendElbows: false)
            c.gvhmrFootContact = true
            let cmd = c.daemonCommandLine(oscIP: "1.2.3.4", oscPort: 9100, skeletonBack: nil)
            let expected = #""C:\BR\gvhmr\.venv\Scripts\python.exe" "C:\BR\gvhmr\gvhmr_daemon.py" --listen udp://0.0.0.0:5000?overrun_nonfatal=1&fifo_size=16384 --osc-ip 1.2.3.4 --osc-port 9100 --height 174 --k 5 --six-point --foot-contact --out-hz 120 --lead-ms 50"#
            t.check(cmd == expected, "GVHMR foot-contact mismatch:\n  got:  \(cmd)\n  want: \(expected)")
        }

        t.test("GVHMR Native rotations toggle adds --native-rot") {
            var c = config(model: .gvhmr, sendElbows: false)
            c.gvhmrNativeRot = true
            let cmd = c.daemonCommandLine(oscIP: "1.2.3.4", oscPort: 9100, skeletonBack: nil)
            let expected = #""C:\BR\gvhmr\.venv\Scripts\python.exe" "C:\BR\gvhmr\gvhmr_daemon.py" --listen udp://0.0.0.0:5000?overrun_nonfatal=1&fifo_size=16384 --osc-ip 1.2.3.4 --osc-port 9100 --height 174 --k 5 --six-point --native-rot --out-hz 120 --lead-ms 50"#
            t.check(cmd == expected, "GVHMR native-rot mismatch:\n  got:  \(cmd)\n  want: \(expected)")
        }

        t.test("GVHMR both experimental flags: --foot-contact before --native-rot, both before --skeleton-back/--out-hz") {
            var c = config(model: .gvhmr, sendElbows: false)
            c.gvhmrFootContact = true; c.gvhmrNativeRot = true
            let cmd = c.daemonCommandLine(oscIP: "192.168.1.50", oscPort: 9000, skeletonBack: "10.0.0.2:5001")
            let expected = #""C:\BR\gvhmr\.venv\Scripts\python.exe" "C:\BR\gvhmr\gvhmr_daemon.py" --listen udp://0.0.0.0:5000?overrun_nonfatal=1&fifo_size=16384 --osc-ip 192.168.1.50 --osc-port 9000 --height 174 --k 5 --six-point --foot-contact --native-rot --skeleton-back 10.0.0.2:5001 --out-hz 120 --lead-ms 50"#
            t.check(cmd == expected, "GVHMR both-flags mismatch:\n  got:  \(cmd)\n  want: \(expected)")
            // Ordering invariants spelled out: foot-contact precedes native-rot, and both
            // precede the skeleton-back/out-hz tail.
            let fc = cmd.range(of: " --foot-contact")!.lowerBound
            let nr = cmd.range(of: " --native-rot")!.lowerBound
            let sb = cmd.range(of: " --skeleton-back")!.lowerBound
            let oh = cmd.range(of: " --out-hz")!.lowerBound
            t.check(fc < nr && nr < sb && sb < oh,
                    "flag order must be --foot-contact < --native-rot < --skeleton-back < --out-hz")
        }

        t.test("NLF never gets --foot-contact / --native-rot even when the GVHMR toggles are on") {
            var c = config(model: .nlf, sendElbows: false)
            c.gvhmrFootContact = true; c.gvhmrNativeRot = true
            let cmd = c.daemonCommandLine(oscIP: "1.2.3.4", oscPort: 9100, skeletonBack: nil)
            t.check(!cmd.contains("--foot-contact"), "NLF must never pass --foot-contact (got \(cmd))")
            t.check(!cmd.contains("--native-rot"), "NLF must never pass --native-rot (got \(cmd))")
        }

        // The --listen target follows the (pinned) stream port, so the daemon binds the
        // exact UDP socket the Mac streams H.264 to.
        t.test("GVHMR --listen port follows the configured stream port") {
            let c = PCOffloadConfig.make(
                host: "10.0.0.9", user: "u", mac: "AA:BB:CC:DD:EE:FF", model: .gvhmr,
                oscIP: "1.2.3.4", oscPort: 9000, relayViaMac: false, relayPort: 9001,
                heightCm: 174, sendElbows: false, mirror: true,
                streamW: 1280, streamH: 720, streamFPS: 30, bitrateMbps: 10,
                politeMode: false, fpsCap: 0, streamPort: 5000, cameraName: nil)
            let cmd = c.daemonCommandLine(oscIP: "1.2.3.4", oscPort: 9000, skeletonBack: nil)
            t.check(cmd.contains("--listen udp://0.0.0.0:5000?overrun_nonfatal=1&fifo_size=16384"),
                    "listen URL must target the stream port with the overrun/fifo query (got \(cmd))")
        }
    }

    /// A deterministic config for the launch tests (real `.make` path construction under
    /// the test's `FEVER_PC_BRIDGE`). Direct OSC route; the caller tweaks flags as needed.
    private static func config(model: PCModel, sendElbows: Bool) -> PCOffloadConfig {
        PCOffloadConfig.make(
            host: "10.0.0.9", user: "u", mac: "AA:BB:CC:DD:EE:FF", model: model,
            oscIP: "192.168.1.50", oscPort: 9000,
            relayViaMac: false, relayPort: 9001,
            heightCm: 174, sendElbows: sendElbows, mirror: true, predictionLeadMs: 50,
            gvhmrK: 5, gvhmrMirror: false, gvhmrFlipX: false, gvhmrMoving: false,
            streamW: 1280, streamH: 720, streamFPS: 30, bitrateMbps: 10,
            politeMode: false, fpsCap: 0, streamPort: 5000, cameraName: nil)
    }
}
