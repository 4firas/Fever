import Foundation
import simd
import FeverCore

/// Wire-level parity pins for the 1:1 PinoFBT port: the shipping handedness
/// default and the `/rotation` slot coverage. (The old MediaPipe coordinate-map /
/// rest-relative-rebase parity tests were removed with that pipeline — the live
/// wire now comes straight from `PinoSolver`.)
enum WireParityTests {

    static func run(_ t: TestRunner) {
        t.test("PARITY: shipping DEFAULT is mirror-ON (webcam shows the user mirrored)") {
            UserDefaults.standard.removeObject(forKey: "mirrorTracking")
            let cfg = TrackingConfig()
            t.check(cfg.mirrorTracking == true,
                    "default mirrorTracking must be true: \(cfg.mirrorTracking)")
        }

        // ── ROTATION COVERAGE — ALL 8 (PinoFBT parity) ─────────────────────────
        t.test("ROTATION SLOTS: all 8 body trackers carry /rotation (PinoFBT parity)") {
            for s in ["1", "2", "3", "4", "5", "6", "7", "8"] {
                t.check(OSCSender.rotationSlots.contains(s), "slot \(s) must carry rotation")
            }
            t.check(OSCSender.rotationSlots.count == 8,
                    "exactly 8 rotation slots, got \(OSCSender.rotationSlots.count)")
        }
    }
}
