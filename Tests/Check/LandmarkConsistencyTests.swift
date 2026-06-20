import Foundation
import simd
import FeverCore

/// LandmarkConsistency: temporal L/R anti-swap + visibility-hysteresis gating.
enum LandmarkConsistencyTests {

    static func pose(_ pairs: [(BlazePose.Landmark, SIMD3<Float>, Float)]) -> PoseResult {
        var lm = [NormalizedLandmark](repeating: NormalizedLandmark(position: .zero, visibility: 0.9),
                                      count: 33)
        for (l, p, v) in pairs { lm[l.rawValue] = NormalizedLandmark(position: p, visibility: v) }
        return PoseResult(landmarks: lm, timestamp: 0)
    }

    static func run(_ t: TestRunner) {
        t.test("CONSISTENCY: transposed L/R ankles are re-assigned to match previous") {
            let c = LandmarkConsistency()
            // Frame 1 establishes prev: left at −x, right at +x.
            _ = c.process(pose([(.leftAnkle, SIMD3(-0.10, -0.8, 0), 0.9),
                                (.rightAnkle, SIMD3(0.10, -0.8, 0), 0.9)]))
            // Frame 2: MediaPipe TRANSPOSED the labels (left now at +x, right at −x).
            let out = c.process(pose([(.leftAnkle, SIMD3(0.10, -0.8, 0), 0.9),
                                      (.rightAnkle, SIMD3(-0.10, -0.8, 0), 0.9)]))
            t.check(out[.leftAnkle].position.x < 0,
                    "left ankle re-assigned to its side (x<0): \(out[.leftAnkle].position.x)")
            t.check(out[.rightAnkle].position.x > 0,
                    "right ankle re-assigned to its side (x>0): \(out[.rightAnkle].position.x)")
        }

        t.test("CONSISTENCY: a stable L/R pair is NOT swapped (small real motion)") {
            let c = LandmarkConsistency()
            _ = c.process(pose([(.leftAnkle, SIMD3(-0.10, -0.8, 0), 0.9),
                                (.rightAnkle, SIMD3(0.10, -0.8, 0), 0.9)]))
            // Tiny real motion, same sides — must not trigger a false swap.
            let out = c.process(pose([(.leftAnkle, SIMD3(-0.09, -0.8, 0), 0.9),
                                      (.rightAnkle, SIMD3(0.11, -0.8, 0), 0.9)]))
            t.check(out[.leftAnkle].position.x < 0 && out[.rightAnkle].position.x > 0,
                    "no false swap on small motion")
        }

        t.test("CONSISTENCY: an occluded (low-visibility) landmark is gated to absent") {
            let c = LandmarkConsistency()
            _ = c.process(pose([(.leftAnkle, SIMD3(-0.10, -0.8, 0), 0.9)]))   // seed engaged
            let out = c.process(pose([(.leftAnkle, SIMD3(-0.10, -0.8, 0), 0.2)]))  // occluded
            t.check(out[.leftAnkle].visibility == 0 && out[.leftAnkle].presence == 0,
                    "occluded ankle gated to absent so the predictor holds-last: vis=\(out[.leftAnkle].visibility)")
        }

        t.test("CONSISTENCY: visibility gating has hysteresis (no flicker at the band)") {
            let c = LandmarkConsistency()
            _ = c.process(pose([(.leftAnkle, SIMD3(-0.1, -0.8, 0), 0.9)]))  // engaged
            // 0.45 is below vOn(0.5) but ABOVE vOff(0.35) → stays engaged (hysteresis).
            let out = c.process(pose([(.leftAnkle, SIMD3(-0.1, -0.8, 0), 0.45)]))
            t.check(out[.leftAnkle].visibility > 0,
                    "0.45 between vOff and vOn stays engaged (hysteresis): \(out[.leftAnkle].visibility)")
        }
    }
}
