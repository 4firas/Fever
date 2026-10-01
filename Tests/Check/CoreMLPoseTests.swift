import CoreVideo
import Foundation
import simd
import FeverCore

/// Coverage for the native CoreML pose backend (`CoreMLPoseLandmarker`) — the real
/// PinoFBT NLF graph, in-process.
///
/// Pins the two things that are easy to get silently wrong:
///   1. **the joint slice** — the graph emits 1048 localizer queries and the SMPL-24
///      joints are the LAST 24 (1024…1047). If that index moved, the skeleton would
///      still "look like numbers" but the anatomy would be nonsense, so we assert the
///      topology (head above pelvis, ankles below knees — the model's +Y-down frame).
///   2. **the presence gate** — a no-person frame collapses the skeleton to ~2 cm
///      across, so `hasTracked` must be 0 there; a real body is ~0.3-0.5 m.
///
/// Skipped (not failed) when the model isn't installed on this machine.
enum CoreMLPoseTests {

    static func register(_ t: TestRunner) async {
        guard let paths = CoreMLModelPaths.resolve() else {
            print("ok - CoreMLPoseTests skipped (no pose.mlmodelc installed)")
            return
        }

        let landmarker = try? CoreMLPoseLandmarker(modelURL: paths.model)
        t.test("CoreMLPoseLandmarker: loads the compiled NLF graph") {
            t.check(landmarker != nil, "model at \(paths.model.path) loads")
        }
        guard let lm = landmarker else { return }

        // Inference happens out here: the harness's test bodies are synchronous.
        guard let pb = blankBuffer(CoreMLPoseLandmarker.inputSide) else {
            t.check(false, "could not make a test pixel buffer"); return
        }
        let pose = await lm.detect(pb, at: 1.0)
        let again = await lm.detect(pb, at: 1.1)
        lm.reset()

        t.test("CoreMLPoseLandmarker: blank frame → SMPL-24 anatomy decode") {
            guard let pose else { t.check(false, "detect returned nil"); return }
            t.check(pose.joints3D.count == SMPLJoint.count, "24 joints3D (got \(pose.joints3D.count))")
            t.check(pose.joints2D.count == SMPLJoint.count, "24 joints2D")
            t.check(pose.width == CoreMLPoseLandmarker.inputSide, "reports the fed frame size")

            // +Y-down camera space: the head is ABOVE (smaller y) the pelvis, the
            // ankles BELOW the knees, and the spine climbs monotonically. All of these
            // only hold if the 1024…1047 slice really is the SMPL-24 skeleton.
            let head = pose.joints3D[SMPLJoint.head.rawValue]
            let pelvis = pose.joints3D[SMPLJoint.pelvis.rawValue]
            let knee = pose.joints3D[SMPLJoint.leftKnee.rawValue]
            let ankle = pose.joints3D[SMPLJoint.leftAnkle.rawValue]
            let spine1 = pose.joints3D[SMPLJoint.spine1.rawValue]
            let chest = pose.joints3D[SMPLJoint.spine3.rawValue]
            t.check(head.y < pelvis.y, "head above pelvis (+Y down): head \(head.y) < pelvis \(pelvis.y)")
            t.check(ankle.y > knee.y, "ankle below knee: \(ankle.y) > \(knee.y)")
            t.check(spine1.y < pelvis.y && chest.y < spine1.y,
                    "spine climbs: pelvis \(pelvis.y) > spine1 \(spine1.y) > chest \(chest.y)")

            // A blank frame has no body: the graph's guess is degenerate and must not
            // claim tracking (otherwise Fever would stream a phantom skeleton).
            t.check(!pose.isTracked, "no person in frame → hasTracked 0 (got \(pose.hasTracked))")
        }

        t.test("CoreMLPoseLandmarker: repeated frames are stable") {
            guard let pose, let again else { t.check(false, "detect returned nil"); return }
            var same = true
            for i in 0..<SMPLJoint.count where simd_distance(pose.joints3D[i], again.joints3D[i]) > 1e-4 {
                same = false
            }
            t.check(same, "same input frame → same joints")
            t.check(again.timestamp == 1.1, "timestamp is the capture time")
        }
    }

    // MARK: - helpers

    private static func blankBuffer(_ side: Int) -> CVPixelBuffer? {
        var pb: CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault, side, side, kCVPixelFormatType_32BGRA,
                            [kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary] as CFDictionary, &pb)
        guard let buf = pb else { return nil }
        CVPixelBufferLockBaseAddress(buf, [])
        if let base = CVPixelBufferGetBaseAddress(buf) {
            memset(base, 0, CVPixelBufferGetDataSize(buf))
        }
        CVPixelBufferUnlockBaseAddress(buf, [])
        return buf
    }
}
