import CoreML
import CoreVideo
import Foundation
import FeverCore

/// Coverage for the person detector (`DetectionLandmarker`) and its YOLO-style decode.
///
/// The decode is the part that fails *quietly*: an 8400-anchor head with the class
/// scores offset by 4, coordinates in input pixels (not normalized), and ANE-padded
/// strides. Getting any of that wrong yields a box that still looks plausible, so the
/// decode is checked against a synthetic head where the right answer is known.
enum DetectionTests {

    static func register(_ t: TestRunner) async {
        // --- synthetic head: one person anchor above threshold, one below, one other class
        func anchor(_ a: MLMultiArray, _ stride: Int, _ comp: Int, _ i: Int,
                    cx: Float, cy: Float, w: Float, h: Float, person: Float, other: Float) {
            let p = a.dataPointer.assumingMemoryBound(to: UInt16.self)
            func set(_ c: Int, _ v: Float) { p[i * stride + c * comp] = Float16(v).bitPattern }
            set(0, cx); set(1, cy); set(2, w); set(3, h)
            set(4, person)                     // class 0 = person
            set(4 + 1, other)                  // class 1 = bicycle, say
        }

        let shape: [NSNumber] = [1, 10, 84]
        guard let head = try? MLMultiArray(shape: shape, dataType: .float16) else {
            t.check(false, "could not allocate a synthetic head"); return
        }
        let stride = 84, comp = 1
        // Clear, then plant: anchor 3 is the person (cx 320, cy 320, 160x320 → a body in
        // the middle of a 640² frame at 0.9 confidence); anchor 7 is a weak person;
        // anchor 9 is a confident *bicycle*, which must never be taken for a person.
        let p = head.dataPointer.assumingMemoryBound(to: UInt16.self)
        for i in 0..<(10 * 84) { p[i] = Float16(0).bitPattern }
        anchor(head, stride, comp, 3, cx: 320, cy: 320, w: 160, h: 320, person: 0.9, other: 0)
        anchor(head, stride, comp, 7, cx: 100, cy: 100, w: 80, h: 80, person: 0.30, other: 0)
        anchor(head, stride, comp, 9, cx: 500, cy: 500, w: 100, h: 100, person: 0, other: 0.99)

        let box = DetectionLandmarker.bestPerson(in: head, confThreshold: 0.45)
        t.test("DetectionLandmarker.decode: picks the best PERSON anchor, normalized") {
            guard let box else { t.check(false, "no box decoded"); return }
            t.close(box.score, 0.9, tol: 0.01, "score is the person anchor's")
            // cx 320 of 640 → 0.5 centre; 160x320 px → 0.25x0.5 normalized
            t.close(box.centerX, 0.5, tol: 0.01, "centre x")
            t.check(abs(box.h / box.w - 2.0) < 0.02, "aspect preserved (h = 2w here)")
            t.close(box.y, 0.25, tol: 0.01, "top edge")
            t.close(box.h, 0.5, tol: 0.01, "height")
        }

        t.test("DetectionLandmarker.decode: the confidence gate drops weak anchors") {
            t.check(DetectionLandmarker.bestPerson(in: head, confThreshold: 0.95) == nil,
                    "nothing clears 0.95")
            let low = DetectionLandmarker.bestPerson(in: head, confThreshold: 0.25)
            t.close(low?.score ?? 0, 0.9, tol: 0.01, "still the strongest person above 0.25")
        }

        t.test("DetectionLandmarker.decode: a confident non-person class is not a person") {
            // Person score is 0 on the bicycle anchor, so with the gate above it the
            // decoder must fall back to the real person, never the bicycle.
            let b = DetectionLandmarker.bestPerson(in: head, confThreshold: 0.5)
            t.check(b != nil && abs((b?.centerX ?? 0) - 0.5) < 0.01, "chose the person anchor")
        }

        t.test("DetectionLandmarker.decode: an empty head has nobody") {
            guard let empty = try? MLMultiArray(shape: shape, dataType: .float16) else { return }
            let q = empty.dataPointer.assumingMemoryBound(to: UInt16.self)
            for i in 0..<(10 * 84) { q[i] = Float16(0).bitPattern }
            t.check(DetectionLandmarker.bestPerson(in: empty, confThreshold: 0.1) == nil, "no anchors")
        }

        // --- the real compiled model, if installed
        guard let paths = CoreMLModelPaths.resolve(kind: .detection),
              let detector = try? DetectionLandmarker(modelURL: paths.model) else {
            print("ok - DetectionLandmarker model skipped (no detection.mlmodelc installed)")
            return
        }
        var pb: CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault, 640, 640, kCVPixelFormatType_32BGRA,
                            [kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary] as CFDictionary, &pb)
        let blank = pb
        if let blank {
            // Wall-clock cost of one detection: decides the cadence the pipeline can
            // afford (the model is small, but if it falls back to the CPU it is not).
            var samples: [Double] = []
            for _ in 0..<20 {
                let t0 = ProcessInfo.processInfo.systemUptime
                _ = detector.detect(blank)
                samples.append((ProcessInfo.processInfo.systemUptime - t0) * 1000)
            }
            let mean = samples.reduce(0,+)/Double(samples.count)
            print(String(format: "     [detect] mean %.1f ms  min %.1f  max %.1f  (n=%d)",
                         mean, samples.min() ?? 0, samples.max() ?? 0, samples.count))
        }
        let result = blank.flatMap { detector.detect($0) }
        t.test("DetectionLandmarker: the compiled model loads and rejects a blank frame") {
            t.check(result == nil, "no person in an empty frame (got \(String(describing: result)))")
        }
    }
}
