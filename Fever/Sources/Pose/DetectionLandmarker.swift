import CoreML
import CoreVideo
import Foundation

/// One detected person, in normalized source-frame coordinates (0…1, top-left).
public struct PersonBox: Sendable, Equatable {
    public var x: Float        // left
    public var y: Float        // top
    public var w: Float
    public var h: Float
    public var score: Float

    public init(x: Float, y: Float, w: Float, h: Float, score: Float) {
        self.x = x; self.y = y; self.w = w; self.h = h; self.score = score
    }

    public var centerX: Float { x + w / 2 }
    public var centerY: Float { y + h / 2 }
    public var area: Float { w * h }

    /// Grow the box by `fraction` of its size on every side (a person crop needs a
    /// little air around the body — feet and hands clip otherwise).
    public func expanded(by fraction: Float) -> PersonBox {
        PersonBox(x: x - w * fraction, y: y - h * fraction,
                  w: w * (1 + 2 * fraction), h: h * (1 + 2 * fraction), score: score)
    }

    /// Intersection-over-union, for de-duplicating overlapping people.
    public func iou(_ o: PersonBox) -> Float {
        let x0 = max(x, o.x), y0 = max(y, o.y)
        let x1 = min(x + w, o.x + o.w), y1 = min(y + h, o.y + o.h)
        let iw = max(0, x1 - x0), ih = max(0, y1 - y0)
        let inter = iw * ih
        let union = area + o.area - inter
        return union > 0 ? inter / union : 0
    }
}

/// Person detector — the vendor's `detection.mlmodelc` (YOLO-style, 640×640 BGRA in,
/// `box_raw [1, 8400, 84]` fp16 out: 4 box values + 80 class scores per anchor).
///
/// PinoFBT runs this alongside the pose model (its `InferenceManager` holds
/// `detectionModel`, `detectionTargetBuffer`, `_activeBox`/`_safeLastBox`,
/// `confThreshold`): the box drives the on-screen rectangle and keeps a stale-but-sane
/// box alive across frames where the detector dips. This mirrors that.
public final class DetectionLandmarker: @unchecked Sendable {

    /// Model input side (the graph's image constraint).
    public static let inputSide = 640
    /// COCO class 0 is `person`.
    public static let personClass = 0

    private let model: MLModel
    private let lock = NSLock()
    private let input = ImageInputBuffer(side: DetectionLandmarker.inputSide)
    /// Confidence gate. Below this the frame yields no box and the caller holds the last one.
    public var confThreshold: Float = 0.45

    public init(modelURL: URL) throws {
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .all
        self.model = try MLModel(contentsOf: modelURL, configuration: cfg)
    }

    public convenience init?() {
        guard let p = CoreMLModelPaths.resolve(kind: .detection) else { return nil }
        try? self.init(modelURL: p.model)
    }

    /// Best person box in the frame, or nil when nothing clears the threshold.
    public func detect(_ pixelBuffer: CVPixelBuffer) -> PersonBox? {
        lock.lock()
        defer { lock.unlock() }
        guard let prepared = input.fill(from: pixelBuffer) else { return nil }
        guard let provider = try? MLDictionaryFeatureProvider(
            dictionary: ["image": MLFeatureValue(pixelBuffer: prepared)]) else { return nil }
        guard let out = try? model.prediction(from: provider),
              let raw = out.featureValue(for: "box_raw")?.multiArrayValue else { return nil }
        return Self.bestPerson(in: raw, confThreshold: confThreshold)
    }

    // MARK: - decode

    /// Pick the highest-scoring person anchor above `confThreshold`, then suppress
    /// anchors that overlap it (an 8400-anchor head fires many boxes per body).
    public static func bestPerson(in raw: MLMultiArray, confThreshold: Float) -> PersonBox? {
        let anchors = raw.shape[1].intValue
        let stride = raw.strides[1].intValue          // element stride; ANE/GPU pads it
        let comp = raw.strides[2].intValue
        let side = Float(inputSide)
        let ptr = raw.dataPointer.assumingMemoryBound(to: UInt16.self)
        func value(_ a: Int, _ c: Int) -> Float {
            Float(Float16(bitPattern: ptr[a * stride + c * comp]))
        }
        var best: PersonBox?
        for a in 0..<anchors {
            let score = value(a, 4 + personClass)
            guard score >= confThreshold else { continue }
            let cx = value(a, 0), cy = value(a, 1)
            let w = value(a, 2), h = value(a, 3)
            guard w > 0, h > 0 else { continue }
            let box = PersonBox(x: (cx - w / 2) / side, y: (cy - h / 2) / side,
                                w: w / side, h: h / side, score: score)
            if best == nil || box.score > best!.score { best = box }
        }
        return best
    }
}
