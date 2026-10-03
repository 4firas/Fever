import Accelerate
import CoreVideo
import Foundation

/// A reusable BGRA input buffer for a CoreML image input of `side`×`side`.
///
/// One place owns the frame→model-input scaling so the pose and detection models
/// can't drift apart: the whole frame (aspect stretched, which is what CoreML's own
/// image constraint does with a camera buffer) or a normalized sub-rectangle of it
/// (the detector's person box, used as a crop). vImage reads the source rows through
/// their real `rowBytes`, so a sub-rectangle is just a `vImage_Buffer` pointed inside
/// the source — no copy.
final class ImageInputBuffer {
    let side: Int
    private var buffer: CVPixelBuffer?

    init(side: Int) { self.side = side }

    /// Fill from the whole source frame (stretched to the model's square input).
    func fill(from src: CVPixelBuffer) -> CVPixelBuffer? {
        fill(from: src, rect: CGRect(x: 0, y: 0, width: 1, height: 1))
    }

    /// Fill from a normalized sub-rectangle of the source frame (0…1, top-left origin).
    func fill(from src: CVPixelBuffer, box: PersonBox, margin: Float = 0) -> CVPixelBuffer? {
        let b = box.expanded(by: margin)
        return fill(from: src, rect: CGRect(x: CGFloat(b.x), y: CGFloat(b.y),
                                            width: CGFloat(b.w), height: CGFloat(b.h)))
    }

    private func fill(from src: CVPixelBuffer, rect: CGRect) -> CVPixelBuffer? {
        if buffer == nil {
            var pb: CVPixelBuffer?
            CVPixelBufferCreate(kCFAllocatorDefault, side, side, kCVPixelFormatType_32BGRA,
                                [kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary] as CFDictionary, &pb)
            buffer = pb
        }
        guard let dst = buffer,
              CVPixelBufferGetPixelFormatType(src) == kCVPixelFormatType_32BGRA else { return nil }

        let srcW = CVPixelBufferGetWidth(src), srcH = CVPixelBufferGetHeight(src)
        // Normalized → pixels, clamped into the frame. A box that leaves the frame
        // (person half out of shot) must still produce a usable crop.
        let x0 = max(0, min(srcW - 1, Int((rect.minX * CGFloat(srcW)).rounded(.down))))
        let y0 = max(0, min(srcH - 1, Int((rect.minY * CGFloat(srcH)).rounded(.down))))
        let x1 = max(x0 + 1, min(srcW, Int((rect.maxX * CGFloat(srcW)).rounded(.up))))
        let y1 = max(y0 + 1, min(srcH, Int((rect.maxY * CGFloat(srcH)).rounded(.up))))
        let cropW = x1 - x0, cropH = y1 - y0

        CVPixelBufferLockBaseAddress(src, .readOnly)
        CVPixelBufferLockBaseAddress(dst, [])
        defer {
            CVPixelBufferUnlockBaseAddress(dst, [])
            CVPixelBufferUnlockBaseAddress(src, .readOnly)
        }
        guard let sb = CVPixelBufferGetBaseAddress(src), let db = CVPixelBufferGetBaseAddress(dst) else { return nil }
        let srcRow = CVPixelBufferGetBytesPerRow(src)
        // Sub-buffer pointing at the crop inside the source rows.
        var s = vImage_Buffer(data: sb.advanced(by: y0 * srcRow + x0 * 4),
                              height: vImagePixelCount(cropH), width: vImagePixelCount(cropW),
                              rowBytes: srcRow)
        var d = vImage_Buffer(data: db, height: vImagePixelCount(side), width: vImagePixelCount(side),
                              rowBytes: CVPixelBufferGetBytesPerRow(dst))
        guard vImageScale_ARGB8888(&s, &d, nil, vImage_Flags(kvImageHighQualityResampling)) == kvImageNoError
        else { return nil }
        return dst
    }
}
