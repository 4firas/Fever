import Foundation
import simd

/// Decoded reply from the pose sidecar. Arrays are 33-long when `found`, else empty.
public struct SidecarReply: Sendable {
    public let found: Bool
    public let world: [SIMD3<Float>]
    public let visibility: [Float]
    public let presence: [Float]
    public let image: [SIMD2<Float>]

    public init(found: Bool, world: [SIMD3<Float>], visibility: [Float],
                presence: [Float], image: [SIMD2<Float>]) {
        self.found = found
        self.world = world
        self.visibility = visibility
        self.presence = presence
        self.image = image
    }
}

/// Length-prefixed binary framing shared with `Sidecar/pose_server.py`.
/// Every wire message = `len: u32 LE` followed by `len` bytes of body.
public enum SidecarProtocol {
    public static let reqMagic: UInt32 = 0xF0E1D2C3
    public static let repMagic: UInt32 = 0xC3D2E1F0
    static let n = 33

    /// Returns a length-prefixed request frame (`len:u32` + body). `rgb` is
    /// tightly packed RGB888, `width*height*3` bytes.
    public static func encodeRequest(seq: UInt32, tMicros: UInt64,
                                     width: Int, height: Int, rgb: Data) -> Data {
        var b = Data()
        func u32(_ v: UInt32) { var x = v.littleEndian; b.append(Data(bytes: &x, count: 4)) }
        func u64(_ v: UInt64) { var x = v.littleEndian; b.append(Data(bytes: &x, count: 8)) }
        func u16(_ v: UInt16) { var x = v.littleEndian; b.append(Data(bytes: &x, count: 2)) }
        func u8(_ v: UInt8) { var x = v; b.append(Data(bytes: &x, count: 1)) }
        u32(reqMagic); u32(seq); u64(tMicros); u16(UInt16(width)); u16(UInt16(height)); u8(0)
        b.append(rgb)
        var len = UInt32(b.count).littleEndian
        var out = Data(bytes: &len, count: 4)
        out.append(b)
        return out
    }

    /// Decodes a reply BODY (length prefix already stripped by the reader).
    public static func decodeReply(_ body: Data) -> (seq: UInt32, reply: SidecarReply)? {
        let bytes = [UInt8](body)
        guard bytes.count >= 9 else { return nil }
        func u32(_ off: Int) -> UInt32 {
            UInt32(bytes[off]) | UInt32(bytes[off + 1]) << 8
            | UInt32(bytes[off + 2]) << 16 | UInt32(bytes[off + 3]) << 24
        }
        guard u32(0) == repMagic else { return nil }
        let seq = u32(4)
        let found = bytes[8] == 1
        if !found {
            return (seq, SidecarReply(found: false, world: [], visibility: [], presence: [], image: []))
        }
        let floatCount = n * 3 + n + n + n * 2
        guard bytes.count >= 9 + floatCount * 4 else { return nil }
        func f32(_ off: Int) -> Float {
            let bits = UInt32(bytes[off]) | UInt32(bytes[off + 1]) << 8
            | UInt32(bytes[off + 2]) << 16 | UInt32(bytes[off + 3]) << 24
            return Float(bitPattern: bits)
        }
        var p = 9
        var world = [SIMD3<Float>](repeating: .zero, count: n)
        for i in 0..<n { world[i] = SIMD3(f32(p + i*12), f32(p + i*12 + 4), f32(p + i*12 + 8)) }
        p += n * 12
        var vis = [Float](repeating: 0, count: n)
        for i in 0..<n { vis[i] = f32(p + i*4) }
        p += n * 4
        var pres = [Float](repeating: 0, count: n)
        for i in 0..<n { pres[i] = f32(p + i*4) }
        p += n * 4
        var image = [SIMD2<Float>](repeating: .zero, count: n)
        for i in 0..<n { image[i] = SIMD2(f32(p + i*8), f32(p + i*8 + 4)) }
        return (seq, SidecarReply(found: true, world: world, visibility: vis, presence: pres, image: image))
    }
}
