import Foundation
import simd

/// Wire codec for the NLF (Neural Localizer Fields) onnxruntime sidecar.
///
/// Faithful to the proven sidecar protocol (PinoFBT-derived, personal-study only):
///   stdin  <- int32 W (LE), int32 H (LE), then W*H*3 raw RGB uint8 (top-left origin)
///   stdout -> one JSON line: {"ht":0/1,"box":[..],"j2":[[x,y]*24],"j3":[[x,y,z]*24],"w","h"}
/// The model carries its own detector + tracker (last_box/use_tracker/has_tracked)
/// and vertical-orientation auto-correction internally, so the caller just streams frames.
public enum NLFProtocol {

    /// int32 W (LE) + int32 H (LE) + raw RGB. No magic/seq/header — the sidecar
    /// reads `<ii` then exactly W*H*3 bytes.
    public static func encodeRequest(width: Int, height: Int, rgb: Data) -> Data {
        var out = Data(capacity: 8 + rgb.count)
        var w = Int32(width).littleEndian
        var h = Int32(height).littleEndian
        withUnsafeBytes(of: &w) { out.append(contentsOf: $0) }
        withUnsafeBytes(of: &h) { out.append(contentsOf: $0) }
        out.append(rgb)
        return out
    }

    private struct Reply: Decodable {
        let ht: Double
        let box: [Double]?
        let j2: [[Double]]?
        let j3: [[Double]]?
        let w: Int?
        let h: Int?
        let err: String?
        let ready: Bool?
    }

    /// True if this stdout line is the sidecar's startup `{"ready":true}` handshake.
    public static func isReady(_ line: String) -> Bool {
        guard let data = line.data(using: .utf8),
              let r = try? JSONDecoder().decode(Reply.self, from: data) else { return false }
        return r.ready == true
    }

    /// Decode one reply line into an SMPLPose. A not-detected / error frame
    /// (`ht<=0.5`, missing joints, or `err`) decodes to an untracked pose — never nil,
    /// so a momentary miss does NOT tear the sidecar down.
    public static func decodeReply(_ line: String, timestamp: Double) -> SMPLPose? {
        guard let data = line.data(using: .utf8),
              let r = try? JSONDecoder().decode(Reply.self, from: data) else { return nil }
        guard r.err == nil, r.ht > 0.5,
              let j3 = r.j3, let j2 = r.j2,
              j3.count == SMPLJoint.count, j2.count == SMPLJoint.count else {
            return .untracked(timestamp: timestamp)
        }
        let p3 = j3.map { SIMD3<Float>(Float($0[0]), Float($0[1]), Float($0[2])) }
        let p2 = j2.map { SIMD2<Float>(Float($0[0]), Float($0[1])) }
        return SMPLPose(joints3D: p3, joints2D: p2, hasTracked: Float(r.ht), timestamp: timestamp,
                        width: r.w ?? 0, height: r.h ?? 0)
    }
}
