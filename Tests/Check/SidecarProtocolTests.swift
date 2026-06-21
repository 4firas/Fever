import Foundation
import simd
import FeverCore

enum SidecarProtocolTests {
    static func run(_ t: TestRunner) {
        t.test("SidecarProtocol decodes a found reply") {
            var body = Data()
            func u32(_ v: UInt32) { var x = v.littleEndian; body.append(Data(bytes: &x, count: 4)) }
            func u8(_ v: UInt8) { var x = v; body.append(Data(bytes: &x, count: 1)) }
            func f32(_ v: Float) { var x = v.bitPattern.littleEndian; body.append(Data(bytes: &x, count: 4)) }
            u32(SidecarProtocol.repMagic); u32(42); u8(1)
            for i in 0..<33 { f32(Float(i)); f32(Float(i) + 0.5); f32(Float(i) + 0.25) } // world xyz
            for i in 0..<33 { f32(Float(i) / 33) }                                        // vis
            for _ in 0..<33 { f32(1) }                                                    // pres
            for i in 0..<33 { f32(Float(i) / 100); f32(Float(i) / 200) }                  // image xy

            guard let (seq, r) = SidecarProtocol.decodeReply(body) else {
                t.check(false, "decodeReply returned nil"); return
            }
            t.check(seq == 42, "seq parsed")
            t.check(r.found, "found true")
            t.check(r.world.count == 33 && r.image.count == 33, "33 landmarks")
            t.close(r.world[5].y, 5.5, tol: 1e-4, "world y[5] == 5.5")
            t.close(r.image[10].x, 0.10, tol: 1e-4, "image x[10] == 0.10")
            t.close(r.visibility[3], 3.0 / 33.0, tol: 1e-5, "vis[3]")
        }

        t.test("SidecarProtocol decodes a not-found reply") {
            var body = Data()
            func u32(_ v: UInt32) { var x = v.littleEndian; body.append(Data(bytes: &x, count: 4)) }
            func u8(_ v: UInt8) { var x = v; body.append(Data(bytes: &x, count: 1)) }
            u32(SidecarProtocol.repMagic); u32(5); u8(0)
            guard let (seq, r) = SidecarProtocol.decodeReply(body) else {
                t.check(false, "decodeReply returned nil for not-found"); return
            }
            t.check(seq == 5 && !r.found, "not-found parsed")
            t.check(r.world.isEmpty, "empty world for not-found")
        }

        t.test("SidecarProtocol rejects bad magic") {
            var body = Data([0,0,0,0, 0,0,0,0, 0])
            t.check(SidecarProtocol.decodeReply(body) == nil, "bad magic -> nil")
            body = Data([1, 2])
            t.check(SidecarProtocol.decodeReply(body) == nil, "short body -> nil")
            body = Data([0,0,0,0, 0,0,0,0]) // 8 bytes, one short of the 9-byte minimum
            t.check(SidecarProtocol.decodeReply(body) == nil, "8-byte body -> nil")
        }

        t.test("SidecarProtocol rejects a truncated found reply") {
            var body = Data()
            func u32(_ v: UInt32) { var x = v.littleEndian; body.append(Data(bytes: &x, count: 4)) }
            func u8(_ v: UInt8) { var x = v; body.append(Data(bytes: &x, count: 1)) }
            func f32(_ v: Float) { var x = v.bitPattern.littleEndian; body.append(Data(bytes: &x, count: 4)) }
            u32(SidecarProtocol.repMagic); u32(7); u8(1) // valid found header...
            for i in 0..<10 { f32(Float(i)) }            // ...but only 10 floats of payload (torn IPC)
            t.check(SidecarProtocol.decodeReply(body) == nil, "truncated found body -> nil")
        }

        t.test("SidecarProtocol request is length-prefixed with magic") {
            let req = SidecarProtocol.encodeRequest(seq: 9, tMicros: 1000,
                                                    width: 2, height: 1, rgb: Data([1,2,3, 4,5,6]))
            let b = [UInt8](req)
            let len = UInt32(b[0]) | UInt32(b[1]) << 8 | UInt32(b[2]) << 16 | UInt32(b[3]) << 24
            t.check(Int(len) == req.count - 4, "request length prefix correct")
            let magic = UInt32(b[4]) | UInt32(b[5]) << 8 | UInt32(b[6]) << 16 | UInt32(b[7]) << 24
            t.check(magic == SidecarProtocol.reqMagic, "request magic present")
            // body = 21-byte header + 6 rgb bytes = 27.
            t.check(Int(len) == 27, "body length 21+6 == 27, got \(len)")
        }
    }
}
