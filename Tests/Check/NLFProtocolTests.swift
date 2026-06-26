import Foundation
import simd
import FeverCore

/// Coverage for the NLF sidecar wire codec (`NLFProtocol`) — previously untested.
/// Pins the stdin framing (int32-LE W/H + raw RGB) and the reply policy: a tracked
/// frame decodes to a real pose, a parseable-but-not-detected / error frame decodes
/// to a NON-nil untracked pose (so a momentary miss never tears the sidecar down),
/// and an unparseable line decodes to nil (a dead/garbled sidecar).
enum NLFProtocolTests {

    private static func j2() -> String {
        (0..<24).map { "[\(Double($0)),\(Double($0) + 0.5)]" }.joined(separator: ",")
    }
    private static func j3() -> String {
        (0..<24).map { "[\(Double($0)),\(Double($0) * 2),\(Double($0) * 3)]" }.joined(separator: ",")
    }

    static func run(_ t: TestRunner) {
        t.test("NLFProtocol.encodeRequest = int32-LE W + int32-LE H + raw RGB") {
            let rgb = Data((0..<(2 * 3 * 3)).map { UInt8($0) })   // 2x3 RGB = 18 bytes
            let bytes = [UInt8](NLFProtocol.encodeRequest(width: 2, height: 3, rgb: rgb))
            t.check(bytes.count == 8 + 18, "total = 8-byte header + W*H*3")
            t.check(bytes[0] == 2 && bytes[1] == 0 && bytes[2] == 0 && bytes[3] == 0, "width little-endian")
            t.check(bytes[4] == 3 && bytes[5] == 0 && bytes[6] == 0 && bytes[7] == 0, "height little-endian")
            t.check(Array(bytes[8...]) == [UInt8](rgb), "raw RGB appended verbatim")
        }

        t.test("NLFProtocol.isReady detects the {\"ready\":true} handshake only") {
            t.check(NLFProtocol.isReady("{\"ready\":true}"), "ready handshake → true")
            t.check(!NLFProtocol.isReady("{\"ht\":1.0,\"j3\":[]}"), "a normal reply → false")
            t.check(!NLFProtocol.isReady("not json"), "garbage → false")
        }

        t.test("NLFProtocol.decodeReply: a tracked frame carries 24 joints") {
            let line = "{\"ht\":1.0,\"j2\":[\(j2())],\"j3\":[\(j3())],\"w\":640,\"h\":480}"
            let pose = NLFProtocol.decodeReply(line, timestamp: 1.0)
            t.check(pose != nil, "tracked frame decodes")
            t.check(pose?.isTracked == true, "isTracked true (ht>0.5)")
            t.check(pose?.joints3D.count == 24 && pose?.joints2D.count == 24, "24 j3 + 24 j2 carried")
            t.check(pose?.joints3D[5].y == 10, "joint values mapped (j3[5] = (5,10,15))")
        }

        t.test("NLFProtocol.decodeReply: parseable-but-not-tracked → NON-nil untracked") {
            // low confidence
            let lowHt = "{\"ht\":0.2,\"j2\":[\(j2())],\"j3\":[\(j3())]}"
            t.check(NLFProtocol.decodeReply(lowHt, timestamp: 0).map { !$0.isTracked } == true,
                    "ht<=0.5 → untracked, not nil")
            // incomplete (no joints)
            t.check(NLFProtocol.decodeReply("{\"ht\":1.0}", timestamp: 0).map { !$0.isTracked } == true,
                    "missing joints → untracked, not nil")
            // explicit error
            t.check(NLFProtocol.decodeReply("{\"err\":\"boom\"}", timestamp: 0).map { !$0.isTracked } == true,
                    "err field → untracked, not nil")
            // wrong joint count
            let short = "{\"ht\":1.0,\"j2\":[[0,0]],\"j3\":[[0,0,0]]}"
            t.check(NLFProtocol.decodeReply(short, timestamp: 0).map { !$0.isTracked } == true,
                    "wrong joint count → untracked, not nil")
        }

        t.test("NLFProtocol.decodeReply: an unparseable line → nil (dead sidecar)") {
            t.check(NLFProtocol.decodeReply("not json{{", timestamp: 0) == nil, "garbage → nil")
            t.check(NLFProtocol.decodeReply("", timestamp: 0) == nil, "empty → nil")
        }
    }
}
