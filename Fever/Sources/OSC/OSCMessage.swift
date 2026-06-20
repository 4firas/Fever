import Foundation

/// Minimal OSC 1.0 message encoder. We only need `,fff` and `,ffff` tag sets
/// for VRChat tracker position/rotation, so this hand-rolled encoder is faster
/// and dependency-free on the inference-thread hot path.
public struct OSCMessage {
    public let address: String
    public let arguments: [OSCArgument]

    public init(address: String, arguments: [OSCArgument]) {
        self.address = address
        self.arguments = arguments
    }

    /// Serialize to an OSC packet (address + typetags + args, 4-byte aligned).
    ///
    /// Builds a SINGLE `Data` with capacity reserved up front and appends bytes
    /// in place, instead of allocating a separate `Data`/`[UInt8]` per field and
    /// concatenating. The wire layout is byte-for-byte identical: NUL-terminated
    /// 4-aligned address + type-tag strings, then big-endian arguments.
    public func encoded() -> Data {
        let addr = address.utf8
        let addrBlock = paddedLength(addr.count)          // NUL-terminated, 4-aligned
        let tagBlock = paddedLength(1 + arguments.count)  // "," + one char per arg
        var data = Data()
        data.reserveCapacity(addrBlock + tagBlock + arguments.count * 4)

        // Address string + NUL terminator + NUL pad.
        data.append(contentsOf: addr)
        appendPadding(&data, from: addr.count, to: addrBlock)

        // Type tag string: "," then one tag char per argument, NUL-pad to 4.
        data.append(0x2C) // ","
        for a in arguments { data.append(a.typeTagByte) }
        appendPadding(&data, from: 1 + arguments.count, to: tagBlock)

        // Big-endian argument payloads (each already a multiple of 4).
        for arg in arguments { arg.encode(into: &data) }
        return data
    }

    /// Length of an OSC string of `n` content bytes once a NUL terminator is
    /// added and the whole is NUL-padded up to a 4-byte boundary.
    @inline(__always)
    private func paddedLength(_ n: Int) -> Int { ((n / 4) + 1) * 4 }

    /// Append NUL bytes to grow `data`'s field from `current` to `target` length.
    @inline(__always)
    private func appendPadding(_ data: inout Data, from current: Int, to target: Int) {
        for _ in current..<target { data.append(0) }
    }
}

public enum OSCArgument {
    case float(Float)
    case int(Int32)

    var typeTag: String {
        switch self {
        case .float: return "f"
        case .int:   return "i"
        }
    }

    /// Single ASCII type-tag byte ("f" / "i"), avoiding a String + utf8 walk.
    @inline(__always)
    var typeTagByte: UInt8 {
        switch self {
        case .float: return 0x66 // 'f'
        case .int:   return 0x69 // 'i'
        }
    }

    func encoded() -> Data {
        var d = Data()
        encode(into: &d)
        return d
    }

    /// Append this argument's 4-byte big-endian payload directly to `data`,
    /// without allocating an intermediate `Data`.
    @inline(__always)
    func encode(into data: inout Data) {
        switch self {
        case .float(let v):
            // OSC floats are IEEE-754 big-endian.
            withUnsafeBytes(of: v.bitPattern.bigEndian) { data.append(contentsOf: $0) }
        case .int(let v):
            withUnsafeBytes(of: v.bigEndian) { data.append(contentsOf: $0) }
        }
    }
}
