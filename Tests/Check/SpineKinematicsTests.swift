import Foundation
import simd
import FeverCore

/// Locks the reverse-engineered PinoFBT spine math (Math.swift `calcRootRotation`
/// / `calcChestRotation` / `shortestArc`) against ground-truth vectors captured by
/// EMULATING the compiled `fast_kinematics.pyd` cores (Unicorn). Any drift in the
/// joint choices, the cross/order, or the quaternion convention trips these.
/// Oracle: `~/Downloads/fbt_re/oracle_vectors.json` (a1=fwd(0,0,1), a2=up(0,1,0)).
enum SpineKinematicsTests {

    static func run(_ t: TestRunner) {
        func mk(_ pairs: [(Int, Float, Float, Float)]) -> [SIMD3<Float>] {
            var w = [SIMD3<Float>](repeating: .zero, count: 24)
            for (i, x, y, z) in pairs { w[i] = SIMD3<Float>(x, y, z) }
            return w
        }
        // Quaternions compared sign-agnostically (q and -q are the same rotation).
        func qclose(_ q: simd_quatf, _ e: [Float]) -> Bool {
            let v = q.vector
            let d = abs(v.x * e[0] + v.y * e[1] + v.z * e[2] + v.w * e[3])
            return abs(d - 1) < 3e-3
        }

        struct Case { let n: String; let w: [SIMD3<Float>]; let root: [Float]; let chest: [Float] }
        let cases = [
            Case(n: "rest",
                 w: mk([(1, 0.07, -0.08, 0), (2, -0.07, -0.08, 0), (3, 0, 0.1, 0), (6, 0, 0.22, 0),
                        (9, 0, 0.3, 0), (13, 0.08, 0.42, 0), (14, -0.08, 0.42, 0)]),
                 root: [0, 0, 0, 1], chest: [0, 0, -0.70711, 0.70711]),
            Case(n: "yaw30",
                 w: mk([(1, 0.06062, -0.08, -0.035), (2, -0.06062, -0.08, 0.035), (3, 0, 0.1, 0),
                        (6, 0, 0.22, 0), (9, 0, 0.3, 0), (13, 0.06928, 0.42, -0.04), (14, -0.06928, 0.42, 0.04)]),
                 root: [0, 0.25882, 0, 0.96593], chest: [-0.18301, 0.18301, -0.68301, 0.68301]),
            Case(n: "pitch20",
                 w: mk([(1, 0.07, -0.07518, -0.02736), (2, -0.07, -0.07518, -0.02736), (3, 0, 0.09397, 0.0342),
                        (6, 0, 0.20673, 0.07524), (9, 0, 0.28191, 0.10261), (13, 0.08, 0.39467, 0.14365),
                        (14, -0.08, 0.39467, 0.14365)]),
                 root: [0.17365, 0, 0, 0.98481], chest: [0.12279, 0.12279, -0.69636, 0.69636]),
            Case(n: "roll15bend",
                 w: mk([(1, 0.08832, -0.05916, 0), (2, -0.04691, -0.09539, 0), (3, -0.02588, 0.09659, 0),
                        (6, -0.05694, 0.2125, 0), (9, -0.07765, 0.28978, 0), (13, -0.03143, 0.42639, 0),
                        (14, -0.18598, 0.38498, 0)]),
                 root: [0, 0, 0.13053, 0.99144], chest: [0, 0, -0.60876, 0.79335]),
        ]
        for c in cases {
            t.test("calc_root \(c.n) == PinoFBT oracle") {
                t.check(qclose(calcRootRotation(c.w), c.root),
                        "calc_root \(c.n): got \(calcRootRotation(c.w).vector) expect \(c.root)")
            }
            t.test("calc_chest \(c.n) == PinoFBT oracle") {
                t.check(qclose(calcChestRotation(c.w), c.chest),
                        "calc_chest \(c.n): got \(calcChestRotation(c.w).vector) expect \(c.chest)")
            }
        }

        // get_rotation == canonical shortest-arc (x->y is +90° about Z).
        t.test("shortestArc x->y is (0,0,.707,.707)") {
            let q = shortestArc(SIMD3<Float>(1, 0, 0), SIMD3<Float>(0, 1, 0))
            t.check(qclose(q, [0, 0, 0.70711, 0.70711]), "shortestArc x->y: \(q.vector)")
        }
        t.test("shortestArc parallel -> identity") {
            let q = shortestArc(SIMD3<Float>(0, 2, 0), SIMD3<Float>(0, 5, 0))
            t.check(qclose(q, [0, 0, 0, 1]), "shortestArc parallel: \(q.vector)")
        }
    }
}
