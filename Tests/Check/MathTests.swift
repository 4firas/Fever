import Foundation
import simd
import FeverCore

/// Direct unit coverage for the pure SIMD helpers in `Utils/Math.swift`.
///
/// Focus: `frameFromTwoAxes` — the degenerate / parallel-axis HOLD-LAST branch
/// that the whole rotation rework was built around. The integration tests
/// deliberately steer clear of the singular region, so a regression that drops
/// the parallel-axis check (`|cross| < 1e-4`) or flips a length threshold would
/// reintroduce the 90° hip/limb snap on vertical limbs WITHOUT tripping any
/// other test. These assertions pin that branch directly.
enum MathTests {
    static func run(_ t: TestRunner) {
        // A distinctive sentinel orientation: any code path that returns the
        // hold-last quat must return THIS one, bit-for-bit.
        let sentinel = simd_quatf(angle: .pi / 3, axis: SIMD3<Float>(1, 0, 0))

        t.test("frameFromTwoAxes parallel axes -> holdLast exactly") {
            // primary ∥ secondary (both +Y): cross == 0 → degenerate → hold-last.
            // This is the singularity guard; without it the frame collapses and
            // simd_quatf produces a non-rotation (the vertical-limb 90° snap).
            let q = frameFromTwoAxes(primary: SIMD3<Float>(0, 1, 0),
                                     secondary: SIMD3<Float>(0, 1, 0),
                                     holdLast: sentinel)
            t.check(q.vector == sentinel.vector,
                    "parallel axes must return holdLast unchanged: \(q.vector) vs \(sentinel.vector)")
        }

        t.test("frameFromTwoAxes anti-parallel axes -> holdLast exactly") {
            // secondary = -primary is also degenerate (cross == 0).
            let q = frameFromTwoAxes(primary: SIMD3<Float>(0, 1, 0),
                                     secondary: SIMD3<Float>(0, -1, 0),
                                     holdLast: sentinel)
            t.check(q.vector == sentinel.vector,
                    "anti-parallel axes must return holdLast unchanged: \(q.vector)")
        }

        t.test("frameFromTwoAxes zero primary -> holdLast exactly") {
            // |primary| ≈ 0 fails the pl > 1e-5 guard → hold-last (no NaN normalize).
            let q = frameFromTwoAxes(primary: SIMD3<Float>(0, 0, 0),
                                     secondary: SIMD3<Float>(1, 0, 0),
                                     holdLast: sentinel)
            t.check(q.vector == sentinel.vector,
                    "zero primary must return holdLast unchanged: \(q.vector)")
        }

        t.test("frameFromTwoAxes zero secondary -> holdLast exactly") {
            // |secondary| ≈ 0 fails the sl > 1e-5 guard → hold-last.
            let q = frameFromTwoAxes(primary: SIMD3<Float>(0, 1, 0),
                                     secondary: SIMD3<Float>(0, 0, 0),
                                     holdLast: sentinel)
            t.check(q.vector == sentinel.vector,
                    "zero secondary must return holdLast unchanged: \(q.vector)")
        }

        t.test("frameFromTwoAxes perpendicular axes -> finite orthonormal rotation") {
            // primary=+Y, secondary=+X: well-conditioned, must build a proper
            // right-handed rotation (NOT hold-last).
            let q = frameFromTwoAxes(primary: SIMD3<Float>(0, 1, 0),
                                     secondary: SIMD3<Float>(1, 0, 0),
                                     holdLast: sentinel)
            let v = q.vector
            t.check(v.x.isFinite && v.y.isFinite && v.z.isFinite && v.w.isFinite,
                    "perpendicular case produced non-finite quat: \(v)")
            // Unit quaternion.
            t.close(simd_length(v), 1, tol: 1e-4, "quat must be unit length")
            // It must NOT be the hold-last sentinel (it built a real frame).
            t.check(v != sentinel.vector,
                    "perpendicular case must build a new frame, not hold-last")

            // The reconstructed rotation matrix columns must be orthonormal.
            let m = simd_float3x3(q)
            let c0 = m.columns.0, c1 = m.columns.1, c2 = m.columns.2
            t.close(simd_length(c0), 1, tol: 1e-4, "column 0 unit length")
            t.close(simd_length(c1), 1, tol: 1e-4, "column 1 unit length")
            t.close(simd_length(c2), 1, tol: 1e-4, "column 2 unit length")
            t.close(simd_dot(c0, c1), 0, tol: 1e-4, "columns 0,1 orthogonal")
            t.close(simd_dot(c0, c2), 0, tol: 1e-4, "columns 0,2 orthogonal")
            t.close(simd_dot(c1, c2), 0, tol: 1e-4, "columns 1,2 orthogonal")
            // Right-handed: c0 × c1 == c2.
            let cross = simd_cross(c0, c1)
            t.close(cross.x, c2.x, tol: 1e-4, "right-handed basis x")
            t.close(cross.y, c2.y, tol: 1e-4, "right-handed basis y")
            t.close(cross.z, c2.z, tol: 1e-4, "right-handed basis z")
            // local +Y (column 1) must point along the primary direction.
            t.close(c1.y, 1, tol: 1e-4, "local +Y aligns with primary (+Y)")
        }
    }
}
