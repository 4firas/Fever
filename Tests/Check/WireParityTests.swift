import Foundation
import simd
import FeverCore

/// Wire-level parity pins, VERIFIED by a live OSC capture diff of Fever vs
/// PinoFBT (the 1:1 benchmark, same front-webcam-on-Mac setup).
///
/// ── Handedness (position X sign) ──────────────────────────────────────────
/// PinoFBT outputs the anatomical-LEFT limb at NEGATIVE X and tracks correctly
/// in VRChat, so Fever must land on the same side. The capture showed:
///   • mirror OFF (det +1, negate Y,Z only): Fever left → −X  ✓ matches PinoFBT
///   • mirror ON  (det −1):                  Fever left → +X  ✗ mirrored
/// The non-obvious reason: the MediaPipe **Tasks API** (Fever) emits world-X with
/// the OPPOSITE sign to the legacy GPU graph PinoFBT bundles — so Fever must NOT
/// negate X even though PinoFBT's binary does. Shipping default = mirror OFF.
///
/// ── Rotation coverage ─────────────────────────────────────────────────────
/// Capture showed Fever's HIP rotation is clean but CHEST/FOOT rotations are
/// broken, so only the hip carries /rotation for now (the rest are position-only,
/// IK-solved). Re-widen once the per-bone rotation solver is fixed.
enum WireParityTests {

    /// Net MediaPipe-world → VRChat position linear map: MediaPipeFrame's axis-fix
    /// (x, −y, z·zSign) then CoordinateMapper.toVRChatPosition (mirror, −z, scale).
    static func net(_ v: SIMD3<Float>, mirror: Bool) -> SIMD3<Float> {
        let solver = SIMD3<Float>(v.x, -v.y, v.z * MediaPipeFrame.defaultZSign)
        let m = CoordinateMapper(userHeightMeters: 1.8, referenceHeightMeters: 1.8,
                                 mirrorHorizontally: mirror)
        return m.toVRChatPosition(solver)
    }

    static func det(mirror: Bool) -> Float {
        let cx = net(SIMD3(1, 0, 0), mirror: mirror)
        let cy = net(SIMD3(0, 1, 0), mirror: mirror)
        let cz = net(SIMD3(0, 0, 1), mirror: mirror)
        return simd_determinant(simd_float3x3(cx, cy, cz))
    }

    static func run(_ t: TestRunner) {
        t.test("PARITY: shipping DEFAULT is mirror-OFF (matches PinoFBT, Tasks-API frame)") {
            UserDefaults.standard.removeObject(forKey: "mirrorTracking")
            let cfg = TrackingConfig()
            t.check(cfg.mirrorTracking == false,
                    "default mirrorTracking must be false (Tasks-API: left→−X like PinoFBT): \(cfg.mirrorTracking)")
        }

        t.test("PARITY: the shipping (mirror-OFF) net map preserves the Tasks-API frame (det +1)") {
            t.check(det(mirror: false) > 0,
                    "shipping net map must be det +1 (negate Y,Z only): \(det(mirror: false))")
            t.check(det(mirror: true) < 0,
                    "mirror-on is the det −1 reflection we do NOT ship: \(det(mirror: true))")
        }

        t.test("PARITY: anatomical left vs right land on opposite avatar sides") {
            let left  = net(SIMD3( 0.12, 0.9, 0.0), mirror: false)
            let right = net(SIMD3(-0.12, 0.9, 0.0), mirror: false)
            t.check(left.x * right.x < 0, "left/right map to opposite X: L=\(left.x) R=\(right.x)")
        }

        // ── ROTATION MODE — ABSOLUTE, not rest-relative (PinoFBT ground truth) ──
        // PinoFBT streams ABSOLUTE world orientations (its hip yaw tracks the turn;
        // limbs carry real orientation). Fever's rest-relative rebase ZEROED every
        // joint at the Recenter pose, which both diverged from PinoFBT and MASKED a
        // broken rotation solver. Default = absolute.
        t.test("ROTATION MODE: default is ABSOLUTE (rest-relative OFF)") {
            UserDefaults.standard.removeObject(forKey: "rotationRestRelative")
            let cfg = TrackingConfig()
            t.check(cfg.rotationRestRelative == false,
                    "default rotationRestRelative must be false (absolute, like PinoFBT): \(cfg.rotationRestRelative)")
        }

        t.test("ROTATION MODE: absolute rebase PRESERVES a real orientation (no zeroing)") {
            let rb = RotationRebaser(smoothingFactor: 0)
            let live = simd_quatf(angle: 40 * .pi / 180, axis: SIMD3<Float>(1, 0, 0))
            let out = rb.rebase(.leftElbow, live: live, captureNow: false)
            t.check(abs(simd_dot(out.vector, live.vector)) > 0.999,
                    "absolute rebase must emit the live orientation")
        }

        // ── ROTATION COVERAGE — ALL 8 (PinoFBT parity; solver now fixed) ───────
        t.test("ROTATION SLOTS: all 8 body trackers carry /rotation (PinoFBT parity)") {
            for s in ["1", "2", "3", "4", "5", "6", "7", "8"] {
                t.check(OSCSender.rotationSlots.contains(s), "slot \(s) must carry rotation")
            }
            t.check(OSCSender.rotationSlots.count == 8, "exactly 8 rotation slots, got \(OSCSender.rotationSlots.count)")
        }
    }
}
