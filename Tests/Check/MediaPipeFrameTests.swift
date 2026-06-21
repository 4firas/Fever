import Foundation
import simd
import FeverCore

enum MediaPipeFrameTests {
    static func run(_ t: TestRunner) {
        t.test("FloorOriginLatch freezes on first call") {
            let latch = FloorOriginLatch()
            let o1 = latch.latchOriginXZ(SIMD2(2, 3))
            let o2 = latch.latchOriginXZ(SIMD2(9, 9))
            t.check(o1 == SIMD2(2, 3) && o2 == SIMD2(2, 3), "origin XZ latched to first call")
            let f1 = latch.latchFloor(-1.0); let f2 = latch.latchFloor(5.0)
            t.check(f1 == -1.0 && f2 == -1.0, "floor latched to first call")
            latch.reset()
            t.check(latch.latchFloor(7.0) == 7.0, "reset re-latches")
        }

        t.test("MediaPipeFrame negates Y so shoulders sit above hips") {
            var world = [SIMD3<Float>](repeating: .zero, count: 33)
            let vis = [Float](repeating: 1, count: 33)
            // MediaPipe is y-DOWN: shoulders (above hips) have the SMALLER y.
            world[BlazePose.Landmark.leftShoulder.rawValue]  = SIMD3(-0.2, -0.5, 0.0)
            world[BlazePose.Landmark.rightShoulder.rawValue] = SIMD3( 0.2, -0.5, 0.0)
            world[BlazePose.Landmark.leftHip.rawValue]  = SIMD3(-0.1, 0.0, 0.0)
            world[BlazePose.Landmark.rightHip.rawValue] = SIMD3( 0.1, 0.0, 0.0)
            world[BlazePose.Landmark.leftAnkle.rawValue]  = SIMD3(-0.1, 0.9, 0.0)
            world[BlazePose.Landmark.rightAnkle.rawValue] = SIMD3( 0.1, 0.9, 0.0)
            let reply = SidecarReply(found: true, world: world, visibility: vis, presence: vis,
                                     image: [SIMD2<Float>](repeating: SIMD2(0.5, 0.5), count: 33))
            guard let pose = MediaPipeFrame.toSolverFrame(reply, latch: FloorOriginLatch()) else {
                t.check(false, "toSolverFrame returned nil for a valid torso"); return
            }
            let ls = pose[.leftShoulder].position, lh = pose[.leftHip].position
            t.check(ls.y > lh.y, "shoulders are ABOVE hips after y-negation (solver +Y up)")
            t.check(pose.landmarks.count == 33, "33 landmarks out")
            t.check(pose.imagePoints.count == 33, "33 image points out")
        }

        t.test("MediaPipeFrame centres hip XZ to ~origin and floors the feet") {
            var world = [SIMD3<Float>](repeating: .zero, count: 33)
            let vis = [Float](repeating: 1, count: 33)
            // Hip offset +2m in X (absolute camera frame); feet 0.9 below in MP y-down.
            world[BlazePose.Landmark.leftShoulder.rawValue]  = SIMD3(2.0 - 0.2, -0.5, 0.0)
            world[BlazePose.Landmark.rightShoulder.rawValue] = SIMD3(2.0 + 0.2, -0.5, 0.0)
            world[BlazePose.Landmark.leftHip.rawValue]  = SIMD3(2.0 - 0.1, 0.0, 0.0)
            world[BlazePose.Landmark.rightHip.rawValue] = SIMD3(2.0 + 0.1, 0.0, 0.0)
            world[BlazePose.Landmark.leftAnkle.rawValue]  = SIMD3(2.0 - 0.1, 0.9, 0.0)
            world[BlazePose.Landmark.rightAnkle.rawValue] = SIMD3(2.0 + 0.1, 0.9, 0.0)
            let reply = SidecarReply(found: true, world: world, visibility: vis, presence: vis, image: [])
            guard let pose = MediaPipeFrame.toSolverFrame(reply, latch: FloorOriginLatch()) else {
                t.check(false, "toSolverFrame nil"); return
            }
            let hipMidX = (pose[.leftHip].position.x + pose[.rightHip].position.x) * 0.5
            t.close(hipMidX, 0, tol: 1e-4, "hip X centred to ~0 after origin latch")
            let footY = Swift.min(pose[.leftAnkle].position.y, pose[.rightAnkle].position.y)
            t.close(footY, 0, tol: 1e-4, "lowest foot floored to Y~0")
            // absent image -> 33 NaN points.
            t.check(pose.imagePoints.count == 33 && pose.imagePoints[0].x.isNaN, "absent image -> NaN points")
        }

        t.test("MediaPipeFrame floors every landmark regardless of presence") {
            var world = [SIMD3<Float>](repeating: .zero, count: 33)
            let vis = [Float](repeating: 1, count: 33)        // torso guard passes
            var pres = [Float](repeating: 1, count: 33)
            world[BlazePose.Landmark.leftShoulder.rawValue]  = SIMD3(-0.2, -0.5, 0.0)
            world[BlazePose.Landmark.rightShoulder.rawValue] = SIMD3( 0.2, -0.5, 0.0)
            world[BlazePose.Landmark.leftHip.rawValue]  = SIMD3(-0.1, 0.0, 0.0)
            world[BlazePose.Landmark.rightHip.rawValue] = SIMD3( 0.1, 0.0, 0.0)
            world[BlazePose.Landmark.leftAnkle.rawValue]  = SIMD3(-0.1, 0.9, 0.0)
            world[BlazePose.Landmark.rightAnkle.rawValue] = SIMD3( 0.1, 0.9, 0.0)
            // Right hip reads as present (visibility 1) but presence drops to 0 this frame.
            pres[BlazePose.Landmark.rightHip.rawValue] = 0
            let reply = SidecarReply(found: true, world: world, visibility: vis, presence: pres, image: [])
            guard let pose = MediaPipeFrame.toSolverFrame(reply, latch: FloorOriginLatch()) else {
                t.check(false, "toSolverFrame nil"); return
            }
            // floor = lowest foot in solver +Y = -0.9 (ankles at MP y 0.9 -> -0.9).
            // After floor subtraction both hips land at +0.9, regardless of presence.
            let leftHipY = pose[.leftHip].position.y
            let rightHipY = pose[.rightHip].position.y
            t.close(leftHipY, 0.9, tol: 1e-4, "present hip floored to +0.9")
            t.close(rightHipY, 0.9, tol: 1e-4, "presence-0 hip floored by the SAME amount, not left ~1m off")
            t.close(rightHipY, leftHipY, tol: 1e-4, "both hips stay rigid (equal Y) after floor")
        }

        t.test("MediaPipeFrame returns nil without a torso") {
            let world = [SIMD3<Float>](repeating: .zero, count: 33)
            let vis = [Float](repeating: 0, count: 33)  // nothing visible
            let reply = SidecarReply(found: true, world: world, visibility: vis, presence: vis, image: [])
            t.check(MediaPipeFrame.toSolverFrame(reply, latch: FloorOriginLatch()) == nil,
                    "no torso -> nil")
            let notFound = SidecarReply(found: false, world: [], visibility: [], presence: [], image: [])
            t.check(MediaPipeFrame.toSolverFrame(notFound, latch: FloorOriginLatch()) == nil,
                    "not-found -> nil")
        }
    }
}
