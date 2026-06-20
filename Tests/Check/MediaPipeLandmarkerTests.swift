import Foundation
import CoreVideo
import simd
import FeverCore

/// Injectable fake so the landmarker can be tested without the live sidecar.
final class FakeService: PoseInferenceService, @unchecked Sendable {
    let reply: SidecarReply
    init(_ r: SidecarReply) { reply = r }
    func infer(rgb: Data, width: Int, height: Int, tMicros: UInt64) async -> SidecarReply? { reply }
    func reset() {}
}

enum MediaPipeLandmarkerTests {
    static func run(_ t: TestRunner) async {
        var world = [SIMD3<Float>](repeating: .zero, count: 33)
        let vis = [Float](repeating: 1, count: 33)
        world[BlazePose.Landmark.leftShoulder.rawValue]  = SIMD3(-0.2, -0.5, 0)
        world[BlazePose.Landmark.rightShoulder.rawValue] = SIMD3( 0.2, -0.5, 0)
        world[BlazePose.Landmark.leftHip.rawValue]  = SIMD3(-0.1, 0, 0)
        world[BlazePose.Landmark.rightHip.rawValue] = SIMD3( 0.1, 0, 0)
        let reply = SidecarReply(found: true, world: world, visibility: vis, presence: vis,
                                 image: [SIMD2<Float>](repeating: SIMD2(0.5, 0.5), count: 33))
        let lm = MediaPipePoseLandmarker(service: FakeService(reply))

        var pb: CVPixelBuffer?
        CVPixelBufferCreate(nil, 64, 64, kCVPixelFormatType_32BGRA, nil, &pb)

        // Async work happens OUTSIDE the synchronous t.test closures.
        let pose = await lm.detect(pb!, at: 12.5)
        t.test("MediaPipePoseLandmarker builds a PoseResult from the service") {
            guard let pose else { t.check(false, "detect returned nil"); return }
            t.check(pose.timestamp == 12.5, "timestamp stamped from caller")
            t.check(pose[.leftShoulder].position.y > pose[.leftHip].position.y, "y-up after conversion")
            t.check(pose.landmarks.count == 33, "33 landmarks")
        }

        let none = MediaPipePoseLandmarker(
            service: FakeService(SidecarReply(found: false, world: [], visibility: [], presence: [], image: [])))
        let nonePose = await none.detect(pb!, at: 1.0)
        t.test("MediaPipePoseLandmarker returns nil when the service finds nothing") {
            t.check(nonePose == nil, "no body -> nil PoseResult")
        }
    }
}
