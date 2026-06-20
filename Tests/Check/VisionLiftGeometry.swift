import simd
import Foundation
import FeverCore

/// Pure 2D->metric-skeleton geometry, extracted from the retired
/// `VisionPoseLandmarker` so the OSC wire tests keep a synthetic-skeleton
/// generator. Contains NO Apple Vision dependency — it is anthropometric/simd
/// math (`MonocularDepthLift` + BlazePose). Test-target scaffolding only.
enum VisionLiftGeometry {

    /// Reads a present landmark from the scratch arrays, or nil if absent.
    @inline(__always)
    static func value(_ raw: [SIMD2<Float>], _ present: [Bool],
                      _ l: BlazePose.Landmark) -> SIMD2<Float>? {
        present[l.rawValue] ? raw[l.rawValue] : nil
    }

    /// Hip-root origin = midpoint of the hips, or the available torso center.
    static func rootOrigin(_ raw: [SIMD2<Float>], present: [Bool]) -> SIMD2<Float> {
        let lHip = value(raw, present, .leftHip), rHip = value(raw, present, .rightHip)
        if let l = lHip, let r = rHip { return (l + r) * 0.5 }
        if let l = lHip { return l }
        if let r = rHip { return r }
        if let ls = value(raw, present, .leftShoulder),
           let rs = value(raw, present, .rightShoulder) { return (ls + rs) * 0.5 }
        var sum = SIMD2<Float>.zero
        var count = 0
        for i in 0..<present.count where present[i] { sum += raw[i]; count += 1 }
        guard count > 0 else { return .zero }
        return sum / Float(count)
    }

    /// Vertical body span (normalized height-units) from the highest available
    /// head point down to the lowest available foot point, for metric scaling.
    static func bodySpan(_ raw: [SIMD2<Float>], present: [Bool]) -> Float {
        @inline(__always) func y(_ l: BlazePose.Landmark) -> Float? { value(raw, present, l)?.y }
        let top = [y(.nose), y(.leftEye), y(.rightEye), y(.leftEar), y(.rightEar)]
            .compactMap { $0 }.max()
        let bottom = [y(.leftAnkle), y(.rightAnkle), y(.leftKnee), y(.rightKnee)]
            .compactMap { $0 }.min()
        if let t = top, let b = bottom, t > b { return t - b }
        let sh = [y(.leftShoulder), y(.rightShoulder)].compactMap { $0 }.max()
        let hp = [y(.leftHip), y(.rightHip)].compactMap { $0 }.min()
        if let s = sh, let h = hp, s > h { return (s - h) * 3.0 }
        return 0
    }

    /// Builds the 33-slot BlazePose `PoseResult` from lifted joints in the stable
    /// camera/world frame: hip-relative metric XY (·k) + foreshortening Z, fixed-
    /// length retarget, hip-world translation with a latched XZ origin, heel/toe
    /// synthesis, and a latched floor at Y≈0. Returns nil if the torso is absent.
    static func assemble(raw: [SIMD2<Float>],
                         present: [Bool],
                         root: SIMD2<Float>,
                         k: Float,
                         depthLift: MonocularDepthLift,
                         imagePoints: [SIMD2<Float>],
                         time: TimeInterval) -> PoseResult? {
        var metricXY = [SIMD2<Float>](repeating: .zero, count: 33)
        for i in 0..<33 where present[i] {
            metricXY[i] = (raw[i] - root) * k
        }
        let z = depthLift.depths(metricXY: metricXY, present: present)

        var rel = [SIMD3<Float>](repeating: .zero, count: 33)
        for i in 0..<33 where present[i] {
            rel[i] = SIMD3<Float>(metricXY[i].x, metricXY[i].y, z[i])
        }

        rel = depthLift.retarget(rel, present: present)

        let hipWorld = SIMD3<Float>(root.x * k, root.y * k, 0)
        let origin = depthLift.latchOriginXZ(SIMD2<Float>(hipWorld.x, hipWorld.z))
        let centeredWorld = SIMD3<Float>(hipWorld.x - origin.x, hipWorld.y, hipWorld.z - origin.y)

        var lm = [NormalizedLandmark](repeating: NormalizedLandmark(position: .zero,
                                                                    visibility: 0, presence: 0),
                                      count: 33)
        for i in 0..<33 where present[i] {
            lm[i] = NormalizedLandmark(position: rel[i] + centeredWorld, visibility: 1, presence: 1)
        }

        let haveShoulders = lm[.leftShoulder].presence > 0 && lm[.rightShoulder].presence > 0
        let haveHip = lm[.leftHip].presence > 0 || lm[.rightHip].presence > 0
        guard haveShoulders, haveHip else { return nil }

        for (ankle, heel) in [(BlazePose.Landmark.leftAnkle, BlazePose.Landmark.leftHeel),
                              (BlazePose.Landmark.rightAnkle, BlazePose.Landmark.rightHeel)] {
            guard lm[ankle].presence > 0 else { continue }
            let a = lm[ankle].position
            lm[heel] = NormalizedLandmark(position: SIMD3<Float>(a.x, a.y - 0.06, a.z - 0.06),
                                          visibility: 1, presence: 1)
        }
        for (ankle, toe) in [(BlazePose.Landmark.leftAnkle, BlazePose.Landmark.leftFootIndex),
                             (BlazePose.Landmark.rightAnkle, BlazePose.Landmark.rightFootIndex)] {
            guard lm[ankle].presence > 0 else { continue }
            let a = lm[ankle].position
            lm[toe] = NormalizedLandmark(position: SIMD3<Float>(a.x, a.y - 0.08, a.z + 0.12),
                                         visibility: 1, presence: 1)
        }

        let footSlots: [BlazePose.Landmark] = [.leftAnkle, .rightAnkle,
                                               .leftHeel, .rightHeel,
                                               .leftFootIndex, .rightFootIndex]
        var lowestFoot: Float? = nil
        for s in footSlots where lm[s].presence > 0 {
            let y = lm[s].position.y
            lowestFoot = lowestFoot.map { Swift.min($0, y) } ?? y
        }
        if let lf = lowestFoot {
            let floor = depthLift.latchFloor(lf)
            for i in 0..<33 where lm[i].presence > 0 {
                lm[i].position.y -= floor
            }
        }

        return PoseResult(landmarks: lm, timestamp: time, imagePoints: imagePoints)
    }
}
