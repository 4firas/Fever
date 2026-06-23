import Foundation
import simd
import CoreVideo
import FeverCore

@main
struct FeverCheck {

    static func main() async {
        // Headless benchmark mode: `swift run FeverCheck --bench [N]`.
        // Runs the full solve→encode process chain N times and prints avg
        // microseconds/frame. Non-failing (always exits 0) so it is CI-safe.
        if CommandLine.arguments.contains("--bench") {
            let n = CommandLine.arguments.compactMap { Int($0) }.first ?? 2000
            await runBenchmark(iterations: n)
            exit(0)
        }

        // STUB OSC WIRE: bind a real UDP listener on 127.0.0.1:9000, send the
        // assembled trackers over the real OSCSender, decode the captured wire
        // bytes, and assert the leg-tracker geometry (foot below knee, feet near
        // floor) on the ACTUAL bytes. `swift run FeverCheck --osc-stub`.
        if CommandLine.arguments.contains("--osc-stub") {
            let ok = await OSCWireStub.run()
            exit(ok ? 0 : 1)
        }

        // ROTATION STUB OSC WIRE: stream the assembled trackers with `/rotation`
        // ENABLED through the real OSCSender to `--host`/`--port` (default
        // 127.0.0.1:9000) and verify the rotation contract (rotation present for
        // slots 1-8, no head/rotation, bounded euler, no (0,0,0) positions). Used
        // by the installed-binary stub-wire verification. `swift run FeverCheck
        // --rot-stub [--host H] [--port P]`.
        if CommandLine.arguments.contains("--rot-stub") {
            let ok = await RotationWireStub.runCLI()
            exit(ok ? 0 : 1)
        }

        // LIVE SIDECAR: launch the real Python sidecar and time round-trips,
        // proving the Swift<->Python IPC end to end. `swift run FeverCheck --live-sidecar`.
        if CommandLine.arguments.contains("--live-sidecar") {
            let ok = await LiveSidecarCheck.run()
            exit(ok ? 0 : 1)
        }

        let t = TestRunner()

        testOSCMessage(t)
        testQuaternionEulerRoundTrip(t)
        testCoordinateMapper(t)
        await testJointSolver(t)
        await testFootTrackerAtAnkle(t)
        testOneEuroFilter(t)
        testTrackerAssembler(t)
        testJointPredictor(t)
        testTrackingDefaults(t)
        testMonocularDepthLift(t)
        GeometrySanity.run(t)
        CoordinateModelRegression.run(t)
        HipDynamics.run(t)
        await WireDefenseTests.run(t)
        await FinalizeWireTests.run(t)
        RotationTests.run(t)
        await RotationWireStub.run(t: t)
        SidecarProtocolTests.run(t)
        MathTests.run(t)
        LevelEstimatorTests.run(t)
        BodyStabilizerTests.run(t)
        LeveledBoxTests.run(t)
        WireParityTests.run(t)
        RotationSolverTests.run(t)
        YawStabilizerTests.run(t)
        ConfigPersistenceTests.run(t)
        MediaPipeFrameTests.run(t)
        PoseSidecarPathTests.run(t)
        await MediaPipeLandmarkerTests.run(t)
        FootExaggerationTests.run(t)
        LandmarkConsistencyTests.run(t)

        let summary = t.finalSummary()
        print(summary)
        if t.failed > 0 {
            FileHandle.standardError.write(Data("FAILED: \(t.failed) assertion(s) failed\n".utf8))
            exit(1)
        }
        exit(0)
    }

    // MARK: - 1. OSCMessage.encoded()

    static func testOSCMessage(_ t: TestRunner) {
        t.test("OSCMessage.encoded /position fff") {
            let msg = OSCMessage(address: "/tracking/trackers/1/position",
                                 arguments: [.float(0), .float(1), .float(0)])
            let data = msg.encoded()
            let bytes = [UInt8](data)

            // 4-byte aligned overall.
            t.check(bytes.count % 4 == 0, "packet length not 4-aligned: \(bytes.count)")

            // Address block: NUL-terminated and zero-padded to a 4-byte boundary.
            let addr = Array("/tracking/trackers/1/position".utf8)
            // Address length 29 → padded block must be 32 (next multiple of 4 that
            // leaves room for at least one NUL terminator).
            let addrBlockLen = ((addr.count / 4) + 1) * 4
            t.check(addrBlockLen == 32, "address block length expected 32, got \(addrBlockLen)")
            t.check(Array(bytes[0..<addr.count]) == addr, "address bytes mismatch")
            // Every byte from the address end through the block boundary is NUL.
            var addrPadAllZero = true
            for i in addr.count..<addrBlockLen where bytes[i] != 0 { addrPadAllZero = false }
            t.check(addrPadAllZero, "address padding not all NUL")
            t.check(bytes[addr.count] == 0, "address not NUL-terminated")

            // Type tag block: ",fff" then NUL pad to 4-byte boundary → 8 bytes.
            let tagStart = addrBlockLen
            let expectedTag: [UInt8] = Array(",fff".utf8) + [0, 0, 0, 0] // 4 + 4 pad = 8
            let tagBlock = Array(bytes[tagStart..<(tagStart + 8)])
            t.check(tagBlock == expectedTag, "type tag block mismatch: \(tagBlock)")

            // Three big-endian float32 args.
            let argStart = tagStart + 8
            let a0 = Array(bytes[argStart..<(argStart + 4)])
            let a1 = Array(bytes[(argStart + 4)..<(argStart + 8)])
            let a2 = Array(bytes[(argStart + 8)..<(argStart + 12)])
            t.check(a0 == [0x00, 0x00, 0x00, 0x00], "arg0 (0.0) bytes: \(a0)")
            t.check(a1 == [0x3F, 0x80, 0x00, 0x00], "arg1 (1.0) bytes: \(a1)")
            t.check(a2 == [0x00, 0x00, 0x00, 0x00], "arg2 (0.0) bytes: \(a2)")

            // Total = 32 + 8 + 12 = 52.
            t.check(bytes.count == 52, "total length expected 52, got \(bytes.count)")
        }

        t.test("OSCMessage.encoded int arg") {
            let msg = OSCMessage(address: "/x", arguments: [.int(1)])
            let bytes = [UInt8](msg.encoded())
            t.check(bytes.count % 4 == 0, "int packet not 4-aligned")
            // address "/x" (2) + NUL + pad = 4 ; tag ",i" + NUL + pad = 4 ; int 4.
            // big-endian int32(1) = 0x00000001
            let arg = Array(bytes.suffix(4))
            t.check(arg == [0x00, 0x00, 0x00, 0x01], "int32(1) big-endian: \(arg)")
        }
    }

    // MARK: - 2. quaternionToEulerZXYDegrees round-trip

    /// Build q = Ry(y)*Rx(x)*Rz(z) from Euler degrees (VRChat ZXY composition).
    static func zxyQuat(_ x: Float, _ y: Float, _ z: Float) -> simd_quatf {
        let d = Float.pi / 180
        let qx = simd_quatf(angle: x * d, axis: SIMD3<Float>(1, 0, 0))
        let qy = simd_quatf(angle: y * d, axis: SIMD3<Float>(0, 1, 0))
        let qz = simd_quatf(angle: z * d, axis: SIMD3<Float>(0, 0, 1))
        return qy * qx * qz
    }

    static func testQuaternionEulerRoundTrip(_ t: TestRunner) {
        let cases: [(Float, Float, Float)] = [
            (0, 0, 0),
            (0, 90, 0),     // pure +90 yaw
            (30, 0, 0),     // pure +30 about X (pitch)
            (0, 0, 30),     // pure +30 roll about Z
            (15, -40, 25),  // combo
        ]
        for (x, y, z) in cases {
            t.test("ZXY euler round-trip (\(x),\(y),\(z))") {
                let q = zxyQuat(x, y, z)
                let e = quaternionToEulerZXYDegrees(q)
                t.check(e.x.isFinite && e.y.isFinite && e.z.isFinite,
                        "euler produced NaN/inf: \(e)")
                let qBack = zxyQuat(e.x, e.y, e.z)
                // Compare rotations via |dot| (sign-agnostic), > 0.999.
                let dot = abs(simd_dot(q.vector, qBack.vector))
                t.check(dot > 0.999, "round-trip dot too low: \(dot) for euler \(e)")
            }
        }

        // Gimbal-lock guard: x ≈ +90° (sinX ≈ 1) must not NaN.
        t.test("ZXY euler gimbal lock no NaN") {
            let q = zxyQuat(90, 35, 0)
            let e = quaternionToEulerZXYDegrees(q)
            t.check(e.x.isFinite && e.y.isFinite && e.z.isFinite,
                    "gimbal-lock euler NaN/inf: \(e)")
            // x should be recovered near +90.
            t.close(e.x, 90, tol: 0.5, "gimbal-lock pitch")
            // Reconstruction should still reproduce the rotation.
            let qBack = zxyQuat(e.x, e.y, e.z)
            let dot = abs(simd_dot(q.vector, qBack.vector))
            t.check(dot > 0.999, "gimbal-lock round-trip dot: \(dot)")
        }
    }

    // MARK: - 3. CoordinateMapper

    static func testCoordinateMapper(_ t: TestRunner) {
        t.test("CoordinateMapper position mirror+scale+handedness") {
            let m = CoordinateMapper(userHeightMeters: 1.5,
                                     referenceHeightMeters: 1.8,
                                     mirrorHorizontally: true)
            let out = m.toVRChatPosition(SIMD3<Float>(0.20, 0.90, 0.30))
            t.close(out.x, -0.16667, tol: 1e-3, "pos.x")
            t.close(out.y, 0.75, tol: 1e-3, "pos.y")
            t.close(out.z, -0.25, tol: 1e-3, "pos.z")
        }

        // The CoordinateMapper *constructor* default is UN-mirrored (the mapper is a
        // pure value type; the mirror state is chosen by the caller). The two mirror
        // states must produce X positions of OPPOSITE sign (same magnitude) while
        // Y/Z are identical — this pins the symmetry of the X flip. NOTE: the
        // app-level persisted preference `UserSettings.mirrorTracking` now defaults
        // to TRUE (X negates with Z to complete the right→left handedness flip and
        // match PinoFBT); that is a separate setting from this bare-mapper default.
        t.test("CoordinateMapper default is un-mirrored") {
            let def = CoordinateMapper(userHeightMeters: 1.8)
            t.check(def.mirrorHorizontally == false,
                    "default CoordinateMapper must be un-mirrored")
            t.close(def.mirrorSignX, 1, tol: 1e-6, "default mirrorSignX must be +1")
        }

        t.test("CoordinateMapper mirror states give opposite X sign") {
            let unmir = CoordinateMapper(userHeightMeters: 1.8,
                                         referenceHeightMeters: 1.8,
                                         mirrorHorizontally: false)
            let mir = CoordinateMapper(userHeightMeters: 1.8,
                                       referenceHeightMeters: 1.8,
                                       mirrorHorizontally: true)
            let p = SIMD3<Float>(0.20, 0.90, 0.30)
            let a = unmir.toVRChatPosition(p)
            let b = mir.toVRChatPosition(p)
            // X flips sign, same magnitude; Y and Z are unaffected by the mirror.
            t.check(a.x * b.x < 0, "X must flip sign across mirror states: \(a.x) vs \(b.x)")
            t.close(a.x, -b.x, tol: 1e-6, "X magnitude preserved across mirror")
            t.close(a.y, b.y, tol: 1e-6, "Y unaffected by mirror")
            t.close(a.z, b.z, tol: 1e-6, "Z unaffected by mirror")
            // Concrete expected values (scale 1.0): un-mirrored +x, mirrored -x.
            t.close(a.x, 0.20, tol: 1e-6, "un-mirrored x = +x")
            t.close(b.x, -0.20, tol: 1e-6, "mirrored x = -x")
        }

        t.test("CoordinateMapper euler +90 yaw (mirror)") {
            let m = CoordinateMapper(userHeightMeters: 1.5,
                                     referenceHeightMeters: 1.8,
                                     mirrorHorizontally: true)
            let yaw = simd_quatf(angle: .pi / 2, axis: SIMD3<Float>(0, 1, 0))
            let e = m.toVRChatEulerDegrees(yaw)
            t.close(e.x, 0, tol: 1e-2, "yaw->x")
            t.close(e.y, 90, tol: 1e-2, "yaw->y")
            t.close(e.z, 0, tol: 1e-2, "yaw->z")
        }

        t.test("CoordinateMapper euler +30 roll (no mirror)") {
            let m = CoordinateMapper(userHeightMeters: 1.5,
                                     referenceHeightMeters: 1.8,
                                     mirrorHorizontally: false)
            let roll = simd_quatf(angle: 30 * .pi / 180, axis: SIMD3<Float>(0, 0, 1))
            let e = m.toVRChatEulerDegrees(roll)
            t.close(e.x, 0, tol: 1e-2, "roll->x")
            t.close(e.y, 0, tol: 1e-2, "roll->y")
            t.close(e.z, 30, tol: 1e-2, "roll->z")
        }

        t.test("CoordinateMapper scale fallback ref=0") {
            let m = CoordinateMapper(userHeightMeters: 1.5,
                                     referenceHeightMeters: 0,
                                     mirrorHorizontally: false)
            t.close(m.scale, 1, tol: 1e-6, "scale fallback for ref=0")
            let out = m.toVRChatPosition(SIMD3<Float>(1, 2, 3))
            t.check(out.x.isFinite && out.y.isFinite && out.z.isFinite,
                    "ref=0 produced non-finite position: \(out)")
            // With scale 1, no mirror: x*1, y*1, z*-1.
            t.close(out.x, 1, tol: 1e-6, "ref=0 x")
            t.close(out.z, -3, tol: 1e-6, "ref=0 z")
        }

        t.test("CoordinateMapper scale fallback ref=NaN") {
            let m = CoordinateMapper(userHeightMeters: 1.5,
                                     referenceHeightMeters: .nan,
                                     mirrorHorizontally: false)
            t.close(m.scale, 1, tol: 1e-6, "scale fallback for ref=NaN")
            let out = m.toVRChatPosition(SIMD3<Float>(1, 2, 3))
            t.check(out.x.isFinite && out.y.isFinite && out.z.isFinite,
                    "ref=NaN produced non-finite position: \(out)")
        }
    }

    // MARK: - 4. JointSolver

    static func testJointSolver(_ t: TestRunner) async {
        // Build the synthetic T-pose via the stub landmarker, then solve.
        let stub = StubPoseLandmarker()
        let dummy = makeDummyPixelBuffer()
        guard let pose = await stub.detect(dummy, at: 0) else {
            t.test("JointSolver stub detect") {
                t.check(false, "StubPoseLandmarker.detect returned nil")
            }
            return
        }

        let config = TrackingConfig()
        let solver = JointSolver(settings: config)
        let joints = solver.solve(pose)

        t.test("JointSolver returns 9 joints, all types") {
            t.check(joints.count == 9, "expected 9 joints, got \(joints.count)")
            let types = Set(joints.map { $0.type })
            t.check(types == Set(JointType.allCases),
                    "joint types incomplete: \(types)")
        }

        t.test("JointSolver no NaN positions/rotations") {
            for j in joints {
                let p = j.position
                t.check(p.x.isFinite && p.y.isFinite && p.z.isFinite,
                        "\(j.type) position NaN/inf: \(p)")
                let v = j.rotation.vector
                t.check(v.x.isFinite && v.y.isFinite && v.z.isFinite && v.w.isFinite,
                        "\(j.type) rotation NaN/inf: \(v)")
                // Unit-ish quaternion.
                let len = simd_length(v)
                t.check(len > 0.5, "\(j.type) degenerate quaternion length \(len)")
            }
        }

        t.test("JointSolver hip centered between hips") {
            guard let hip = joints.first(where: { $0.type == .hip }) else {
                t.check(false, "no hip joint"); return
            }
            // Hip x should be ~ midpoint of the two hip landmarks (c-0.10, c+0.10) = c = 0.5,
            // scaled by jointSize (1.0). hipLength default 0 so no spine offset.
            t.close(hip.position.x, 0.5, tol: 1e-2, "hip x centered")
        }

        // Degenerate / collinear limb must not produce NaN — exercise the fallback
        // by building a pose where shoulder, elbow, wrist are exactly collinear.
        t.test("JointSolver collinear limb no NaN (fallback)") {
            var lms = pose.landmarks
            // Make left arm perfectly collinear along +x (straight arm).
            lms[BlazePose.Landmark.leftShoulder.rawValue] =
                NormalizedLandmark(position: SIMD3<Float>(0.30, 0.55, 0), visibility: 0.9)
            lms[BlazePose.Landmark.leftElbow.rawValue] =
                NormalizedLandmark(position: SIMD3<Float>(0.40, 0.55, 0), visibility: 0.9)
            lms[BlazePose.Landmark.leftWrist.rawValue] =
                NormalizedLandmark(position: SIMD3<Float>(0.50, 0.55, 0), visibility: 0.9)
            // Also collapse a foot (heel == toe) to hit the bone-fallback there.
            lms[BlazePose.Landmark.leftHeel.rawValue] =
                NormalizedLandmark(position: SIMD3<Float>(0.10, 0.97, 0), visibility: 0.9)
            lms[BlazePose.Landmark.leftFootIndex.rawValue] =
                NormalizedLandmark(position: SIMD3<Float>(0.10, 0.97, 0), visibility: 0.9)
            let degenPose = PoseResult(landmarks: lms, timestamp: 0)
            let dj = solver.solve(degenPose)
            t.check(dj.count == 9, "degenerate solve count \(dj.count)")
            for j in dj {
                let v = j.rotation.vector
                t.check(v.x.isFinite && v.y.isFinite && v.z.isFinite && v.w.isFinite,
                        "degenerate \(j.type) rotation NaN: \(v)")
                t.check(j.position.x.isFinite && j.position.y.isFinite && j.position.z.isFinite,
                        "degenerate \(j.type) position NaN: \(j.position)")
            }
        }
    }

    // MARK: - 4b. Foot tracker at ankle vs toe

    /// The `footTrackersAtAnkle` toggle (default true) must place the foot/ankle
    /// tracker at the ANKLE landmark, not the synthesized toe (foot-index). Solve
    /// the stub T-pose both ways and check the foot joint positions: ankle y=0.95,
    /// toe y=1.00 in the stub, so the two modes are cleanly distinguishable.
    static func testFootTrackerAtAnkle(_ t: TestRunner) async {
        let stub = StubPoseLandmarker()
        let dummy = makeDummyPixelBuffer()
        guard let pose = await stub.detect(dummy, at: 0) else {
            t.test("FootTrackerAtAnkle stub detect") {
                t.check(false, "StubPoseLandmarker.detect returned nil")
            }
            return
        }

        // Stub landmark anchors (jointSize default 1.0, so position == landmark).
        let heelY: Float = 0.97    // lm[29]/lm[30] — PinoFBT foot point
        let toeY: Float = 1.00     // lm[31]/lm[32]
        let footX: Float = 0.10    // |c ± 0.10|

        t.test("Foot tracker placed at the HEEL when footTrackersAtAnkle true (PinoFBT foot point)") {
            let cfg = TrackingConfig()
            cfg.footTrackersAtAnkle = true
            let joints = JointSolver(settings: cfg).solve(pose)
            guard let lf = joints.first(where: { $0.type == .leftFoot }),
                  let rf = joints.first(where: { $0.type == .rightFoot }) else {
                t.check(false, "missing foot joints"); return
            }
            // Y must be at the heel, NOT the toe.
            t.close(lf.position.y, heelY, tol: 1e-3, "leftFoot y at heel")
            t.close(rf.position.y, heelY, tol: 1e-3, "rightFoot y at heel")
            t.check(abs(lf.position.y - toeY) > 1e-2,
                    "leftFoot must NOT sit at the toe in heel-mode: \(lf.position.y)")
            // X at the heel landmark (left negative, right positive of center).
            t.close(lf.position.x, 0.5 - footX, tol: 1e-3, "leftFoot x at heel")
            t.close(rf.position.x, 0.5 + footX, tol: 1e-3, "rightFoot x at heel")
        }

        t.test("Foot tracker placed at TOE when footTrackersAtAnkle false") {
            let cfg = TrackingConfig()
            cfg.footTrackersAtAnkle = false
            let joints = JointSolver(settings: cfg).solve(pose)
            guard let lf = joints.first(where: { $0.type == .leftFoot }) else {
                t.check(false, "missing left foot joint"); return
            }
            t.close(lf.position.y, toeY, tol: 1e-3, "leftFoot y at toe when heel-mode off")
            t.check(abs(lf.position.y - heelY) > 1e-2,
                    "leftFoot must move off the heel when heel-mode off: \(lf.position.y)")
        }
    }

    // MARK: - 5. OneEuroFilter

    static func testOneEuroFilter(_ t: TestRunner) {
        t.test("OneEuroFilter converges on constant") {
            var f = OneEuroFilter(minCutoff: 1.0, beta: 0.007)
            let value: Float = 3.5
            var out: Float = 0
            var time: TimeInterval = 0
            for _ in 0..<200 {
                out = f.filter(value, at: time)
                time += 1.0 / 60.0
                t.check(out.isFinite, "filter produced non-finite: \(out)")
            }
            t.close(out, value, tol: 1e-3, "constant convergence")
        }

        t.test("OneEuroFilter step moves monotonically toward target") {
            var f = OneEuroFilter(minCutoff: 1.0, beta: 0.007)
            var time: TimeInterval = 0
            // Prime at 0.
            _ = f.filter(0, at: time); time += 1.0 / 60.0
            for _ in 0..<10 { _ = f.filter(0, at: time); time += 1.0 / 60.0 }
            // Apply a step to 10 and verify each output increases toward it.
            var prev: Float = 0
            for i in 0..<60 {
                let out = f.filter(10, at: time); time += 1.0 / 60.0
                t.check(out.isFinite, "step filter non-finite: \(out)")
                t.check(out >= prev - 1e-5,
                        "step output not monotonic at \(i): \(out) < \(prev)")
                t.check(out <= 10 + 1e-3, "step overshoot: \(out)")
                prev = out
            }
            t.check(prev > 5, "step did not approach target, last=\(prev)")
        }
    }

    // MARK: - 6. TrackerAssembler

    static func testTrackerAssembler(_ t: TestRunner) {
        t.test("TrackerAssembler default slot map + enabled set") {
            let enabled: Set<JointType> = [.hip, .leftFoot, .rightFoot]
            let asm = TrackerAssembler(enabled: enabled,
                                       slotMap: TrackerAssembler.defaultSlotMap)
            let mapper = CoordinateMapper(userHeightMeters: 1.8,
                                          referenceHeightMeters: 1.8,
                                          mirrorHorizontally: true)
            let q = simd_quatf(angle: 0, axis: SIMD3<Float>(0, 1, 0))
            // Build one joint of every type so we can check filtering + head.
            let joints: [VRJoint] = JointType.allCases.map { type in
                VRJoint(type: type,
                        position: SIMD3<Float>(0.1, 0.2, 0.3),
                        rotation: q,
                        confidence: 1)
            }
            let (body, head) = asm.assemble(joints, mapper: mapper)

            // Body must contain exactly the 3 enabled numbered joints.
            t.check(body.count == 3, "body count expected 3, got \(body.count)")
            let bySlot = Dictionary(uniqueKeysWithValues: body.map { ($0.slot, $0) })
            t.check(bySlot["1"] != nil, "slot 1 (hip) missing")
            t.check(bySlot["2"] != nil, "slot 2 (leftFoot) missing")
            t.check(bySlot["3"] != nil, "slot 3 (rightFoot) missing")

            // Disabled joints (chest=4, knees, elbows) must be excluded.
            let slots = Set(body.map { $0.slot })
            t.check(!slots.contains("4"), "chest should be excluded")
            t.check(!slots.contains("5") && !slots.contains("6"), "knees should be excluded")
            t.check(!slots.contains("7") && !slots.contains("8"), "elbows should be excluded")

            // Head reference produced independently of the numbered enabled set.
            t.check(head != nil, "head reference not produced")
            t.check(head?.slot == "head", "head slot incorrect: \(String(describing: head?.slot))")
        }

        t.test("TrackerAssembler head produced even when not in enabled") {
            // enabled does NOT contain .head; head must still be returned.
            let asm = TrackerAssembler(enabled: [.hip],
                                       slotMap: TrackerAssembler.defaultSlotMap)
            let mapper = CoordinateMapper(userHeightMeters: 1.8)
            let q = simd_quatf(angle: 0, axis: SIMD3<Float>(0, 1, 0))
            let joints = [
                VRJoint(type: .head, position: .zero, rotation: q),
                VRJoint(type: .hip, position: .zero, rotation: q),
            ]
            let (body, head) = asm.assemble(joints, mapper: mapper)
            t.check(head != nil, "head must be produced regardless of enabled set")
            t.check(body.count == 1 && body.first?.slot == "1", "only hip body tracker expected")
        }
    }

    // MARK: - 7. JointPredictor (predictive gap-fill)

    /// Builds a single-landmark `PoseResult` (count 1) at index 0; remaining
    /// helper for deterministic predictor tests.
    static func mkPose(_ x: Float, _ y: Float, _ z: Float,
                       present: Bool, t: Double) -> PoseResult {
        let vis: Float = present ? 0.9 : 0
        let pos = present ? SIMD3<Float>(x, y, z) : SIMD3<Float>(.nan, .nan, .nan)
        let lm = NormalizedLandmark(position: pos, visibility: vis, presence: vis)
        let img = present ? SIMD2<Float>(x, y) : SIMD2<Float>(.nan, .nan)
        return PoseResult(landmarks: [lm], timestamp: t, imagePoints: [img])
    }

    static func testJointPredictor(_ t: TestRunner) {
        let dt = 1.0 / 30.0

        t.test("JointPredictor passes present landmarks through unchanged") {
            let p = JointPredictor(count: 1)
            var time = 0.0
            for i in 0..<5 {
                let x = 0.2 + Float(i) * 0.05
                let out = p.predict(mkPose(x, 0.5, 0.1, present: true, t: time))
                t.close(out.landmarks[0].position.x, x, tol: 1e-6, "present x passthrough")
                t.close(out.landmarks[0].position.y, 0.5, tol: 1e-6, "present y passthrough")
                t.close(out.imagePoints[0].x, x, tol: 1e-6, "present image x passthrough")
                t.check(out.landmarks[0].presence > 0.5, "present keeps confidence")
                time += dt
            }
        }

        t.test("JointPredictor synthesizes a brief gap with decaying motion") {
            let p = JointPredictor(count: 1, maxHoldSeconds: 0.3,
                                   velocityEMA: 1.0, velocityDecay: 0.85,
                                   reappearBlendFrames: 0)
            var time = 0.0
            // Move steadily +0.02/frame in x so velocity ≈ +0.6/s.
            var x: Float = 0.2
            for _ in 0..<6 {
                _ = p.predict(mkPose(x, 0.5, 0.0, present: true, t: time))
                x += 0.02; time += dt
            }
            let lastGoodX = x - 0.02
            // First missing frame: should predict forward (x increases).
            let m1 = p.predict(mkPose(0, 0, 0, present: false, t: time))
            time += dt
            t.check(m1.landmarks[0].position.x > lastGoodX,
                    "held frame should extrapolate forward: \(m1.landmarks[0].position.x)")
            t.check(m1.landmarks[0].presence > 0 && m1.landmarks[0].presence < 1,
                    "held frame has decaying confidence: \(m1.landmarks[0].presence)")
            t.check(m1.imagePoints[0].x.isFinite, "held image point finite")
            // Second missing frame: velocity decayed, so the step is SMALLER than
            // the first (eases to a stop, no fly-off).
            let prevX = m1.landmarks[0].position.x
            let m2 = p.predict(mkPose(0, 0, 0, present: false, t: time))
            time += dt
            let step1 = prevX - lastGoodX
            let step2 = m2.landmarks[0].position.x - prevX
            t.check(step2 < step1 + 1e-6, "second held step must not exceed first (decay): \(step2) vs \(step1)")
            t.check(step2 > 0, "still moving forward but slower: \(step2)")
        }

        t.test("JointPredictor drops landmark after the hold window") {
            let p = JointPredictor(count: 1, maxHoldSeconds: 0.1)  // 3 frames @30
            var time = 0.0
            _ = p.predict(mkPose(0.5, 0.5, 0.0, present: true, t: time)); time += dt
            _ = p.predict(mkPose(0.52, 0.5, 0.0, present: true, t: time)); time += dt
            // Hold for several missing frames, well past 0.1 s.
            var lastOut: PoseResult!
            for _ in 0..<8 {
                lastOut = p.predict(mkPose(0, 0, 0, present: false, t: time)); time += dt
            }
            t.check(lastOut.landmarks[0].presence == 0,
                    "beyond hold window presence must be 0: \(lastOut.landmarks[0].presence)")
            t.check(lastOut.imagePoints[0].x.isNaN,
                    "beyond hold window image point must be NaN")
        }

        t.test("JointPredictor blends on reappearance (no snap)") {
            let p = JointPredictor(count: 1, maxHoldSeconds: 0.3,
                                   velocityEMA: 1.0, velocityDecay: 0.85,
                                   reappearBlendFrames: 3)
            var time = 0.0
            var x: Float = 0.2
            for _ in 0..<6 { _ = p.predict(mkPose(x, 0.5, 0.0, present: true, t: time)); x += 0.02; time += dt }
            // One missing frame.
            let held = p.predict(mkPose(0, 0, 0, present: false, t: time)); time += dt
            let heldX = held.landmarks[0].position.x
            // Reappear FAR from the prediction (a would-be snap).
            let realX: Float = 0.9
            let blended = p.predict(mkPose(realX, 0.5, 0.0, present: true, t: time))
            let bx = blended.landmarks[0].position.x
            t.check(bx > heldX && bx < realX,
                    "blended output between prediction and real, not snapped: held=\(heldX) blend=\(bx) real=\(realX)")
        }

        t.test("JointPredictor clamps image points to [0,1]") {
            let p = JointPredictor(count: 1, velocityEMA: 1.0, velocityDecay: 1.0,
                                   reappearBlendFrames: 0)
            var time = 0.0
            // Drive image x toward the right edge fast so extrapolation overshoots 1.
            _ = p.predict(mkPose(0.90, 0.5, 0.0, present: true, t: time)); time += dt
            _ = p.predict(mkPose(0.98, 0.5, 0.0, present: true, t: time)); time += dt
            let m = p.predict(mkPose(0, 0, 0, present: false, t: time))
            t.check(m.imagePoints[0].x <= 1.0 + 1e-6 && m.imagePoints[0].x >= 0,
                    "image x clamped to [0,1]: \(m.imagePoints[0].x)")
        }

        t.test("JointPredictor never-seen landmark stays absent") {
            let p = JointPredictor(count: 1)
            let out = p.predict(mkPose(0, 0, 0, present: false, t: 0))
            t.check(out.landmarks[0].presence == 0, "unseen landmark presence 0")
            t.check(out.imagePoints[0].x.isNaN, "unseen landmark image NaN")
        }

        t.test("JointPredictor is deterministic across runs") {
            func run() -> [Float] {
                let p = JointPredictor(count: 1)
                var time = 0.0; var x: Float = 0.3; var xs: [Float] = []
                for i in 0..<10 {
                    let present = !(i == 5 || i == 6)  // two-frame gap
                    let out = p.predict(mkPose(x, 0.5, 0.0, present: present, t: time))
                    xs.append(out.landmarks[0].position.x)
                    if present { x += 0.03 }
                    time += dt
                }
                return xs
            }
            let a = run(); let b = run()
            t.check(a == b, "predictor not deterministic: \(a) vs \(b)")
        }
    }

    // MARK: - Tracking defaults (height + tracker set + slot map)

    /// The OSC tracker-framing rework locked these defaults: user height 1.74 m,
    /// all 8 numbered body trackers enabled, and a slot map that covers all 8.
    /// (Head is the always-on position-only reference, not a numbered slot.)
    static func testTrackingDefaults(_ t: TestRunner) {
        t.test("Default user height is 1.74 m") {
            // Clear any persisted override so we read the code default.
            UserDefaults.standard.removeObject(forKey: "userHeightMeters")
            UserDefaults.standard.removeObject(forKey: "enabledJoints")
            let cfg = TrackingConfig()
            t.close(Float(cfg.userHeightMeters), 1.74, tol: 1e-4,
                    "default userHeightMeters must be 1.74: \(cfg.userHeightMeters)")
        }

        t.test("Default enabled joints are all 8 body trackers") {
            UserDefaults.standard.removeObject(forKey: "enabledJoints")
            let cfg = TrackingConfig()
            let expected: Set<JointType> = [.hip, .chest,
                                            .leftElbow, .rightElbow,
                                            .leftKnee, .rightKnee,
                                            .leftFoot, .rightFoot]
            t.check(cfg.enabledJoints == expected,
                    "default enabledJoints must be all 8: \(cfg.enabledJoints)")
            t.check(!cfg.enabledJoints.contains(.head),
                    "head must NOT be a numbered enabled tracker")
        }

        t.test("oscPort didSet clamps to the valid UDP range") {
            let cfg = TrackingConfig()
            cfg.oscPort = 70000
            t.check(cfg.oscPort == 65535,
                    "oscPort above 65535 must clamp to 65535: \(cfg.oscPort)")
            cfg.oscPort = 0
            t.check(cfg.oscPort == 1,
                    "oscPort below 1 must clamp to 1: \(cfg.oscPort)")
        }

        t.test("Stale sub-1.0 coefficients clamp up to the tasteful defaults on load") {
            // A persisted value below 1.0 is a stale rotation-era setting (old
            // -1...1 range / 0.0 default) that would collapse the hip/feet onto
            // their neutral. Loading must clamp it back up to the position-gain
            // default rather than honoring the inert value.
            UserDefaults.standard.set(0.0, forKey: "hipExaggerateCoefficient")
            UserDefaults.standard.set(0.0, forKey: "hipTwistCoefficient")
            UserDefaults.standard.set(0.0, forKey: "stepStrideCoefficient")
            UserDefaults.standard.set(0.0, forKey: "stepLiftCoefficient")
            let cfg = TrackingConfig()
            t.close(Float(cfg.hipExaggerateCoefficient), 2.0, tol: 1e-4,
                    "stale hipExaggerateCoefficient must clamp to 2.0: \(cfg.hipExaggerateCoefficient)")
            t.close(Float(cfg.hipTwistCoefficient), 1.4, tol: 1e-4,
                    "stale hipTwistCoefficient must clamp to 1.4: \(cfg.hipTwistCoefficient)")
            t.close(Float(cfg.stepStrideCoefficient), 1.6, tol: 1e-4,
                    "stale stepStrideCoefficient must clamp to 1.6: \(cfg.stepStrideCoefficient)")
            t.close(Float(cfg.stepLiftCoefficient), 1.3, tol: 1e-4,
                    "stale stepLiftCoefficient must clamp to 1.3: \(cfg.stepLiftCoefficient)")
            UserDefaults.standard.removeObject(forKey: "hipExaggerateCoefficient")
            UserDefaults.standard.removeObject(forKey: "hipTwistCoefficient")
            UserDefaults.standard.removeObject(forKey: "stepStrideCoefficient")
            UserDefaults.standard.removeObject(forKey: "stepLiftCoefficient")
        }

        t.test("Slot map covers all 8 body trackers, head excluded") {
            let map = TrackerAssembler.defaultSlotMap
            let body: [JointType] = [.hip, .chest,
                                     .leftElbow, .rightElbow,
                                     .leftKnee, .rightKnee,
                                     .leftFoot, .rightFoot]
            for j in body {
                t.check(map[j] != nil, "slotMap must map \(j)")
            }
            t.check(map[.head] == nil, "slotMap must NOT contain head (it is the reference)")
            // All 8 slots are distinct "1".."8".
            let slots = Set(body.compactMap { map[$0] })
            t.check(slots.count == 8, "8 distinct slots expected: \(slots)")
        }
    }

    // MARK: - Monocular depth lift (stable scale + foreshortening depth)

    /// The OSC-path depth lift must (1) hold a FIXED scale derived from the
    /// least-foreshortened vertical body extent (NOT from on-screen widths, the
    /// old foreshortening-pulse cause), so lateral rotation / width jitter does
    /// NOT change scale, and (2) synthesize a real per-joint depth so a bent leg
    /// has the knee distinctly FORWARD of the hip→ankle line (un-collapses IK).
    static func testMonocularDepthLift(_ t: TestRunner) {
        let ref: Float = 1.8

        // A standing-ish constellation in aspect-corrected normalized units.
        // Origin lower-left, +Y up. `widthJitter` perturbs the SHOULDER WIDTH (a
        // lateral, foreshortening-prone measurement the OLD scale used) WITHOUT
        // changing the vertical head→ankle extent — exactly the rotation case
        // that used to pulse the skeleton and must now leave the fixed scale put.
        func makeXY(widthJitter: Float) -> ([SIMD2<Float>], [Bool]) {
            var xy = [SIMD2<Float>](repeating: .zero, count: 33)
            var present = [Bool](repeating: false, count: 33)
            func set(_ l: BlazePose.Landmark, _ x: Float, _ y: Float) {
                xy[l.rawValue] = SIMD2<Float>(x, y); present[l.rawValue] = true
            }
            let c: Float = 0.5
            // Head/face points give the extent estimator its TOP (least
            // foreshortened upright body span: head → ankle).
            set(.nose, c, 0.92)
            set(.leftEye, c - 0.03, 0.93); set(.rightEye, c + 0.03, 0.93)
            set(.leftEar, c - 0.05, 0.92); set(.rightEar, c + 0.05, 0.92)
            set(.leftShoulder, c - 0.18 - widthJitter, 0.55)
            set(.rightShoulder, c + 0.18 + widthJitter, 0.55)
            set(.leftHip, c - 0.10, 0.40)
            set(.rightHip, c + 0.10, 0.40)
            set(.leftKnee, c - 0.10, 0.25)
            set(.rightKnee, c + 0.10, 0.25)
            set(.leftAnkle, c - 0.10, 0.10)
            set(.rightAnkle, c + 0.10, 0.10)
            return (xy, present)
        }

        t.test("Fixed scale is invariant to lateral width jitter (no pulsing)") {
            let lift = MonocularDepthLift(referenceHeight: ref)
            // Seed many clean frames so the latch converges.
            var (xy, present) = makeXY(widthJitter: 0)
            var s0: Float = 0
            for _ in 0..<200 { s0 = lift.stableScale(xy: xy, present: present) ?? s0 }
            // Now collapse the shoulder width by 0.20 (a strong torso-yaw
            // foreshortening) — the OLD width-based scale would have jumped; the
            // new vertical-extent scale must NOT move (width is irrelevant to it).
            (xy, present) = makeXY(widthJitter: -0.20)
            let s1 = lift.stableScale(xy: xy, present: present) ?? s0
            let rel = abs(s1 - s0) / s0
            t.check(rel < 0.01, "lateral width change must not move fixed scale: \(rel)")
            t.check(s0 > 0 && s0.isFinite, "scale finite/positive: \(s0)")
        }

        t.test("Bent leg: knee resolves forward of the hip→ankle line in Z") {
            let lift = MonocularDepthLift(referenceHeight: ref)
            // Build a metric XY where the knee is foreshortened: thigh+shank
            // project SHORTER than their true length because the knee swings
            // toward the camera. Place hip/knee/ankle nearly vertical in XY so
            // the 2D projected bone length is well below the true bone length.
            var metricXY = [SIMD2<Float>](repeating: .zero, count: 33)
            var present = [Bool](repeating: false, count: 33)
            func set(_ l: BlazePose.Landmark, _ x: Float, _ y: Float) {
                metricXY[l.rawValue] = SIMD2<Float>(x, y); present[l.rawValue] = true
            }
            // Need shoulders+hips present for the chain roots.
            set(.leftShoulder, -0.2, 0.5); set(.rightShoulder, 0.2, 0.5)
            set(.leftHip, -0.1, 0.0); set(.rightHip, 0.1, 0.0)
            // Left leg foreshortened: hip(0,0) → knee(~0,-0.15) → ankle(~0,-0.30).
            // Thigh true ≈ 1.8*0.245 = 0.441 m; projected ≈ 0.15 m → big dz.
            set(.leftKnee, -0.1, -0.15)
            set(.leftAnkle, -0.1, -0.30)
            // Right leg straight in-plane (control): full-length projection.
            set(.rightKnee, 0.1, -0.44)
            set(.rightAnkle, 0.1, -0.88)

            let z = lift.depths(metricXY: metricXY, present: present)
            let zHipL = z[BlazePose.Landmark.leftHip.rawValue]
            let zKneeL = z[BlazePose.Landmark.leftKnee.rawValue]
            let zAnkleL = z[BlazePose.Landmark.leftAnkle.rawValue]
            // Hip is the root (≈0). Knee must be clearly forward (+Z) of it, and
            // forward of the ankle — i.e. NOT coplanar (the collapse condition).
            t.check(zKneeL > zHipL + 0.05,
                    "bent knee must be forward of hip in Z: knee=\(zKneeL) hip=\(zHipL)")
            t.check(zKneeL > zAnkleL + 0.02,
                    "bent knee must be forward of ankle in Z: knee=\(zKneeL) ankle=\(zAnkleL)")
            t.check(zKneeL.isFinite && zAnkleL.isFinite, "leg depths finite")
        }

        t.test("Depth sign is temporally stable (no flip across frames)") {
            let lift = MonocularDepthLift(referenceHeight: ref)
            var metricXY = [SIMD2<Float>](repeating: .zero, count: 33)
            var present = [Bool](repeating: false, count: 33)
            func set(_ l: BlazePose.Landmark, _ x: Float, _ y: Float) {
                metricXY[l.rawValue] = SIMD2<Float>(x, y); present[l.rawValue] = true
            }
            set(.leftShoulder, -0.2, 0.5); set(.rightShoulder, 0.2, 0.5)
            set(.leftHip, -0.1, 0.0); set(.rightHip, 0.1, 0.0)
            set(.leftKnee, -0.1, -0.15); set(.leftAnkle, -0.1, -0.30)
            var prevSign: Float = 0
            for i in 0..<30 {
                let z = lift.depths(metricXY: metricXY, present: present)
                let zKnee = z[BlazePose.Landmark.leftKnee.rawValue]
                let zHip = z[BlazePose.Landmark.leftHip.rawValue]
                let sign: Float = (zKnee - zHip) >= 0 ? 1 : -1
                if i > 0 {
                    t.check(sign == prevSign,
                            "knee depth sign must not flip frame \(i): \(sign) vs \(prevSign)")
                }
                prevSign = sign
            }
        }
    }

    // MARK: - Benchmark (headless, non-failing)

    /// Runs the full per-frame process chain N times and reports avg µs/frame.
    /// Chain: StubPoseLandmarker.detect → JointPredictor (gap-fill) →
    /// LandmarkStabilizer (One-Euro 33×3) → JointSolver → QuaternionStabilizer
    /// (per-joint SLERP) → CoordinateMapper → TrackerAssembler → OSC encode
    /// (position + rotation per tracker). Vision I/O is intentionally excluded —
    /// this measures the CPU solve/encode budget that must fit comfortably inside
    /// the 33 ms (30 fps) frame budget. The JointPredictor is in the chain so the
    /// reported cost includes the predictive gap-fill work.
    static func runBenchmark(iterations n: Int) async {
        let config = TrackingConfig()
        let stub = StubPoseLandmarker()
        let dummy = makeDummyPixelBuffer()

        // A representative synthetic pose (the stub T-pose), captured once.
        guard let basePose = await stub.detect(dummy, at: 0) else {
            print("BENCH: stub detect failed"); return
        }

        let predictor = JointPredictor(count: basePose.landmarks.count)
        let stabilizer = LandmarkStabilizer(minCutoff: config.stabilizerMinCutoffF,
                                            beta: config.stabilizerBetaF)
        let quat = QuaternionStabilizer(smoothingFactor: config.rotationSmoothingF)
        let solver = JointSolver(settings: config)
        let mapper = CoordinateMapper(userHeightMeters: config.userHeightMetersF,
                                      referenceHeightMeters: 1.8,
                                      mirrorHorizontally: true)
        let assembler = TrackerAssembler(enabled: config.enabledJoints,
                                         slotMap: config.slotMap)

        var sink = 0   // consume output so the chain isn't optimized away.

        // Warm up (filters/stabilizer prime; JIT/branch-predict settle).
        for i in 0..<200 {
            let t = Double(i) / 30.0
            let pose = PoseResult(landmarks: basePose.landmarks, timestamp: t)
            let predicted = predictor.predict(pose)
            let stab = stabilizer.stabilize(predicted)
            var joints = solver.solve(stab)
            for k in joints.indices { joints[k] = quat.stabilize(joints[k]) }
            let asm = assembler.assemble(joints, mapper: mapper)
            for tr in asm.body {
                sink &+= encodeTracker(tr)
            }
            if let h = asm.head { sink &+= encodeTracker(h) }
        }

        let start = DispatchTime.now().uptimeNanoseconds
        for i in 0..<n {
            let t = Double(i + 1000) / 30.0
            let pose = PoseResult(landmarks: basePose.landmarks, timestamp: t)
            let predicted = predictor.predict(pose)
            let stab = stabilizer.stabilize(predicted)
            var joints = solver.solve(stab)
            for k in joints.indices { joints[k] = quat.stabilize(joints[k]) }
            let asm = assembler.assemble(joints, mapper: mapper)
            for tr in asm.body {
                sink &+= encodeTracker(tr)
            }
            if let h = asm.head { sink &+= encodeTracker(h) }
        }
        let end = DispatchTime.now().uptimeNanoseconds

        let totalNs = Double(end - start)
        let perFrameUs = (totalNs / Double(n)) / 1000.0
        let budgetPct = (perFrameUs / 1000.0) / 33.0 * 100.0

        print("BENCH: \(n) iterations, full solve→encode chain (Vision I/O excluded)")
        print(String(format: "BENCH: avg %.3f µs/frame  (%.4f%% of the 33 ms 30fps budget)",
                     perFrameUs, budgetPct))
        print("BENCH: sink=\(sink)  (anti-DCE checksum)")
    }

    /// Encode one tracker's /position + /rotation OSC messages, returning the
    /// total encoded byte count (consumed as an anti-dead-code sink).
    static func encodeTracker(_ t: OSCTracker) -> Int {
        let pos = OSCMessage(address: "/tracking/trackers/\(t.slot)/position",
                             arguments: [.float(t.position.x),
                                         .float(t.position.y),
                                         .float(t.position.z)]).encoded()
        let rot = OSCMessage(address: "/tracking/trackers/\(t.slot)/rotation",
                             arguments: [.float(t.eulerDegrees.x),
                                         .float(t.eulerDegrees.y),
                                         .float(t.eulerDegrees.z)]).encoded()
        return pos.count + rot.count
    }

    // MARK: - helpers

    static func makeDummyPixelBuffer() -> CVPixelBuffer {
        var pb: CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault, 16, 16,
                            kCVPixelFormatType_32BGRA, nil, &pb)
        return pb!
    }
}
