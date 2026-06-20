import Foundation
import FeverCore

enum PoseSidecarPathTests {
    static func run(_ t: TestRunner) {
        t.test("SidecarPaths resolves the dev layout") {
            let fm = FileManager.default
            let tmp = NSTemporaryDirectory() + "fever-paths-\(UInt32.random(in: 0..<999_999))"
            try? fm.createDirectory(atPath: tmp + "/Sidecar/.venv/bin", withIntermediateDirectories: true)
            fm.createFile(atPath: tmp + "/Sidecar/.venv/bin/python3", contents: Data())
            fm.createFile(atPath: tmp + "/Sidecar/pose_server.py", contents: Data())
            try? fm.createDirectory(atPath: tmp + "/Models", withIntermediateDirectories: true)
            fm.createFile(atPath: tmp + "/Models/pose_landmarker_full.task", contents: Data())

            guard let p = SidecarPaths.resolve(bundle: Bundle.main, projectRoot: tmp) else {
                t.check(false, "resolve returned nil for a valid dev layout"); return
            }
            t.check(p.python.hasSuffix("/Sidecar/.venv/bin/python3"), "python path")
            t.check(p.script.hasSuffix("/Sidecar/pose_server.py"), "script path")
            t.check(p.model.hasSuffix("pose_landmarker_full.task"), "model path")
            try? fm.removeItem(atPath: tmp)
        }

        t.test("SidecarPaths returns nil when the model is missing") {
            let fm = FileManager.default
            let tmp = NSTemporaryDirectory() + "fever-paths-\(UInt32.random(in: 0..<999_999))"
            try? fm.createDirectory(atPath: tmp + "/Sidecar/.venv/bin", withIntermediateDirectories: true)
            fm.createFile(atPath: tmp + "/Sidecar/.venv/bin/python3", contents: Data())
            fm.createFile(atPath: tmp + "/Sidecar/pose_server.py", contents: Data())
            // No Models/ -> unresolved.
            t.check(SidecarPaths.resolve(bundle: Bundle.main, projectRoot: tmp) == nil,
                    "missing model -> nil")
            try? fm.removeItem(atPath: tmp)
        }
    }
}
