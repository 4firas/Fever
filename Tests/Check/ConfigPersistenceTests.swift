import Foundation
import FeverCore

/// Pins that editing OSC settings PERSISTS to UserDefaults — the bug behind
/// "I set the Quest IP in the app but it kept sending to 127.0.0.1". The Settings
/// TextField binds `$config.oscHost`; setting it MUST survive into a fresh config
/// (i.e. the next launch / pipeline rebuild reads the new value).
enum ConfigPersistenceTests {
    static func run(_ t: TestRunner) {
        t.test("oscHost set on the config persists across a reload") {
            UserDefaults.standard.removeObject(forKey: "oscHost")
            let cfg = TrackingConfig()
            cfg.oscHost = "192.168.100.215"
            let reloaded = TrackingConfig()
            t.check(reloaded.oscHost == "192.168.100.215",
                    "oscHost must persist to UserDefaults; reloaded = \(reloaded.oscHost)")
            UserDefaults.standard.removeObject(forKey: "oscHost")
        }

        t.test("oscPort set on the config persists across a reload") {
            UserDefaults.standard.removeObject(forKey: "oscPort")
            let cfg = TrackingConfig()
            cfg.oscPort = 9001
            let reloaded = TrackingConfig()
            t.check(reloaded.oscPort == 9001,
                    "oscPort must persist to UserDefaults; reloaded = \(reloaded.oscPort)")
            UserDefaults.standard.removeObject(forKey: "oscPort")
        }
    }
}
