import Foundation
import FeverCore

/// Guards the wire slot map (`TrackerMapPino`) against silently diverging from the
/// `JointType` ↔ slot mapping the UI and solver use — a divergence would mislabel
/// trackers or send a joint to the wrong VRChat slot with no test catching it.
enum SlotMapTests {
    static func run(_ t: TestRunner) {
        t.test("SLOT-MAP parity: TrackerMapPino slots round-trip through JointType") {
            t.check(TrackerMapPino.slots.count == 8, "8 numbered body slots")
            t.check(Set(TrackerMapPino.slots.map(\.path)) == Set(["1", "2", "3", "4", "5", "6", "7", "8"]),
                    "slot paths are exactly 1…8")
            for slot in TrackerMapPino.slots {
                if let j = JointType.forPinoSlot(slot.path) {
                    t.check(j.pinoSlot == slot.path, "wire slot \(slot.path) round-trips via JointType.pinoSlot")
                } else {
                    t.check(false, "no JointType maps to wire slot \(slot.path)")
                }
            }
            // Handedness sanity: slot 7 = L_ankle (left), slot 8 = R_ankle (right).
            if let l7 = JointType.forPinoSlot("7"), let r8 = JointType.forPinoSlot("8") {
                t.check(l7.isLeft && !r8.isLeft, "slot 7 is left, slot 8 is right")
            } else {
                t.check(false, "slots 7/8 must map to JointTypes")
            }
        }
    }
}
