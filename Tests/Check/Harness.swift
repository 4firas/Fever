import Foundation

/// Minimal headless assertion harness for the CLT-only fallback test runner.
/// Records pass/fail counts and prints a TAP-ish summary; the @main entry exits
/// non-zero if anything failed so `swift run FeverCheck` is CI-usable.
final class TestRunner {
    private(set) var passed = 0
    private(set) var failed = 0
    private(set) var testCount = 0
    private var current = ""

    func test(_ name: String, _ body: () -> Void) {
        testCount += 1
        current = name
        let before = failed
        body()
        if failed == before {
            print("ok - \(name)")
        }
    }

    func check(_ condition: Bool, _ message: @autoclosure () -> String) {
        if condition {
            passed += 1
        } else {
            failed += 1
            print("not ok - [\(current)] \(message())")
        }
    }

    /// Float approximate-equality check with absolute tolerance.
    func close(_ a: Float, _ b: Float, tol: Float, _ message: @autoclosure () -> String) {
        let ok = a.isFinite && b.isFinite && abs(a - b) <= tol
        check(ok, "\(message()) (got \(a), want \(b) ± \(tol))")
    }

    func finalSummary() -> String {
        let total = passed + failed
        return "TEST SUMMARY: \(testCount) tests, \(total) assertions, \(passed) passed, \(failed) failed"
    }
}
