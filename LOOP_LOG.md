# Auto-improve loop log

Branch: `loop/auto-improve` — autonomous, multi-agent quality improvement of Fever.

Each entry is one committed change that passed the real build + test suite and
survived an adversarial skeptic panel (majority could not refute it).

## Baseline (verified before any change)

- `swift build` → exit 0 (linker search-path warnings are benign CLT-only noise)
- `swift run FeverCheck` → exit 0 · **102 tests, 933 assertions, 0 failed**
- `Sidecar/.venv/bin/python Sidecar/test_pose_server.py` → exit 0 · well-formed reply
- No `swiftlint` / `swift-format` installed (minimal-footprint host); style is enforced by review, not a linter.

Stack: SwiftPM (tools 6.2, Swift 6 language mode, `.macOS(.v26)`), library
`FeverCore` + executables `Fever` (SwiftUI app) and `FeverCheck` (headless test
runner). Python MediaPipe sidecar under `Sidecar/`.

---
