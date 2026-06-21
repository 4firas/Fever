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

## Iteration 1 — 2026-06-22

Workflow: 6 blind explorers (38 raw findings) → merge (36 candidates) → 3-judge
panel → selector (3 independent picks) → conditional design → worktree
implementers → 3-skeptic adversarial refute → completeness critic. **3 chosen,
3 survived (every skeptic panel 0/3 refuted), 0 rejected.**

Integration verified on-branch: `swift build` green · `swift run FeverCheck` =
102 tests / 933 assertions / 0 failed · real MediaPipe sidecar smoke test (the
live child, exercising the modified Python loop) = OK exit 0.

### `9dbaf93` Bound sidecar reply frame length to prevent unbounded allocation
- **What:** Cap the declared reply-frame length (`maxReplyBytes = 4096`) in
  `PoseSidecar.readFrame()` before `readExact(Int(len))`; `teardown()` on overflow.
- **Why:** The sidecar is a restart-on-crash native child; a crash/partial-write
  desync of the length framing turned any bogus length into an OOM-scale
  allocation or a permanent hang of the inference queue. Now a desync becomes a
  clean child restart. Real reply body is a fixed 933 bytes, far under the cap.
- **Evidence:** build green; 102/933/0; valid-frame path provably unchanged
  (largest real body 933 ≤ 4096). 3/3 skeptics could not refute.
- **Integration note:** the implementer's comment/commit said the body is "1065
  bytes" — verified wrong (it's 933 = `9 + 924`, which the Python test asserts);
  corrected to 933 before commit.

### `2bfd041` Keep pose sidecar alive on per-frame inference errors
- **What:** Wrap `detect_for_video` (+ frame reshape/Image build) in try/except in
  `Sidecar/pose_server.py`; on any exception log to stderr and emit a `found=0`
  reply for that seq instead of crashing.
- **Why:** An unhandled per-frame error killed the whole child, breaking the
  one-reply-per-seq IPC contract and forcing an EOF-detected restart + multi-second
  model reload that froze all tracking. The `found=0` path already existed.
- **Evidence:** `py_compile` OK; build green; 102/933/0; live sidecar smoke test
  still OK. 3/3 skeptics could not refute (the one raised edge case — a sticky
  non-monotonic-timestamp fault — was shown not to apply: timestamps come from a
  real-time clock that always advances).

### `8273827` Fix solver frame docs to describe MediaPipe source, not the removed Vision path
- **What:** Comments only. Correct the canonical coordinate/handedness reference in
  `CoordinateMapper`, `JointSolver`, `VRJoint`, and the `TrackingPipeline` diagram
  to the actual post-pivot solver frame (hip-origin, y-down→up, `z*zSign` via
  `MediaPipeFrame`); drop the dead `PoseAdapter` / Vision-17-joint / heightEstimation
  references; swap the stale `QuaternionStabilizer` mention for `RotationRebaser`.
- **Why:** These headers are the reference for the project's most failure-prone
  seam; a wrong mental model here is exactly how the sign/handedness bugs this
  branch keeps fixing get reintroduced.
- **Evidence:** docs-only; build green; 102/933/0 (byte-identical behavior). 3/3
  skeptics could not refute. Noted incomplete: stale "Vision" doc references remain
  in the same + other files → queued for a later iteration.

**Critic — high-confidence work remaining (true):** (1) request-path crash twin in
`pose_server.py` (unguarded `struct.unpack_from` + `np.frombuffer/.reshape` on a
short/truncated frame — the sibling of the two fixes above); (2) real logic bug in
`LandmarkConsistency` — the L/R anti-swap swaps `lm[]/img[]` but not the parallel
`engaged[]/prev[]` hysteresis state; (3) untested crash-guards (`JointSolver`
count!=33, `SidecarProtocol` short found-body); (4) dead no-op ternary in
`FootMotionState`. Dry streak: 0/2. Continuing.

---
