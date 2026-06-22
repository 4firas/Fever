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

## Iteration 2 — 2026-06-22

Same workflow shape (40 raw findings → 36 candidates → judge panel → selector →
worktree implementers → 3-skeptic refute → critic). **3 chosen, 3 survived
(0/3 refuted each), 0 rejected.** Integration verified: build green; `swift run
FeverCheck` = 105 tests / 941 assertions / 0 failed, reproduced over 5 consecutive
clean runs.

This round's picks were deliberately conservative (the selector favoured items the
asset-free headless suite can fully verify): one dead-code removal + two
test-coverage additions. The carry-over seeding from iteration 1 did **not** take
effect (workflow logged `+0 carried-over critic seeds` — the `args` object didn't
reach the script), but the blind fan-out independently re-discovered the same
high-value items, so nothing was lost.

### `8025d7f` Drop no-op ternary in FootMotionState swing ramp
- **What:** `Self.ramp(lift, feet[i].seeded ? liftFull : liftFull, liftNone)` →
  `Self.ramp(lift, liftFull, liftNone)` (both ternary arms were identical; the
  foot is always seeded by that point anyway).
- **Why:** A no-op conditional on the per-frame swing-ramp path invites a future
  reader to "fix" one branch and silently shift the step/stride exaggeration.
- **Evidence:** provable no-op (git history shows both arms were `liftFull` from
  birth); build green; 105/941/0; STEP tests unchanged. 3/3 skeptics could not
  refute. Commit message reworded from the implementer's to drop an inaccurate
  "lost distinction" narrative the skeptics flagged.

### `13d269d` Test decodeReply rejects truncated found-reply bodies
- **What:** Two new assertions in `SidecarProtocolTests`: a found header followed
  by only 10 floats (far under the 933-byte floor) must decode to nil, plus an
  8-byte body one short of the 9-byte minimum.
- **Why:** The exact-length floor guard in `decodeReply` is the only thing
  stopping a torn/partial IPC frame from an out-of-bounds `f32()` read; its
  rejection side was previously unexercised (the old bad-body test exited at the
  earlier `>=9` guard). +1 test / +2 assertions.
- **Evidence:** build green; 105/941/0; new test passes deterministically. 3/3
  skeptics could not refute.

### `744e48a` Add tests for oscPort clamp and stale coefficient load-clamps
- **What:** Two new tests pinning the `oscPort` didSet clamp (70000→65535, 0→1,
  the UInt16/NWEndpoint.Port trap guard) and the load-time `>=1.0` clamp that
  rescues hip/step gains from stale sub-1.0 preferences (2.0/1.4/1.6/1.3).
- **Why:** A regression removing either clamp would silently reintroduce a process
  trap or the "hip collapses onto stance centre" bug with no test failure.
- **Evidence:** build green; 105/941/0; +2 tests / +6 assertions. 3/3 skeptics
  could not refute (minor noted nit: mutates `UserDefaults.standard`, matching the
  surrounding `testTrackingDefaults` convention; no in-suite contamination).

**Critic — high-confidence work remaining (true):** (1) **`MediaPipeFrame`
floor-anchor gates feet on `presence > 0` while shoulders/hip use `visibility >
0.5` in the same function** — and the latch runs (one-shot, until Recenter) before
any visibility gating, so a single occluded first frame permanently biases every
tracker's vertical position. One-line fix matching the adjacent pattern; **the
standout correctness bug.** (2) Python sidecar still crashes on a <21-byte request
frame (`struct.unpack_from` sits before the try/except). (3) `ensureRunning` READY
handshake uses a blocking `availableData` read so the documented 30s deadline can
never fire → possible permanent startup freeze. (4) `PoseSidecar` leaks an FD +
armed `readabilityHandler` on every crash-restart.

**Meta-finding (skeptics, not the critic):** the FeverCheck suite is non-deterministic
under *concurrent* runs — the UDP wire tests bind hardcoded ports 9000/9111 and
contend when multiple `FeverCheck` processes run at once (as in the workflow's
parallel verification). Sequential runs are clean. Worth fixing with ephemeral
ports to make parallel verification trustworthy.

Dry streak: 0/2. Continuing — next iteration steers toward the floor-anchor
correctness fix (now unit-testable via the existing `MediaPipeFrameTests`).

---

## Iteration 3 — 2026-06-22

The selector picked exactly the right work this round (correctness over
cleanup, per the sharpened steering): **2 correctness fixes + 1 singularity
test**. Two operational notes:

- **First attempt was an infra miss, not a dry result.** The harness
  `isolation:'worktree'` could not find the repo from this session's CWD (which
  resets to the non-repo home dir), so all three implementers failed at worktree
  creation — zero code touched. Durable fix: implementers now create their own
  worktree via absolute `git -C` commands (CWD-independent). The run was then
  *resumed* (cached analysis reused) and completed normally.
- Carry-forward now works via a hardcoded fallback (`+4 carried-over critic
  seeds`) after `args` failed to thread through in iter 2.

Integration verified on-branch: build green; `swift run FeverCheck` = **111
tests / 961 assertions / 0 failed**; real MediaPipe sidecar **survived a 10-byte
short frame and answered the next valid request** (live end-to-end probe).

### `bfd55bc` Fix floor-anchor leaving low-presence joints ~1m out of place
- **What:** In `MediaPipeFrame.toSolverFrame`, the per-landmark vertical floor
  shift was `for i in 0..<33 where lm[i].presence > 0` — gated, while the XZ
  origin subtraction above is unconditional. Make the Y shift unconditional, and
  change the lowest-foot search gate from `presence > 0` to `present()`
  (visibility > 0.5, matching the torso gating).
- **Why:** MediaPipe emits all 33 landmarks every frame; a body landmark whose
  presence dips to 0 (distinct from visibility) kept its raw Y while the rest of
  the skeleton shifted by the latched floor (~standing height), teleporting that
  joint ~1m and reintroducing the 1-frame limb-spaz the pipeline suppresses.
  Because the latch is one-shot until Recenter, a bad first frame biases the
  whole session.
- **Evidence:** build green; 111/961; new regression test confirmed to FAIL on
  old code (`got -0.0, want 0.9`) and PASS on the fix. 3/3 skeptics could not refute.

### `4ac3461` Add direct unit tests for frameFromTwoAxes singularity guard
- **What:** New `Tests/Check/MathTests.swift` (+ one wire-up line) directly
  covering the degenerate/parallel-axis hold-last branch of `frameFromTwoAxes`
  (the guard the rotation rework is built around): parallel/anti-parallel/zero
  axes return holdLast unchanged; a perpendicular pair builds a finite, unit,
  right-handed orthonormal frame.
- **Why:** Integration tests deliberately avoid the singular region, so flipping
  the cross-length threshold would silently reintroduce the 90° limb snap.
- **Evidence:** test-only (source untouched); build green; 111/961; the implementer
  confirmed a threshold-flip mutation fails the perpendicular test. 3/3 skeptics
  could not refute. (Corrected the implementer's stale "+8 tests/+25 assertions"
  accounting in the commit message — actual is +5 tests / +17 assertions.)

### `b1edcd2` Guard sidecar header parse against short request frames
- **What:** In `pose_server.py`, drop a request body shorter than the 21-byte
  fixed header (no decodable seq → can't reply) instead of crashing in
  `struct.unpack_from`, which runs before the recoverable-error try/except.
- **Why:** A framing hiccup that delivered a sub-header frame killed the child →
  EOF → restart + multi-second model reload (a reload loop on persistent desync).
- **Evidence:** build green; py_compile OK; 111/961; **live probe: the real
  sidecar survived a 10-byte frame and answered the next valid request**.
- **Adversarial caveat acted on:** the implementer also added a *second* guard
  (truncated-payload `continue`). One skeptic correctly refuted it: the existing
  try/except already answers `found=0` for an under-declared payload, so an early
  `continue` with no reply would instead hang the Swift reader (no read timeout).
  **I dropped the second guard** and kept only the crash-preventing first guard;
  the truncated-payload case keeps its graceful `found=0` path.

**Critic — high-confidence work remaining (true):** (1) `PoseSidecar` restart
leaks a stderr FD + dispatch read-source and never reaps the SIGTERM'd child
(teardown only nils 3 refs) — compounds with the now-hardened restart paths
toward EMFILE/zombies; (2) the L/R anti-swap in `LandmarkConsistency` corrects
ankles+knees but NOT heels/foot-indices, so a corrected foot derives orientation
from the *other* foot's heel→toe vector in the profile-transpose case it targets
(real correctness gap, headless-testable); (3) `safeSlerp`/`angleBetween`
NaN/zero-length guards have no direct tests.

Dry streak: 0/2. Continuing.

---
