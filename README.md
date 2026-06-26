# Fever

Markerless full-body tracking for VRChat, from a single webcam.

No straps. No base stations. No SteamVR. Fever points your Mac's camera at you,
works out where your body is in 3D, and streams the result to VRChat as OSC
trackers — hips, chest, feet, knees, elbows. Stand up, calibrate once, dance.

It's a native macOS app: Swift, AVFoundation, on-device pose estimation, and a
hand-rolled OSC stack. Nothing leaves your network.

## Why

Full-body tracking usually means buying hardware — Vive pucks, a SlimeVR set,
lighthouses screwed to your walls. That's a few hundred dollars and an afternoon
of setup before you can kick your legs in-game. The phone apps that exist tend to
paywall your knees and feet, and lean the moment you turn sideways.

Fever is the version I wanted: one camera you already own, every tracker
unlocked, and tracking that stays still when you do.

## What it does

- The full eight-tracker VRChat set — hip, chest, both feet, both knees, both
  elbows — plus a head reference point that keeps the play space aligned.
- Runs live off the built-in camera or any USB webcam.
- Calibrate in VRChat: stand in a T-pose and run the headset's tracker
  calibration. Fever streams absolute poses, so VRChat owns the alignment.
- Smoothed where it counts. A One-Euro filter feeds a forward-predicting
  upsampler that fills the gap between inferences, so the stream is both smooth
  and low-latency; the on-screen skeleton draws raw landmarks so the preview
  stays glued to you.
- Talks to VRChat over OSC/UDP. A standalone Quest works over Wi-Fi with no PC and
  no driver.

## Requirements

- macOS 26+, Apple Silicon
- A webcam
- VRChat with OSC enabled (it is by default)
- Enough room and light that your whole body fits in frame

## Build

Fever builds straight from the Swift toolchain — there's no Xcode project.

```sh
swift build -c release
./Scripts/bundle.sh      # assembles + signs Fever.app, installs to /Applications
```

Run the checks:

```sh
swift run FeverCheck
```

(`swift test` needs an xctest host that the Command-Line-Tools toolchain doesn't
ship, so the test suite runs as a plain executable that links the same library
and exits non-zero on failure.)

## Using it

1. Launch Fever, grant camera access.
2. In Settings, point the OSC host at the device running VRChat (your headset or
   PC) and leave the port at `9000`.
3. In VRChat, set your **real height** under Tracking & IK and make sure OSC is on.
4. Step back until your whole body is in frame.
5. Run VRChat's calibration — T-pose, arms out, feet on the floor.

Two things matter more than they should: your VRChat height has to match your
actual height or your feet sink through the floor, and you want an even, well-lit
background. A single camera infers depth, so fast forward/back motion is the soft
spot — that's the part I'm still hammering on.

## Under the hood

Every frame runs through a pose model, gets lifted into a metric, hip-rooted 3D
skeleton, smoothed, and solved into a transform per VR joint. Capture, inference,
and the network send each live on their own path, so one slow frame never stalls
the preview.

What goes on the wire is VRChat's tracker contract, exactly:

- OSC/UDP to port **9000**
- `/tracking/trackers/{1..8}/position` — three float metres
- `/tracking/trackers/{1..8}/rotation` — three float **Euler degrees, ZXY order**
  (not a quaternion)
- `/tracking/trackers/head/position` — the anchor VRChat re-origins the space to,
  which cancels absolute-position drift
- Fixed slot assignment, one body part per slot, every frame — no multiplexing

Getting that format right — handedness, the head anchor, hold-last-valid so a
dropped limb doesn't teleport — is most of the work. The details are in the source
and the commits.

The code is three SwiftPM targets:

```
FeverCore    pure logic: PinoFBT IK solve, One-Euro, fps-mux upsampling, OSC encoding
Fever        the @main app: SwiftUI window, camera preview, skeleton overlay, CLI
FeverCheck   headless assertion runner (swift run FeverCheck)
```

## Status

Working — tracks live, streams eight trackers with rotation to VRChat. Current
focus: depth on the forward/back axis, foot orientation, and frame rate.

## License

MIT — see [LICENSE](LICENSE).
