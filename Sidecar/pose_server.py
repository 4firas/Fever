#!/usr/bin/env python3
"""Fever pose sidecar: stdin RGB frames -> stdout 33 BlazePose world landmarks.

Dumb, single-purpose. The Swift app owns the camera, UI, solver, and OSC; this
process only runs MediaPipe Pose Landmarker and returns landmarks.

Wire protocol (length-prefixed framing; every message = len:u32 LE + body):
  request body : magic:u32=0xF0E1D2C3 | seq:u32 | t_micros:u64 | w:u16 | h:u16 | fmt:u8(0=RGB888) | rgb[w*h*3]
  reply  body  : magic:u32=0xC3D2E1F0 | seq:u32 | found:u8 | (if found) world[33*3 f32] vis[33 f32] pres[33 f32] image[33*2 f32]
All little-endian. Landmark order = BlazePose index 0..32.
"""
import os
# Quiet MediaPipe / glog C++ logging on stderr (set before importing mediapipe).
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("GLOG_logtostderr", "0")
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")
import sys
import struct
import argparse
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

REQ_MAGIC = 0xF0E1D2C3
REP_MAGIC = 0xC3D2E1F0
NLM = 33  # BlazePose landmark count


def _read_exact(stream, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            return None  # EOF
        buf.extend(chunk)
    return bytes(buf)


def _read_frame(stream):
    hdr = _read_exact(stream, 4)
    if hdr is None:
        return None
    (length,) = struct.unpack("<I", hdr)
    return _read_exact(stream, length)


def _write_frame(stream, body):
    stream.write(struct.pack("<I", len(body)))
    stream.write(body)
    stream.flush()


def _pack_reply(seq, result):
    found = 1 if (result and result.pose_world_landmarks) else 0
    head = struct.pack("<IIB", REP_MAGIC, seq, found)
    if not found:
        return head
    world = result.pose_world_landmarks[0]
    image = result.pose_landmarks[0]
    w = struct.pack("<%df" % (NLM * 3), *[c for lm in world for c in (lm.x, lm.y, lm.z)])
    vis = struct.pack("<%df" % NLM, *[lm.visibility for lm in world])
    pres = struct.pack("<%df" % NLM, *[lm.presence for lm in world])
    img = struct.pack("<%df" % (NLM * 2), *[c for lm in image for c in (lm.x, lm.y)])
    return head + w + vis + pres + img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("FEVER_MODEL", ""))
    args = ap.parse_args()
    if not args.model or not os.path.exists(args.model):
        sys.stderr.write("FATAL: model not found: %r\n" % args.model)
        sys.stderr.flush()
        return 2

    opts = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=args.model),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.PoseLandmarker.create_from_options(opts)
    sys.stderr.write("READY %s\n" % mp.__version__)
    sys.stderr.flush()

    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    while True:
        body = _read_frame(stdin)
        if body is None:
            break
        magic, seq, t_micros, width, height, fmt = struct.unpack_from("<IIQHHB", body, 0)
        if magic != REQ_MAGIC:
            sys.stderr.write("bad magic %x\n" % magic)
            sys.stderr.flush()
            continue
        try:
            rgb = np.frombuffer(body, dtype=np.uint8, offset=21, count=width * height * 3)
            rgb = rgb.reshape((height, width, 3))
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            result = landmarker.detect_for_video(image, int(t_micros // 1000))
        except Exception as exc:
            # A recoverable inference error must not crash the process: the IPC
            # framing requires every seq to get exactly one reply. Answer found=0
            # and keep serving so we don't trigger an EOF restart + model reload.
            sys.stderr.write("inference error seq=%d: %r\n" % (seq, exc))
            sys.stderr.flush()
            result = None
        _write_frame(stdout, _pack_reply(seq, result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
