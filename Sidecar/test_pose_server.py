"""Spawns pose_server.py, sends one synthetic frame, asserts a well-formed reply."""
import os, sys, struct, subprocess, numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, "Sidecar", ".venv", "bin", "python3")
SERVER = os.path.join(ROOT, "Sidecar", "pose_server.py")
MODEL = os.path.join(ROOT, "Models", "pose_landmarker_full.task")
REQ_MAGIC, REP_MAGIC = 0xF0E1D2C3, 0xC3D2E1F0


def _frame(body):
    return struct.pack("<I", len(body)) + body


def main():
    w, h = 256, 256
    rgb = (np.random.rand(h, w, 3) * 255).astype(np.uint8).tobytes()
    body = struct.pack("<IIQHHB", REQ_MAGIC, 7, 1000, w, h, 0) + rgb
    p = subprocess.Popen([PY, SERVER, "--model", MODEL],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # MediaPipe/glog prints init lines to stderr before READY; scan until READY.
    ready = False
    for _ in range(500):
        line = p.stderr.readline()
        if not line:
            break
        if line.startswith(b"READY"):
            ready = True
            break
    assert ready, "did not see READY on stderr"
    p.stdin.write(_frame(body)); p.stdin.flush()
    ln = p.stdout.read(4); (length,) = struct.unpack("<I", ln)
    reply = p.stdout.read(length)
    magic, seq, found = struct.unpack_from("<IIB", reply, 0)
    assert magic == REP_MAGIC, hex(magic)
    assert seq == 7, seq
    assert found in (0, 1), found
    if found:
        assert length == 9 + 924, length
    p.stdin.close(); p.terminate()
    print("OK framing seq=%d found=%d len=%d" % (seq, found, length))


if __name__ == "__main__":
    main()
