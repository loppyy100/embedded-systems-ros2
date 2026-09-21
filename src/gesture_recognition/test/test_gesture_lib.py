"""Tests for the gesture logic, using synthetic hands.

`gesture_lib` deliberately has no OpenCV, MediaPipe or ROS imports, so these run
anywhere Python does:

    python3 src/gesture_recognition/test/test_gesture_lib.py      # no pytest needed
    python3 -m pytest src/gesture_recognition/test                # or with pytest

The synthetic hands are built from a simple model: each finger has a base
knuckle and a direction.  An extended finger puts its PIP and TIP along that
direction; a bent finger folds the TIP back towards the palm.  That is enough
to exercise every branch of `classify` without a camera.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gesture_recognition import gesture_lib as gl  # noqa: E402

WRIST_POS = (0.50, 0.90)
KNUCKLES = {            # finger -> (mcp x, mcp y)
    "index": (0.42, 0.70),
    "middle": (0.50, 0.70),
    "ring": (0.57, 0.70),
    "pinky": (0.63, 0.72),
}
UP = (0.0, -1.0)


def _add(base, direction, length):
    return (base[0] + direction[0] * length, base[1] + direction[1] * length, 0.0)


def _unit(vec):
    norm = math.hypot(vec[0], vec[1])
    return (vec[0] / norm, vec[1] / norm)


def make_hand(thumb=True, index=True, middle=True, ring=True, pinky=True,
              thumb_dir="up", pinch=False, scale=1.0, origin=(0.0, 0.0)):
    """Build 21 synthetic landmarks. Returns a list of (x, y, z)."""
    points = [None] * 21

    def place(i, xy):
        x = origin[0] + (xy[0] - 0.5) * scale + 0.5
        y = origin[1] + (xy[1] - 0.5) * scale + 0.5
        points[i] = (x, y, 0.0)

    place(gl.WRIST, WRIST_POS)

    states = {"index": index, "middle": middle, "ring": ring, "pinky": pinky}
    joints = {
        "index": (gl.INDEX_MCP, gl.INDEX_PIP, gl.INDEX_DIP, gl.INDEX_TIP),
        "middle": (gl.MIDDLE_MCP, gl.MIDDLE_PIP, gl.MIDDLE_DIP, gl.MIDDLE_TIP),
        "ring": (gl.RING_MCP, gl.RING_PIP, gl.RING_DIP, gl.RING_TIP),
        "pinky": (gl.PINKY_MCP, gl.PINKY_PIP, gl.PINKY_DIP, gl.PINKY_TIP),
    }

    for name, (mcp_i, pip_i, dip_i, tip_i) in joints.items():
        base = KNUCKLES[name]
        pip = _add(base, UP, 0.07)
        place(mcp_i, base)
        place(pip_i, pip[:2])
        if states[name]:
            place(dip_i, _add(base, UP, 0.12)[:2])
            place(tip_i, _add(base, UP, 0.17)[:2])
        else:
            # Folded: the tip curls back past the knuckle towards the palm, so
            # both the PIP angle and the wrist-reach test read "bent".
            place(dip_i, _add(base, UP, 0.04)[:2])
            place(tip_i, _add(base, UP, 0.01)[:2])

    # --- thumb ---------------------------------------------------------
    thumb_cmc = (0.46, 0.86)
    place(gl.THUMB_CMC, thumb_cmc)
    if thumb:
        out = _unit((-0.75, -0.66)) if thumb_dir == "up" else _unit((-0.75, 0.66))
        mcp = _add(thumb_cmc, out, 0.06)
        place(gl.THUMB_MCP, mcp[:2])
        place(gl.THUMB_IP, _add(thumb_cmc, out, 0.13)[:2])
        place(gl.THUMB_TIP, _add(thumb_cmc, out, 0.20)[:2])
    else:
        # Tucked across the palm: the tip moves towards the pinky knuckle.
        inward = _unit((0.85, -0.30))
        place(gl.THUMB_MCP, _add(thumb_cmc, inward, 0.05)[:2])
        place(gl.THUMB_IP, _add(thumb_cmc, inward, 0.10)[:2])
        place(gl.THUMB_TIP, _add(thumb_cmc, inward, 0.15)[:2])

    if pinch:
        contact = (0.44, 0.645)
        place(gl.THUMB_TIP, contact)
        place(gl.INDEX_TIP, (contact[0] + 0.004, contact[1] + 0.004))
        place(gl.INDEX_DIP, (0.43, 0.60))

    assert all(p is not None for p in points), "landmark not filled in"
    return points


def check(label, points, expected):
    actual, states = gl.classify(points)
    ok = actual == expected
    print("  %-14s expected %-11s got %-11s %s   %s"
          % (label, expected, actual, "OK" if ok else "FAIL",
             "".join("1" if states[f] else "0" for f in gl.FINGER_ORDER)))
    return ok


# ---------------------------------------------------------------------------
# Tests (named test_* so pytest collects them too)
# ---------------------------------------------------------------------------

CASES = [
    ("thumbs up", dict(thumb=True, index=False, middle=False, ring=False,
                       pinky=False, thumb_dir="up"), gl.THUMB_UP),
    ("thumbs down", dict(thumb=True, index=False, middle=False, ring=False,
                         pinky=False, thumb_dir="down"), gl.THUMB_DOWN),
    ("victory", dict(thumb=False, index=True, middle=True, ring=False,
                     pinky=False), gl.VICTORY),
    ("rock", dict(thumb=False, index=True, middle=False, ring=False,
                  pinky=True), gl.ROCK),
    ("ok", dict(thumb=True, index=False, middle=True, ring=True,
                pinky=True, pinch=True), gl.OK),
    ("heart", dict(thumb=True, index=False, middle=False, ring=False,
                   pinky=False, pinch=True), gl.HEART),
    ("open palm", dict(thumb=True, index=True, middle=True, ring=True,
                       pinky=True), gl.OPEN_PALM),
    ("fist", dict(thumb=False, index=False, middle=False, ring=False,
                  pinky=False), gl.FIST),
    ("point", dict(thumb=False, index=True, middle=False, ring=False,
                   pinky=False), gl.POINT),
]


def test_gestures():
    failures = []
    for label, kwargs, expected in CASES:
        if not check(label, make_hand(**kwargs), expected):
            failures.append(label)
    assert not failures, "misclassified: %s" % failures


def test_scale_invariance():
    """The same pose must classify identically near and far from the camera."""
    failures = []
    for label, kwargs, expected in CASES:
        for scale in (0.35, 1.0, 1.8):
            points = make_hand(scale=scale, **kwargs)
            actual, _ = gl.classify(points)
            if actual != expected:
                failures.append("%s @ scale %.2f -> %s" % (label, scale, actual))
    print("  scale invariance: %s" % ("OK" if not failures else failures))
    assert not failures, failures


def test_translation_invariance():
    """Moving the hand around the frame must not change the result."""
    failures = []
    for label, kwargs, expected in CASES:
        for origin in ((-0.25, -0.2), (0.0, 0.0), (0.3, 0.15)):
            points = make_hand(origin=origin, scale=0.6, **kwargs)
            actual, _ = gl.classify(points)
            if actual != expected:
                failures.append("%s @ %s -> %s" % (label, origin, actual))
    print("  translation invariance: %s" % ("OK" if not failures else failures))
    assert not failures, failures


def test_finger_states():
    states = gl.finger_states(make_hand(thumb=True, index=True, middle=False,
                                        ring=False, pinky=True))
    print("  finger states: %s" % states)
    assert states == {"thumb": True, "index": True, "middle": False,
                      "ring": False, "pinky": True}


def test_smoother_rejects_single_frame_noise():
    """One bad frame in a run of good ones must not change the output."""
    smoother = gl.GestureSmoother(window=5, min_votes=3)
    for _ in range(5):
        smoother.update("Right", gl.VICTORY)
    assert smoother.update("Right", gl.VICTORY) == gl.VICTORY

    # A single spurious FIST is outvoted by the VICTORY frames still in window.
    assert smoother.update("Right", gl.FIST) == gl.VICTORY
    print("  smoother rejects single-frame noise: OK")


def test_smoother_accepts_sustained_change():
    smoother = gl.GestureSmoother(window=5, min_votes=3)
    for _ in range(5):
        smoother.update("Right", gl.VICTORY)
    results = [smoother.update("Right", gl.OK) for _ in range(4)]
    assert results[-1] == gl.OK, results
    print("  smoother accepts sustained change after %d frames: OK"
          % (results.index(gl.OK) + 1))


def test_smoother_separates_hands():
    smoother = gl.GestureSmoother(window=3, min_votes=2)
    for _ in range(3):
        smoother.update("Left", gl.FIST)
        smoother.update("Right", gl.OPEN_PALM)
    assert smoother.update("Left", gl.FIST) == gl.FIST
    assert smoother.update("Right", gl.OPEN_PALM) == gl.OPEN_PALM

    smoother.prune({"Right"})
    assert smoother.update("Left", gl.OK) == gl.UNKNOWN  # history was dropped
    print("  smoother keeps hands independent: OK")


def test_angle_helper():
    assert abs(gl.angle_at((0, 1, 0), (0, 0, 0), (0, -1, 0)) - 180.0) < 1e-6
    assert abs(gl.angle_at((1, 0, 0), (0, 0, 0), (0, 1, 0)) - 90.0) < 1e-6
    print("  angle helper: OK")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for test in tests:
        print("%s:" % test.__name__)
        try:
            test()
        except AssertionError as exc:
            print("  FAILED: %s" % exc)
            failed += 1
    print("\n%d/%d test functions passed" % (len(tests) - failed, len(tests)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
