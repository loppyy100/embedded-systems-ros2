"""Hand geometry and gesture classification.

Deliberately free of ROS, OpenCV and MediaPipe imports: everything here works on
plain `(x, y, z)` tuples, so it can be unit-tested on any machine
(`python3 -m pytest src/gesture_recognition/test`) and is shared by both the
ROS2 node and the standalone script.

--------------------------------------------------------------------------
How this differs from the lab guide, and why
--------------------------------------------------------------------------
The guide detects an extended finger with `landmarks[tip].y < landmarks[pip].y`
and notes two weaknesses itself: half-bent fingers get misread, and the MCP
joint is ignored.  Its suggested fix, `tip.y < pip.y < mcp.y`, still compares
raw image-space Y, so it only holds while the hand points straight up - tilt
the hand 90 degrees and every finger reads as bent.

We use two rotation-invariant tests per finger instead:

  1. *Straightness* - the interior angle at the PIP joint (MCP-PIP-TIP).  A
     straight finger is near 180 degrees; a bent one drops sharply.  An angle
     does not care which way the hand is facing.
  2. *Reach* - the tip must be further from the wrist than the PIP is.  This
     catches the case a pure angle test misses: a finger that is straight but
     folded down at the knuckle, as in a flat "paw" hand.

The thumb is handled the same way but with its own reference.  The guide
switches on left/right handedness to pick an X-axis direction; that breaks as
soon as the palm faces away or the image is mirrored.  Instead we ask whether
the thumb tip has moved *away* from the pinky knuckle relative to its own IP
joint, which is true for an extended thumb on either hand at any orientation.

All distance thresholds are expressed as a fraction of `hand_scale` (the
wrist-to-middle-knuckle distance) so they hold whether the hand is close to the
camera or far away - the guide's "dynamic threshold adjustment" idea, applied
everywhere rather than only to the OK gesture.
"""

import math
from collections import Counter, deque

# ---------------------------------------------------------------------------
# MediaPipe Hands landmark indices (21 points per hand)
# ---------------------------------------------------------------------------
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

# finger name -> (mcp, pip, tip)
FINGER_JOINTS = {
    "index": (INDEX_MCP, INDEX_PIP, INDEX_TIP),
    "middle": (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP),
    "ring": (RING_MCP, RING_PIP, RING_TIP),
    "pinky": (PINKY_MCP, PINKY_PIP, PINKY_TIP),
}
FINGER_ORDER = ("thumb", "index", "middle", "ring", "pinky")

# Gesture labels.  Plain ASCII on purpose: cv2.putText uses Hershey vector
# fonts, which have no emoji glyphs - drawing "👍" renders as a box.  The
# emoji in the guide's table are reproduced as drawn shapes in overlay.py.
THUMB_UP = "THUMB_UP"
THUMB_DOWN = "THUMB_DOWN"
VICTORY = "VICTORY"
ROCK = "ROCK"
OK = "OK"
HEART = "HEART"
OPEN_PALM = "OPEN_PALM"
FIST = "FIST"
POINT = "POINT"
UNKNOWN = "UNKNOWN"

# Human-readable captions for the HUD.
GESTURE_CAPTIONS = {
    THUMB_UP: "Thumbs Up",
    THUMB_DOWN: "Thumbs Down",
    VICTORY: "Victory",
    ROCK: "Rock",
    OK: "OK",
    HEART: "Heart",
    OPEN_PALM: "Open Palm",
    FIST: "Fist",
    POINT: "Pointing",
    UNKNOWN: "",
}


class GestureConfig:
    """Tunable thresholds.  Ratios are multiples of `hand_scale`."""

    def __init__(self,
                 straight_deg=155.0,
                 thumb_straight_deg=150.0,
                 reach_ratio=1.05,
                 thumb_away_ratio=1.03,
                 pinch_ratio=0.38,
                 thumb_direction_ratio=0.45):
        self.straight_deg = straight_deg
        self.thumb_straight_deg = thumb_straight_deg
        self.reach_ratio = reach_ratio
        self.thumb_away_ratio = thumb_away_ratio
        self.pinch_ratio = pinch_ratio
        self.thumb_direction_ratio = thumb_direction_ratio


DEFAULT_CONFIG = GestureConfig()


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def to_points(landmarks):
    """MediaPipe landmark list -> list of (x, y, z) tuples.

    Coordinates are normalised: x and y to [0, 1] of the frame (y grows
    downward, as in image coordinates), z roughly in the same unit as x,
    negative towards the camera.
    """
    return [(lm.x, lm.y, lm.z) for lm in landmarks]


def distance(a, b):
    """3D Euclidean distance between two (x, y, z) points."""
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def distance_2d(a, b):
    """2D distance - MediaPipe's z is noisier than x/y, so some checks skip it."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def angle_at(a, b, c):
    """Interior angle at vertex `b` of the path a-b-c, in degrees (0..180)."""
    v1 = (a[0] - b[0], a[1] - b[1], a[2] - b[2])
    v2 = (c[0] - b[0], c[1] - b[1], c[2] - b[2])
    n1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2 + v1[2] ** 2)
    n2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2 + v2[2] ** 2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cos = (v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]) / (n1 * n2)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def hand_scale(points):
    """Wrist-to-middle-knuckle distance: a stable proxy for apparent hand size.

    Every ratio threshold in this module is multiplied by this, which is what
    makes recognition independent of how far the hand is from the camera.
    """
    return max(distance(points[WRIST], points[MIDDLE_MCP]), 1e-6)


# ---------------------------------------------------------------------------
# Finger state
# ---------------------------------------------------------------------------

def finger_states(points, cfg=DEFAULT_CONFIG):
    """Return {finger_name: True if extended}.

    See the module docstring for why this uses joint angles rather than the
    guide's raw Y comparison.
    """
    states = {}
    wrist = points[WRIST]

    for name, (mcp, pip, tip) in FINGER_JOINTS.items():
        straight = angle_at(points[mcp], points[pip], points[tip]) >= cfg.straight_deg
        reaching = (distance(points[tip], wrist)
                    > distance(points[pip], wrist) * cfg.reach_ratio)
        states[name] = bool(straight and reaching)

    # Thumb: straight at the IP joint, and abducted away from the palm.
    # Tucking the thumb across the palm moves its tip *towards* the pinky
    # knuckle, so comparing tip and IP distances to that knuckle detects the
    # tuck on either hand without needing a handedness label.
    thumb_straight = (angle_at(points[THUMB_MCP], points[THUMB_IP], points[THUMB_TIP])
                      >= cfg.thumb_straight_deg)
    thumb_away = (distance(points[THUMB_TIP], points[PINKY_MCP])
                  > distance(points[THUMB_IP], points[PINKY_MCP]) * cfg.thumb_away_ratio)
    states["thumb"] = bool(thumb_straight and thumb_away)

    return states


def is_pinching(points, scale, cfg=DEFAULT_CONFIG):
    """True when the thumb and index fingertips are touching.

    Threshold scales with hand size, so the same pinch registers at any
    distance from the camera.
    """
    return distance(points[THUMB_TIP], points[INDEX_TIP]) < cfg.pinch_ratio * scale


def thumb_direction(points, scale, cfg=DEFAULT_CONFIG):
    """'up', 'down' or 'side' - where the thumb points in the image.

    Image Y grows downward, so a negative delta means the tip is above the
    knuckle on screen.
    """
    dy = (points[THUMB_TIP][1] - points[THUMB_MCP][1]) / scale
    if dy < -cfg.thumb_direction_ratio:
        return "up"
    if dy > cfg.thumb_direction_ratio:
        return "down"
    return "side"


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify(points, cfg=DEFAULT_CONFIG):
    """Classify one hand.  Returns (gesture_label, finger_states_dict)."""
    scale = hand_scale(points)
    states = finger_states(points, cfg)

    thumb = states["thumb"]
    index = states["index"]
    middle = states["middle"]
    ring = states["ring"]
    pinky = states["pinky"]

    pinch = is_pinching(points, scale, cfg)

    # Both OK and Heart start from a thumb-index contact; the other three
    # fingers tell them apart.  Tested first because a pinched index finger
    # reads as "bent" and would otherwise fall through to FIST or POINT.
    if pinch:
        if middle and ring and pinky:
            return OK, states
        if not (middle or ring or pinky):
            return HEART, states

    if index and pinky and not middle and not ring:
        return ROCK, states

    if index and middle and not ring and not pinky:
        return VICTORY, states

    if thumb and not (index or middle or ring or pinky):
        direction = thumb_direction(points, scale, cfg)
        if direction == "up":
            return THUMB_UP, states
        if direction == "down":
            return THUMB_DOWN, states
        return UNKNOWN, states

    if thumb and index and middle and ring and pinky:
        return OPEN_PALM, states

    if not (thumb or index or middle or ring or pinky):
        return FIST, states

    if index and not (middle or ring or pinky):
        return POINT, states

    return UNKNOWN, states


# ---------------------------------------------------------------------------
# Temporal smoothing
# ---------------------------------------------------------------------------

class GestureSmoother:
    """Majority vote over a sliding window, per hand.

    Raw per-frame classification flickers: a hand passing from fist to victory
    spends a few frames in between and emits garbage labels.  Requiring the
    same label in `min_votes` of the last `window` frames removes almost all of
    it, at the cost of a short delay (window/fps seconds) when switching.

    This is the guide's "state hold mechanism", plus a `min_votes` floor - a
    plain `most_common(1)` still returns a label when the window is evenly
    split between three different guesses.
    """

    def __init__(self, window=5, min_votes=3):
        if min_votes > window:
            raise ValueError("min_votes cannot exceed window")
        self.window = window
        self.min_votes = min_votes
        self._history = {}
        self._stable = {}

    def update(self, hand_id, gesture):
        """Feed one frame's raw label, get the smoothed label back."""
        history = self._history.setdefault(hand_id, deque(maxlen=self.window))
        history.append(gesture)

        label, votes = Counter(history).most_common(1)[0]
        if votes >= self.min_votes and label != UNKNOWN:
            self._stable[hand_id] = label
        elif votes >= self.min_votes and label == UNKNOWN:
            # A confident UNKNOWN clears the held gesture rather than freezing
            # the last one on screen after the hand relaxes.
            self._stable[hand_id] = UNKNOWN

        return self._stable.get(hand_id, UNKNOWN)

    def forget(self, hand_id):
        """Drop state for a hand that has left the frame."""
        self._history.pop(hand_id, None)
        self._stable.pop(hand_id, None)

    def prune(self, live_ids):
        """Forget every hand id not in `live_ids`."""
        for hand_id in list(self._history):
            if hand_id not in live_ids:
                self.forget(hand_id)


def caption(gesture):
    """Display text for a gesture label."""
    return GESTURE_CAPTIONS.get(gesture, gesture)
