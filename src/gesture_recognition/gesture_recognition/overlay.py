"""Presentation layer: skeleton drawing, animated effects and the HUD.

Only needs OpenCV and NumPy - no MediaPipe, no ROS.  The skeleton is drawn by
hand rather than with `mp.solutions.drawing_utils` so bones can be coloured by
finger state, which turns the overlay into a live debugging view: if a gesture
is misread you can see immediately *which* finger the classifier got wrong.

A note on emoji.  `cv2.putText` renders Hershey vector fonts, which contain no
emoji glyphs, so the table in the lab guide ("Thumbs Up 👍", "Heart 🤞") cannot
be drawn as text - it comes out as empty boxes.  The heart is therefore drawn
as a filled parametric curve, and every label is ASCII.
"""

import math
import time
from collections import deque

import cv2
import numpy as np

from . import gesture_lib as gl

# Bones grouped per finger so each can take the colour of its own state.
FINGER_BONES = {
    "thumb": [(0, 1), (1, 2), (2, 3), (3, 4)],
    "index": [(0, 5), (5, 6), (6, 7), (7, 8)],
    "middle": [(9, 10), (10, 11), (11, 12)],
    "ring": [(13, 14), (14, 15), (15, 16)],
    "pinky": [(0, 17), (17, 18), (18, 19), (19, 20)],
}
PALM_BONES = [(5, 9), (9, 13), (13, 17)]

COLOUR_EXTENDED = (80, 255, 120)    # BGR - green
COLOUR_BENT = (80, 120, 255)        # BGR - red/orange
COLOUR_PALM = (200, 200, 200)
COLOUR_JOINT = (255, 255, 255)

ANIMATION_DURATION = 1.2            # seconds
MAX_SCALE = 2.6


def to_pixels(points, width, height):
    """Normalised landmarks -> integer pixel coordinates."""
    return [(int(p[0] * width), int(p[1] * height)) for p in points]


def draw_skeleton(frame, pixels, states):
    """Draw the 21-point hand skeleton, colouring each finger by its state."""
    for bone in PALM_BONES:
        cv2.line(frame, pixels[bone[0]], pixels[bone[1]], COLOUR_PALM, 2)

    for finger, bones in FINGER_BONES.items():
        colour = COLOUR_EXTENDED if states.get(finger) else COLOUR_BENT
        for a, b in bones:
            cv2.line(frame, pixels[a], pixels[b], colour, 3)

    for i, point in enumerate(pixels):
        radius = 6 if i in (gl.WRIST, gl.THUMB_TIP, gl.INDEX_TIP,
                            gl.MIDDLE_TIP, gl.RING_TIP, gl.PINKY_TIP) else 3
        cv2.circle(frame, point, radius, COLOUR_JOINT, -1)


# ---------------------------------------------------------------------------
# Heart shape
# ---------------------------------------------------------------------------

def _heart_points(center, size, samples=60):
    """Classic parametric heart, sampled into a polygon.

        x = 16 sin^3 t
        y = 13 cos t - 5 cos 2t - 2 cos 3t - cos 4t

    The y term is negated because image coordinates grow downward.
    """
    cx, cy = center
    pts = []
    for i in range(samples):
        t = 2.0 * math.pi * i / samples
        x = 16 * math.sin(t) ** 3
        y = (13 * math.cos(t) - 5 * math.cos(2 * t)
             - 2 * math.cos(3 * t) - math.cos(4 * t))
        pts.append((int(cx + x * size / 16.0), int(cy - y * size / 16.0)))
    return np.array([pts], dtype=np.int32)


def draw_heart(frame, center, size, alpha, colour=(90, 60, 255)):
    """Alpha-blended filled heart.

    OpenCV has no per-draw alpha, so the shape goes onto a scratch copy of the
    frame which is then blended back - the standard trick for transparency.
    """
    if alpha <= 0.01 or size < 1:
        return
    layer = frame.copy()
    cv2.fillPoly(layer, _heart_points(center, size), colour)
    cv2.polylines(layer, _heart_points(center, size), True, (255, 255, 255), 2)
    cv2.addWeighted(layer, alpha, frame, 1.0 - alpha, 0, dst=frame)


# ---------------------------------------------------------------------------
# Multi-layer text
# ---------------------------------------------------------------------------

def draw_glow_text(frame, text, org, scale=1.0, colour=(255, 255, 255),
                   outline=2, alpha=1.0, outline_colour=(0, 0, 0)):
    """Centred text with a dark outline, readable over any background.

    Two OpenCV details drive this implementation, and the guide's three-layer
    recipe (`font_scale = scale * (0.8 + i*0.2)`, `thickness = 2 + i*2`) trips
    over both:

    1. Varying *font_scale* per layer makes the layers different widths, so the
       larger ones protrude at both ends - that reads as doubled text, not as a
       halo.
    2. Varying *thickness* is no better, because a Hershey glyph's advance
       width grows with stroke thickness.  Same string, same origin, thicker
       stroke: the characters spread out and the tail of the label drifts
       sideways.  The layers cannot line up.

    So the outline is drawn by stamping the text at several small pixel offsets
    at a *constant* scale and thickness, then putting the bright core on top.
    Identical metrics every time, therefore perfect registration.

    Animated growth still happens - EffectManager scales the whole label over
    time, which is where a size change belongs.
    """
    if not text:
        return
    canvas = frame if alpha >= 0.99 else frame.copy()

    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(1, int(round(2 * scale)))
    (width, height), _ = cv2.getTextSize(text, font, scale, thickness)
    pos = (int(org[0] - width / 2), int(org[1] + height / 2))

    if outline > 0:
        offsets = [(-outline, 0), (outline, 0), (0, -outline), (0, outline),
                   (-outline, -outline), (outline, -outline),
                   (-outline, outline), (outline, outline)]
        for dx, dy in offsets:
            cv2.putText(canvas, text, (pos[0] + dx, pos[1] + dy), font, scale,
                        outline_colour, thickness, cv2.LINE_AA)

    cv2.putText(canvas, text, pos, font, scale, colour, thickness, cv2.LINE_AA)

    if alpha < 0.99:
        cv2.addWeighted(canvas, alpha, frame, 1.0 - alpha, 0, dst=frame)


# ---------------------------------------------------------------------------
# Effects
# ---------------------------------------------------------------------------

class EffectManager:
    """Fire-and-forget animations keyed by hand id.

    Each effect stores its start time and anchor point.  Progress is derived
    from the wall clock rather than a frame counter, so animations run at the
    same speed whether the pipeline manages 10 or 30 fps.
    """

    def __init__(self, duration=ANIMATION_DURATION):
        self.duration = duration
        self._active = {}          # hand_id -> [start_time, (x, y), gesture]

    def trigger(self, hand_id, gesture, position):
        """Start (or restart) an effect for this hand."""
        self._active[hand_id] = [time.time(), position, gesture]

    def draw(self, frame):
        now = time.time()
        for hand_id in list(self._active):
            start, position, gesture = self._active[hand_id]
            progress = (now - start) / self.duration
            if progress >= 1.0:
                del self._active[hand_id]
                continue

            scale = 1.0 + (MAX_SCALE - 1.0) * progress
            alpha = 1.0 - progress ** 2        # quadratic fade-out

            if gesture == gl.HEART:
                x, y = position
                draw_heart(frame, (x, int(y - 60 * progress)),
                           30 * scale, alpha)
            else:
                draw_glow_text(frame, gl.caption(gesture), position,
                               scale=0.9 * scale, colour=(120, 220, 255),
                               alpha=alpha)


# ---------------------------------------------------------------------------
# HUD
# ---------------------------------------------------------------------------

class FpsCounter:
    """Rolling-average frame rate over the last `window` frames."""

    def __init__(self, window=30):
        self._stamps = deque(maxlen=window)

    def tick(self):
        self._stamps.append(time.time())

    @property
    def fps(self):
        if len(self._stamps) < 2:
            return 0.0
        span = self._stamps[-1] - self._stamps[0]
        return (len(self._stamps) - 1) / span if span > 0 else 0.0


def draw_hud(frame, fps, lines, title="Gesture Recognition"):
    """Translucent info panel in the top-left corner."""
    height = 34 + 26 * (1 + len(lines))
    panel = frame.copy()
    cv2.rectangle(panel, (10, 10), (330, 10 + height), (25, 25, 25), -1)
    cv2.addWeighted(panel, 0.55, frame, 0.45, 0, dst=frame)

    cv2.putText(frame, title, (24, 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "FPS: %.1f" % fps, (24, 68), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (120, 220, 255), 1, cv2.LINE_AA)

    y = 94
    for line in lines:
        cv2.putText(frame, line, (24, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (200, 255, 200), 1, cv2.LINE_AA)
        y += 26


def draw_hand_label(frame, pixels, gesture, handedness):
    """Name the gesture just above the hand it belongs to."""
    text = gl.caption(gesture)
    if not text:
        return
    xs = [p[0] for p in pixels]
    ys = [p[1] for p in pixels]
    anchor = ((min(xs) + max(xs)) // 2, max(min(ys) - 24, 30))
    draw_glow_text(frame, "%s: %s" % (handedness, text), anchor, scale=0.7)
