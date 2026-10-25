"""The perception + logic + presentation pipeline, in one reusable object.

The lab guide describes a three-layer architecture:

    Perception   camera frames -> MediaPipe Hands -> 21 landmarks per hand
    Logic        landmarks -> finger states -> gesture label -> smoothing
    Presentation skeleton, effects, HUD

`HandPipeline.process()` runs all three on one frame.  Both entry points - the
ROS2 node (`gesture_node.py`) and the no-ROS fallback (`standalone.py`) - use
this same object, so there is exactly one copy of the recognition behaviour.

Two MediaPipe APIs are supported, because the one the lab guide uses no longer
exists in current releases:

    mediapipe < 0.10.30    mp.solutions.hands      (Ubuntu 20.04 VM, 0.10.9)
    mediapipe >= 0.10.30   tasks HandLandmarker    (newer Pythons, e.g. 3.12)

Both run the same hand landmark model and return the same 21 points, so
everything after detection is identical.  The Tasks API needs the model as a
file, shipped in `models/hand_landmarker.task`.
"""

import os
import time

import cv2
import mediapipe as mp

from . import gesture_lib as gl
from . import overlay

MODEL_NAME = "hand_landmarker.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")


def find_model(model_path=None):
    """Locate hand_landmarker.task: explicit path, installed share dir, source tree."""
    candidates = []
    if model_path:
        candidates.append(model_path)
    try:
        from ament_index_python.packages import get_package_share_directory
        candidates.append(os.path.join(
            get_package_share_directory("gesture_recognition"), "models", MODEL_NAME))
    except Exception:                                       # noqa: BLE001
        pass    # not running under ROS, or the package is not installed
    candidates.append(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "models", MODEL_NAME))

    for path in candidates:
        if os.path.isfile(path):
            return os.path.abspath(path)
    raise FileNotFoundError(
        "MediaPipe hand model not found. Download it with:\n"
        "  curl -L -o src/gesture_recognition/models/%s %s"
        % (MODEL_NAME, MODEL_URL))


class _SolutionsHands:
    """The legacy `mp.solutions.hands` API, as used in the lab guide."""

    def __init__(self, max_hands, detection_confidence, tracking_confidence,
                 model_complexity):
        # model_complexity=0 is the lite model: noticeably faster, which matters
        # inside a VM where there is no GPU and the camera is passed through USB.
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            model_complexity=model_complexity,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )

    def detect(self, frame_rgb):
        """Returns a list of (handedness label or None, 21 landmarks)."""
        # Marking the array non-writeable lets MediaPipe skip an internal copy.
        frame_rgb.flags.writeable = False
        result = self._hands.process(frame_rgb)
        if not result.multi_hand_landmarks:
            return []
        handedness = result.multi_handedness or []
        return [(handedness[i].classification[0].label if i < len(handedness) else None,
                 landmarks.landmark)
                for i, landmarks in enumerate(result.multi_hand_landmarks)]

    def close(self):
        self._hands.close()


class _TasksHands:
    """The MediaPipe Tasks `HandLandmarker`, which replaced `mp.solutions`."""

    def __init__(self, max_hands, detection_confidence, tracking_confidence,
                 model_path):
        from mediapipe.tasks.python import BaseOptions, vision

        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            # VIDEO mode tracks hands between frames instead of re-detecting
            # every frame, like static_image_mode=False in the legacy API.
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_ms = -1

    def detect(self, frame_rgb):
        """Returns a list of (handedness label or None, 21 landmarks)."""
        # VIDEO mode rejects a timestamp that is not strictly increasing.
        now_ms = max(int(time.monotonic() * 1000), self._last_ms + 1)
        self._last_ms = now_ms
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._landmarker.detect_for_video(image, now_ms)
        handedness = result.handedness or []
        return [(handedness[i][0].category_name if i < len(handedness) and handedness[i] else None,
                 landmarks)
                for i, landmarks in enumerate(result.hand_landmarks)]

    def close(self):
        self._landmarker.close()


class HandPipeline:

    def __init__(self,
                 max_hands=2,
                 detection_confidence=0.6,
                 tracking_confidence=0.5,
                 model_complexity=0,
                 smoothing_window=5,
                 min_votes=3,
                 config=None,
                 draw=True,
                 model_path=None):
        if hasattr(mp, "solutions"):
            self._hands = _SolutionsHands(max_hands, detection_confidence,
                                          tracking_confidence, model_complexity)
        else:
            self._hands = _TasksHands(max_hands, detection_confidence,
                                      tracking_confidence, find_model(model_path))
        self.config = config or gl.DEFAULT_CONFIG
        self.smoother = gl.GestureSmoother(window=smoothing_window,
                                           min_votes=min_votes)
        self.effects = overlay.EffectManager()
        self.fps = overlay.FpsCounter()
        self.draw = draw
        self._last_stable = {}

    def process(self, frame_bgr):
        """Run one frame through the pipeline.

        Returns `(annotated_frame, gestures)` where `gestures` maps a hand id
        such as "Right" to its smoothed gesture label.  The input frame is
        annotated in place when `draw` is enabled.
        """
        self.fps.tick()
        height, width = frame_bgr.shape[:2]

        # MediaPipe wants RGB.
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        hands = self._hands.detect(frame_rgb)

        gestures = {}
        live_ids = set()

        for i, (label, landmarks) in enumerate(hands):
            label = label or "Hand%d" % i

            # Two hands can carry the same handedness label for a frame or
            # two while tracking settles; suffix duplicates so their
            # smoothing histories never merge.
            hand_id = label
            suffix = 1
            while hand_id in live_ids:
                hand_id = "%s_%d" % (label, suffix)
                suffix += 1
            live_ids.add(hand_id)

            points = gl.to_points(landmarks)
            raw_gesture, states = gl.classify(points, self.config)
            stable = self.smoother.update(hand_id, raw_gesture)
            gestures[hand_id] = stable

            pixels = overlay.to_pixels(points, width, height)

            # Fire an animation only on the *transition* into a gesture,
            # otherwise it would restart on every single frame.
            if stable != self._last_stable.get(hand_id) and stable != gl.UNKNOWN:
                anchor = (pixels[gl.WRIST][0], pixels[gl.MIDDLE_TIP][1])
                self.effects.trigger(hand_id, stable, anchor)
            self._last_stable[hand_id] = stable

            if self.draw:
                overlay.draw_skeleton(frame_bgr, pixels, states)
                overlay.draw_hand_label(frame_bgr, pixels, stable, hand_id)

        # Drop smoothing history for hands that have left the frame.
        self.smoother.prune(live_ids)
        for hand_id in list(self._last_stable):
            if hand_id not in live_ids:
                del self._last_stable[hand_id]

        if self.draw:
            self.effects.draw(frame_bgr)
            lines = ["%s: %s" % (hid, gl.caption(g) or "-")
                     for hid, g in sorted(gestures.items())]
            if not lines:
                lines = ["no hand detected"]
            overlay.draw_hud(frame_bgr, self.fps.fps, lines)

        return frame_bgr, gestures

    def primary(self, gestures):
        """Pick one gesture to publish when several hands are visible.

        Prefers a real gesture over UNKNOWN, and otherwise takes the
        lowest-sorted hand id so the answer is stable frame to frame.
        """
        return self.primary_hand(gestures)[1]

    def primary_hand(self, gestures):
        """Like `primary()`, but also says which hand: ("Left", "VICTORY").

        The hand id is "" when no hand shows a gesture.  With the frame
        mirrored (the default), "Left" is the user's own left hand.
        """
        for hand_id in sorted(gestures):
            if gestures[hand_id] != gl.UNKNOWN:
                return hand_id, gestures[hand_id]
        return "", gl.UNKNOWN

    def close(self):
        self._hands.close()
