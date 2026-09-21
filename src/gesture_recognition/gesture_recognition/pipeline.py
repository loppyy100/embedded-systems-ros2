"""The perception + logic + presentation pipeline, in one reusable object.

The lab guide describes a three-layer architecture:

    Perception   camera frames -> MediaPipe Hands -> 21 landmarks per hand
    Logic        landmarks -> finger states -> gesture label -> smoothing
    Presentation skeleton, effects, HUD

`HandPipeline.process()` runs all three on one frame.  Both entry points - the
ROS2 node (`gesture_node.py`) and the no-ROS fallback (`standalone.py`) - use
this same object, so there is exactly one copy of the recognition behaviour.
"""

import cv2
import mediapipe as mp

from . import gesture_lib as gl
from . import overlay


class HandPipeline:

    def __init__(self,
                 max_hands=2,
                 detection_confidence=0.6,
                 tracking_confidence=0.5,
                 model_complexity=0,
                 smoothing_window=5,
                 min_votes=3,
                 config=None,
                 draw=True):
        # model_complexity=0 is the lite model: noticeably faster, which matters
        # inside a VM where there is no GPU and the camera is passed through USB.
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            model_complexity=model_complexity,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
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

        # MediaPipe wants RGB.  Marking the array non-writeable lets it skip an
        # internal copy - a small but free speed-up.
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        result = self._hands.process(frame_rgb)

        gestures = {}
        live_ids = set()

        if result.multi_hand_landmarks:
            handedness_list = result.multi_handedness or []
            for i, landmarks in enumerate(result.multi_hand_landmarks):
                label = "Hand%d" % i
                if i < len(handedness_list):
                    label = handedness_list[i].classification[0].label

                # Two hands can carry the same handedness label for a frame or
                # two while tracking settles; suffix duplicates so their
                # smoothing histories never merge.
                hand_id = label
                suffix = 1
                while hand_id in live_ids:
                    hand_id = "%s_%d" % (label, suffix)
                    suffix += 1
                live_ids.add(hand_id)

                points = gl.to_points(landmarks.landmark)
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
        for hand_id in sorted(gestures):
            if gestures[hand_id] != gl.UNKNOWN:
                return gestures[hand_id]
        return gl.UNKNOWN

    def close(self):
        self._hands.close()
