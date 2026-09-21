"""Run the gesture recogniser without ROS at all.

Useful in three situations that come up constantly in this lab:

  * MediaPipe is installed but ROS2 is not (or you are on Windows/macOS);
  * `cam2image` will not see the webcam through VMware's USB passthrough and
    you want to prove the recognition itself works;
  * you are on a headless or camera-less machine and want to run the whole
    thing against a recorded video file.

    python3 standalone.py                       # default camera
    python3 standalone.py --source 2            # /dev/video2
    python3 standalone.py --source clip.mp4     # a video file
    python3 standalone.py --source clip.mp4 --loop --no-window --print

The capture runs on its own thread with a single latest-frame slot, matching
the ROS2 node's structure so the two behave identically.
"""

import argparse
import sys
import threading
import time

import cv2

from .pipeline import HandPipeline
from . import gesture_lib as gl


class Capture:
    """Threaded frame grabber with a latest-frame-wins slot.

    `VideoCapture.read()` returns the oldest buffered frame.  If the consumer
    is slower than the camera, those buffers fill and the displayed image falls
    progressively further behind.  Draining them on a dedicated thread keeps
    the latency flat.
    """

    def __init__(self, source, loop=False):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError("cannot open video source %r" % (source,))
        self.loop = loop
        self._frame = None
        self._lock = threading.Lock()
        self._running = True
        self._ended = False
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                if self.loop:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                self._ended = True
                break
            with self._lock:
                self._frame = frame

    def read(self):
        with self._lock:
            frame, self._frame = self._frame, None
        return frame

    @property
    def ended(self):
        return self._ended

    def release(self):
        self._running = False
        self._thread.join(timeout=1.0)
        self.cap.release()


def parse_source(value):
    """'0' -> camera index 0; anything else -> a file path or stream URL."""
    try:
        return int(value)
    except ValueError:
        return value


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="0",
                        help="camera index, video file or stream URL (default: 0)")
    parser.add_argument("--loop", action="store_true",
                        help="restart the video file when it ends")
    parser.add_argument("--no-flip", dest="flip", action="store_false",
                        help="do not mirror the frame")
    parser.add_argument("--no-window", dest="window", action="store_false",
                        help="run headless (implies --print)")
    parser.add_argument("--print", dest="echo", action="store_true",
                        help="log each gesture change to stdout")
    parser.add_argument("--max-hands", type=int, default=2)
    parser.add_argument("--model-complexity", type=int, default=0, choices=(0, 1))
    parser.add_argument("--detection-confidence", type=float, default=0.6)
    parser.add_argument("--record", default=None,
                        help="write the annotated video to this path")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv if argv is not None else sys.argv[1:])
    if not args.window:
        args.echo = True

    capture = Capture(parse_source(args.source), loop=args.loop)
    pipeline = HandPipeline(max_hands=args.max_hands,
                            model_complexity=args.model_complexity,
                            detection_confidence=args.detection_confidence)
    writer = None
    last_printed = None

    try:
        while True:
            frame = capture.read()
            if frame is None:
                if capture.ended:
                    break
                time.sleep(0.005)
                continue

            if args.flip:
                frame = cv2.flip(frame, 1)

            annotated, gestures = pipeline.process(frame)
            primary = pipeline.primary(gestures)

            if args.echo and primary != last_printed:
                print("[%.1f fps] %s" % (pipeline.fps.fps, gl.caption(primary) or "-"),
                      flush=True)
                last_printed = primary

            if args.record:
                if writer is None:
                    height, width = annotated.shape[:2]
                    writer = cv2.VideoWriter(
                        args.record, cv2.VideoWriter_fourcc(*"mp4v"),
                        20.0, (width, height))
                writer.write(annotated)

            if args.window:
                cv2.imshow("Gesture Recognition (q to quit)", annotated)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        capture.release()
        pipeline.close()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
