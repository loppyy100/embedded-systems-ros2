"""Experiment 2 - subscribe to the camera topic and recognise hand gestures.

    ros2 run image_tools cam2image                  # publishes sensor_msgs/Image on /image
    ros2 run gesture_recognition gesture_node       # this node

Publishes:
    /gesture              std_msgs/String     smoothed gesture of the primary hand
    /gesture/annotated    sensor_msgs/Image   frame with skeleton, labels and effects

Parameters (all settable with --ros-args -p name:=value):
    image_topic           str    default "/image"
    show_window           bool   default True   - open an OpenCV preview window
    publish_annotated     bool   default True
    flip                  bool   default True   - mirror the frame (selfie view)
    max_hands             int    default 2
    detection_confidence  float  default 0.6
    tracking_confidence   float  default 0.5
    model_complexity      int    default 0      - 0 = lite model, 1 = full
    smoothing_window      int    default 5
    min_votes             int    default 3

--------------------------------------------------------------------------
Why there are three threads
--------------------------------------------------------------------------
MediaPipe needs 20-50 ms per frame on a VM's CPU, while `cam2image` publishes
at ~30 fps.  Running inference inside the subscription callback would block the
executor, the queue would back up, and the preview would drift seconds behind
reality.  So the work is split, with a *latest-frame-wins* handover between
stages - old frames are dropped, never queued:

    executor thread   receives Image messages, stores the newest one
    worker thread     converts + runs MediaPipe + publishes results
    main thread       cv2.imshow (GUI toolkits want the main thread)

This is the "asynchronous processing pipeline" the guide suggests, with the
guide's unbounded `queue.put(frame)` replaced by a single slot, because an
unbounded queue in front of a slower consumer grows without limit.
"""

import threading

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String

from .pipeline import HandPipeline
from . import gesture_lib as gl


def image_msg_to_bgr(msg):
    """sensor_msgs/Image -> BGR ndarray, without cv_bridge.

    cv_bridge is genuinely awkward in a fresh VM (it is a compiled package and
    is easy to break by pip-installing a different NumPy), and the conversion
    is a handful of lines, so we do it directly.

    `msg.step` is the row stride in bytes and may exceed width*channels when
    rows are padded, hence the reshape-then-slice rather than a direct reshape.
    """
    encoding = msg.encoding.lower()
    buffer = np.frombuffer(msg.data, dtype=np.uint8)

    if encoding in ("bgr8", "rgb8"):
        rows = buffer.reshape(msg.height, msg.step)
        img = rows[:, : msg.width * 3].reshape(msg.height, msg.width, 3)
        if encoding == "rgb8":
            img = img[:, :, ::-1]
        return np.ascontiguousarray(img)

    if encoding in ("bgra8", "rgba8"):
        rows = buffer.reshape(msg.height, msg.step)
        img = rows[:, : msg.width * 4].reshape(msg.height, msg.width, 4)
        code = cv2.COLOR_BGRA2BGR if encoding == "bgra8" else cv2.COLOR_RGBA2BGR
        return cv2.cvtColor(np.ascontiguousarray(img), code)

    if encoding in ("mono8", "8uc1"):
        rows = buffer.reshape(msg.height, msg.step)
        gray = np.ascontiguousarray(rows[:, : msg.width])
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if encoding in ("yuv422", "yuv422_yuy2", "yuyv"):
        rows = buffer.reshape(msg.height, msg.step)
        yuv = np.ascontiguousarray(rows[:, : msg.width * 2]
                                   ).reshape(msg.height, msg.width, 2)
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_YUY2)

    raise ValueError("unsupported image encoding: %r" % msg.encoding)


def bgr_to_image_msg(frame, header, encoding="bgr8"):
    """BGR ndarray -> sensor_msgs/Image."""
    msg = Image()
    msg.header = header
    msg.height, msg.width = frame.shape[:2]
    msg.encoding = encoding
    msg.is_bigendian = 0
    msg.step = frame.shape[1] * frame.shape[2]
    msg.data = np.ascontiguousarray(frame).tobytes()
    return msg


class GestureNode(Node):

    def __init__(self):
        super().__init__("gesture_recognition_node")

        self.declare_parameter("image_topic", "/image")
        self.declare_parameter("show_window", True)
        self.declare_parameter("publish_annotated", True)
        self.declare_parameter("flip", True)
        self.declare_parameter("max_hands", 2)
        self.declare_parameter("detection_confidence", 0.6)
        self.declare_parameter("tracking_confidence", 0.5)
        self.declare_parameter("model_complexity", 0)
        self.declare_parameter("smoothing_window", 5)
        self.declare_parameter("min_votes", 3)

        topic = self.get_parameter("image_topic").value
        self.show_window = bool(self.get_parameter("show_window").value)
        self.publish_annotated = bool(self.get_parameter("publish_annotated").value)
        self.flip = bool(self.get_parameter("flip").value)

        self.pipeline = HandPipeline(
            max_hands=int(self.get_parameter("max_hands").value),
            detection_confidence=float(self.get_parameter("detection_confidence").value),
            tracking_confidence=float(self.get_parameter("tracking_confidence").value),
            model_complexity=int(self.get_parameter("model_complexity").value),
            smoothing_window=int(self.get_parameter("smoothing_window").value),
            min_votes=int(self.get_parameter("min_votes").value),
        )

        # cam2image publishes best-effort by default.  A RELIABLE subscriber
        # will not match a BEST_EFFORT publisher in ROS2, and the symptom is
        # simply that no messages ever arrive - a classic lab time-sink.
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.subscription = self.create_subscription(
            Image, topic, self.image_callback, sensor_qos)

        self.gesture_pub = self.create_publisher(String, "/gesture", 10)
        self.annotated_pub = self.create_publisher(Image, "/gesture/annotated", 1)

        # --- handover slots -------------------------------------------------
        self._incoming = None              # newest raw message, executor -> worker
        self._incoming_lock = threading.Lock()
        self._incoming_event = threading.Event()

        self._display = None               # newest annotated frame, worker -> main
        self._display_lock = threading.Lock()

        self._running = True
        self._frames_seen = 0
        self._warned_empty = False

        self._worker = threading.Thread(target=self._process_loop, daemon=True)
        self._worker.start()

        self.create_timer(5.0, self._check_camera)
        self.get_logger().info("Listening for images on %s" % topic)

    # -- executor thread ---------------------------------------------------

    def image_callback(self, msg):
        """Keep only the newest frame; drop anything the worker did not reach."""
        with self._incoming_lock:
            self._incoming = msg
        self._incoming_event.set()
        self._frames_seen += 1

    def _check_camera(self):
        if self._frames_seen == 0 and not self._warned_empty:
            self._warned_empty = True
            self.get_logger().warn(
                "No images received yet. Check that the camera is publishing:\n"
                "  ros2 topic list            # is /image there?\n"
                "  ros2 run image_tools cam2image\n"
                "In VMware also check VM > Removable Devices > your webcam, "
                "and that 'ls /dev/video*' shows a device.")

    # -- worker thread -----------------------------------------------------

    def _process_loop(self):
        while self._running:
            if not self._incoming_event.wait(timeout=0.5):
                continue
            self._incoming_event.clear()

            with self._incoming_lock:
                msg = self._incoming
                self._incoming = None
            if msg is None:
                continue

            try:
                frame = image_msg_to_bgr(msg)
            except Exception as exc:                       # noqa: BLE001
                self.get_logger().error("Image conversion failed: %s" % exc)
                continue

            if self.flip:
                # Mirror so the preview behaves like a mirror. MediaPipe's
                # Left/Right labels assume this selfie view, so flipping also
                # makes the handedness label match the user's actual hand.
                frame = cv2.flip(frame, 1)

            annotated, gestures = self.pipeline.process(frame)

            out = String()
            out.data = self.pipeline.primary(gestures)
            self.gesture_pub.publish(out)

            if self.publish_annotated:
                self.annotated_pub.publish(bgr_to_image_msg(annotated, msg.header))

            if self.show_window:
                with self._display_lock:
                    self._display = annotated

    # -- main thread -------------------------------------------------------

    def display_loop(self):
        """Pump the OpenCV window. Returns when the user presses q or ESC."""
        window = "Gesture Recognition (q to quit)"
        while self._running and rclpy.ok():
            with self._display_lock:
                frame = self._display
                self._display = None

            if frame is not None:
                cv2.imshow(window, frame)

            # waitKey also drives the GUI event loop, so it must run even when
            # there is no new frame - otherwise the window freezes.
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), 27):
                break

        cv2.destroyAllWindows()

    def shutdown(self):
        self._running = False
        self._incoming_event.set()
        if self._worker.is_alive():
            self._worker.join(timeout=2.0)
        self.pipeline.close()


def main(args=None):
    rclpy.init(args=args)
    node = GestureNode()

    # rclpy spins on a background thread so the main thread is free for the
    # OpenCV window; on Linux, GUI calls off the main thread are unreliable.
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        if node.show_window:
            node.display_loop()
        else:
            spin_thread.join()
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
