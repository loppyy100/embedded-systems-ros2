"""A stand-in for `ros2 run image_tools cam2image`.

`cam2image` can only read a real camera device.  That is a problem twice over
in this course: VMware's USB passthrough often refuses to hand the webcam to
the guest, and a remote GPU server has no webcam at all.

This node publishes the same `sensor_msgs/Image` on the same `/image` topic,
but the source can be a camera index, a video file, or a generated test
pattern - so the rest of the pipeline can be developed and demonstrated with
no camera present.

    ros2 run gesture_recognition image_source                          # camera 0
    ros2 run gesture_recognition image_source --ros-args -p source:=clip.mp4
    ros2 run gesture_recognition image_source --ros-args -p source:=pattern

Parameters:
    source        str    "0" | "/dev/video2" | "clip.mp4" | "pattern"
    topic         str    default "/image"
    fps           float  default 30.0
    width/height  int    default 640/480
    loop          bool   default True   - rewind video files at EOF
"""

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image


def make_pattern(width, height, tick):
    """A moving colour-bar and circle pattern, so you can see frames flowing."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    bars = 8
    for i in range(bars):
        x0 = int(width * i / bars)
        x1 = int(width * (i + 1) / bars)
        shade = int(255 * (i + 1) / bars)
        frame[:, x0:x1] = (shade, 255 - shade, (shade * 2) % 256)
    x = int(width / 2 + (width / 3) * np.sin(tick / 12.0))
    y = int(height / 2 + (height / 4) * np.cos(tick / 9.0))
    cv2.circle(frame, (x, y), 40, (255, 255, 255), -1)
    cv2.putText(frame, "test pattern %d" % tick, (16, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    return frame


class ImageSource(Node):

    def __init__(self):
        super().__init__("image_source")

        self.declare_parameter("source", "0")
        self.declare_parameter("topic", "/image")
        self.declare_parameter("fps", 30.0)
        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)
        self.declare_parameter("loop", True)

        source = str(self.get_parameter("source").value)
        topic = self.get_parameter("topic").value
        fps = float(self.get_parameter("fps").value)
        self.width = int(self.get_parameter("width").value)
        self.height = int(self.get_parameter("height").value)
        self.loop = bool(self.get_parameter("loop").value)
        self.tick = 0

        # Match cam2image's QoS so existing subscribers connect unchanged.
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.publisher_ = self.create_publisher(Image, topic, qos)

        self.cap = None
        if source == "pattern":
            self.get_logger().info("Publishing a generated test pattern on %s" % topic)
        else:
            try:
                index = int(source)
            except ValueError:
                index = source
            self.cap = cv2.VideoCapture(index)
            if not self.cap.isOpened():
                self.get_logger().error(
                    "Cannot open source %r - falling back to the test pattern. "
                    "Check 'ls /dev/video*' for available cameras." % source)
                self.cap = None
            else:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.get_logger().info("Publishing %r on %s" % (source, topic))

        self.create_timer(1.0 / max(fps, 1.0), self.publish_frame)

    def publish_frame(self):
        if self.cap is None:
            frame = make_pattern(self.width, self.height, self.tick)
        else:
            ok, frame = self.cap.read()
            if not ok:
                if self.loop:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = self.cap.read()
                if not ok:
                    self.get_logger().warn("Source exhausted.", once=True)
                    return

        self.tick += 1

        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera"
        msg.height, msg.width = frame.shape[:2]
        msg.encoding = "bgr8"
        msg.is_bigendian = 0
        msg.step = frame.shape[1] * 3
        msg.data = np.ascontiguousarray(frame).tobytes()
        self.publisher_.publish(msg)

    def destroy_node(self):
        if self.cap is not None:
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ImageSource()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
