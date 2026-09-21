"""Lab 2, section 2.6 - timer callback publisher.

A *callback* is a function you hand to ROS2 and never call yourself.  ROS2 calls
it for you when something happens.  Here the "something" is a timer expiring;
in `subscriber.py` it is a message arriving.

Run alongside `ros2 run callback_demo subscriber`, then inspect the wiring with
`rqt_graph` or `ros2 topic echo /topic`.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class MinimalPublisher(Node):

    def __init__(self):
        super().__init__("minimal_publisher")

        # 10 is the queue depth: how many outgoing messages ROS2 will buffer for
        # a slow subscriber before it starts dropping the oldest ones.
        self.publisher_ = self.create_publisher(String, "topic", 10)

        timer_period = 1.0  # seconds -> publish once per second
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        """Called by ROS2 every `timer_period` seconds - never called directly."""
        msg = String()
        msg.data = "Hello World: %d" % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Published: "%s"' % msg.data)
        self.i += 1


def main(args=None):
    rclpy.init(args=args)
    publisher = MinimalPublisher()
    try:
        rclpy.spin(publisher)
    except KeyboardInterrupt:
        pass
    finally:
        publisher.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
