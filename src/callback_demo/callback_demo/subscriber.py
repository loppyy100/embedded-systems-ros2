"""Lab 2, section 2.6 - subscriber with a message callback.

The subscriber never polls.  `rclpy.spin()` parks the process; when a message
lands on `/topic`, ROS2 wakes it up and runs `listener_callback` with that
message as the argument.

For the publisher and subscriber to connect, three things must match exactly:
  * the topic name        ("topic")
  * the message type      (std_msgs/msg/String)
  * compatible QoS        (both use the default here)
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__("minimal_subscriber")
        self.subscription = self.create_subscription(
            String,                  # message type
            "topic",                 # topic name
            self.listener_callback,  # the callback ROS2 will invoke
            10,                      # queue depth
        )

    def listener_callback(self, msg):
        # std_msgs/msg/String has a single field `data`, hence msg.data.
        self.get_logger().info('Received: "%s"' % msg.data)


def main(args=None):
    rclpy.init(args=args)
    subscriber = MinimalSubscriber()
    try:
        rclpy.spin(subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        subscriber.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
