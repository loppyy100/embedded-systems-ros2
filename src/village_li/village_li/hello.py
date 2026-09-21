"""Lab 2, section 2.4 - the smallest possible ROS2 node.

The five steps every ROS2 Python node follows:
  1. import the libraries
  2. initialise the client library (rclpy.init)
  3. create a node object
  4. spin the node so it stays alive
  5. shut the client library down
"""

import rclpy
from rclpy.node import Node


def main(args=None):
    rclpy.init(args=args)

    # Node names may not contain spaces or punctuation such as '!'.
    hello_node = Node("hello1")
    hello_node.get_logger().info("hello!")

    try:
        rclpy.spin(hello_node)          # blocks until Ctrl+C
    except KeyboardInterrupt:
        pass
    finally:
        hello_node.destroy_node()
        # In Galactic, shutdown() raises if rclpy is already down.
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
