"""Lab 2, section 2.1 - the same node written object-oriented instead of procedural.

`hello.py` creates a bare `Node` object (procedural style, POP).  Here we instead
*inherit* from `Node`, which is what every real ROS2 program does: the class then
owns its publishers, timers and state.

Li Si is the village writer.  He publishes one line of his novel per second on
the `/novel` topic; any number of readers may subscribe to it.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class WriterNode(Node):
    """Publishes a line of text on /novel once per second."""

    def __init__(self, name):
        # super().__init__(name) runs the parent Node constructor, which is what
        # actually registers this node with the ROS graph under `name`.
        super().__init__(name)
        self.get_logger().info("Hello everyone, I am %s, I am a student!" % name)

        # create_publisher(msg_type, topic_name, queue_depth)
        self.publisher_ = self.create_publisher(String, "novel", 10)
        self.timer = self.create_timer(1.0, self.write_line)
        self.i = 1

    def write_line(self):
        msg = String()
        # std_msgs/msg/String holds exactly one field: `string data`.
        # Check it yourself with:  ros2 interface show std_msgs/msg/String
        msg.data = "Episode %d: Traveler %d encounters an old friend" % (self.i, self.i)
        self.publisher_.publish(msg)
        self.get_logger().info('Published: "%s"' % msg.data)
        self.i += 1


def main(args=None):
    rclpy.init(args=args)
    node = WriterNode("li_si")
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
