"""Lab 1, question 1 - customise the turtle's movement trajectory.

The lab's `patrol.py` publishes one Twist per second and hard-codes a
forward/turn pattern.  Two problems with that:

  * turtlesim stops the turtle if it does not receive a `cmd_vel` within about
    one second, so a 1 Hz publisher makes the turtle stutter;
  * `angular.z` is a *velocity* (rad/s), not an angle.  Publishing `1.57` for
    one second happens to turn 90 degrees only because the duration is 1 s.
    Change the period and the shape silently breaks.

This node fixes both.  A trajectory is a list of `(linear, angular, duration)`
segments; a 20 Hz timer walks through them using the wall clock, so a "turn 90
degrees" segment is expressed as `angle / speed` seconds and stays correct
whatever the publish rate is.

Usage:
    ros2 run turtlesim turtlesim_node
    ros2 run turtle_patrol patrol                              # default: star
    ros2 run turtle_patrol patrol --ros-args -p shape:=flower
    ros2 run turtle_patrol patrol --ros-args -p shape:=square -p size:=3.0
    ros2 run turtle_patrol patrol --ros-args -p shape:=spiral -p loop:=true

Shapes: square, triangle, star, hexagon, flower, spiral.
"""

import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from turtlesim.srv import SetPen, TeleportAbsolute

PUBLISH_HZ = 20.0

# Pen colour per shape, purely cosmetic (r, g, b).
PEN_COLOURS = {
    "square": (255, 120, 0),
    "triangle": (0, 200, 120),
    "star": (255, 40, 90),
    "hexagon": (120, 100, 255),
    "flower": (255, 0, 200),
    "spiral": (0, 190, 255),
}


def polygon(sides, size, speed=2.0, turn_speed=1.5):
    """Regular polygon: drive an edge, turn the exterior angle, repeat."""
    edge_time = size / speed
    turn_angle = 2.0 * math.pi / sides
    turn_time = turn_angle / turn_speed
    segments = []
    for _ in range(sides):
        segments.append((speed, 0.0, edge_time))
        segments.append((0.0, turn_speed, turn_time))
    return segments


def star(points=5, size=3.0, speed=2.5, turn_speed=2.5):
    """Five-pointed star: the turn between edges is the *exterior* angle of the
    {5/2} star polygon, i.e. 144 degrees, not 72."""
    edge_time = size / speed
    turn_angle = 2.0 * math.pi * 2 / points   # 144 deg for points=5
    turn_time = turn_angle / turn_speed
    segments = []
    for _ in range(points):
        segments.append((speed, 0.0, edge_time))
        segments.append((0.0, turn_speed, turn_time))
    return segments


def flower(petals=8, size=2.0, speed=3.2):
    """Trace a full circle (a petal), rotate by 360/petals, repeat.

    Driving v and w together traces a circle of radius v/w, so to close one
    petal we hold both for exactly one revolution: 2*pi / w seconds.
    """
    w = speed / size                      # radius = v/w = size
    petal_time = 2.0 * math.pi / w
    gap_turn_speed = 2.0
    gap_time = (2.0 * math.pi / petals) / gap_turn_speed
    segments = []
    for _ in range(petals):
        segments.append((speed, w, petal_time))
        segments.append((0.0, gap_turn_speed, gap_time))
    return segments


def spiral(turns=5, growth=0.35, w=2.0, step=0.1):
    """Archimedean-ish spiral: constant angular rate, linear speed ramping up.

    Radius = v/w, so growing v at a constant rate grows the radius the same way.
    Built from many short segments because v changes continuously.
    """
    segments = []
    total_time = turns * (2.0 * math.pi / w)
    t = 0.0
    while t < total_time:
        v = 0.4 + growth * t
        segments.append((v, w, step))
        t += step
    return segments


def build_trajectory(shape, size):
    if shape == "square":
        return polygon(4, size)
    if shape == "triangle":
        return polygon(3, size)
    if shape == "hexagon":
        return polygon(6, size * 0.6)
    if shape == "star":
        return star(5, size)
    if shape == "flower":
        return flower(8, size * 0.6)
    if shape == "spiral":
        return spiral()
    raise ValueError(
        "unknown shape %r - choose from square, triangle, hexagon, star, "
        "flower, spiral" % shape
    )


class Patrol(Node):

    def __init__(self):
        super().__init__("patrol_node")

        self.declare_parameter("shape", "star")
        self.declare_parameter("size", 2.5)
        self.declare_parameter("loop", False)
        self.declare_parameter("turtle", "turtle1")
        self.declare_parameter("recentre", True)

        self.shape = self.get_parameter("shape").value
        self.size = float(self.get_parameter("size").value)
        self.loop = bool(self.get_parameter("loop").value)
        self.recentre = bool(self.get_parameter("recentre").value)
        turtle = self.get_parameter("turtle").value

        self.segments = build_trajectory(self.shape, self.size)
        self.index = 0
        self.segment_started = None
        self.finished = False

        self.publisher_ = self.create_publisher(Twist, "/%s/cmd_vel" % turtle, 10)
        self.pen_client = self.create_client(SetPen, "/%s/set_pen" % turtle)
        self.teleport_client = self.create_client(
            TeleportAbsolute, "/%s/teleport_absolute" % turtle)

        self.get_logger().info(
            "Drawing '%s' - %d segments, size %.1f, loop=%s"
            % (self.shape, len(self.segments), self.size, self.loop))

        self._prepare_canvas()
        self.timer = self.create_timer(1.0 / PUBLISH_HZ, self.tick)

    # -- one-shot service calls -------------------------------------------

    def _prepare_canvas(self):
        """Centre the turtle and set a pen colour, if turtlesim is up.

        These are *services*, not topics: a request/response call rather than a
        fire-and-forget stream.  We do not block on the result - if turtlesim
        is not running yet the drawing still works, it just starts wherever the
        turtle happens to be.
        """
        if self.recentre and self.teleport_client.wait_for_service(timeout_sec=2.0):
            req = TeleportAbsolute.Request()
            req.x, req.y, req.theta = 5.544445, 5.544445, 0.0
            self.teleport_client.call_async(req)

        if self.pen_client.wait_for_service(timeout_sec=2.0):
            r, g, b = PEN_COLOURS.get(self.shape, (255, 255, 255))
            req = SetPen.Request()
            req.r, req.g, req.b = r, g, b
            req.width = 3
            req.off = 0
            self.pen_client.call_async(req)

    # -- the 20 Hz control loop -------------------------------------------

    def tick(self):
        if self.finished:
            return

        now = self.get_clock().now().nanoseconds / 1e9
        if self.segment_started is None:
            self.segment_started = now

        linear, angular, duration = self.segments[self.index]

        if now - self.segment_started >= duration:
            self.index += 1
            self.segment_started = now
            if self.index >= len(self.segments):
                if self.loop:
                    self.index = 0
                else:
                    self.stop()
                    return
            linear, angular, duration = self.segments[self.index]

        msg = Twist()
        msg.linear.x = float(linear)
        msg.angular.z = float(angular)
        self.publisher_.publish(msg)

    def stop(self):
        self.finished = True
        self.publisher_.publish(Twist())   # all-zero Twist = halt
        self.get_logger().info("Trajectory complete. Ctrl+C to exit.")


def main(args=None):
    rclpy.init(args=args)
    node = Patrol()
    print("The little turtle starts its patrol!")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
