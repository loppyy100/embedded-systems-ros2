"""Turn recognised gestures into CyberDog motions.

    ros2 run cyberdog_gesture_bridge bridge                                 # dry run
    ros2 run cyberdog_gesture_bridge bridge --ros-args -p dry_run:=false    # real dog

Subscribes to `/gesture` (std_msgs/String) from the gesture recogniser and
sends the mapped locomotion command over LCM.

--------------------------------------------------------------------------
Safety, and why the debounce matters
--------------------------------------------------------------------------
The recogniser publishes at camera rate, so a held gesture arrives ~30 times a
second.  Firing a motion on every message would flood the control board and
make the dog jitter.  Two guards prevent that:

  * **hold frames** - a gesture must be reported `hold_frames` times in a row
    before it fires.  This is on top of the smoothing already done in the
    recogniser: that one removes per-frame flicker, this one makes sure the
    user actually *meant* it.
  * **cooldown** - once an action fires, nothing new fires for `cooldown`
    seconds, so a motion gets to finish before the next one interrupts it.

`dry_run` defaults to **true**: the node logs the command it would send and
touches no hardware.  Set it to false only with the dog on the floor, clear of
obstacles, and with someone holding the remote.
"""

import threading
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from .dog_driver import DogDriver

# gesture label -> (driver method name, args, human description)
GESTURE_ACTIONS = {
    "OK":         ("stand",         (),              "stand up"),
    "THUMB_UP":   ("two_leg_stand", (),              "stand on two legs"),
    "THUMB_DOWN": ("lie_down",      (),              "lie down (pure damper)"),
    "VICTORY":    ("shake_hand",    (),              "shake hands"),
    "OPEN_PALM":  ("halt",          (),              "stop and stand still"),
    "FIST":       ("walk",          (0.3, 0.0, 0.0), "walk forward"),
    "ROCK":       ("walk",          (0.0, 0.0, 0.8), "turn in place"),
    "HEART":      ("dance",         (),              "dance"),
    "POINT":      ("nod",           (0.3,),          "look up"),
}


class GestureBridge(Node):

    def __init__(self):
        super().__init__("cyberdog_gesture_bridge")

        self.declare_parameter("dry_run", True)
        self.declare_parameter("gesture_topic", "/gesture")
        self.declare_parameter("hold_frames", 8)
        self.declare_parameter("cooldown", 3.0)
        self.declare_parameter("enabled", True)

        self.dry_run = bool(self.get_parameter("dry_run").value)
        self.hold_frames = int(self.get_parameter("hold_frames").value)
        self.cooldown = float(self.get_parameter("cooldown").value)
        self.enabled = bool(self.get_parameter("enabled").value)
        topic = self.get_parameter("gesture_topic").value

        self.driver = DogDriver(dry_run=self.dry_run,
                                logger=self.get_logger().info)

        self._candidate = None
        self._streak = 0
        self._last_fired = 0.0
        self._last_action = None
        self._busy = threading.Lock()

        self.create_subscription(String, topic, self.on_gesture, 10)

        if self.dry_run:
            self.get_logger().info(
                "DRY RUN - commands will be logged, not sent. "
                "Pass -p dry_run:=false when the robot is connected.")
        else:
            self.get_logger().warn(
                "LIVE MODE - the robot will move. Keep the area clear.")
        self.get_logger().info("Mapping: " + ", ".join(
            "%s->%s" % (g, spec[2]) for g, spec in sorted(GESTURE_ACTIONS.items())))

    def on_gesture(self, msg):
        gesture = msg.data

        if gesture != self._candidate:
            self._candidate = gesture
            self._streak = 1
            return
        self._streak += 1

        if not self.enabled or gesture not in GESTURE_ACTIONS:
            return
        if self._streak < self.hold_frames:
            return
        if time.time() - self._last_fired < self.cooldown:
            return
        # Require the user to drop the gesture before it can fire again,
        # otherwise holding a pose retriggers on every cooldown expiry.
        if gesture == self._last_action:
            return

        self._fire(gesture)

    def _fire(self, gesture):
        method_name, args, description = GESTURE_ACTIONS[gesture]
        self._last_fired = time.time()
        self._last_action = gesture
        self.get_logger().info("Gesture %s -> %s" % (gesture, description))

        # Some actions (stand, dance) block for seconds while they wait on the
        # control board. Running them on a worker keeps the subscription
        # callback responsive; the lock drops any request that arrives while
        # one is still running.
        def run():
            if not self._busy.acquire(blocking=False):
                self.get_logger().warn("Still executing a motion - ignoring %s"
                                       % gesture)
                return
            try:
                getattr(self.driver, method_name)(*args)
            except Exception as exc:                        # noqa: BLE001
                self.get_logger().error("Action %s failed: %s" % (method_name, exc))
            finally:
                self._busy.release()

        threading.Thread(target=run, daemon=True).start()

    def shutdown(self):
        try:
            if not self.dry_run:
                # Leave the robot in a safe posture rather than mid-stride.
                self.driver.halt()
                time.sleep(0.2)
                self.driver.lie_down()
        finally:
            self.driver.close()


def main(args=None):
    rclpy.init(args=args)
    node = GestureBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
