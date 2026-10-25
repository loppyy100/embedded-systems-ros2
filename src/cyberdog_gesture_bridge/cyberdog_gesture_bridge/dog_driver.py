"""CyberDog locomotion driver - named actions on top of the raw LCM interface.

This wraps the `Robot_Ctrl` pattern from `basic/basic_motion/main.py` so the
rest of the code can say `dog.sit()` instead of filling in `mode`, `gait_id`,
`life_count` and a duration by hand.

Two details from the vendor sample that are easy to get wrong and are handled
here:

  * **life_count** - the control board only acts on a command when this counter
    *changes*.  Resending an identical message does nothing.  `send()`
    increments it for you.
  * **heartbeat** - the board expects traffic at roughly 10 Hz and will fall
    back to a safe state if it stops hearing from us.  A background thread
    republishes the last command to maintain it.

`dry_run=True` makes every action log what it would have sent and return
immediately, with no LCM traffic and no `lcm` import.  That is the default in
the bridge node, so the gesture pipeline can be demonstrated and marked with no
robot present.
"""

import time
from threading import Thread, Lock

from .robot_control_cmd_lcmt import robot_control_cmd_lcmt
from .robot_control_response_lcmt import robot_control_response_lcmt

# Mode / gait ids, from the CyberDog locomotion interface documentation:
# https://miroboticslab.github.io/blogs/#/en/cyberdog_loco_en
MODE_PURE_DAMPER = 7        # limp / lie down safely
MODE_LOCOMOTION = 11        # walking gaits, driven by vel_des
MODE_RECOVERY_STAND = 12    # stand up from any posture
MODE_POSE = 21              # body pose interpolation (rpy_des / pos_des)
MODE_MOTION = 62            # a preinstalled motion sequence, chosen by gait_id
MODE_TWOLEG_STAND = 64

# gait_id values for MODE_MOTION, from MOTION_ID in the locomotion repo's
# common/include/control_flags_release.hpp
MOTION_SHAKE_HAND = 2       # "hi five right": sit, offer the right paw, shake (~7.5 s)
MOTION_SIT_DOWN = 3         # sit and wiggle; stays sitting until the next command

GAIT_TROT_MEDIUM = 3
GAIT_TROT_FAST = 10
GAIT_TROT_SLOW = 27
GAIT_AUTO = 26              # automatic frequency selection
GAIT_POSE_HEIGHT = 5        # gait_id used with MODE_POSE for body height

DEFAULT_STEP_HEIGHT = [0.06, 0.06]


class DogDriver:

    def __init__(self, dry_run=True, logger=None):
        self.dry_run = dry_run
        self.log = logger or (lambda msg: print(msg))
        # Nothing for the heartbeat to repeat until the first real command: an
        # all-zero message is mode 0, which the board treats as an e-stop.
        self.cmd_msg = None
        self.rec_msg = robot_control_response_lcmt()
        self._send_lock = Lock()
        self._delay_cnt = 0
        self._mode_ok = 0
        self._gait_ok = 0
        self._running = True
        self._life = 0
        self._lc_send = None
        self._lc_recv = None
        self._threads = []

        if not self.dry_run:
            self._connect()

    # -- transport ---------------------------------------------------------

    def _connect(self):
        # Imported lazily: `lcm` is not pip-installable on every machine, and
        # dry-run users should never need it.
        import lcm

        self._lc_recv = lcm.LCM("udpm://239.255.76.67:7670?ttl=255")
        self._lc_send = lcm.LCM("udpm://239.255.76.67:7671?ttl=255")
        self._lc_recv.subscribe("robot_control_response", self._on_response)

        self._threads = [
            Thread(target=self._recv_loop, daemon=True),
            Thread(target=self._heartbeat_loop, daemon=True),
        ]
        for thread in self._threads:
            thread.start()
        self.log("LCM connected (multicast 239.255.76.67)")

    def _on_response(self, channel, data):
        self.rec_msg = robot_control_response_lcmt().decode(data)
        # order_process_bar is a 0-100 progress value; >=95 means the board has
        # essentially finished the commanded motion.
        if self.rec_msg.order_process_bar >= 95:
            self._mode_ok = self.rec_msg.mode
            self._gait_ok = self.rec_msg.gait_id
        else:
            self._mode_ok = 0

    def _recv_loop(self):
        while self._running:
            self._lc_recv.handle()
            time.sleep(0.002)

    def _heartbeat_loop(self):
        while self._running:
            with self._send_lock:
                if self.cmd_msg is not None and self._delay_cnt > 20:   # ~10 Hz at a 5 ms tick
                    self._lc_send.publish("robot_control_cmd", self.cmd_msg.encode())
                    self._delay_cnt = 0
                self._delay_cnt += 1
            time.sleep(0.005)

    # -- command plumbing --------------------------------------------------

    def send(self, msg, label=""):
        """Publish a command. Increments life_count so the board acts on it."""
        self._life = (self._life + 1) % 128   # life_count is an int8_t
        msg.life_count = self._life

        if self.dry_run:
            self.log("[dry-run] %s  mode=%d gait=%d vel=%s duration=%d"
                     % (label or "cmd", msg.mode, msg.gait_id,
                        [round(v, 2) for v in msg.vel_des], msg.duration))
            return

        with self._send_lock:
            self.cmd_msg = msg
            self._delay_cnt = 50       # force an immediate publish
            self._lc_send.publish("robot_control_cmd", msg.encode())

    def wait_finish(self, mode, gait_id, timeout=10.0):
        """Block until the board reports the motion done, or timeout."""
        if self.dry_run:
            return True
        deadline = time.time() + timeout
        while self._running and time.time() < deadline:
            if self._mode_ok == mode and self._gait_ok == gait_id:
                return True
            time.sleep(0.005)
        return False

    def _base_msg(self, mode, gait_id=0, duration=0):
        msg = robot_control_cmd_lcmt()
        msg.mode = mode
        msg.gait_id = gait_id
        msg.contact = 15          # 0b1111 - all four feet on the ground
        msg.duration = duration
        msg.step_height = list(DEFAULT_STEP_HEIGHT)
        return msg

    # -- named actions -----------------------------------------------------

    def stand(self):
        """Recovery stand - works from any posture. Needs >5 s per the vendor docs."""
        self.send(self._base_msg(MODE_RECOVERY_STAND, 0, 5000), "stand")
        self.wait_finish(MODE_RECOVERY_STAND, 0)

    def lie_down(self):
        """Pure damper: motors go limp and the dog settles. The safe stop."""
        self.send(self._base_msg(MODE_PURE_DAMPER, 0, 4000), "lie_down")

    def shake_hand(self):
        self.send(self._base_msg(MODE_MOTION, MOTION_SHAKE_HAND, 0), "shake_hand")
        # The simulator reports motions as pose control rather than mode 62,
        # so the timeout - the motion's own length - is what normally ends this.
        self.wait_finish(MODE_MOTION, MOTION_SHAKE_HAND, timeout=8.0)

    def sit(self):
        """Stop and sit down. The next command stands the dog back up first."""
        self.halt()
        time.sleep(0.5)
        self.send(self._base_msg(MODE_MOTION, MOTION_SIT_DOWN, 0), "sit")
        self.wait_finish(MODE_MOTION, MOTION_SIT_DOWN, timeout=3.0)

    def two_leg_stand(self):
        if not self.dry_run and self.rec_msg.mode != MODE_RECOVERY_STAND:
            # Only reachable from a plain stand - mid-walk the board answers
            # "Cannot transition from 11 to 64".
            self.stand()
        self.send(self._base_msg(MODE_TWOLEG_STAND, 0, 0), "two_leg_stand")
        self.wait_finish(MODE_TWOLEG_STAND, 0)

    def walk(self, vx=0.0, vy=0.0, vyaw=0.0, gait=GAIT_AUTO):
        """Continuous locomotion.

        duration=0 means "hold until the next command", which is also what lets
        a walk command interrupt an in-progress pose interpolation.
        """
        if not self.dry_run and self.rec_msg.mode not in (MODE_LOCOMOTION, MODE_RECOVERY_STAND):
            # The board only accepts locomotion from a standing posture
            # ("Cannot transition from 0 to 11"), so get up first.
            self.stand()
        msg = self._base_msg(MODE_LOCOMOTION, gait, 0)
        msg.vel_des = [float(vx), float(vy), float(vyaw)]
        self.send(msg, "walk")

    def halt(self):
        """Stop walking but stay standing."""
        msg = self._base_msg(MODE_LOCOMOTION, GAIT_AUTO, 0)
        msg.vel_des = [0.0, 0.0, 0.0]
        self.send(msg, "halt")

    def set_height(self, height=0.25, duration=400):
        msg = self._base_msg(MODE_POSE, GAIT_POSE_HEIGHT, duration)
        msg.pos_des = [0.0, 0.0, float(height)]
        self.send(msg, "set_height")

    def nod(self, pitch=0.3, duration=500):
        """Tilt the body. rpy_des is roll/pitch/yaw in radians."""
        msg = self._base_msg(MODE_POSE, 0, duration)
        msg.rpy_des = [0.0, float(pitch), 0.0]
        self.send(msg, "nod")

    def dance(self):
        """A short celebration: look up, look down, then spin in place."""
        self.nod(0.3, 500)
        time.sleep(0.5)
        self.nod(-0.3, 300)
        time.sleep(0.3)
        self.nod(0.0, 300)
        time.sleep(0.3)
        self.walk(0.0, 0.0, 0.8)
        time.sleep(2.0)
        self.halt()

    # -- lifecycle ---------------------------------------------------------

    def close(self):
        self._running = False
        for thread in self._threads:
            thread.join(timeout=1.0)
