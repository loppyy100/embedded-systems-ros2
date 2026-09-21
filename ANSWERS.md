# Lab question answers

---

## Lab 1, Question 1 — Customise the little turtle's movement trajectory

**Code:** `src/turtle_patrol/turtle_patrol/patrol_node.py`
**Run:** `ros2 run turtle_patrol patrol --ros-args -p shape:=star`

### The bug in the guide's `patrol.py`

The example publishes one `Twist` per second and uses `angular.z = 1.57` with
the comment *"turn left 90 degrees (approx 1.57 radians)"*. That comment is
the mistake. `angular.z` is an angular **velocity** in rad/s, not an angle.

Publishing 1.57 rad/s turns 90° only because the timer period happens to be
exactly 1.0 s, and `1.57 rad/s × 1 s = 1.57 rad`. Change `timer_period` to 0.5
and every corner silently becomes 45°.

There is a second problem: turtlesim stops the turtle if it has not received a
`cmd_vel` for about one second, so a 1 Hz publisher makes the turtle move in
visible stutters rather than smoothly.

### The fix

A trajectory is a list of `(linear, angular, duration)` segments, published at
20 Hz. A turn is expressed as the angle divided by the turn rate:

```python
turn_angle = 2 * math.pi / sides       # exterior angle of the polygon
turn_time  = turn_angle / turn_speed   # seconds needed at that rate
```

Now the shape stays correct whatever the publish rate is, and the motion is
smooth. Segment timing uses the node clock rather than counting timer ticks, so
a dropped tick does not distort the shape.

### Shapes implemented

| Shape | Construction |
|---|---|
| `square`, `triangle`, `hexagon` | drive an edge, turn the exterior angle (360/n), repeat |
| `star` | 5 edges with a **144°** turn between them — the exterior angle of the {5/2} star polygon, not 72° |
| `flower` | hold `v` and `w` together for one full revolution (traces a circle of radius `v/w`), rotate 360/petals, repeat |
| `spiral` | constant `w`, `v` ramping up — since radius = `v/w`, growing `v` linearly grows the radius linearly |

Verified by integrating the segments offline: every closed shape returns to
within 0.04 turtlesim units of its starting point with a net turn of ~360°.

The node also uses two turtlesim **services** — `/turtle1/teleport_absolute`
to recentre before drawing and `/turtle1/set_pen` to colour each shape — which
demonstrates the request/response side of question 2 alongside the topic side.

---

## Lab 1, Question 2 — How does information communicate in ROS2?

**Code:** `src/callback_demo/` (publisher + subscriber)
**Run:** `ros2 run callback_demo publisher` and `ros2 run callback_demo subscriber`

### The four mechanisms

ROS2 gives nodes four ways to exchange information. They differ in *who
initiates* and *whether there is a reply*:

| Mechanism | Shape | Reply? | Use it for |
|---|---|---|---|
| **Topic** | publish / subscribe, many-to-many | no | continuous data streams — camera frames, velocities, sensor readings |
| **Service** | request / response, one-to-one | yes, once | short, immediate operations — "set the pen colour", "reset the simulation" |
| **Action** | goal / feedback / result | yes, repeatedly | long-running tasks you want progress on and can cancel — "navigate to this point" |
| **Parameter** | named values on a node | — | configuration — resolution, thresholds, topic names |

The central design point is that all of these are **anonymous**. A publisher
does not know or care who is listening; a subscriber does not know who sent the
message. Nodes are coupled only by the topic *name* and the message *type*,
which is what lets you swap `cam2image` for our own `image_source` node without
changing a line of the recogniser.

### Why topics are the right tool for images

A camera produces frames whether or not anyone wants them, there may be zero or
several consumers (the recogniser, `rqt`, a recorder), and a frame that arrives
late is worthless. That is exactly the publish/subscribe model, which is why
`/image` is a topic and not a service.

### What must match for two nodes to connect

1. **Topic name** — `/topic` and `/Topic` are different topics.
2. **Message type** — publishing `String` and subscribing `Int32` on the same
   name simply never connects.
3. **Compatible QoS** — the one that catches people out. A *reliable*
   subscriber will not match a *best-effort* publisher, and ROS2 reports no
   error: messages just never arrive. `cam2image` publishes best-effort, which
   is why `gesture_node.py` explicitly subscribes with a best-effort profile.

Check all three with:

```bash
ros2 topic list -t             # names with their types
ros2 topic info /image --verbose   # QoS of every publisher and subscriber
ros2 topic echo /gesture       # see the data
rqt_graph                      # see who is connected to whom
```

### The demonstration

`callback_demo` publishes `Hello World: N` on `/topic` once per second and
receives it in a second node. Neither node names the other anywhere — they
find each other purely through the name `/topic` and the type
`std_msgs/msg/String`. `rqt_graph` draws the resulting edge.

The word *callback* is the key idea in both nodes. You never call
`timer_callback` or `listener_callback` yourself. You hand the function to
ROS2, `rclpy.spin()` parks the process, and ROS2 invokes your function when the
timer expires or a message lands. The program is driven by events, not by a
loop you wrote.

---

## Experiment 2 — Design notes

**Code:** `src/gesture_recognition/`

The guide's §4 asks for the three-layer architecture, and lists several
"existing issues" and "improvement plans". Here is what was actually done about
each.

### Layers

| Layer | File | Responsibility |
|---|---|---|
| Perception | `pipeline.py` | frames → MediaPipe Hands → 21 landmarks per hand |
| Logic | `gesture_lib.py` | landmarks → finger states → gesture label → smoothing |
| Presentation | `overlay.py` | skeleton, animated effects, HUD |

`gesture_lib.py` imports nothing but `math` and `collections`, which is what
makes the whole classifier unit-testable with no camera.

### Issue 1 — "using only two joints may cause false detection"

The guide detects an extended finger with `landmarks[tip].y < landmarks[pip].y`
and proposes `tip.y < pip.y < mcp.y` as the fix. That is better, but it still
compares raw image-space Y, so it only works while the hand points upward.
**Tilt the hand sideways and every finger reads as bent.**

Two rotation-invariant tests are used instead:

1. **Straightness** — the interior angle at the PIP joint (MCP–PIP–TIP). A
   straight finger is near 180°; the angle does not care which way the hand
   faces.
2. **Reach** — the tip must be further from the wrist than the PIP is. This
   catches the case a pure angle test misses: a finger that is straight but
   folded down at the knuckle.

A finger counts as extended only if both hold.

### Issue 2 — thumb handling

The guide switches on left/right handedness to pick an X-axis direction. That
breaks when the palm faces away, or when the image is mirrored — and the
recogniser *does* mirror the frame for a selfie view.

Instead: the thumb is extended if it is straight at the IP joint **and** its tip
has moved away from the pinky knuckle relative to its own IP joint. Tucking the
thumb across the palm always moves the tip *towards* the pinky, on either hand,
at any orientation. No handedness label is needed at all.

### Issue 3 — dynamic thresholds

The guide applies hand-size scaling only to the OK gesture. Here *every*
distance threshold is a multiple of `hand_scale` (the wrist-to-middle-knuckle
distance), so recognition is unaffected by how far the hand is from the camera.
The tests verify this explicitly at three different scales.

### Issue 4 — temporal stability

Raw per-frame classification flickers: a hand moving from fist to victory spends
a few frames in between emitting nonsense. A sliding window of 5 frames takes a
majority vote, with a `min_votes` floor — a plain `most_common(1)` still returns
an answer when the window is split three ways between different guesses. The
window is kept per hand, so two hands never contaminate each other.

Cost: a ~3-frame (~0.1 s) delay when deliberately changing gesture. Measured in
`test_smoother_accepts_sustained_change`.

### Issue 5 — the processing pipeline

The guide suggests a capture thread with `queue.put(frame)`. An unbounded queue
in front of a slower consumer grows without limit, and latency grows with it.
MediaPipe needs 20–50 ms per frame on a VM CPU while the camera publishes at
30 fps, so the queue *would* be the slower side.

The node uses a **single slot** instead — a new frame overwrites the previous
one if it has not been picked up. Old frames are dropped, never queued, so
latency stays flat. Three stages:

```
executor thread   receives Image messages          → stores newest
worker thread     converts, runs MediaPipe, publishes
main thread       cv2.imshow (GUI toolkits require the main thread)
```

### A practical note on emoji

The guide's gesture table uses emoji (👍 ✌️ 👌). `cv2.putText` renders Hershey
vector fonts, which have no emoji glyphs — drawing them produces empty boxes.
All on-screen labels are therefore ASCII, and the heart is drawn as a filled
parametric curve:

```
x = 16 sin³t,    y = 13 cos t − 5 cos 2t − 2 cos 3t − cos 4t
```

### Deliberate deviation from the guide

The guide defines *Thumbs Down* as "thumb bent + four fingers extended". That
describes an open hand with a tucked thumb, not a thumbs-down. Thumbs up and
thumbs down are the same hand shape — thumb out, four fingers closed — differing
only in **orientation**, so both are detected from that shape plus the direction
the thumb points relative to its own knuckle.
