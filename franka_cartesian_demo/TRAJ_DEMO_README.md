# Franka Trajectory Listener + RViz Plot + Run on Keypress

This demo uses a trajectory format with shape `(steps, 7)`:

`[x, y, z, roll, pitch, yaw, gripper_width]`

## Nodes

1. `trajectory_listener_executor.py`
- Subscribes to `/demo/trajectory` (`std_msgs/Float64MultiArray`)
- Immediately visualizes the trajectory in RViz by publishing:
  - `/demo/trajectory_marker` (`visualization_msgs/Marker`)
  - `/demo/trajectory_path` (`nav_msgs/Path`)
  - `/demo/trajectory_poses` (`geometry_msgs/PoseArray`)
- Waits for keyboard commands:
  - `r` + Enter: run loaded trajectory
  - `p` + Enter: print status
  - `q` + Enter: quit

2. `trajectory_test_publisher.py`
- Publishes a sequence of trajectories `traj1 -> traj2 -> traj3` to `/demo/trajectory`

Queue behavior in executor:
- Incoming trajectories are queued in order.
- Executor shows the **next** trajectory in RViz.
- Press `r` to run one trajectory.
- After it finishes, executor shows the following one and waits for next `r`.

---

## Prerequisites
- Franka + MoveIt stack works on your machine
- `franka_gripper` is running if you want gripper commands executed

---

## Running Steps

### Step 1) Start MoveIt + robot bringup
Open Terminal A:

```bash
source /opt/ros/jazzy/setup.bash
source ~/franka_ws/install/setup.bash
ros2 launch franka_fr3_moveit_config moveit.launch.py robot_ip:=192.168.1.11 use_fake_hardware:=false
```

### Step 2) Start trajectory listener/executor
Open Terminal B:

```bash
source /opt/ros/jazzy/setup.bash
source ~/franka_ws/install/setup.bash
python3 /home/binzhao/.openclaw/workspace/franka_cartesian_demo/trajectory_listener_executor.py
```

You should see logs indicating listener is ready.

### Step 3) Publish test trajectory sequence (traj1, traj2, traj3)
Open Terminal C:

```bash
source /opt/ros/jazzy/setup.bash
source ~/franka_ws/install/setup.bash
python3 /home/binzhao/.openclaw/workspace/franka_cartesian_demo/trajectory_test_publisher.py
```

Listener should print:
- `Received trajectory: steps=..., n_dim=7`
- `Trajectory is now visible in RViz. Press 'r' + Enter to run.`

### Step 4) Run trajectory
Go back to Terminal B and type:

```text
r
```
then press Enter.

### Step 5) Optional commands
In Terminal B:
- `p` + Enter → print loaded waypoint count/status
- `q` + Enter → stop listener node

---

## RViz Setup (one-time)
In RViz, set Fixed Frame to `fr3_link0`.

Add displays:
1. **Path** → topic: `/demo/trajectory_path`
2. **PoseArray** → topic: `/demo/trajectory_poses`
3. **Marker** (optional) → topic: `/demo/trajectory_marker`

After this one-time setup, new received trajectories appear automatically.

---

## Data Format Details
`Float64MultiArray.layout.dim` must contain:
- `dim[0] = steps`
- `dim[1] = 7`

Flattened row-major data layout:

`[wp0_x, wp0_y, wp0_z, wp0_roll, wp0_pitch, wp0_yaw, wp0_gripper, wp1_x, ...]`

---

## Safety
- Keep E-stop reachable.
- Start with conservative trajectories.
- Make sure waypoints are collision-free and reachable.
- Test in open workspace before close-contact tasks.
