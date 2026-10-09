#!/usr/bin/env python3
import math
import select
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional

import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from geometry_msgs.msg import Point, Pose, PoseArray, PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import ColorRGBA, Float64MultiArray
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker

from franka_msgs.action import Grasp, Move
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.srv import GetCartesianPath
from sensor_msgs.msg import JointState
from pymoveit2 import MoveIt2


@dataclass
class TrajectoryItem:
    traj_id: int
    waypoints: List["Waypoint"]


@dataclass
class Waypoint:
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float
    gripper: float  # interpreted as target gripper width [m]


def euler_to_quat_xyzw(roll: float, pitch: float, yaw: float):
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return [qx, qy, qz, qw]


class TrajectoryListenerExecutor(Node):
    def __init__(self):
        super().__init__("franka_trajectory_listener_executor")
        self.cb_group = ReentrantCallbackGroup()

        self.base_link = "fr3_link0"
        self.ee_link = "fr3_link8"#"fr3_hand_tcp"
        self.group_name = "fr3_arm"
        self.joint_names = [
            "fr3_joint1",
            "fr3_joint2",
            "fr3_joint3",
            "fr3_joint4",
            "fr3_joint5",
            "fr3_joint6",
            "fr3_joint7",
        ]

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.moveit2 = MoveIt2(
            node=self,
            joint_names=self.joint_names,
            base_link_name=self.base_link,
            end_effector_name=self.ee_link,
            group_name=self.group_name,
            callback_group=self.cb_group,
        )

        self.gripper_move_client = ActionClient(self, Move, "/franka_gripper/move")
        self.gripper_grasp_client = ActionClient(self, Grasp, "/franka_gripper/grasp")

        self.cartesian_client = self.create_client(GetCartesianPath, "/compute_cartesian_path")
        self.exec_traj_client = ActionClient(self, ExecuteTrajectory, "/execute_trajectory")

        self._last_joint_state: Optional[JointState] = None
        self.create_subscription(JointState, "/joint_states", self._on_joint_state, 20)

        self.traj_sub = self.create_subscription(
            Float64MultiArray,
            "/demo/trajectory",
            self._on_trajectory,
            10,
        )

        self.marker_pub = self.create_publisher(Marker, "/demo/trajectory_marker", 10)
        self.path_pub = self.create_publisher(Path, "/demo/trajectory_path", 10)
        self.poses_pub = self.create_publisher(PoseArray, "/demo/trajectory_poses", 10)

        self._traj_queue: Deque[TrajectoryItem] = deque()
        self._active_item: Optional[TrajectoryItem] = None
        self._queue_lock = threading.Lock()
        self._traj_seq = 0

        # Tunables for speed/smoothness
        self.speed_scale = 0.18  # <1.0 => slower
        self.cartesian_max_step = 0.004  # smaller => denser/smoother path

        self.get_logger().info("Trajectory listener/executor ready.")
        self.get_logger().info("Waiting for trajectory on /demo/trajectory ...")

    def wait_for_moveit(self, timeout_sec=25.0):
        required = ["/plan_kinematic_path", "/compute_cartesian_path"]
        t0 = time.time()
        while time.time() - t0 < timeout_sec:
            names = [name for name, _ in self.get_service_names_and_types()]
            svc_ok = all(s in names for s in required)
            act_ok = self.exec_traj_client.wait_for_server(timeout_sec=0.2)
            if svc_ok and act_ok:
                self.get_logger().info("MoveIt planning services and execute action are available.")
                return True
            time.sleep(0.2)
        self.get_logger().error("MoveIt not available in time.")
        return False

    def _on_joint_state(self, msg: JointState):
        self._last_joint_state = msg

    def _parse_layout(self, msg: Float64MultiArray):
        # Expect shape (steps, 7)
        if len(msg.layout.dim) < 2:
            return None, None
        d0 = msg.layout.dim[0]
        d1 = msg.layout.dim[1]
        return d0.size, d1.size

    def _on_trajectory(self, msg: Float64MultiArray):
        steps, n_dim = self._parse_layout(msg)
        if steps is None:
            self.get_logger().error("Trajectory layout missing dimensions. Expect (steps, 7).")
            return
        if n_dim != 7:
            self.get_logger().error(f"Trajectory n_dim must be 7, got {n_dim}.")
            return
        if len(msg.data) != steps * n_dim:
            self.get_logger().error(
                f"Trajectory data size mismatch: got {len(msg.data)}, expected {steps*n_dim}."
            )
            return

        wps: List[Waypoint] = []
        for i in range(steps):
            k = i * n_dim
            wps.append(
                Waypoint(
                    x=float(msg.data[k + 0]),
                    y=float(msg.data[k + 1]),
                    z=float(msg.data[k + 2]),
                    roll=float(msg.data[k + 3]),
                    pitch=float(msg.data[k + 4]),
                    yaw=float(msg.data[k + 5]),
                    gripper=float(msg.data[k + 6]),
                )
            )

        with self._queue_lock:
            self._traj_seq += 1
            item = TrajectoryItem(traj_id=self._traj_seq, waypoints=wps)
            self._traj_queue.append(item)
            qlen = len(self._traj_queue)

        self.get_logger().info(
            f"Received trajectory #{item.traj_id}: steps={steps}, n_dim={n_dim}, queued={qlen}"
        )

        # If nothing active, show first queued trajectory immediately in RViz.
        with self._queue_lock:
            if self._active_item is None and self._traj_queue:
                preview = self._traj_queue[0]
            else:
                preview = None

        if preview is not None:
            self.publish_marker(preview.waypoints)
            self.get_logger().info(
                f"Trajectory #{preview.traj_id} is visible in RViz. Press 'r' + Enter to run."
            )

    def publish_marker(self, wps: List[Waypoint]):
        # clear old
        clear = Marker()
        clear.header.frame_id = self.base_link
        clear.header.stamp = self.get_clock().now().to_msg()
        clear.ns = "demo_traj"
        clear.id = 0
        clear.action = Marker.DELETEALL
        self.marker_pub.publish(clear)

        stamp = self.get_clock().now().to_msg()

        # line strip marker
        m = Marker()
        m.header.frame_id = self.base_link
        m.header.stamp = stamp
        m.ns = "demo_traj"
        m.id = 1
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.005
        m.color = ColorRGBA(r=0.0, g=1.0, b=0.2, a=1.0)

        # pose array + path
        pose_array = PoseArray()
        pose_array.header.frame_id = self.base_link
        pose_array.header.stamp = stamp

        path = Path()
        path.header.frame_id = self.base_link
        path.header.stamp = stamp

        for wp in wps:
            p = Point(x=wp.x, y=wp.y, z=wp.z)
            m.points.append(p)

            q = euler_to_quat_xyzw(wp.roll, wp.pitch, wp.yaw)
            pose = Pose()
            pose.position.x = wp.x
            pose.position.y = wp.y
            pose.position.z = wp.z
            pose.orientation.x = q[0]
            pose.orientation.y = q[1]
            pose.orientation.z = q[2]
            pose.orientation.w = q[3]
            pose_array.poses.append(pose)

            ps = PoseStamped()
            ps.header.frame_id = self.base_link
            ps.header.stamp = stamp
            ps.pose = pose
            path.poses.append(ps)

        self.marker_pub.publish(m)

        # points marker
        pnts = Marker()
        pnts.header.frame_id = self.base_link
        pnts.header.stamp = stamp
        pnts.ns = "demo_traj"
        pnts.id = 2
        pnts.type = Marker.POINTS
        pnts.action = Marker.ADD
        pnts.scale.x = 0.01
        pnts.scale.y = 0.01
        pnts.color = ColorRGBA(r=1.0, g=0.4, b=0.0, a=1.0)
        pnts.points = list(m.points)
        self.marker_pub.publish(pnts)

        self.poses_pub.publish(pose_array)
        self.path_pub.publish(path)

        self.get_logger().info("Published trajectory to /demo/trajectory_marker, /demo/trajectory_path, /demo/trajectory_poses")

    @staticmethod
    def _wait_future(future, timeout_sec: float = 5.0, poll_sec: float = 0.01):
        t0 = time.time()
        while not future.done() and (time.time() - t0) < timeout_sec:
            time.sleep(poll_sec)
        return future.done()

    def _gripper_move(self, width: float, speed: float = 0.02):
        width = max(0.0, min(0.08, width))
        if not self.gripper_move_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().warn("/franka_gripper/move not available; skipping gripper command")
            return
        goal = Move.Goal()
        goal.width = width
        goal.speed = speed
        f = self.gripper_move_client.send_goal_async(goal)
        self._wait_future(f, timeout_sec=3.0)

    def _compute_cartesian_robot_trajectory(self, wps: List[Waypoint]):
        if not self.cartesian_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().error("/compute_cartesian_path service unavailable")
            return None

        req = GetCartesianPath.Request()
        req.header.frame_id = self.base_link
        req.group_name = self.group_name
        req.link_name = self.ee_link
        req.max_step = self.cartesian_max_step
        req.jump_threshold = 0.0
        req.avoid_collisions = True

        # Use latest joint state as start state when available
        if self._last_joint_state is not None:
            req.start_state.joint_state = self._last_joint_state

        for wp in wps:
            q = euler_to_quat_xyzw(wp.roll, wp.pitch, wp.yaw)
            pose = Pose()
            pose.position.x = wp.x
            pose.position.y = wp.y
            pose.position.z = wp.z
            pose.orientation.x = q[0]
            pose.orientation.y = q[1]
            pose.orientation.z = q[2]
            pose.orientation.w = q[3]
            req.waypoints.append(pose)
        import sys
        import moveit_msgs

        self.get_logger().info(f"sys.executable = {sys.executable}")
        self.get_logger().info(f"moveit_msgs.__file__ = {moveit_msgs.__file__}")
        self.get_logger().info(f"GetCartesianPath module = {GetCartesianPath.__module__}")
        self.get_logger().info(f"type(req) = {type(req)}")
        self.get_logger().info(f"type(req).__module__ = {type(req).__module__}")

        fut = self.cartesian_client.call_async(req)
        if not self._wait_future(fut, timeout_sec=10.0):
            self.get_logger().error("compute_cartesian_path timeout")
            return None
        if fut.result() is None:
            self.get_logger().error("compute_cartesian_path call failed")
            return None

        res = fut.result()
        self.get_logger().info(f"Cartesian path fraction: {res.fraction:.3f}")
        if res.fraction < 0.95:
            self.get_logger().warn("Low cartesian fraction; trajectory may be incomplete.")
        print("fraction:", res.fraction)
        print("joint traj points:", len(res.solution.joint_trajectory.points))
        if res.solution.joint_trajectory.points:
            print("first jt point:", res.solution.joint_trajectory.points[0].positions)
            print("last  jt point:", res.solution.joint_trajectory.points[-1].positions)
        return res.solution

    @staticmethod
    def _dur_to_sec(d):
        return float(d.sec) + float(d.nanosec) * 1e-9

    def _retime_trajectory_slow(self, robot_trajectory):
        # Slow down and smooth execution by stretching time.
        # speed_scale in (0,1]: 0.2 => ~5x slower than nominal
        s = max(0.05, min(1.0, self.speed_scale))
        jt = robot_trajectory.joint_trajectory
        for p in jt.points:
            t = self._dur_to_sec(p.time_from_start)
            t_scaled = t / s
            p.time_from_start.sec = int(t_scaled)
            p.time_from_start.nanosec = int((t_scaled - int(t_scaled)) * 1e9)

            if p.velocities:
                p.velocities = [v * s for v in p.velocities]
            if p.accelerations:
                p.accelerations = [a * (s * s) for a in p.accelerations]
        return robot_trajectory

    def _execute_robot_trajectory(self, robot_trajectory):
        # import pdb; pdb.set_trace()
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = robot_trajectory
        fut_goal = self.exec_traj_client.send_goal_async(goal)
        if not self._wait_future(fut_goal, timeout_sec=5.0):
            self.get_logger().error("ExecuteTrajectory goal send timeout")
            return False
        goal_handle = fut_goal.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("ExecuteTrajectory goal rejected")
            return False

        fut_result = goal_handle.get_result_async()
        if not self._wait_future(fut_result, timeout_sec=120.0):
            self.get_logger().error("ExecuteTrajectory result timeout")
            return False
        result_wrap = fut_result.result()
        if result_wrap is None:
            self.get_logger().error("ExecuteTrajectory returned no result")
            return False
        err_code = result_wrap.result.error_code.val
        self.get_logger().info(f"ExecuteTrajectory result error_code={err_code}")
        return err_code == 1

    def _run_gripper_timeline(self, wps: List[Waypoint], total_time_sec: float):
        if total_time_sec <= 0.0:
            return

        n = len(wps)
        t0 = time.time()
        last_cmd = None
        for i, wp in enumerate(wps):
            target_t = (i / max(1, n - 1)) * total_time_sec
            while time.time() - t0 < target_t:
                time.sleep(0.01)
            if last_cmd is None or abs(wp.gripper - last_cmd) > 1e-3:
                self._gripper_move(wp.gripper)
                last_cmd = wp.gripper

    def execute_trajectory(self):
        with self._queue_lock:
            if self._active_item is not None:
                self.get_logger().warn("A trajectory is already running. Please wait.")
                return
            if not self._traj_queue:
                self.get_logger().warn("No trajectory queued. Publish to /demo/trajectory first.")
                return
            self._active_item = self._traj_queue.popleft()
            item = self._active_item
            remaining = len(self._traj_queue)

        wps = item.waypoints
        self.get_logger().info(
            f"Running trajectory #{item.traj_id} ({len(wps)} waypoints). Remaining queued={remaining}"
        )
        robot_traj = self._compute_cartesian_robot_trajectory(wps)
        if robot_traj is None:
            with self._queue_lock:
                self._active_item = None
            return

        robot_traj = self._retime_trajectory_slow(robot_traj)

        pts = robot_traj.joint_trajectory.points
        total_t = 0.0
        if pts:
            p = pts[-1].time_from_start
            total_t = float(p.sec) + float(p.nanosec) * 1e-9

        gt = threading.Thread(target=self._run_gripper_timeline, args=(wps, total_t), daemon=True)
        gt.start()
        ok = self._execute_robot_trajectory(robot_traj)

        with self._queue_lock:
            self._active_item = None
            next_item = self._traj_queue[0] if self._traj_queue else None
            qleft = len(self._traj_queue)

        if ok:
            self.get_logger().info(f"Trajectory #{item.traj_id} done.")
        else:
            self.get_logger().error(f"Trajectory #{item.traj_id} failed.")

        if next_item is not None:
            self.publish_marker(next_item.waypoints)
            self.get_logger().info(
                f"Next trajectory #{next_item.traj_id} is visible in RViz. Press 'r' + Enter to run. (queued={qleft})"
            )
        else:
            self.get_logger().info("Queue empty. Waiting for new trajectory.")


def keyboard_loop(node: TrajectoryListenerExecutor, stop_evt: threading.Event):
    print("\nKeyboard commands:")
    print("  r + Enter : run loaded trajectory")
    print("  p + Enter : print status")
    print("  q + Enter : quit node\n")

    while not stop_evt.is_set():
        rlist, _, _ = select.select([sys.stdin], [], [], 0.2)
        if not rlist:
            continue
        cmd = sys.stdin.readline().strip().lower()
        if cmd == "r":
            node.execute_trajectory()
        elif cmd == "p":
            with node._queue_lock:
                qn = len(node._traj_queue)
                active = node._active_item.traj_id if node._active_item else None
                next_id = node._traj_queue[0].traj_id if node._traj_queue else None
            node.get_logger().info(
                f"Status: active={active}, next={next_id}, queued={qn}, speed_scale={node.speed_scale:.2f}, max_step={node.cartesian_max_step:.4f}"
            )
        elif cmd == "q":
            stop_evt.set()
            return


def main():
    rclpy.init()
    node = TrajectoryListenerExecutor()

    if not node.wait_for_moveit(timeout_sec=30.0):
        node.destroy_node()
        rclpy.shutdown()
        return

    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    stop_evt = threading.Event()
    kb_thread = threading.Thread(target=keyboard_loop, args=(node, stop_evt), daemon=True)
    kb_thread.start()

    try:
        while not stop_evt.is_set():
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        stop_evt.set()
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
