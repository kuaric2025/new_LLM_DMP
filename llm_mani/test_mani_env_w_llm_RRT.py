"""Quick smoke test for ManiEnv with FrankaPanda.

Runs a few end-effector moves and gripper open/close in GUI.

Usage:
  python manipulation_env/test_mani_env.py
"""

import argparse
from dataclasses import dataclass
import json
import os
import sys
import threading
import time
from typing import Optional
from pathlib import Path

import cv2
import numpy as np

import torch
# if torch.cuda.is_available():
#     torch.cuda.init()  # Force early CUDA init

DISPLAY_AVAILABLE = bool(os.environ.get("DISPLAY"))
_WINDOW_CREATED = False
_DISPLAY_FAILED = False

# add repo root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manipulation_env.env import ManiEnv
from manipulation_env.robot import FrankaPanda

# ROS 2 imports (available via rebuilt Jazzy bindings for Python 3.10)
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from builtin_interfaces.msg import Time as TimeMsg
from sensor_msgs.msg import CameraInfo, Image
from message_filters import ApproximateTimeSynchronizer, Subscriber as MFSubscriber
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage

import pybullet as p
import math

# cwd = os.getcwd()
# print(cwd)
from utils.llm_utils import load_llm_config, apply_action_ids, ask_gpt4o
from utils.manipulation_utils import (
    sam3_inference,
    plate_drop_offset_xy_grid_4,
    pixel_depth_to_camera_frame,
    world_to_camera_frame,
    world_to_robot_frame,
    pose4x4_to_xyzrpy,
    get_robot_pose_from_tcp,
    anygrasp_demo,
    euler_closest_to_ref,
    shortest_yaw,
)

LLM_CONFIG, ACTION_ID_MAPPING, SYSTEM_PROMPT = load_llm_config()

def compute_intrinsics(width: int, height: int, fov_deg: float) -> tuple[float, float, float, float]:
    """Return (fx, fy, cx, cy) for a pinhole camera given FOV."""
    fov_rad = np.deg2rad(fov_deg)
    fx = (width / 2.0) / np.tan(fov_rad / 2.0)
    fy = (height / 2.0) / np.tan(fov_rad / 2.0)
    cx = width / 2.0
    cy = height / 2.0
    return fx, fy, cx, cy


def rotation_axes_summary(label: str, R: np.ndarray) -> None:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    print(f"{label}:")
    print(f"    x-axis {np.round(R[:, 0], 4)}")
    print(f"    y-axis {np.round(R[:, 1], 4)}")
    print(f"    z-axis {np.round(R[:, 2], 4)}")


def tcp_tracking_summary(label: str, commanded_xyzrpy: np.ndarray, robot) -> np.ndarray:
    actual_pos, actual_quat = robot.get_eef()
    actual_pos = np.asarray(actual_pos, dtype=np.float64)
    actual_quat = np.asarray(actual_quat, dtype=np.float64)

    commanded_xyzrpy = np.asarray(commanded_xyzrpy, dtype=np.float64).reshape(-1)
    target_pos = commanded_xyzrpy[:3]
    target_quat = np.asarray(
        p.getQuaternionFromEuler(tuple(float(v) for v in commanded_xyzrpy[3:6])),
        dtype=np.float64,
    )

    pos_err = actual_pos - target_pos
    q_err = p.getDifferenceQuaternion(actual_quat.tolist(), target_quat.tolist())
    ang_err = 2.0 * np.arctan2(np.linalg.norm(q_err[:3]), abs(q_err[3]))
    print(
        f"{label}: pos_err={np.round(pos_err, 4)} m, "
        f"pos_norm={np.linalg.norm(pos_err):.4f} m, "
        f"ang_err={np.degrees(ang_err):.2f} deg"
    )
    return np.concatenate([actual_pos, p.getEulerFromQuaternion(actual_quat)]).astype(np.float32)


@dataclass
class RRTNode:
    pose: np.ndarray
    parent: Optional[int]


def wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    angle = np.asarray(angle, dtype=np.float64)
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def pose_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64).reshape(6)
    b = np.asarray(b, dtype=np.float64).reshape(6)
    delta = b - a
    delta[3:6] = wrap_to_pi(delta[3:6])
    return delta


def pose_distance(a: np.ndarray, b: np.ndarray, rot_weight: float) -> float:
    delta = pose_delta(a, b)
    weighted = np.concatenate([delta[:3], rot_weight * delta[3:6]])
    return float(np.linalg.norm(weighted))


def interpolate_pose(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64).reshape(6)
    delta = pose_delta(a, b)
    pose = a + float(t) * delta
    pose[3:6] = wrap_to_pi(pose[3:6])
    return pose


def steer_pose(a: np.ndarray, b: np.ndarray, step_size: float, rot_weight: float) -> np.ndarray:
    dist = pose_distance(a, b, rot_weight=rot_weight)
    if dist <= step_size:
        return np.asarray(b, dtype=np.float64).reshape(6).copy()
    return interpolate_pose(a, b, step_size / max(dist, 1e-9))


def robot_joint_snapshot(robot) -> dict[int, float]:
    joint_ids = list(robot.arm_controllable_joints) + list(getattr(robot, "finger_joint_ids", []))
    return {joint_id: float(p.getJointState(robot.id, joint_id)[0]) for joint_id in joint_ids}


def restore_robot_joint_snapshot(robot, snapshot: dict[int, float]) -> None:
    for joint_id, joint_pos in snapshot.items():
        p.resetJointState(robot.id, joint_id, joint_pos)


def apply_pose_as_joint_state(robot, pose: np.ndarray, gripper_width: float) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64).reshape(6)
    pos = tuple(float(v) for v in pose[:3])
    orn = p.getQuaternionFromEuler(tuple(float(v) for v in pose[3:6]))
    joint_poses = p.calculateInverseKinematics(
        robot.id,
        robot.eef_id,
        pos,
        orn,
        robot.arm_lower_limits,
        robot.arm_upper_limits,
        robot.arm_joint_ranges,
        robot.arm_rest_poses,
        maxNumIterations=80,
    )
    for i, joint_id in enumerate(robot.arm_controllable_joints):
        p.resetJointState(robot.id, joint_id, float(joint_poses[i]))

    finger_target = float(np.clip(gripper_width, *robot.gripper_range)) / 2.0
    for joint_id in getattr(robot, "finger_joint_ids", []):
        p.resetJointState(robot.id, joint_id, finger_target)
    return np.asarray(joint_poses[: robot.arm_num_dofs], dtype=np.float64)


def pose_within_workspace(env: ManiEnv, pose: np.ndarray) -> bool:
    pose = np.asarray(pose, dtype=np.float64).reshape(6)
    return bool(np.all(pose[:3] >= env.min_pose) and np.all(pose[:3] <= env.max_pose))


def pose_collision_free(
    env: ManiEnv,
    pose: np.ndarray,
    gripper_width: float,
    collision_margin: float,
    ignore_body_ids: Optional[set[int]] = None,
) -> bool:
    if not pose_within_workspace(env, pose):
        return False

    ignore_body_ids = set() if ignore_body_ids is None else set(ignore_body_ids)
    snapshot = robot_joint_snapshot(env.robot)
    try:
        apply_pose_as_joint_state(env.robot, pose, gripper_width)
        p.performCollisionDetection()

        self_contacts = [
            cp for cp in p.getContactPoints(env.robot.id, env.robot.id)
            if cp[3] != cp[4]
        ]
        if self_contacts:
            return False

        body_ids = [
            getattr(env, "table_id", None),
            getattr(env, "bin_id", None),
            getattr(env, "boxID", None),
            *list(getattr(env, "object_ids", {}).values()),
        ]
        for body_id in body_ids:
            if body_id is None or body_id in ignore_body_ids:
                continue
            if p.getClosestPoints(env.robot.id, body_id, distance=max(0.0, collision_margin)):
                return False
        return True
    finally:
        restore_robot_joint_snapshot(env.robot, snapshot)
        p.performCollisionDetection()


def edge_collision_free(
    env: ManiEnv,
    start_pose: np.ndarray,
    end_pose: np.ndarray,
    gripper_width: float,
    edge_step: float,
    collision_margin: float,
    rot_weight: float,
    ignore_body_ids: Optional[set[int]] = None,
) -> bool:
    dist = pose_distance(start_pose, end_pose, rot_weight=rot_weight)
    steps = max(1, int(np.ceil(dist / max(edge_step, 1e-4))))
    for i in range(steps + 1):
        pose = interpolate_pose(start_pose, end_pose, i / steps)
        if not pose_collision_free(
            env,
            pose,
            gripper_width=gripper_width,
            collision_margin=collision_margin,
            ignore_body_ids=ignore_body_ids,
        ):
            return False
    return True


def sample_rrt_pose(
    env: ManiEnv,
    rng: np.random.Generator,
    start_pose: np.ndarray,
    goal_pose: np.ndarray,
    goal_bias: float,
) -> np.ndarray:
    if rng.random() < goal_bias:
        return np.asarray(goal_pose, dtype=np.float64).copy()

    pose = interpolate_pose(start_pose, goal_pose, rng.random())
    pose[:3] += rng.normal(0.0, 0.08, size=3)
    pose[:3] = np.clip(pose[:3], env.min_pose, env.max_pose)
    pose[3:6] = wrap_to_pi(pose[3:6] + rng.normal(0.0, 0.35, size=3))
    return pose


def nearest_rrt_node(tree: list[RRTNode], pose: np.ndarray, rot_weight: float) -> int:
    return min(
        range(len(tree)),
        key=lambda idx: pose_distance(tree[idx].pose, pose, rot_weight=rot_weight),
    )


def extend_rrt_tree(
    env: ManiEnv,
    tree: list[RRTNode],
    target_pose: np.ndarray,
    gripper_width: float,
    step_size: float,
    edge_step: float,
    collision_margin: float,
    rot_weight: float,
    ignore_body_ids: Optional[set[int]] = None,
) -> tuple[str, Optional[int]]:
    nearest_idx = nearest_rrt_node(tree, target_pose, rot_weight=rot_weight)
    nearest_pose = tree[nearest_idx].pose
    new_pose = steer_pose(nearest_pose, target_pose, step_size=step_size, rot_weight=rot_weight)

    if not edge_collision_free(
        env,
        nearest_pose,
        new_pose,
        gripper_width=gripper_width,
        edge_step=edge_step,
        collision_margin=collision_margin,
        rot_weight=rot_weight,
        ignore_body_ids=ignore_body_ids,
    ):
        return "trapped", None

    tree.append(RRTNode(new_pose, nearest_idx))
    new_idx = len(tree) - 1
    if pose_distance(new_pose, target_pose, rot_weight=rot_weight) < 1e-6:
        return "reached", new_idx
    return "advanced", new_idx


def connect_rrt_tree(
    env: ManiEnv,
    tree: list[RRTNode],
    target_pose: np.ndarray,
    gripper_width: float,
    step_size: float,
    edge_step: float,
    collision_margin: float,
    rot_weight: float,
    ignore_body_ids: Optional[set[int]] = None,
) -> tuple[str, Optional[int]]:
    latest_idx: Optional[int] = None
    while True:
        status, latest_idx = extend_rrt_tree(
            env,
            tree,
            target_pose=target_pose,
            gripper_width=gripper_width,
            step_size=step_size,
            edge_step=edge_step,
            collision_margin=collision_margin,
            rot_weight=rot_weight,
            ignore_body_ids=ignore_body_ids,
        )
        if status != "advanced":
            return status, latest_idx


def traceback_rrt_path(tree: list[RRTNode], node_idx: int) -> list[np.ndarray]:
    path: list[np.ndarray] = []
    while node_idx is not None:
        node = tree[node_idx]
        path.append(node.pose)
        node_idx = node.parent
    path.reverse()
    return path


def shortcut_pose_path(
    env: ManiEnv,
    path: list[np.ndarray],
    gripper_width: float,
    edge_step: float,
    collision_margin: float,
    rot_weight: float,
    shortcut_iters: int,
    ignore_body_ids: Optional[set[int]] = None,
) -> list[np.ndarray]:
    if len(path) < 3:
        return path

    rng = np.random.default_rng()
    path = [np.asarray(pose, dtype=np.float64).copy() for pose in path]
    for _ in range(max(0, shortcut_iters)):
        if len(path) < 3:
            break
        i, j = sorted(rng.choice(len(path), size=2, replace=False))
        if j <= i + 1:
            continue
        if edge_collision_free(
            env,
            path[i],
            path[j],
            gripper_width=gripper_width,
            edge_step=edge_step,
            collision_margin=collision_margin,
            rot_weight=rot_weight,
            ignore_body_ids=ignore_body_ids,
        ):
            path = path[: i + 1] + path[j:]
    return path


def densify_pose_path(path: list[np.ndarray], max_step: float, rot_weight: float) -> np.ndarray:
    if not path:
        raise ValueError("Expected non-empty path.")
    dense_path = [np.asarray(path[0], dtype=np.float64).copy()]
    for next_pose in path[1:]:
        curr_pose = dense_path[-1]
        dist = pose_distance(curr_pose, next_pose, rot_weight=rot_weight)
        steps = max(1, int(np.ceil(dist / max(max_step, 1e-4))))
        for i in range(1, steps + 1):
            dense_path.append(interpolate_pose(curr_pose, next_pose, i / steps))
    return np.asarray(dense_path, dtype=np.float32)


def plan_rrt_connect_path(
    env: ManiEnv,
    start_pose: np.ndarray,
    goal_pose: np.ndarray,
    gripper_width: float,
    args: argparse.Namespace,
    ignore_body_ids: Optional[set[int]] = None,
) -> np.ndarray:
    start_pose = np.asarray(start_pose, dtype=np.float64).reshape(6)
    goal_pose = np.asarray(goal_pose, dtype=np.float64).reshape(6)
    ignore_body_ids = set() if ignore_body_ids is None else set(ignore_body_ids)

    if pose_distance(start_pose, goal_pose, rot_weight=args.rrt_rot_weight) < 1e-8:
        return np.stack([start_pose, goal_pose], axis=0).astype(np.float32)

    if edge_collision_free(
        env,
        start_pose,
        goal_pose,
        gripper_width=gripper_width,
        edge_step=args.rrt_edge_step,
        collision_margin=args.rrt_collision_margin,
        rot_weight=args.rrt_rot_weight,
        ignore_body_ids=ignore_body_ids,
    ):
        return densify_pose_path(
            [start_pose, goal_pose],
            max_step=args.rrt_exec_step,
            rot_weight=args.rrt_rot_weight,
        )

    rng = np.random.default_rng()
    start_tree = [RRTNode(start_pose.copy(), None)]
    goal_tree = [RRTNode(goal_pose.copy(), None)]
    tree_a = start_tree
    tree_b = goal_tree
    tree_a_is_start = True

    for _ in range(max(1, args.rrt_max_iters)):
        sample_pose = sample_rrt_pose(
            env,
            rng,
            start_pose=start_pose,
            goal_pose=goal_pose,
            goal_bias=args.rrt_goal_bias,
        )
        status_a, idx_a = extend_rrt_tree(
            env,
            tree_a,
            target_pose=sample_pose,
            gripper_width=gripper_width,
            step_size=args.rrt_step_size,
            edge_step=args.rrt_edge_step,
            collision_margin=args.rrt_collision_margin,
            rot_weight=args.rrt_rot_weight,
            ignore_body_ids=ignore_body_ids,
        )
        if status_a != "trapped" and idx_a is not None:
            connect_pose = tree_a[idx_a].pose
            status_b, idx_b = connect_rrt_tree(
                env,
                tree_b,
                target_pose=connect_pose,
                gripper_width=gripper_width,
                step_size=args.rrt_step_size,
                edge_step=args.rrt_edge_step,
                collision_margin=args.rrt_collision_margin,
                rot_weight=args.rrt_rot_weight,
                ignore_body_ids=ignore_body_ids,
            )
            if status_b == "reached" and idx_b is not None:
                path_a = traceback_rrt_path(tree_a, idx_a)
                path_b = traceback_rrt_path(tree_b, idx_b)
                if tree_a_is_start:
                    coarse_path = path_a + list(reversed(path_b[:-1]))
                else:
                    coarse_path = path_b + list(reversed(path_a[:-1]))
                coarse_path = shortcut_pose_path(
                    env,
                    coarse_path,
                    gripper_width=gripper_width,
                    edge_step=args.rrt_edge_step,
                    collision_margin=args.rrt_collision_margin,
                    rot_weight=args.rrt_rot_weight,
                    shortcut_iters=args.rrt_shortcut_iters,
                    ignore_body_ids=ignore_body_ids,
                )
                return densify_pose_path(
                    coarse_path,
                    max_step=args.rrt_exec_step,
                    rot_weight=args.rrt_rot_weight,
                )

        tree_a, tree_b = tree_b, tree_a
        tree_a_is_start = not tree_a_is_start

    print("  RRT-Connect failed to find a collision-free path; falling back to straight-line interpolation.")
    return densify_pose_path(
        [start_pose, goal_pose],
        max_step=args.rrt_exec_step,
        rot_weight=args.rrt_rot_weight,
    )


def execute_pose_trajectory(
    env: ManiEnv,
    traj: np.ndarray,
    gripper_width: float,
    step_stride: int = 1,
    log_tracking: bool = False,
) -> np.ndarray:
    executed_prev = np.asarray(get_robot_pose_from_tcp(env.robot)[:3], dtype=np.float64)
    robot_current_pose = np.append(get_robot_pose_from_tcp(env.robot), gripper_width).astype(np.float32)
    for i in range(traj.shape[1]):
        if i % step_stride != 0 and i != traj.shape[1] - 1:
            continue
        traj_7dof = np.append(traj[:, i], gripper_width).astype(np.float32)
        env.step(traj_7dof)
        if log_tracking:
            actual_pose = tcp_tracking_summary("  Executed waypoint tracking", traj_7dof[:6], env.robot)
            executed_curr = np.asarray(actual_pose[:3], dtype=np.float64)
            p.addUserDebugLine(executed_prev, executed_curr, [0.0, 1.0, 1.0], 3)
            executed_prev = executed_curr
            robot_current_pose = np.append(actual_pose, gripper_width).astype(np.float32)
        else:
            robot_current_pose = np.append(get_robot_pose_from_tcp(env.robot), gripper_width).astype(np.float32)
    return robot_current_pose


def generate_circular_wipe_trajectory(
    env: ManiEnv,
    start_pose: np.ndarray,
    radius: float,
    num_points: int,
    loops: int,
) -> np.ndarray:
    start_pose = np.asarray(start_pose, dtype=np.float64).reshape(6)
    radius = max(1e-4, float(radius))
    num_points = max(12, int(num_points))
    loops = max(1, int(loops))

    # Make the current pose lie on the circumference so the wipe starts and ends
    # at the same pose while tracing a small circle on the table plane.
    circle_center = start_pose.copy()
    circle_center[0] -= radius

    path = [start_pose.copy()]
    total_points = num_points * loops
    for i in range(1, total_points + 1):
        theta = 2.0 * np.pi * (i / num_points)
        pose = start_pose.copy()
        pose[0] = circle_center[0] + radius * np.cos(theta)
        pose[1] = circle_center[1] + radius * np.sin(theta)
        clipped_xyz = np.clip(pose[:3], env.min_pose, env.max_pose)
        if np.any(np.abs(clipped_xyz - pose[:3]) > 1e-6):
            print(
                f"  [wipe] circular waypoint clipped by workspace. "
                f"requested={np.round(pose[:3], 4)} clipped={np.round(clipped_xyz, 4)}"
            )
        pose[:3] = clipped_xyz
        path.append(pose)

    path.append(start_pose.copy())
    return np.asarray(path, dtype=np.float32)


def generate_pour_trajectory(
    start_pose: np.ndarray,
    pour_angle_deg: float,
    rot_weight: float,
    return_to_start: bool = True,
) -> np.ndarray:
    start_pose = np.asarray(start_pose, dtype=np.float64).reshape(6)
    pour_pose = start_pose.copy()
    pour_pose[4] = wrap_to_pi(pour_pose[4] - np.deg2rad(float(pour_angle_deg)))

    path = [start_pose.copy(), pour_pose]
    if return_to_start:
        path.append(start_pose.copy())
    return densify_pose_path(path, max_step=0.05, rot_weight=rot_weight)


class SimulatedD435Publisher(Node):
    def __init__(
        self,
        env: ManiEnv,
        width: int = 640,
        height: int = 480,
        fov: float = 70.0,
        near: float = 0.1,
        far: float = 3.0,
        frame_id: str = "camera_color_optical_frame",
        publish_rate_hz: float = 10.0,
        # robot_frame: str = "panda_link0",   # add
        # world_frame: str = "world",          # add
        # publish_tf: bool = True,             # add: optional toggle
    ) -> None:
        super().__init__("simulated_d435")
        self.env = env
        self.width = width
        self.height = height
        self.fov = fov
        self.near = near
        self.far = far
        self.frame_id = frame_id
        self.latest_rgb: Optional[np.ndarray] = None
        self.latest_depth: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()

        self.color_pub = self.create_publisher(Image, "/camera/color/image_raw", 10)
        self.depth_pub = self.create_publisher(Image, "/camera/depth/image_rect_raw", 10)
        self.info_pub = self.create_publisher(CameraInfo, "/camera/color/camera_info", 10)

        self.fx, self.fy, self.cx, self.cy = compute_intrinsics(width, height, fov)
        self.timer = self.create_timer(1.0 / publish_rate_hz, self._publish)

        # # TF broadcaster (static - publish once)
        # if publish_tf:
        #     self._tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        #     self._publish_tf()

    def capture_from_env(self):
        """Capture camera frame from the main thread; PyBullet is not thread-safe."""
        rgb, depth, _ = self.env.capture_rgbd(
            width=self.width,
            height=self.height,
            fov=self.fov,
            near=self.near,
            far=self.far,
            return_seg=False,
        )
        with self._frame_lock:
            self.latest_rgb = rgb.copy()
            self.latest_depth = depth.copy()

    def _publish(self):
        with self._frame_lock:
            if self.latest_rgb is None or self.latest_depth is None:
                return
            rgb = self.latest_rgb.copy()
            depth = self.latest_depth.copy()
        now = self.get_clock().now().to_msg()
        color_msg = self._make_color_msg(rgb, now)
        depth_msg = self._make_depth_msg(depth, now)
        info_msg = self._make_camera_info(now)

        self.color_pub.publish(color_msg)
        self.depth_pub.publish(depth_msg)
        self.info_pub.publish(info_msg)

    def _make_color_msg(self, rgb: np.ndarray, stamp: TimeMsg) -> Image:
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        msg.height = self.height
        msg.width = self.width
        msg.encoding = "rgb8"
        msg.step = self.width * 3
        msg.data = rgb.astype(np.uint8).tobytes()
        return msg

    def _make_depth_msg(self, depth: np.ndarray, stamp: TimeMsg) -> Image:
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id.replace("color", "depth")
        msg.height = self.height
        msg.width = self.width
        msg.encoding = "32FC1"
        msg.step = self.width * 4
        msg.data = depth.astype(np.float32).tobytes()
        return msg

    def _make_camera_info(self, stamp: TimeMsg) -> CameraInfo:
        msg = CameraInfo()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        msg.height = self.height
        msg.width = self.width
        msg.k = [self.fx, 0.0, self.cx, 0.0, self.fy, self.cy, 0.0, 0.0, 1.0]
        msg.p = [self.fx, 0.0, self.cx, 0.0, 0.0, self.fy, self.cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        msg.distortion_model = "plumb_bob"
        return msg

class CameraTripleSubscriber(Node):
    def __init__(
        self,
        color_topic: str = "/camera/color/image_raw",
        depth_topic: str = "/camera/depth/image_rect_raw",
        info_topic: str = "/camera/color/camera_info",
        log_interval: float = 5.0,
    ) -> None:
        super().__init__("camera_triple_subscriber")
        qos = qos_profile_sensor_data
        info_qos = QoSProfile(depth=10)
        self.color_sub = MFSubscriber(self, Image, color_topic, qos_profile=qos)
        self.depth_sub = MFSubscriber(self, Image, depth_topic, qos_profile=qos)
        self.info_sub = MFSubscriber(self, CameraInfo, info_topic, qos_profile=info_qos)

        self.sync = ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub, self.info_sub], queue_size=10, slop=0.1
        )
        self.sync.registerCallback(self._synced_callback)
        self.latest_rgb: Optional[np.ndarray] = None
        self.latest_depth: Optional[np.ndarray] = None
        self.latest_info: Optional[CameraInfo] = None
        self._lock = threading.Lock()
        self._last_log = 0.0
        self._log_interval = max(0.1, log_interval)

    def _synced_callback(self, rgb_msg: Image, depth_msg: Image, info_msg: CameraInfo):
        rgb = self._image_to_numpy(rgb_msg)
        depth = self._depth_to_numpy(depth_msg)
        with self._lock:
            self.latest_rgb = rgb
            self.latest_depth = depth
            self.latest_info = info_msg

    def _image_to_numpy(self, msg: Image) -> Optional[np.ndarray]:
        if msg.encoding not in ("rgb8", "bgr8"):
            self.get_logger().warn(f"Unsupported RGB encoding: {msg.encoding}")
            return None
        data = np.frombuffer(msg.data, dtype=np.uint8)
        row_stride = msg.step
        expected = msg.width * 3
        if row_stride == expected:
            arr = data.reshape((msg.height, msg.width, 3))
        else:
            arr = data.reshape((msg.height, row_stride))[:, :expected].reshape((msg.height, msg.width, 3))
        if msg.encoding == "bgr8":
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        return arr

    def _depth_to_numpy(self, msg: Image) -> Optional[np.ndarray]:
        if msg.encoding == "32FC1":
            arr = np.frombuffer(msg.data, dtype=np.float32).reshape((msg.height, msg.width))
            return arr
        if msg.encoding == "16UC1":
            arr = np.frombuffer(msg.data, dtype=np.uint16).reshape((msg.height, msg.width)).astype(np.float32)
            arr /= 1000.0
            return arr
        self.get_logger().warn(f"Unsupported depth encoding: {msg.encoding}")
        return None

    def get_latest_images(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[CameraInfo]]:
        """Thread-safe access to latest RGB, depth, and camera info."""
        with self._lock:
            return (
                self.latest_rgb.copy() if self.latest_rgb is not None else None,
                self.latest_depth.copy() if self.latest_depth is not None else None,
                self.latest_info,
            )

    def maybe_log_status(self):
        now = time.time()
        if now - self._last_log < self._log_interval:
            return
        self._last_log = now
        with self._lock:
            if self.latest_rgb is None or self.latest_depth is None or self.latest_info is None:
                self.get_logger().info("Waiting for synchronized RGB/depth/CameraInfo...")
                return
            rgb_shape = self.latest_rgb.shape
            depth_shape = self.latest_depth.shape
            fx = self.latest_info.k[0]
            fy = self.latest_info.k[4]
        self.get_logger().info(
            f"Synced frames: RGB {rgb_shape}, Depth {depth_shape}, fx={fx:.2f}, fy={fy:.2f}"
        )

def run_simulation(
    env: ManiEnv,
    actions: list[np.ndarray],
    control_method: str = "end",
    camera_node: Optional[SimulatedD435Publisher] = None,
    show_camera: bool = False,
    triple_subscriber: Optional[CameraTripleSubscriber] = None,
):
    print("Running ManiEnv smoke trajectory...")
    for a in actions:
        env.step(a, control_method=control_method)
        env.step_simulation()
        if show_camera and camera_node is not None:
            maybe_show_camera_frame(camera_node)
        if triple_subscriber is not None:
            triple_subscriber.maybe_log_status()
        time.sleep(0.01) # 0.05

    print("Done. Close the GUI window to exit.")
    while True:
        env.step_simulation()
        if show_camera and camera_node is not None:
            maybe_show_camera_frame(camera_node)
        if triple_subscriber is not None:
            triple_subscriber.maybe_log_status()
        time.sleep(0.005) # 0.01


def start_ros_nodes(nodes: list[Node]) -> tuple[rclpy.executors.Executor, threading.Thread]:
    executor = rclpy.executors.MultiThreadedExecutor()
    for node in nodes:
        executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    return executor, thread


def maybe_show_camera_frame(node: SimulatedD435Publisher, window_name: str = "Simulated D435"):
    global _WINDOW_CREATED, _DISPLAY_FAILED
    if not DISPLAY_AVAILABLE:
        if not _DISPLAY_FAILED:
            print("[camera] DISPLAY not available; skipping on-screen preview")
            _DISPLAY_FAILED = True
        return
    if _DISPLAY_FAILED:
        return
    if node.latest_rgb is None:
        return
    try:
        if not _WINDOW_CREATED:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            _WINDOW_CREATED = True
        rgb = node.latest_rgb
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imshow(window_name, rgb_bgr)
        cv2.waitKey(1)
    except cv2.error as exc:
        print(f"[camera] OpenCV display failed: {exc}")
        _DISPLAY_FAILED = True


def shutdown_ros(executor: Optional[rclpy.executors.Executor]):
    if executor is not None:
        executor.shutdown()
    if rclpy.ok():
        rclpy.shutdown()


def parse_args():
    parser = argparse.ArgumentParser(description="ManiEnv test with simulated D435 publisher")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fov", type=float, default=70.0)
    parser.add_argument("--camera-rate", type=float, default=10.0, help="Publish rate in Hz")
    parser.add_argument("--vis", action="store_true", help="Show PyBullet GUI")
    parser.add_argument("--show-camera", action="store_true", help="Display realtime camera view via OpenCV")
    parser.add_argument("--subscribe-camera", action="store_true", help="Subscribe to RGB+depth+CameraInfo topics for monitoring")
    parser.add_argument("--subscriber-log-interval", type=float, default=5.0, help="Seconds between subscriber status logs")
    parser.add_argument("--prompt", type=str, default="prepare breakfast")

    parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
    parser.add_argument('--max_gripper_width', type=float, default=0.08, help='Maximum gripper width (<=0.1m)')
    parser.add_argument('--gripper_height', type=float, default=0.01, help='Gripper height')
    parser.add_argument(
        '--grasp-center-offset-z',
        type=float,
        default=0.105,
        help='Hand-frame Z offset from panda_hand to the intended grasp center in meters.',
    )
    parser.add_argument('--top_down_grasp', action='store_true', help='Output top-down grasps.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--num-runs', type=int, default=1, help='Number of trials for success rate evaluation (default: 1)')
    parser.add_argument('--seed', type=int, default=None, help='Base seed for object pose randomization; each trial uses seed+trial_idx')
    parser.add_argument('--no-randomize', action='store_true', help='Disable object pose randomization even when num-runs > 1')
    parser.add_argument('--scene-configs', type=str, default=None,
        help='Path to pre-generated scene_configs.json; use same 10 scenes across pipeline tests for fair comparison')
    parser.add_argument('--rrt-step-size', type=float, default=0.10, help='RRT-Connect extension step in weighted pose space.')
    parser.add_argument('--rrt-edge-step', type=float, default=0.03, help='Collision-check discretization along each edge.')
    parser.add_argument('--rrt-exec-step', type=float, default=0.02, help='Dense waypoint spacing for execution after planning.')
    parser.add_argument('--rrt-max-iters', type=int, default=700, help='Maximum bidirectional RRT-Connect growth iterations.')
    parser.add_argument('--rrt-goal-bias', type=float, default=0.25, help='Probability of sampling the exact goal pose.')
    parser.add_argument('--rrt-rot-weight', type=float, default=0.12, help='Orientation weight relative to XYZ in pose distance.')
    parser.add_argument('--rrt-shortcut-iters', type=int, default=80, help='Shortcut smoothing attempts after planning.')
    parser.add_argument('--rrt-collision-margin', type=float, default=0.0, help='Extra PyBullet collision margin for waypoint validity checks.')
    parser.add_argument('--wipe-radius', type=float, default=0.07, help='Radius of the circular wiping motion in meters.')
    parser.add_argument('--wipe-num-points', type=int, default=40, help='Waypoints per wipe loop.')
    parser.add_argument('--wipe-loops', type=int, default=2, help='Number of circular wipe loops to execute.')
    parser.add_argument('--pour-angle-deg', type=float, default=100.0, help='Relative end-effector pitch rotation for pouring.')
    return parser.parse_args()



def main():
    args = parse_args()

    table_path = os.path.join(os.path.dirname(__file__), "../manipulation_env/models/urdf/objects/table/table.urdf")
    block_path = None

    # Task 1
    bin_path = os.path.join(os.path.dirname(__file__), "../manipulation_env/models/urdf/objects/table/bin.urdf")
    banana_path = os.path.join(os.path.dirname(__file__), "../manipulation_env/models/ycb/011_banana/model.urdf")
    plate_path = os.path.join(os.path.dirname(__file__), "../manipulation_env/models/ycb/029_plate/model.urdf")
    apple_path = os.path.join(os.path.dirname(__file__), "../manipulation_env/models/ycb/013_apple/model.urdf")
    orange_path = os.path.join(os.path.dirname(__file__), "../manipulation_env/models/ycb/017_orange/model.urdf")
    
    # Task 2
    pitcher_path = os.path.join(os.path.dirname(__file__), "../manipulation_env/models/ycb/019_pitcher_base/model.urdf")
    e_cups_path = os.path.join(os.path.dirname(__file__), "../manipulation_env/models/ycb/065-e_cups/model.urdf")
    i_cups_path = os.path.join(os.path.dirname(__file__), "../manipulation_env/models/ycb/065-i_cups/model.urdf")
    j_cups_path = os.path.join(os.path.dirname(__file__), "../manipulation_env/models/ycb/065-j_cups/model.urdf")
    g_cups_path = os.path.join(os.path.dirname(__file__), "../manipulation_env/models/ycb/065-g_cups/model.urdf")
    h_cups_path = os.path.join(os.path.dirname(__file__), "../manipulation_env/models/ycb/065-h_cups/model.urdf")

    # Task 3
    sponge_path = os.path.join(os.path.dirname(__file__), "../manipulation_env/models/ycb/026_sponge/model.urdf")
    mug_path = os.path.join(os.path.dirname(__file__), "../manipulation_env/models/ycb/025_mug/model.urdf")
    bowl_path = os.path.join(os.path.dirname(__file__), "../manipulation_env/models/ycb/024_plate/model.urdf")
    # banana, bowl

    # # serve multiple cups of drink
            # ("065-h_cups", [0.55, -0.24, 0.06], [0, 0, 1.0]),
            # ("065-i_cups", [0.55, 0.00, 0.06], [0, 0, 1.0]),
            # ("065-j_cups", [0.55, 0.24, 0.06], [0, 0, 1.0]),
            # ("019_pitcher_base", [-0.2, -0.5, 0.06], [0, 0, 1.0]),

            # # clear the table into bin and clean the table
            # ("011_banana", [0.55, -0.24, 0.06], [0, 0, 1.0]),
            # ("025_mug", [0.65, 0.24, 0.07], [0, 0, 0]),
            # # ("010_potted_meat_can", [0.4, 0.2, 0.06], [0, 0, 1.0]),
            # # ("065-h_cups", [0.55, -0.24, 0.06], [0, 0, 1.0]),
            # ("013_apple", [0.4, -0.24, 0.06], [0, 0, 1.0]),
            # ("026_sponge", [0.2, 0.2, 0.06], [0, 0, 1.0]),

    robot = FrankaPanda(model_path=None)
    # env = ManiEnv(robot, block_path=block_path, table_path=table_path, bin_path=bin_path, banana_path=banana_path, plate_path=plate_path, apple_path=apple_path, orange_path=orange_path, pitcher_path=pitcher_path, e_cups_path=e_cups_path, i_cups_path=i_cups_path, j_cups_path=j_cups_path, sponge_path=sponge_path, mug_path=mug_path, bowl_path=bowl_path, vis=args.vis)
    # config_path = os.path.join(os.path.dirname(__file__), "../manipulation_env/configs/task_1.json")
    env = ManiEnv(
        robot,
        block_path=block_path, table_path=table_path, bin_path=bin_path, banana_path=banana_path, plate_path=plate_path, apple_path=apple_path, orange_path=orange_path, pitcher_path=pitcher_path, e_cups_path=e_cups_path, i_cups_path=i_cups_path, j_cups_path=j_cups_path, sponge_path=sponge_path, mug_path=mug_path, bowl_path=bowl_path, vis=args.vis,
        object_config_path=args.scene_configs,
    )
    rclpy.init()
    nodes: list[Node] = []
    camera_node = SimulatedD435Publisher(
        env,
        width=args.camera_width,
        height=args.camera_height,
        fov=args.camera_fov,
        near=0.1,
        far=3.0,
        publish_rate_hz=args.camera_rate,
    )
    nodes.append(camera_node)

    triple_sub = None
    if args.subscribe_camera:
        triple_sub = CameraTripleSubscriber(
            log_interval=args.subscriber_log_interval,
        )
        nodes.append(triple_sub)
    
    executor, exec_thread = start_ros_nodes(nodes)
    # Capture PyBullet camera frames on the main thread. The ROS timer only
    # publishes this cached frame to avoid OpenGL/PyBullet thread crashes.
    camera_node.capture_from_env()
    time.sleep(0.2)

    scene_configs = None
    if args.scene_configs is not None:
        with open(args.scene_configs, "r", encoding="utf-8") as f:
            data = json.load(f)
        scene_configs = data.get("scenes", data) if isinstance(data, dict) else data
        if not isinstance(scene_configs, list):
            scene_configs = [scene_configs]
        print(f"Loaded {len(scene_configs)} pre-generated scene configs from {args.scene_configs}")

    use_randomize = args.num_runs > 1 and not args.no_randomize and scene_configs is None
    results: list[bool] = []
    
    for trial_idx in range(args.num_runs):
        scene_config = None
        if scene_configs is not None:
            scene_config = scene_configs[trial_idx % len(scene_configs)]
            if args.num_runs > 1:
                sid = int(
                    np.random.default_rng(
                        (args.seed + trial_idx) if args.seed is not None else None
                    ).integers(0, 2**31 - 1)
                )
                print(f"\n\033[36m=== Trial {trial_idx + 1}/{args.num_runs} (scene seed={sid}) ===\033[0m")
        else:
            if args.num_runs > 1:
                if use_randomize:
                    print(f"\n\033[36m=== Trial {trial_idx + 1}/{args.num_runs} (randomized object poses) ===\033[0m")
                else:
                    print(f"\n\033[36m=== Trial {trial_idx + 1}/{args.num_runs} ===\033[0m")

        try:
            # if scene_config is not None:
            #     print(f"Scene config: {scene_config}=====================")
            #     env.reset(scene_config=scene_config)
            # else:
            #     print(f"Scene config: None=====================")
            #     trial_rng = np.random.default_rng((args.seed + trial_idx) if args.seed is not None else None)
            #     env.reset(randomize_poses=use_randomize, rng=trial_rng if use_randomize else None)
            # time.sleep(0.5)  # Wait for camera to capture new scene
            if triple_sub is not None:
                rgb, depth, cam_info = triple_sub.get_latest_images()
                if rgb is not None:
                    print(f"RGB shape: {rgb.shape}, dtype: {rgb.dtype}")
                    print(f"RGB min: {rgb.min()}, max: {rgb.max()}, mean: {rgb.mean()}")
                    print(f"RGB pixel at (y=100, x=200): R={rgb[100, 200, 0]}, G={rgb[100, 200, 1]}, B={rgb[100, 200, 2]}")
                if depth is not None:
                    print(f"Depth shape: {depth.shape}, dtype: {depth.dtype}")
                    print(f"Depth min: {depth.min()}, max: {depth.max()}, mean: {depth.mean()}")
                    print(f"Depth value at (100, 200): {depth[100, 200]} meters")
            #===========================MLLMs======================================
        
            # Get RGB image from subscriber or camera node
            rgb_image = None
            depth_image = None
            if triple_sub is not None:
                rgb, depth, _ = triple_sub.get_latest_images()
                if rgb is not None:
                    rgb_image = rgb
                    depth_image = depth
                else:
                    print("Warning: triple_sub has no RGB image yet, waiting...")
                    time.sleep(1.0)  # Wait a bit more
                    rgb, depth, _ = triple_sub.get_latest_images()
                    if rgb is not None:
                        rgb_image = rgb
                        depth_image = depth
            elif camera_node is not None and camera_node.latest_rgb is not None:
                rgb_image = camera_node.latest_rgb.copy()
                _, depth_image, _ = env.capture_rgbd(
                    width=args.camera_width, height=args.camera_height, fov=args.camera_fov,
                    near=0.1, far=3.0, return_seg=False,
                )
            
            # Fallback to file path if no RGB available
            if rgb_image is None:
                image_path = os.environ.get('LLM_IMAGE_PATH', '/home/mhumais/Downloads/55.png')
                image_bgr = cv2.imread(str(image_path))
                if image_bgr is not None:
                    print(f"Warning: No RGB image available, using file path: {image_path}")
                    rgb_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                else:
                    print(
                        f"Warning: No RGB image available and fallback file is missing: {image_path}. "
                        "Capturing directly from env."
                    )
                    rgb_image, depth_image, _ = env.capture_rgbd(
                        width=args.camera_width, height=args.camera_height, fov=args.camera_fov,
                        near=0.1, far=3.0, return_seg=False,
                    )
                if depth_image is None:
                    _, depth_image, _ = env.capture_rgbd(
                    width=args.camera_width, height=args.camera_height, fov=args.camera_fov,
                    near=0.1, far=3.0, return_seg=False,
                    )
                prompt = os.environ.get('LLM_PROMPT', args.prompt)
                reply = ask_gpt4o(
                    prompt=prompt,
                    system_prompt=SYSTEM_PROMPT,
                    image_arrays=[rgb_image],
                )
            else:
                print(f"Using RGB image from camera: shape={rgb_image.shape}, dtype={rgb_image.dtype}")
                prompt = os.environ.get('LLM_PROMPT', args.prompt)
                reply = ask_gpt4o(
                    prompt=prompt,
                    system_prompt=SYSTEM_PROMPT,
                    image_arrays=[rgb_image],
                )
            
            # # build lib for segmentation masks
            # object_list = ['banana', 'orange', 'apple', 'plate']
            # # object_list = ['apple']
            # segmap_list = {}
            # for obj in object_list:
            #     # import pdb; pdb.set_trace()
            #     print(f"  Segmenting {obj}...")
            #     segmap = sam3_inference(rgb_image, obj)
            #     segmap_list[obj] = segmap
            #     # import pdb; pdb.set_trace()
            # torch.save(segmap_list, "task_12.pt")
            # # breakfast_preparation: task_1.pt
            # import pdb; pdb.set_trace()
            
            
            # loaded = torch.load("task_1.pt")
            # np.array(loaded['banana'][0]).shape
            
            # f = open("breakfast_segmap.json", "w", encoding="utf-8")
            # try:
            #     json.dump(segmap_list, f, ensure_ascii=False, indent=4)
            # finally:
            #     f.close()
        
            # segmap_list = torch.load("task_12.pt")
        
            # Add action_id to each step from config mapping
            reply = apply_action_ids(reply, ACTION_ID_MAPPING)
        
            # Print formatted JSON response
            print("\n=== GPT-4o Response (JSON) ===")
            print(json.dumps(reply, indent=2))
            num_steps = len(reply)
        
            gripper_val = 0.15
            drop_id = 0
            carried_object_id: Optional[int] = None
        
            robot_current_pose = get_robot_pose_from_tcp(env.robot)
            
            print("\n=== Parsed Action Plan ===")
            for step in reply:
                env.clean_traj_plot()
                print(f"\033[32mStep {step.get('step_id', 'N/A')}: {step.get('action', 'N/A')} (action_id: {step.get('action_id', 'N/A')})\033[0m")
                if 'target_object' in step and 'action_id' in step:
                    color = step['target_object'].get('attributes')
                    if color == 'white':
                        color = 'blue'
                    if color is not None:
                        color = str(color).strip()
                    # import pdb; pdb.set_trace()
                    obj_name = step['target_object'].get('name')
                    
                    obj_name = '' if obj_name is None else str(obj_name).strip()
                    obj = f"{color} {obj_name}".strip() if color else obj_name
                    if obj_name == 'kettle':
                        obj = 'pitcher handle'
                    if obj_name == 'bowl':
                        obj_name = 'plate'
                    # import pdb; pdb.set_trace()
                    # if obj == 'bowl':
                    #     obj = 'plate'
                    print(f"  Target Object: {obj}")
                    action_id = step['action_id']
                    if action_id is not None:
                        action_id = int(action_id)
                    print(f"  Action ID: {action_id}")
                    action_name = step['action']
                    print(f"  Action Name: {action_name}")
                    step_id = step.get('step_id', 'N/A')
                    # import pdb; pdb.set_trace()
                    
                    
                    # gripper_val = float(robot_current_pose[-1])
                    # msg = JointTrajectory()
                    # msg.header.stamp = camera_node.get_clock().now().to_msg()
                    # msg.header.frame_id = "panda_link0"
                    # msg.joint_names = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
                    # action_id = 3
                    dt = 0.05
                    if action_id == 0:
                        print(f"  SAM3 prompt: {obj}")
                        is_bin_target = obj.lower() in {"gray bin", "grey bin", "bin"} or obj_name.lower() == "bin"
                        bin_id = None
                        if is_bin_target:
                            bin_id = env.object_ids.get("bin") if hasattr(env, "object_ids") else None
                            if bin_id is None:
                                bin_id = getattr(env, "bin_id", None)
                            if bin_id is None:
                                bin_id = getattr(env, "binID", None)
                        if is_bin_target and bin_id is not None:
                            _, bin_depth, bin_seg = env.capture_rgbd(
                                width=args.camera_width,
                                height=args.camera_height,
                                fov=args.camera_fov,
                                near=0.1,
                                far=3.0,
                                return_seg=True,
                            )
                            bin_id = int(bin_id)
                            bin_seg = np.asarray(bin_seg, dtype=np.int32)
                            bin_mask = (bin_seg & ((1 << 24) - 1)) == bin_id
                            if np.any(bin_mask):
                                depth_image = bin_depth
                                segmap = torch.from_numpy(bin_mask)
                                print(
                                    f"  Using PyBullet bin mask: pixels={int(bin_mask.sum())}, "
                                    f"object_id={bin_id}"
                                )
                            else:
                                unique_seg = np.unique(bin_seg)
                                print(
                                    f"  PyBullet bin mask is empty for object_id={bin_id}; "
                                    f"seg unique sample={unique_seg[:10]}. Falling back to SAM3."
                                )
                                segmap = sam3_inference(rgb_image, obj)
                        else:
                            if is_bin_target:
                                print("  No PyBullet bin id found; falling back to SAM3.")
                            segmap = sam3_inference(rgb_image, obj)
                        # segmap = torch.tensor(segmap_list[obj][0]).to(device)
                        # segmap =segmap_list[obj][0]
                        if segmap is not None:
                        # aa = True
                        # if aa:
                            # import pdb; pdb.set_trace()
                            # segmap = segmap.permute(2, 1, 0).squeeze(2) # real-time segmentation
                            segmap = segmap.squeeze(0)
                            print(f"  Segmap: {segmap.shape}")

                            segmap_gray = segmap.detach().cpu().numpy().astype(np.uint8)
                            segmap_gray = (segmap_gray > 0).astype(np.uint8) * 255
                            from PIL import Image as PILImage
                            PILImage.fromarray(segmap_gray, mode="L").save(
                                f"gray_bin_segmap_step_{step_id}.png"
                            )
                            print(f"  Saved gray bin segmap: gray_bin_segmap_step_{step_id}.png")
                            ys, xs = torch.where(segmap)
                            # xs, ys = torch.where(segmap)
                            # # import pdb; pdb.set_trace()
                            # center_y = ys.float().mean()   # row coordinate (v)
                            # center_x = xs.float().mean()   # col coordinate (u)
                            # # If you want integer pixel center:
                            # center_y_int = int(center_y.round().item())
                            # center_x_int = int(center_x.round().item())
        
                            center_y = ys.float().median().item()   # more robust to asymmetric masks
                            center_x = xs.float().median().item()
                            # import pdb; pdb.set_trace()
                            center_y_int = int(center_y)
                            center_x_int = int(center_x)
        
                            
                            print(f"  Center (y, x): ({center_y_int}, {center_x_int})") # 329, 294
                            
                            center_z_m = depth_image[center_y_int, center_x_int] # 1.138107180595398
                            print(f"  Center Z (m): {center_z_m}")
                            
                            #==============get pose in robot frame======================================
                            T_c2r = np.array([[ 1.0, 0.0, 0.0, 0.6000 ],
                                            [ 0.0, -1.0, 0.0, 0.0 ],
                                            [ 0.0, 0.0, -1.0, 1.200 ],
                                            [ 0.0, 0.0, 0.0, 1.0 ]])
                            T_r2c = np.linalg.inv(T_c2r)
        
                            T_o2c = np.array([
                                    [1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0,-1, 0],
                                    [0, 0, 0, 1],
                                ])
        
                            # cam_info = np.array([457.0073621574767,  0, 320.0,
                            #             0, 342.75552161810754, 240.0,
                            #                                 0,  0,  1])
                            cam_info = np.array([342.7555,  0, 320.0,
                                        0, 342.7555, 240.0,
                                                            0,  0,  1])
                            # # import pdb; pdb.set_trace()
                            # pos_camera = pixel_depth_to_camera_frame(
                            #             center_x_int, center_y_int, center_z_m, cam_info
                            #         )
        
                            # Backproject all segmented pixels to 3D
                            pos_camera = pixel_depth_to_camera_frame(depth_image, segmap.cpu().numpy(), cam_info)
        
                            # Empirical correction: camera-frame offset (X, Y, Z) from ground-truth comparison.
                            # Tune per object if needed; default derived from apple.
                            # OBJECT_OFFSETS_CAM = {
                            #     "apple": np.array([-0.019, -0.022, 0.060]),
                            #     "banana": np.array([-0.019, -0.022, 0.060]),  # same as apple; validate & tune
                            #     "orange": np.array([-0.019, -0.022, 0.060]),
                            #     "plate": np.array([-0.019, -0.022, 0.060]),
                            # }
                            # offset_cam = OBJECT_OFFSETS_CAM.get(obj_name, np.array([-0.019, -0.022, 0.060]))
                            # pos_camera = np.asarray(pos_camera, dtype=np.float64) + offset_cam
        
                            # # Debug: compare with ground truth (optional)
                            # # obj_id_key = {"kettle": "pitcher"}.get(obj_name, obj_name)
                            # obj_id = env.object_ids.get('apple')
                            # if obj_id is not None:
                            #     # Better ground truth: AABB center (geometric center) instead of link origin
                            #     aabb_min, aabb_max = p.getAABB(obj_id)
                            #     center_world = 0.5 * (np.array(aabb_min) + np.array(aabb_max))
                            #     pos_world_h = np.append(center_world, 1.0)
                            #     pos_cam_gt = (T_r2c @ pos_world_h)[:3]
                            #     print(f"Ground truth in camera:", pos_cam_gt)
                            #     print("Predicted from segmap (after offset):", pos_camera)
        
                            # pos_camera = pixel_depth_to_camera_frame(center_y_int, center_x_int, center_z_m, cam_info_2)
                            pos_camera_ = np.eye(4)
                            pos_camera_[:3, 3] = pos_camera.reshape(3)
                            # import pdb; pdb.set_trace()
        
                            # # anygrasp pose in camera frame
                            if step_id < num_steps and reply[step_id]['action'] == "grasp":
                            #     print("AnyGrasp Predicting=========")
                                # grasp = anygrasp_demo(rgb_image, depth_image, np.array(segmap.permute(1,0).cpu()), step_id, args) # real-time segmentation
                                grasp = anygrasp_demo(rgb_image, depth_image, np.array(segmap.cpu()), step_id, args) # pre-computed segmentation
                                rotation_axes_summary("  AnyGrasp camera-frame axes", grasp.rotation_matrix)
                                # Map AnyGrasp's gripper frame to the Panda hand frame.
                                # This remaps the nearly vertical AnyGrasp x-axis onto the
                                # Panda hand's local z-axis, which is the axis IK uses.
                                R_grasp_to_panda = np.array([
                                    [0.0, 0.0, 1.0],
                                    [0.0, 1.0, 0.0],
                                    [-1.0, 0.0, 0.0],
                                ], dtype=np.float64)
                                grasp_rotation_panda = np.asarray(grasp.rotation_matrix, dtype=np.float64) @ R_grasp_to_panda
                                pos_camera_[:3, :3] = grasp_rotation_panda
                                pos_camera_[:3, 3] = grasp.translation
                                rotation_axes_summary("  Panda-aligned camera-frame axes", grasp_rotation_panda)
        
                            pos_robot = T_c2r @ pos_camera_
                            print(f"  Pos robot: {pos_robot}")
                            rotation_axes_summary("  Robot-frame grasp axes", pos_robot[:3, :3])
        
                            # Offset is expressed in the Panda hand frame. The default
                            # 0.105 m matches panda_hand -> panda_grasptarget in the URDF,
                            # but this can be tuned if AnyGrasp's translation is already
                            # closer to the true contact center.
                            GRASP_CENTER_OFFSET = np.array(
                                [0.0, -0.0, -args.grasp_center_offset_z], #World-frame offset delta: [ 0.014  -0.03    0.0837]
                                dtype=np.float64,
                            )
                            FINGER_LENGTH =  0 # 0.1034  # m
        
                            # 根据你的手爪，在这里选对轴：
                            # GRIPPER_AXIS = np.array([1.0, 0.0, 0.0])  # 如果手指沿 +X
                            GRIPPER_AXIS = np.array([0.0, 0.0, 1.0])  # 如果手指沿 +Z
        
                            TCP_OFFSET_WITH_FINGER = GRASP_CENTER_OFFSET + FINGER_LENGTH * GRIPPER_AXIS
                            print(f"  Hand-frame TCP offset: {TCP_OFFSET_WITH_FINGER}")
                            offset_world_delta = pos_robot[:3, :3] @ (-TCP_OFFSET_WITH_FINGER)
                            print(f"  World-frame offset delta: {np.round(offset_world_delta, 4)}")
        
                            T_offset_inv = np.eye(4)
                            T_offset_inv[:3, 3] = -TCP_OFFSET_WITH_FINGER
                            pos_robot_tcp = pos_robot @ T_offset_inv  # TCP pose in robot frame
                            # pos_robot_tcp = T_offset_inv @ pos_robot
                            rotation_axes_summary("  Robot-frame TCP axes", pos_robot_tcp[:3, :3])
                            
                            
                            
                            #==================plan motion with RRT-Connect======================================
                            if step_id == 1:
                                robot_initial_pose = get_robot_pose_from_tcp(env.robot) # in world frame    
                            else:
                                robot_initial_pose = robot_current_pose

                            robot_initial_pose[6] = gripper_val #0.08
                            print(f"  Robot initial pose: {robot_initial_pose}")
                            robot_end_pose = pos_robot_tcp

                            ## check the closest euler representation
                            xyzrpy_end = pose4x4_to_xyzrpy(robot_end_pose, gripper_val)
                            xyzrpy_end[3:6] = euler_closest_to_ref(
                                xyzrpy_end[3:6],
                                robot_initial_pose[3:6]
                            )
                            target_quat = p.getQuaternionFromEuler(tuple(float(v) for v in xyzrpy_end[3:6]))
                            target_R = np.array(p.getMatrixFromQuaternion(target_quat), dtype=np.float64).reshape(3, 3)
                            current_quat = env.robot.get_eef()[1]
                            current_R = np.array(p.getMatrixFromQuaternion(current_quat), dtype=np.float64).reshape(3, 3)
                            rotation_axes_summary("  Current Panda hand axes", current_R)
                            rotation_axes_summary("  Commanded Panda hand axes", target_R)
                            # import pdb; pdb.set_trace()
                            # if obj_name == "cup":
                                # xyzrpy_end[1] -= 0.03
                                # xyzrpy_end[0] -= 0.02
                                # xyzrpy_end[2] += 0.005
                            # if obj_name == "cup" and reply[step_id]['action'] == "grasp":
                            #     xyzrpy_end[2] += 0.01
                            if reply[step_id]['action'] == "drop":
                                drop_id += 1
                                xyzrpy_end[2] += 0.18
                                xyzrpy_end[2] += 0.05 * drop_id
                                # Avoid overlap: place objects on predefined grid
                                if obj_name == 'plate':
                                    offset_xy = plate_drop_offset_xy_grid_4(drop_id, radius=0.06)
                                    xyzrpy_end[0] += offset_xy[0]
                                    xyzrpy_end[1] += offset_xy[1]
                                    print(f"  Plate drop offset: {offset_xy}")
                            if reply[step_id]['action'] == "pour":
                                xyzrpy_end[2] += 0.25
                                # xyzrpy_end[3] = 0
                                # xyzrpy_end[4] = math.pi

                                ref_yaw = float(robot_initial_pose[5])
                                yaw_candidates = [
                                    float(xyzrpy_end[5]),
                                    float(xyzrpy_end[5] + np.pi),
                                    float(xyzrpy_end[5] - np.pi),
                                ]
                                best_yaw = min(
                                    yaw_candidates,
                                    key=lambda y: abs(np.arctan2(np.sin(y - ref_yaw), np.cos(y - ref_yaw))),
                                )
                                xyzrpy_end[5] = shortest_yaw(ref_yaw, best_yaw)

                            planning_ignore_ids: set[int] = set()
                            if carried_object_id is not None:
                                planning_ignore_ids.add(carried_object_id)
                            if action_name == "grasp":
                                target_object_id = env.object_ids.get(obj_name)
                                if target_object_id is not None:
                                    planning_ignore_ids.add(target_object_id)

                            print(f"  RRT goal pose: {np.round(xyzrpy_end[:6], 4)}")
                            traj_path = plan_rrt_connect_path(
                                env,
                                start_pose=np.asarray(robot_initial_pose[:6], dtype=np.float64),
                                goal_pose=np.asarray(xyzrpy_end[:6], dtype=np.float64),
                                gripper_width=gripper_val,
                                args=args,
                                ignore_body_ids=planning_ignore_ids,
                            )
                            traj = traj_path.T
                            env.traj_plot(traj[:3,:].T)
                            # Use dense waypoint execution while carrying an object to reduce slip.
                            step_stride = 1
                            robot_current_pose = execute_pose_trajectory(
                                env,
                                traj,
                                gripper_width=gripper_val,
                                step_stride=step_stride,
                                log_tracking=True,
                            )
                    elif action_id == 1: # wipe
                        print("wipe")
                        wipe_start_pose = np.asarray(robot_current_pose[:6], dtype=np.float64)
                        traj_path = generate_circular_wipe_trajectory(
                            env,
                            start_pose=wipe_start_pose,
                            radius=args.wipe_radius,
                            num_points=args.wipe_num_points,
                            loops=args.wipe_loops,
                        )
                        traj = traj_path.T
                        env.traj_plot(traj[:3,:].T)
                        step_stride = 1
                        robot_current_pose = execute_pose_trajectory(
                            env,
                            traj,
                            gripper_width=gripper_val,
                            step_stride=step_stride,
                            log_tracking=False,
                        )
                    elif action_id == 2: # pour
                        print("pour")
                        pour_start_pose = np.asarray(robot_current_pose[:6], dtype=np.float64)
                        traj_path = generate_pour_trajectory(
                            start_pose=pour_start_pose,
                            pour_angle_deg=args.pour_angle_deg,
                            rot_weight=args.rrt_rot_weight,
                            return_to_start=True,
                        )
                        traj = traj_path.T
                        env.traj_plot(traj[:3,:].T)
                        step_stride = 1
                        robot_current_pose = execute_pose_trajectory(
                            env,
                            traj,
                            gripper_width=gripper_val,
                            step_stride=step_stride,
                            log_tracking=False,
                        )
                    elif action_id == 4: # drop
                        # Release in place first so the object can separate under gravity
                        # before the arm retreats upward.
                        keep_pose = np.asarray(robot_current_pose[:6], dtype=np.float32)
                        env.robot.move_ee(keep_pose, control_method="end")
                        released = False
                        for _ in range(240):
                            env.robot.move_gripper(0.08)
                            env.step_simulation()
                            if not env._has_any_finger_contact():
                                released = True
                                # Give the object a short extra settling window after contact breaks.
                                for _ in range(24):
                                    env.robot.move_gripper(0.08)
                                    env.step_simulation()
                                break
        
                        retreat_pose = robot_current_pose.copy()
                        retreat_pose[2] += 0.06
                        retreat_pose[6] = 0.08
                        env.robot.move_ee(np.asarray(retreat_pose[:6], dtype=np.float32), control_method="end")
                        for _ in range(80 if released else 120):
                            env.robot.move_gripper(0.08)
                            env.step_simulation()
                        robot_current_pose = retreat_pose
                        gripper_val = 0.08
                        carried_object_id = None
        
                        # env.robot.move_gripper(0.08)
                        # for _ in range(480):  # ~4× more time
                        #     env.step_simulation()
        
                        # gripper_val= 0.08
                        # traj_7DoF = np.append(traj[:, i], gripper_val)
                        # pt = JointTrajectoryPoint()
                        # pt.positions = traj_7DoF.tolist()
                        # pt.time_from_start = Duration(sec=int(i * dt), nanosec=int((i * dt % 1) * 1e9))
                        # msg.points.append(pt)
                    elif action_id == 3: # grasp
                        print("grasp")
                        # # grasp: move to target and close gripper
        
        
                        # robot_current_pose[6] = 0.00
                        # env.step(robot_current_pose, control_method="end")
                        # gripper_val = 0.00
                        # # Briefly keep clamping after grasp to stabilize contact before transport.
                        # for _ in range(180):
                        #     env.robot.move_gripper(0.00)
                        #     env.step_simulation()
        
                        robot_current_pose[6] = 0.02
                        env.step(robot_current_pose, control_method="end")
        
                        # gripper_val = float(env.robot.get_gripper_width())
                        # import pdb; pdb.set_trace()
                        gripper_val = 0.00
                        gripper_val = max(0.008, float(env.robot.get_gripper_width()) - 0.0045)
                        if obj_name == "cup" or obj_name == "sponge":
                            gripper_val = 0.00
                        else:
                            gripper_val = max(0.008, float(env.robot.get_gripper_width()) - 0.01)
                        print(f" float(env.robot.get_gripper_width()): {float(env.robot.get_gripper_width())}")
                        robot_current_pose[6] = gripper_val
        
                        for _ in range(180):
                            env.robot.move_gripper(gripper_val)
                            env.step_simulation()
                        carried_object_id = env.object_ids.get(obj_name)
        
                        # env.robot.move_gripper(0.00)
                        # import pdb; pdb.set_trace()
                        # for _ in range(480):  # ~4× more time
                        #     env.step_simulation()
                        # gripper_val= 0.00
                        # traj_7DoF = np.append(traj[:, i], gripper_val)
                    #     pt = JointTrajectoryPoint()
                    #     pt.positions = traj_7DoF.tolist()
                    #     pt.time_from_start = Duration(sec=int(i * dt), nanosec=int((i * dt % 1) * 1e9))
                    #     msg.points.append(pt)
            # camera_node.traj_pub.publish(msg)
            #===============================================================

            success = env.check_success()
            results.append(success)
            if args.num_runs > 1 and success:
                print(f"\033[32mTrial {trial_idx + 1} completed successfully.\033[0m")
        except Exception as e:
            if args.num_runs > 1:
                print(f"\033[31mTrial {trial_idx + 1} failed: {e}\033[0m")
            else:
                print(f"\033[31mFailed: {e}\033[0m")
            import traceback
            traceback.print_exc()
            results.append(False)

    if args.num_runs > 1:
        n_ok = sum(results)
        print(f"\n\033[1m=== Success Rate: {n_ok}/{args.num_runs} ({100.0 * n_ok / args.num_runs:.1f}%) ===\033[0m")

    shutdown_ros(executor)


if __name__ == "__main__":
    main()
    import pdb; pdb.set_trace()

# python test_mani_env_w_llm.py --vis --show-camera --subscribe-camera --subscriber-log-interval 5
# --prompt "prepare breakfast with all fruits" YES
# --prompt "clear the table: put all objects into bin" YES
# --prompt "serve multiple cups of drink" YES
# --prompt "clear the table: put all objects into bin, and clean the table" 
# --prompt "serve three plates of breakfast"
# --prompt "clean the table"


# python test_mani_env_w_llm.py --vis --show-camera --subscribe-camera --subscriber-log-interval 5 --prompt "prepare breakfast with all fruits in plate" --checkpoint_path /home/mhumais/Huang/anygrasp_sdk/grasp_detection/log/checkpoint_detection.tar --top_down_grasp --grasp-center-offset-z -0.105 --num-runs 1  --scene-configs scene_4.json