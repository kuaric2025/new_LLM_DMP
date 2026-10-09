"""Quick smoke test for ManiEnv with FrankaPanda.

Runs a few end-effector moves and gripper open/close in GUI.

Usage:
  python manipulation_env/test_mani_env.py
"""

import argparse
import json
import os
import re
import sys
import threading
import time
from typing import Optional
from pathlib import Path

import cv2
import msgpack_numpy
import numpy as np

import torch
from websockets.sync.client import connect as ws_connect
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
def get_robot_joint_positions(robot) -> np.ndarray:
    return np.array(
        [p.getJointState(robot.id, joint_id)[0] for joint_id in robot.arm_controllable_joints],
        dtype=np.float32,
    )


def build_world_from_cam(eye_position, target_position, up_vector) -> np.ndarray:
    eye = np.asarray(eye_position, dtype=np.float64).reshape(3)
    target = np.asarray(target_position, dtype=np.float64).reshape(3)
    up = np.asarray(up_vector, dtype=np.float64).reshape(3)

    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    cam_up = np.cross(right, forward)
    cam_up /= np.linalg.norm(cam_up)
    down = -cam_up

    world_from_cam = np.eye(4, dtype=np.float32)
    world_from_cam[:3, 0] = right.astype(np.float32)
    world_from_cam[:3, 1] = down.astype(np.float32)
    world_from_cam[:3, 2] = forward.astype(np.float32)
    world_from_cam[:3, 3] = eye.astype(np.float32)
    return world_from_cam


def build_intrinsics_matrix(width: int, height: int, fov_deg: float) -> np.ndarray:
    fx, fy, cx, cy = compute_intrinsics(width, height, fov_deg)
    return np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def _is_gemini_rate_limit_error(err: object) -> bool:
    text = err if isinstance(err, str) else str(err)
    return (
        "429" in text
        or "RESOURCE_EXHAUSTED" in text
        or "Quota exceeded" in text
        or "rate limit" in text.lower()
    )


def _gemini_suggested_retry_seconds(err: object) -> Optional[float]:
    """Parse suggested wait from API error text, if present."""
    text = err if isinstance(err, str) else str(err)
    m = re.search(r"retry in ([0-9]+(?:\.[0-9]+)?)s", text, re.IGNORECASE)
    if m:
        return float(m.group(1))
    m = re.search(r"retryDelay['\"]\s*:\s*['\"]([0-9]+)s", text)
    if m:
        return float(m.group(1))
    return None


def _task_requires_action(prompt_text: str, action_word: str) -> bool:
    text = str(prompt_text).lower()
    return action_word.lower() in text


def _plan_has_action(plan: object, action_word: str) -> bool:
    action_word = action_word.lower()

    def _walk(node: object, depth: int = 0) -> bool:
        if depth > 7 or node is None:
            return False
        if isinstance(node, str):
            s = node.lower()
            return action_word in s
        if isinstance(node, dict):
            # Fast path on common fields first.
            for k in ("type", "step_type", "action", "label", "name", "predicate"):
                if k in node and _walk(node.get(k), depth + 1):
                    return True
            for _, v in node.items():
                if _walk(v, depth + 1):
                    return True
            return False
        if isinstance(node, (list, tuple)):
            for item in node:
                if _walk(item, depth + 1):
                    return True
            return False
        return False

    return _walk(plan)


def _refine_prompt_for_missing_action(prompt_text: str, action_word: str) -> str:
    p = str(prompt_text).strip()
    action_word = action_word.lower().strip()
    if action_word == "wipe":
        return (
            p
            + " IMPORTANT: include an explicit WIPE subtask in the symbolic plan. "
            + "After grasping sponge, execute wipe motion on table surface (back-and-forth strokes) "
            + "before placing sponge."
        )
    return p + f" IMPORTANT: include explicit action '{action_word}' in the symbolic plan."


def _compile_tiptop_instruction(prompt_text: str, scene_object_names: list[str]) -> str:
    """
    Build a structured task instruction string for TiPToP perception+translation.
    TiPToP websocket currently accepts only a single text field (`task`), so we embed
    light structure in text to reduce ambiguous pick/place-only interpretations.
    """
    p = str(prompt_text).strip()
    objs = [str(x) for x in scene_object_names if x is not None]
    obj_text = ", ".join(objs) if len(objs) > 0 else "unknown"
    hints: list[str] = []
    if _task_requires_action(p, "wipe"):
        hints.append("must include WIPE action with sponge on table surface before final place")
    if _task_requires_action(p, "pour"):
        hints.append("must include POUR action before release")
    hint_text = "; ".join(hints) if len(hints) > 0 else "optimize for semantically faithful actions"
    return (
        f"Goal: {p}\n"
        f"Scene objects: {obj_text}\n"
        f"Constraints: {hint_text}\n"
        "Return a plan that preserves action semantics, not only generic pick-and-place."
    )


def ask_tiptop_server(
    rgb: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    world_from_cam: np.ndarray,
    task_instruction: str,
    q_init: np.ndarray,
    host: str,
    port: int,
) -> dict:
    uri = f"ws://{host}:{port}"
    obs = {
        "rgb": np.asarray(rgb, dtype=np.uint8),
        "depth": np.asarray(depth, dtype=np.float32),
        "intrinsics": np.asarray(intrinsics, dtype=np.float32),
        "world_from_cam": np.asarray(world_from_cam, dtype=np.float32),
        "task": str(task_instruction),
        "q_init": np.asarray(q_init, dtype=np.float32),
    }
    print(f"Connecting to TiPToP server at {uri} ...")
    t0 = time.time()
    with ws_connect(uri, compression=None, max_size=None) as ws:
        metadata = msgpack_numpy.unpackb(ws.recv())
        print(f"TiPToP server metadata: {metadata}")
        max_plan_s = float(metadata.get("max_planning_time", 60.0))
        rgb_a, dep_a = obs["rgb"], obs["depth"]
        print(
            f"Packing observation: rgb shape={getattr(rgb_a, 'shape', None)}, "
            f"depth shape={getattr(dep_a, 'shape', None)}, task={task_instruction!r}"
        )
        t_pack = time.time()
        payload = msgpack_numpy.packb(obs)
        print(f"packed in {time.time() - t_pack:.2f}s, {len(payload) / 1e6:.2f} MB (wire estimate)")
        ws.send(payload)
        print(
            f"Observation sent; waiting for plan (server may use up to ~{max_plan_s:.0f}s; "
            "this is normal — tqdm lines may still tick from the sim GUI)."
        )

        wait_done = threading.Event()

        def _heartbeat() -> None:
            t_h = time.time()
            while not wait_done.wait(5.0):
                elapsed = time.time() - t_h
                print(f"  ... still waiting for TiPToP plan ({elapsed:.0f}s elapsed)")

        hb = threading.Thread(target=_heartbeat, daemon=True)
        hb.start()
        try:
            raw = ws.recv()
            result = json.loads(raw)
        finally:
            wait_done.set()
    print(f"TiPToP planning finished in {time.time() - t0:.1f}s")
    return result


def remap_grasp_pose_robotiq_to_panda(
    pose6: np.ndarray,
    remap_xyz: np.ndarray,
    remap_rpy: np.ndarray,
) -> np.ndarray:
    """Apply fixed 6D transform: world_T_panda = world_T_robotiq @ robotiq_T_panda."""
    pose6 = np.asarray(pose6, dtype=np.float32).reshape(-1)[:6]
    remap_xyz = np.asarray(remap_xyz, dtype=np.float32).reshape(3)
    remap_rpy = np.asarray(remap_rpy, dtype=np.float32).reshape(3)

    world_T_robotiq = np.eye(4, dtype=np.float32)
    world_T_robotiq[:3, :3] = np.asarray(
        p.getMatrixFromQuaternion(p.getQuaternionFromEuler(tuple(float(v) for v in pose6[3:6]))),
        dtype=np.float32,
    ).reshape(3, 3)
    world_T_robotiq[:3, 3] = pose6[:3]

    robotiq_T_panda = np.eye(4, dtype=np.float32)
    robotiq_T_panda[:3, :3] = np.asarray(
        p.getMatrixFromQuaternion(p.getQuaternionFromEuler(tuple(float(v) for v in remap_rpy))),
        dtype=np.float32,
    ).reshape(3, 3)
    robotiq_T_panda[:3, 3] = remap_xyz

    world_T_panda = world_T_robotiq @ robotiq_T_panda
    R_out = world_T_panda[:3, :3].astype(np.float64)

    # Rotation matrix -> quaternion (xyzw)
    tr = float(np.trace(R_out))
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R_out[2, 1] - R_out[1, 2]) / s
        qy = (R_out[0, 2] - R_out[2, 0]) / s
        qz = (R_out[1, 0] - R_out[0, 1]) / s
    elif R_out[0, 0] > R_out[1, 1] and R_out[0, 0] > R_out[2, 2]:
        s = np.sqrt(1.0 + R_out[0, 0] - R_out[1, 1] - R_out[2, 2]) * 2.0
        qw = (R_out[2, 1] - R_out[1, 2]) / s
        qx = 0.25 * s
        qy = (R_out[0, 1] + R_out[1, 0]) / s
        qz = (R_out[0, 2] + R_out[2, 0]) / s
    elif R_out[1, 1] > R_out[2, 2]:
        s = np.sqrt(1.0 + R_out[1, 1] - R_out[0, 0] - R_out[2, 2]) * 2.0
        qw = (R_out[0, 2] - R_out[2, 0]) / s
        qx = (R_out[0, 1] + R_out[1, 0]) / s
        qy = 0.25 * s
        qz = (R_out[1, 2] + R_out[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R_out[2, 2] - R_out[0, 0] - R_out[1, 1]) * 2.0
        qw = (R_out[1, 0] - R_out[0, 1]) / s
        qx = (R_out[0, 2] + R_out[2, 0]) / s
        qy = (R_out[1, 2] + R_out[2, 1]) / s
        qz = 0.25 * s
    quat_xyzw = np.array([qx, qy, qz, qw], dtype=np.float32)

    out = np.zeros(6, dtype=np.float32)
    out[:3] = world_T_panda[:3, 3]
    out[3:] = np.asarray(p.getEulerFromQuaternion(quat_xyzw.tolist()), dtype=np.float32)
    return out


def execute_tiptop_plan(
    env: ManiEnv,
    plan: dict,
    gripper_action_steps: int = 8,
    sim_control_hz: float = 10.0,
    curobo_interp_hz: float = 50.0,
    grasp_center_offset_z: float = 0.0,
    align_grasp_pose_before_close: bool = False,
    remap_robotiq_to_panda_xyz: np.ndarray | None = None,
    remap_robotiq_to_panda_rpy: np.ndarray | None = None,
) -> None:
    if not isinstance(plan, dict) or "steps" not in plan:
        raise ValueError(f"Unexpected TiPToP plan format: {type(plan)}")

    current_gripper = float(env.robot.get_gripper_width())
    q_init = np.asarray(plan.get("q_init", get_robot_joint_positions(env.robot)), dtype=np.float32).reshape(-1)
    if q_init.shape[0] >= 7:
        q_init = q_init[:7]
        for _ in range(5):
            env.step(np.concatenate([q_init, np.array([current_gripper], dtype=np.float32)]), control_method="joint")

    waypoint_stride = max(1, int(round(curobo_interp_hz / sim_control_hz)))
    executed_steps = 0
    skipped_steps = 0
    offset_applied_steps = 0
    steps = plan.get("steps", [])
    last_predicted_pose6: Optional[np.ndarray] = None
    last_trajectory_end_pose6: Optional[np.ndarray] = None
    remap_xyz = (
        np.asarray(remap_robotiq_to_panda_xyz, dtype=np.float32).reshape(3)
        if remap_robotiq_to_panda_xyz is not None
        else np.zeros(3, dtype=np.float32)
    )
    remap_rpy = (
        np.asarray(remap_robotiq_to_panda_rpy, dtype=np.float32).reshape(3)
        if remap_robotiq_to_panda_rpy is not None
        else np.zeros(3, dtype=np.float32)
    )
    if np.linalg.norm(remap_xyz) > 1e-9 or np.linalg.norm(remap_rpy) > 1e-9:
        print(
            "Applying Robotiq->Panda grasp remap: "
            f"xyz={np.round(remap_xyz, 4)}, rpy={np.round(remap_rpy, 4)}"
        )
    if len(steps) > 0 and isinstance(steps[0], dict):
        print(f"TiPToP first step keys: {sorted(list(steps[0].keys()))}")

    def align_tcp_to_pose6(target_pose6: np.ndarray, max_iters: int = 8) -> tuple[float, float]:
        target_pose6 = np.asarray(target_pose6, dtype=np.float32).reshape(-1)[:6]
        target_quat = np.asarray(
            p.getQuaternionFromEuler(tuple(float(v) for v in target_pose6[3:6])),
            dtype=np.float32,
        )
        final_pos_err = float("inf")
        final_ang_err = float("inf")
        for _ in range(max_iters):
            # Use arm-only Cartesian alignment here. Calling env.step(..., "end") also runs
            # gripper surface-stop logic, which can perturb TCP during pre-close alignment.
            env.robot.move_ee(target_pose6, control_method="end")
            for _ in range(20):
                env.step_simulation()
            cur_pos, cur_quat = env.robot.get_eef()
            cur_pos = np.asarray(cur_pos, dtype=np.float32).reshape(3)
            cur_quat = np.asarray(cur_quat, dtype=np.float32).reshape(4)
            pos_err = float(np.linalg.norm(cur_pos - target_pose6[:3]))
            q_err = p.getDifferenceQuaternion(cur_quat.tolist(), target_quat.tolist())
            ang_err = float(2.0 * np.arctan2(np.linalg.norm(q_err[:3]), abs(q_err[3])))
            final_pos_err, final_ang_err = pos_err, ang_err
            if pos_err < 0.004 and ang_err < np.deg2rad(6.0):
                break
        return final_pos_err, final_ang_err

    def apply_tcp_z_offset_to_pose6(pose6: np.ndarray, offset_z: float) -> np.ndarray:
        """Return a new pose6 translated by TCP-frame +Z offset."""
        pose6 = np.asarray(pose6, dtype=np.float32).reshape(-1)[:6].copy()
        if abs(float(offset_z)) <= 1e-9:
            return pose6
        quat = np.asarray(p.getQuaternionFromEuler(tuple(float(v) for v in pose6[3:6])), dtype=np.float32)
        R = np.asarray(p.getMatrixFromQuaternion(quat.tolist()), dtype=np.float32).reshape(3, 3)
        tcp_z_world = R[:, 2]
        pose6[:3] = pose6[:3] + tcp_z_world * float(offset_z)
        return pose6

    def apply_tcp_z_offset_before_grasp(offset_z: float) -> bool:
        """Move TCP by TCP-frame +Z offset (tool-local axis)."""
        if abs(float(offset_z)) <= 1e-9:
            return False
        ee_pos0, ee_quat0 = env.robot.get_eef()
        ee_pos0 = np.asarray(ee_pos0, dtype=np.float32).reshape(3)
        ee_quat0 = np.asarray(ee_quat0, dtype=np.float32).reshape(4)
        rot_flat = p.getMatrixFromQuaternion(ee_quat0.tolist())
        R = np.asarray(rot_flat, dtype=np.float32).reshape(3, 3)
        tcp_z_world = R[:, 2]
        target_pos = ee_pos0 + tcp_z_world * float(offset_z)
        target_rpy = np.asarray(p.getEulerFromQuaternion(ee_quat0.tolist()), dtype=np.float32)

        # Closed-loop correction in Cartesian space to improve IK convergence.
        for _ in range(3):
            action = np.concatenate([target_pos, target_rpy, np.array([current_gripper], dtype=np.float32)])
            env.step(action, control_method="end")
            ee_pos_now, _ = env.robot.get_eef()
            ee_pos_now = np.asarray(ee_pos_now, dtype=np.float32).reshape(3)
            pos_err = target_pos - ee_pos_now
            if np.linalg.norm(pos_err) < 0.003:
                break
            target_pos += pos_err

        ee_pos1, _ = env.robot.get_eef()
        ee_pos1 = np.asarray(ee_pos1, dtype=np.float32).reshape(3)
        print(
            "Applied pre-grasp TCP offset in TCP Z: "
            f"offset={offset_z:.4f} m, requested_from={np.round(ee_pos0, 4)} "
            f"requested_to={np.round(target_pos, 4)} actual_to={np.round(ee_pos1, 4)} "
            f"actual_delta={np.round(ee_pos1 - ee_pos0, 4)}"
        )
        return True

    symbolic_texts = _build_symbolic_texts(steps)
    total_steps = len(steps)
    for step_i, step in enumerate(steps, start=1):
        step_type = str(step.get("type", step.get("step_type", ""))).lower()
        symbolic_text = symbolic_texts[step_i - 1]

        # Handle common aliases used by planners.
        if step_type in ("trajectory", "joint_trajectory", "motion", "move", "pour", "wipe"):
            positions_raw = (
                step.get("positions")
                if step.get("positions") is not None
                else step.get("trajectory")
            )
            if positions_raw is None:
                positions_raw = step.get("qpos")
            positions = np.asarray(positions_raw if positions_raw is not None else [], dtype=np.float32)
            if positions.size == 0:
                skipped_steps += 1
                continue
            print(f"[Executing symbolic step {step_i}/{total_steps}] {symbolic_text}")
            indices = np.arange(0, len(positions), waypoint_stride)
            if len(indices) == 0 or indices[-1] != len(positions) - 1:
                indices = np.append(indices, len(positions) - 1)
            for idx in indices:
                waypoint = np.asarray(positions[idx], dtype=np.float32).reshape(-1)
                if waypoint.shape[0] == 7:
                    action = np.concatenate([waypoint, np.array([current_gripper], dtype=np.float32)])
                elif waypoint.shape[0] == 8:
                    action = waypoint
                    current_gripper = float(waypoint[-1])
                else:
                    raise ValueError(f"Unexpected trajectory waypoint shape: {waypoint.shape}")
                env.step(action, control_method="joint")
            ee_pos, ee_quat = env.robot.get_eef()
            ee_pos = np.asarray(ee_pos, dtype=np.float32).reshape(3)
            ee_rpy = np.asarray(p.getEulerFromQuaternion(ee_quat), dtype=np.float32).reshape(3)
            last_trajectory_end_pose6 = np.concatenate([ee_pos, ee_rpy]).astype(np.float32)
            executed_steps += 1
        elif step_type in ("reach", "approach", "cartesian", "ee_pose", "pose"):
            # Some planners return Cartesian EE waypoints instead of joint trajectories.
            pose_raw = step.get("pose") if step.get("pose") is not None else step.get("target_pose")
            if pose_raw is None:
                pose_raw = step.get("xyzrpy")
            pose = np.asarray(pose_raw if pose_raw is not None else [], dtype=np.float32).reshape(-1)
            if pose.shape[0] < 6:
                skipped_steps += 1
                print(f"Skipping cartesian step without valid pose: {step}")
                continue
            print(f"[Executing symbolic step {step_i}/{total_steps}] {symbolic_text}")
            pose6 = pose[:6].copy()
            pose6 = remap_grasp_pose_robotiq_to_panda(pose6, remap_xyz, remap_rpy)
            action = np.concatenate([pose6, np.array([current_gripper], dtype=np.float32)])
            env.step(action, control_method="end")
            last_predicted_pose6 = pose6.copy()
            executed_steps += 1
        elif step_type in ("gripper", "grasp", "open", "close"):
            print(f"[Executing symbolic step {step_i}/{total_steps}] {symbolic_text}")
            action_name = str(step.get("action", step_type)).lower()
            if action_name == "close":
                pre_close_target: Optional[np.ndarray] = None
                if align_grasp_pose_before_close:
                    if last_predicted_pose6 is not None:
                        pre_close_target = last_predicted_pose6.copy()
                    elif last_trajectory_end_pose6 is not None:
                        # Joint-only plans: use achieved pre-close pose as proxy, then apply remap.
                        pre_close_target = remap_grasp_pose_robotiq_to_panda(
                            last_trajectory_end_pose6,
                            remap_xyz,
                            remap_rpy,
                        )
                        print(
                            "Joint-only plan detected: applying Robotiq->Panda remap "
                            "to pre-close trajectory endpoint before close..."
                        )
                # Combine remap+offset into one single pre-close alignment move.
                if pre_close_target is not None:
                    source_pose = pre_close_target.copy()
                    pre_close_target = apply_tcp_z_offset_to_pose6(pre_close_target, float(grasp_center_offset_z))
                    print("Aligning TCP to combined pre-close target (remap + offset) ...")
                    pos_err_m, ang_err_rad = align_tcp_to_pose6(pre_close_target)
                    ee_pos_now, ee_quat_now = env.robot.get_eef()
                    ee_pos_now = np.asarray(ee_pos_now, dtype=np.float32).reshape(3)
                    ee_rpy_now = np.asarray(p.getEulerFromQuaternion(ee_quat_now), dtype=np.float32).reshape(3)
                    print(
                        "Pre-close diagnostics: "
                        f"source_pose={np.round(source_pose, 4)} "
                        f"target_pose={np.round(pre_close_target, 4)} "
                        f"actual_pose={np.round(np.concatenate([ee_pos_now, ee_rpy_now]), 4)} "
                        f"pos_err_cm={pos_err_m * 100.0:.2f}, "
                        f"ang_err_deg={np.degrees(ang_err_rad):.2f}"
                    )
                    if abs(float(grasp_center_offset_z)) > 1e-9:
                        offset_applied_steps += 1
                elif apply_tcp_z_offset_before_grasp(float(grasp_center_offset_z)):
                    # Fallback path when no pre-close target pose is available.
                    offset_applied_steps += 1
                current_gripper = float(env.robot.gripper_range[0])
            elif action_name == "open":
                current_gripper = float(env.robot.gripper_range[1])
            else:
                raise ValueError(f"Unknown TiPToP gripper action: {action_name}")
            q_curr = get_robot_joint_positions(env.robot)
            gripper_action = np.concatenate([q_curr, np.array([current_gripper], dtype=np.float32)])
            for _ in range(gripper_action_steps):
                env.step(gripper_action, control_method="joint")
            executed_steps += 1
        else:
            print(f"Skipping unsupported TiPToP step: {step}")
            skipped_steps += 1

    if abs(float(grasp_center_offset_z)) > 1e-9 and offset_applied_steps == 0:
        print(
            "Note: --grasp-center-offset-z is set, but no Cartesian grasp/reach step was found; "
            "plan appears joint-only, so offset was not applied."
        )
    print(
        "TiPToP execution summary: "
        f"executed_steps={executed_steps}, skipped_steps={skipped_steps}, "
        f"offset_applied_steps={offset_applied_steps}"
    )
    if executed_steps == 0:
        raise RuntimeError("TiPToP plan contained no executable steps for this simulator format.")


def print_tiptop_plan_details(plan: dict) -> None:
    if not isinstance(plan, dict):
        print(f"TiPToP plan details: unexpected type={type(plan)}")
        return
    steps = plan.get("steps", [])
    print("\n=== TiPToP Plan Details ===")
    print(f"Plan keys: {sorted(list(plan.keys()))}")
    print(f"Total steps: {len(steps)}")
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            print(f"  Step {i}: non-dict step ({type(step)}) -> {step}")
            continue
        step_keys = sorted(list(step.keys()))
        step_type = str(step.get("type", step.get("step_type", "unknown")))
        print(f"  Step {i}: type={step_type}, keys={step_keys}")

        if any(k in step for k in ("positions", "trajectory", "qpos")):
            traj = (
                step.get("positions")
                if step.get("positions") is not None
                else step.get("trajectory")
            )
            if traj is None:
                traj = step.get("qpos")
            traj_arr = np.asarray(traj if traj is not None else [], dtype=np.float32)
            if traj_arr.size == 0:
                print("    trajectory: empty")
            else:
                print(f"    trajectory: shape={traj_arr.shape}")
                first_wp = np.asarray(traj_arr[0]).reshape(-1)
                last_wp = np.asarray(traj_arr[-1]).reshape(-1)
                print(f"    first waypoint: {np.round(first_wp, 4)}")
                print(f"    last  waypoint: {np.round(last_wp, 4)}")

        if "action" in step:
            print(f"    action: {step.get('action')}")

        if "object" in step:
            print(f"    object: {step.get('object')}")


def _tiptop_symbolic_step_text(step: dict) -> str:
    def _clean_name(val: object) -> Optional[str]:
        if val is None:
            return None
        text = str(val).strip()
        if not text:
            return None
        text = text.strip("()[]{}")
        text = text.replace("_", " ").strip()
        return text or None

    def _extract_from_container(val: object) -> Optional[str]:
        if val is None:
            return None
        if isinstance(val, str):
            return _clean_name(val)
        if isinstance(val, dict):
            for nested_key in ("name", "object", "id", "label", "class", "obj"):
                if nested_key in val and val.get(nested_key) is not None:
                    name = _clean_name(val.get(nested_key))
                    if name:
                        return name
            return None
        if isinstance(val, (list, tuple)):
            for item in val:
                name = _extract_from_container(item)
                if name:
                    return name
        return None

    def _extract_object_name(step_: dict) -> Optional[str]:
        candidate_keys = (
            "object",
            "obj",
            "target",
            "target_object",
            "entity",
            "item",
            "source",
            "tool",
        )
        for key in candidate_keys:
            if key in step_ and step_.get(key) is not None:
                name = _extract_from_container(step_.get(key))
                if name:
                    return name
        # Fallback from planner-style args/parameters lists.
        for key in ("args", "arguments", "parameters", "params"):
            if key in step_ and step_.get(key) is not None:
                arg_name = _extract_from_container(step_.get(key))
                if arg_name:
                    return arg_name
        return None

    def _extract_destination_name(step_: dict) -> Optional[str]:
        for key in ("destination", "to", "place_to", "container", "goal", "dest", "target_container"):
            if key in step_ and step_.get(key) is not None:
                name = _extract_from_container(step_.get(key))
                if name:
                    return name
        return None

    def _extract_verb(step_: dict, step_type_: str, action_: str) -> str:
        for key in ("semantic_action", "label", "name", "op", "operation", "verb", "predicate"):
            if key in step_ and step_.get(key) is not None:
                raw = str(step_.get(key)).strip().lower()
                if raw:
                    # Keep only first token to avoid noisy long labels.
                    token = raw.replace("_", " ").split()[0]
                    if token:
                        return token
        if action_:
            return action_
        return step_type_ if step_type_ else "move"

    step_type = str(step.get("type", step.get("step_type", "unknown"))).lower()
    action = str(step.get("action", "")).lower()
    obj = _extract_object_name(step)
    dest = _extract_destination_name(step)
    verb = _extract_verb(step, step_type, action)

    if step_type in ("trajectory", "joint_trajectory", "motion", "move", "pour", "wipe"):
        if verb in ("pick", "pickup", "grasp", "reach", "approach"):
            text = f"{verb} {obj}" if obj else "move to target"
        elif verb in ("place", "put", "drop"):
            if obj and dest:
                text = f"place {obj} to {dest}"
            elif dest:
                text = f"place object to {dest}"
            else:
                text = "place object"
        else:
            text = f"reach to {obj}" if obj else "move along planned trajectory"
        if step_type == "pour":
            text = f"pour {obj}" if obj else "pour action"
        elif step_type == "wipe":
            text = f"wipe with {obj}" if obj else "wipe action"
    elif step_type in ("gripper", "grasp", "open", "close"):
        if action in ("close", "grasp"):
            text = f"grasp {obj}" if obj else "close gripper"
        elif action == "open" or step_type == "open":
            text = f"release object to {dest}" if dest else "open gripper (release)"
        else:
            text = f"gripper action: {action or step_type}"
    elif step_type in ("pick", "pickup"):
        text = f"pick {obj}" if obj else "pick object"
    elif step_type in ("place", "put"):
        if obj and dest:
            text = f"place {obj} to {dest}"
        elif dest:
            text = f"place object to {dest}"
        else:
            text = f"place {obj}" if obj else "place object"
    else:
        if action and obj:
            text = f"{action} {obj}"
        elif action:
            text = action
        elif obj:
            text = f"interact with {obj}"
        else:
            text = f"unparsed step (type={step_type})"
    return text


def _build_symbolic_texts(steps: list, object_names: Optional[list[str]] = None) -> list[str]:
    """Create readable symbolic strings, with phase-based fallback for generic trajectory plans."""
    object_names = [str(x) for x in (object_names or []) if x is not None]
    texts: list[str] = []
    for step in steps:
        if not isinstance(step, dict):
            texts.append(str(step))
            continue
        text = _tiptop_symbolic_step_text(step)
        texts.append(text)

    # If planner output is generic ("move along planned trajectory"), infer phase from nearest gripper events.
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        text = texts[i]
        st = str(step.get("type", step.get("step_type", ""))).lower()
        if st not in ("trajectory", "joint_trajectory", "motion", "move"):
            continue
        if text != "move along planned trajectory":
            continue

        prev_close_idx = None
        prev_open_idx = None
        next_close_idx = None
        next_open_idx = None

        for j in range(i - 1, -1, -1):
            if not isinstance(steps[j], dict):
                continue
            st_j = str(steps[j].get("type", steps[j].get("step_type", ""))).lower()
            act_j = str(steps[j].get("action", st_j)).lower()
            if prev_close_idx is None and (st_j == "close" or act_j in ("close", "grasp")):
                prev_close_idx = j
            if prev_open_idx is None and (st_j == "open" or act_j == "open"):
                prev_open_idx = j
            if prev_close_idx is not None and prev_open_idx is not None:
                break

        for j in range(i + 1, len(steps)):
            if not isinstance(steps[j], dict):
                continue
            st_j = str(steps[j].get("type", steps[j].get("step_type", ""))).lower()
            act_j = str(steps[j].get("action", st_j)).lower()
            if next_close_idx is None and (st_j == "close" or act_j in ("close", "grasp")):
                next_close_idx = j
            if next_open_idx is None and (st_j == "open" or act_j == "open"):
                next_open_idx = j
            if next_close_idx is not None and next_open_idx is not None:
                break

        # Map each close/open cycle to a known object name when planner omits object labels.
        close_count = 0
        for j in range(0, i + 1):
            if not isinstance(steps[j], dict):
                continue
            st_j = str(steps[j].get("type", steps[j].get("step_type", ""))).lower()
            act_j = str(steps[j].get("action", st_j)).lower()
            if st_j == "close" or act_j in ("close", "grasp"):
                close_count += 1
        cycle_obj = object_names[max(0, close_count - 1)] if close_count > 0 and close_count - 1 < len(object_names) else (object_names[0] if len(object_names) > 0 else None)

        # Phases: pre-grasp, post-grasp transport, post-release retreat.
        if prev_close_idx is None and next_close_idx is not None:
            texts[i] = f"approach {cycle_obj} for grasp" if cycle_obj else "approach object for grasp"
        elif prev_close_idx is not None and (prev_open_idx is None or prev_close_idx > prev_open_idx):
            if next_open_idx is not None:
                texts[i] = (
                    f"transport grasped {cycle_obj} to place target"
                    if cycle_obj
                    else "transport grasped object to place target"
                )
            else:
                texts[i] = f"move while holding {cycle_obj}" if cycle_obj else "move while holding object"
        elif prev_open_idx is not None and (prev_close_idx is None or prev_open_idx > prev_close_idx):
            texts[i] = f"retreat after releasing {cycle_obj}" if cycle_obj else "retreat after release"
        else:
            texts[i] = "reposition for next action"

    return texts


def print_tiptop_symbolic_plan(plan: dict, object_names: Optional[list[str]] = None) -> None:
    if not isinstance(plan, dict):
        print("TiPToP symbolic plan: unavailable (plan is not a dict)")
        return
    steps = plan.get("steps", [])
    print("\n=== TiPToP Symbolic Plan ===")
    if not isinstance(steps, list) or len(steps) == 0:
        print("No steps in plan.")
        return

    symbolic_texts = _build_symbolic_texts(steps, object_names=object_names)
    for i, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            print(f"Step {i}: unknown action ({step})")
            continue
        text = symbolic_texts[i - 1]
        print(f"Step {i}: {text}.")

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

        self.color_pub = self.create_publisher(Image, "/camera/color/image_raw", 10)
        self.depth_pub = self.create_publisher(Image, "/camera/depth/image_rect_raw", 10)
        self.info_pub = self.create_publisher(CameraInfo, "/camera/color/camera_info", 10)

        self.fx, self.fy, self.cx, self.cy = compute_intrinsics(width, height, fov)
        self.timer = self.create_timer(1.0 / publish_rate_hz, self._publish)

        # # TF broadcaster (static - publish once)
        # if publish_tf:
        #     self._tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        #     self._publish_tf()

    def _publish(self):
        rgb, depth, _ = self.env.capture_rgbd(
            width=self.width,
            height=self.height,
            fov=self.fov,
            near=self.near,
            far=self.far,
            return_seg=False,
        )
        self.latest_rgb = rgb.copy()
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

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

from VAE_DMP_mani.data.manipulation_data.read_npy import read_first_last

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
    parser.add_argument("--tiptop-host", type=str, default=os.environ.get("TIPTOP_HOST", "127.0.0.1"))
    parser.add_argument("--tiptop-port", type=int, default=int(os.environ.get("TIPTOP_PORT", "8765")))
    parser.add_argument(
        "--tiptop-max-attempts",
        type=int,
        default=3,
        help="Retry TiPToP planning up to this many times (use 1 on Gemini free tier to save quota)",
    )
    parser.add_argument(
        "--exec-sim-control-hz",
        type=float,
        default=10.0,
        help="Lower value skips more waypoints and runs faster (default: 10.0)",
    )
    parser.add_argument(
        "--exec-curobo-interp-hz",
        type=float,
        default=50.0,
        help="Planner interpolation rate used to compute waypoint stride (default: 50.0)",
    )
    parser.add_argument(
        "--exec-gripper-steps",
        type=int,
        default=8,
        help="Simulation steps to apply each gripper open/close action (default: 8)",
    )
    parser.add_argument(
        "--no-align-grasp-pose",
        action="store_true",
        help="Disable Cartesian alignment to predicted grasp pose before close",
    )
    parser.add_argument("--grasp-remap-x", type=float, default=0.0, help="Robotiq->Panda remap tx (m)")
    parser.add_argument("--grasp-remap-y", type=float, default=0.0, help="Robotiq->Panda remap ty (m)")
    parser.add_argument("--grasp-remap-z", type=float, default=0.0, help="Robotiq->Panda remap tz (m)")
    parser.add_argument("--grasp-remap-roll", type=float, default=0.0, help="Robotiq->Panda remap roll (rad)")
    parser.add_argument("--grasp-remap-pitch", type=float, default=0.0, help="Robotiq->Panda remap pitch (rad)")
    parser.add_argument("--grasp-remap-yaw", type=float, default=0.0, help="Robotiq->Panda remap yaw (rad)")

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
    print(
        f"ManiEnv spawned objects: {sorted(env.object_ids.keys())} "
        f"(from object_config_path={args.scene_configs!r})"
    )
    if len(env.object_ids) == 0:
        print(
            "Warning: no task objects in simulation. "
            "Pass --scene-configs path to a JSON that lists them under scenes[].objects "
            "(e.g. banana, sponge, h_cups keys matching env.py)."
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
                sid = scene_config.get("seed", trial_idx)
                print(f"\n\033[36m=== Trial {trial_idx + 1}/{args.num_runs} (scene seed={sid}) ===\033[0m")
        else:
            if args.num_runs > 1:
                if use_randomize:
                    print(f"\n\033[36m=== Trial {trial_idx + 1}/{args.num_runs} (randomized object poses) ===\033[0m")
                else:
                    print(f"\n\033[36m=== Trial {trial_idx + 1}/{args.num_runs} ===\033[0m")

        try:
            if scene_config is not None:
                print(f"Applying scene config for trial {trial_idx + 1}")
                env.reset(scene_config=scene_config)
            else:
                trial_rng = np.random.default_rng((args.seed + trial_idx) if args.seed is not None else None)
                env.reset(randomize_poses=use_randomize, rng=trial_rng if use_randomize else None)

            # Let physics/camera settle before grabbing perception data.
            for _ in range(20):
                env.step_simulation()
            time.sleep(0.3)
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
        
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            # Get RGB image from subscriber or camera node
            rgb_image = None
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
                print(f"Warning: No RGB image available, using file path: {image_path}")
                rgb_image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
                _, depth_image, _ = env.capture_rgbd(
                    width=args.camera_width, height=args.camera_height, fov=args.camera_fov,
                    near=0.1, far=3.0, return_seg=False,
                )
            else:
                print(f"Using RGB image from camera: shape={rgb_image.shape}, dtype={rgb_image.dtype}")

            prompt = os.environ.get('LLM_PROMPT', args.prompt)
            scene_object_names = sorted(list(env.object_ids.keys()))
            compiled_instruction = _compile_tiptop_instruction(prompt, scene_object_names)
            if triple_sub is not None and triple_sub.latest_info is not None:
                cam_info_msg = triple_sub.latest_info
                intrinsics = np.array(
                    [
                        [cam_info_msg.k[0], 0.0, cam_info_msg.k[2]],
                        [0.0, cam_info_msg.k[4], cam_info_msg.k[5]],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=np.float32,
                )
            else:
                intrinsics = build_intrinsics_matrix(
                    args.camera_width,
                    args.camera_height,
                    args.camera_fov,
                )

            world_from_cam = build_world_from_cam(
                env.camera_eye,
                env.camera_target,
                env.camera_up,
            )
            q_init = get_robot_joint_positions(env.robot)
            tiptop_result = None
            last_error = "TiPToP planning failed"
            max_attempts = max(1, int(args.tiptop_max_attempts))
            prompt_for_attempt = compiled_instruction
            for attempt in range(1, max_attempts + 1):
                tiptop_result = ask_tiptop_server(
                    rgb=rgb_image,
                    depth=depth_image,
                    intrinsics=intrinsics,
                    world_from_cam=world_from_cam,
                    task_instruction=prompt_for_attempt,
                    q_init=q_init,
                    host=args.tiptop_host,
                    port=args.tiptop_port,
                )
                # import pdb; pdb.set_trace()
                if tiptop_result.get("success", False):
                    # Self-correction loop: if user asked for wipe but plan omitted wipe, re-prompt planner.
                    if _task_requires_action(prompt, "wipe"):
                        plan_candidate = tiptop_result.get("plan", {})
                        if not _plan_has_action(plan_candidate, "wipe"):
                            last_error = (
                                "Planner returned success but no explicit 'wipe' action found; "
                                "replanning with clarified instruction."
                            )
                            print(f"TiPToP attempt {attempt}/{max_attempts}: {last_error}")
                            if attempt < max_attempts:
                                prompt_for_attempt = _refine_prompt_for_missing_action(prompt_for_attempt, "wipe")
                                continue
                    break

                last_error = tiptop_result.get("error", "TiPToP planning failed")
                print(f"TiPToP attempt {attempt}/{max_attempts} failed: {last_error}")
                if attempt < max_attempts:
                    if _is_gemini_rate_limit_error(last_error):
                        wait_s = _gemini_suggested_retry_seconds(last_error)
                        if wait_s is None:
                            wait_s = 45.0
                        wait_s = float(np.clip(wait_s, 5.0, 300.0))
                        if "PerDay" in str(last_error) or "PerDayPerProject" in str(
                            last_error
                        ):
                            print(
                                "Note: This often means Gemini free-tier daily quota for this model "
                                "is exhausted (not fixable by waiting a minute). Enable billing / use "
                                "another API key / wait until quota resets. See "
                                "https://ai.google.dev/gemini-api/docs/rate-limits"
                            )
                        print(
                            f"Rate limited — sleeping {wait_s:.1f}s before next TiPToP attempt "
                            f"(avoid burning retries; use --tiptop-max-attempts 1 on free tier)."
                        )
                        time.sleep(wait_s)
                    else:
                        for _ in range(10):
                            env.step_simulation()
                        time.sleep(0.2)

            if tiptop_result is None or not tiptop_result.get("success", False):
                raise RuntimeError(last_error)
            tiptop_plan = tiptop_result.get("plan")
            print("\n=== TiPToP Plan Summary ===")
            print(json.dumps({
                "success": tiptop_result.get("success"),
                "n_steps": len(tiptop_plan.get("steps", [])) if isinstance(tiptop_plan, dict) else None,
                "timing": tiptop_result.get("server_timing"),
            }, indent=2))
            print_tiptop_symbolic_plan(tiptop_plan, object_names=scene_object_names)
            # print_tiptop_plan_details(tiptop_plan)
            execute_tiptop_plan(
                env,
                tiptop_plan,
                gripper_action_steps=max(1, int(args.exec_gripper_steps)),
                sim_control_hz=max(1.0, float(args.exec_sim_control_hz)),
                curobo_interp_hz=max(1.0, float(args.exec_curobo_interp_hz)),
                grasp_center_offset_z=float(args.grasp_center_offset_z),
                align_grasp_pose_before_close=not args.no_align_grasp_pose,
                remap_robotiq_to_panda_xyz=np.array(
                    [args.grasp_remap_x, args.grasp_remap_y, args.grasp_remap_z],
                    dtype=np.float32,
                ),
                remap_robotiq_to_panda_rpy=np.array(
                    [args.grasp_remap_roll, args.grasp_remap_pitch, args.grasp_remap_yaw],
                    dtype=np.float32,
                ),
            )
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

# python /home/mhumais/Huang/DMP/llm_mani/test_mani_env_w_llm_tiptop.py   --tiptop-host 127.0.0.1   --tiptop-port 8765   --tiptop-max-attempts 5   --checkpoint_path /home/mhumais/Huang/anygrasp_sdk/grasp_detection/log/checkpoint_detection.tar   --show-camera --subscribe-camera --subscriber-log-interval 5   --prompt "clear objects into bin and clean the table" --vis --show-camera --scene-configs /home/mhumais/Huang/DMP/llm_mani/T2_scene_1.json --grasp-center-offset-z 0.085 --exec-sim-control-hz 8 --exec-gripper-steps 4



# python /home/mhumais/Huang/DMP/llm_mani/test_mani_env_w_llm_tiptop.py   --tiptop-host 127.0.0.1   --tiptop-port 8765   --tiptop-max-attempts 5   --checkpoint_path /home/mhumais/Huang/anygrasp_sdk/grasp_detection/log/checkpoint_detection.tar   --show-camera --subscribe-camera --subscriber-log-interval 5   --prompt "pour water from cup to bin" --vis --show-camera --scene-configs /home/mhumais/Huang/DMP/llm_mani/T2_scene_1.json --grasp-center-offset-z 0 --exec-sim-control-hz 4 --exec-gripper-steps 4 --grasp-remap-z 0.05 --grasp-remap-yaw 0.785398 