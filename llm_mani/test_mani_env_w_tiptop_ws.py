import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

import msgpack_numpy
import numpy as np
import pybullet as p
import websockets.sync.client


def _rotation_matrix_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """SO(3) matrix -> unit quaternion [x,y,z,w]. Stable Sheppard's method."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(R))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    return (q / (np.linalg.norm(q) + 1e-12)).astype(np.float32)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manipulation_env.env import ManiEnv
from manipulation_env.robot import FrankaPanda


msgpack_numpy.patch()


def compute_intrinsics(width: int, height: int, fov_deg: float) -> np.ndarray:
    """Pinhole K consistent with ManiEnv.capture_rgbd / p.computeProjectionMatrixFOV (vertical FOV)."""
    fov_rad = np.deg2rad(float(fov_deg))
    fy = (float(height) / 2.0) / np.tan(fov_rad / 2.0)
    fx = fy
    cx = float(width) / 2.0
    cy = float(height) / 2.0
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def world_from_camera_from_lookat(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    eye = np.asarray(eye, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)

    z_axis = target - eye
    z_norm = np.linalg.norm(z_axis) + 1e-9
    z_axis = z_axis / z_norm

    x_axis = np.cross(z_axis, up)
    x_norm = np.linalg.norm(x_axis) + 1e-9
    x_axis = x_axis / x_norm

    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-9)

    T = np.eye(4, dtype=np.float32)
    T[:3, 0] = x_axis
    T[:3, 1] = y_axis
    T[:3, 2] = z_axis
    T[:3, 3] = eye
    return T


def get_joint_positions(robot: FrankaPanda) -> np.ndarray:
    q = []
    for joint_id in robot.arm_controllable_joints:
        q.append(float(p.getJointState(robot.id, joint_id)[0]))
    return np.asarray(q, dtype=np.float32)


def save_arm_state_for_planning(robot: FrankaPanda) -> np.ndarray:
    """Snapshot arm q before IK/FK scratch work (avoids GUI 'fast rehearsal' + wrong start pose)."""
    return get_joint_positions(robot).copy()


def restore_arm_state_after_planning(robot: FrankaPanda, q_arm: np.ndarray) -> None:
    set_joint_positions(robot, q_arm)


def set_joint_positions(robot: FrankaPanda, q: np.ndarray) -> None:
    q = np.asarray(q, dtype=np.float32).reshape(-1)
    for i, joint_id in enumerate(robot.arm_controllable_joints):
        p.resetJointState(robot.id, joint_id, float(q[i]))


def open_to_close_scalar_to_width(robot: FrankaPanda, scalar: float) -> float:
    scalar = float(np.clip(scalar, 0.0, 1.0))
    # TiPToP convention: 1.0=close, 0.0=open.
    w_min, w_max = robot.gripper_range
    return float((1.0 - scalar) * (w_max - w_min) + w_min)


def compose_transform(pos: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = np.asarray(p.getMatrixFromQuaternion(quat_xyzw.tolist()), dtype=np.float32).reshape(3, 3)
    T[:3, 3] = pos
    return T


def decompose_transform(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pos = np.asarray(T[:3, 3], dtype=np.float32)
    quat = _rotation_matrix_to_quat_xyzw(T[:3, :3])
    return pos, quat


def _ik_to_pose(
    robot: FrankaPanda,
    target_pos: np.ndarray,
    target_quat_xyzw: np.ndarray,
    rest_pose: np.ndarray | None = None,
) -> np.ndarray:
    if rest_pose is None:
        rest = robot.arm_rest_poses
    else:
        rest = np.asarray(rest_pose, dtype=np.float32).reshape(-1).tolist()
    try:
        joint_poses = p.calculateInverseKinematics(
            robot.id,
            robot.eef_id,
            target_pos.tolist(),
            target_quat_xyzw.tolist(),
            robot.arm_lower_limits,
            robot.arm_upper_limits,
            robot.arm_joint_ranges,
            rest,
            maxNumIterations=120,
            residualThreshold=1e-5,
        )
    except TypeError:
        joint_poses = p.calculateInverseKinematics(
            robot.id,
            robot.eef_id,
            target_pos.tolist(),
            target_quat_xyzw.tolist(),
            robot.arm_lower_limits,
            robot.arm_upper_limits,
            robot.arm_joint_ranges,
            rest,
            maxNumIterations=120,
        )
    return np.asarray(joint_poses[: robot.arm_num_dofs], dtype=np.float32)


def convert_tiptop_waypoint_to_panda(
    robot: FrankaPanda,
    q_waypoint: np.ndarray,
    tiptop_to_panda_offset_xyzrpy: np.ndarray,
    rest_panda: np.ndarray | None = None,
) -> np.ndarray:
    q_waypoint = np.asarray(q_waypoint, dtype=np.float32)
    set_joint_positions(robot, q_waypoint)

    pos, quat = robot.get_eef()
    world_T_tiptop = compose_transform(np.asarray(pos, dtype=np.float32), np.asarray(quat, dtype=np.float32))
    offset_quat = np.asarray(
        p.getQuaternionFromEuler(tuple(float(v) for v in tiptop_to_panda_offset_xyzrpy[3:6])),
        dtype=np.float32,
    )
    tiptop_T_panda = compose_transform(
        np.asarray(tiptop_to_panda_offset_xyzrpy[:3], dtype=np.float32),
        offset_quat,
    )
    world_T_panda = world_T_tiptop @ tiptop_T_panda
    target_pos, target_quat = decompose_transform(world_T_panda)
    # Use previous *Panda* configuration as null-space rest when chaining waypoints.
    # TiPToP joint targets are a different robot model; resting IK on them yields branch
    # switches that look like detours in Cartesian space.
    ik_rest = rest_panda if rest_panda is not None else q_waypoint
    q_sol = _ik_to_pose(robot, target_pos, target_quat, rest_pose=ik_rest)
    # Orientation-aware refinement: iterate a few times if angular error is still high.
    for _ in range(3):
        set_joint_positions(robot, q_sol)
        _, got_quat = robot.get_eef()
        ang_err_deg = float(np.degrees(_quat_angle_error_rad(np.asarray(got_quat, dtype=np.float32), target_quat)))
        if ang_err_deg <= 2.0:
            break
        q_sol = _ik_to_pose(robot, target_pos, target_quat, rest_pose=q_sol)
    return q_sol


def _quat_angle_error_rad(q1_xyzw: np.ndarray, q2_xyzw: np.ndarray) -> float:
    q1 = np.asarray(q1_xyzw, dtype=np.float64)
    q2 = np.asarray(q2_xyzw, dtype=np.float64)
    q1 = q1 / (np.linalg.norm(q1) + 1e-12)
    q2 = q2 / (np.linalg.norm(q2) + 1e-12)
    dot = float(np.clip(abs(np.dot(q1, q2)), -1.0, 1.0))
    return 2.0 * float(np.arccos(dot))


def _collect_trajectory_waypoints(plan_steps: list[dict], max_waypoints: int) -> np.ndarray:
    for step in plan_steps:
        if str(step.get("type", "")) != "trajectory":
            continue
        positions = np.asarray(step.get("positions", []), dtype=np.float32)
        if len(positions) == 0:
            continue
        if len(positions) <= max_waypoints:
            return positions
        idxs = np.linspace(0, len(positions) - 1, max_waypoints, dtype=int)
        return positions[idxs]
    return np.zeros((0, 7), dtype=np.float32)


def auto_calibrate_tiptop_to_panda_zyaw(
    robot: FrankaPanda,
    plan_steps: list[dict],
    base_offset_xyzrpy: np.ndarray,
    z_search_half_span: float,
    z_search_num: int,
    yaw_search_half_span_deg: float,
    yaw_search_num: int,
    max_waypoints: int,
) -> np.ndarray:
    base_offset = np.asarray(base_offset_xyzrpy, dtype=np.float32).copy()
    sample_waypoints = _collect_trajectory_waypoints(plan_steps, max_waypoints=max_waypoints)
    if len(sample_waypoints) == 0:
        print("[calib] No trajectory waypoints found. Skip auto calibration.")
        return base_offset

    q_before = get_joint_positions(robot)
    best_offset = base_offset.copy()
    best_score = np.inf
    z_candidates = np.linspace(
        float(base_offset[2] - z_search_half_span),
        float(base_offset[2] + z_search_half_span),
        int(max(2, z_search_num)),
        dtype=np.float32,
    )
    yaw_candidates = np.linspace(
        float(base_offset[5] - np.deg2rad(yaw_search_half_span_deg)),
        float(base_offset[5] + np.deg2rad(yaw_search_half_span_deg)),
        int(max(2, yaw_search_num)),
        dtype=np.float32,
    )
    print(
        f"[calib] Searching z in [{z_candidates[0]:.4f}, {z_candidates[-1]:.4f}] "
        f"({len(z_candidates)} candidates), yaw in "
        f"[{np.degrees(yaw_candidates[0]):.1f}, {np.degrees(yaw_candidates[-1]):.1f}] deg "
        f"({len(yaw_candidates)} candidates)"
    )

    try:
        for z in z_candidates:
            for yaw in yaw_candidates:
                test_offset = base_offset.copy()
                test_offset[2] = float(z)
                test_offset[5] = float(yaw)
                pos_errs = []
                ang_errs = []

                for q_tip in sample_waypoints:
                    set_joint_positions(robot, q_tip)
                    tip_pos, tip_quat = robot.get_eef()
                    world_T_tiptop = compose_transform(
                        np.asarray(tip_pos, dtype=np.float32),
                        np.asarray(tip_quat, dtype=np.float32),
                    )
                    offset_quat = np.asarray(
                        p.getQuaternionFromEuler(tuple(float(v) for v in test_offset[3:6])),
                        dtype=np.float32,
                    )
                    tiptop_T_panda = compose_transform(np.asarray(test_offset[:3], dtype=np.float32), offset_quat)
                    desired_world_T_panda = world_T_tiptop @ tiptop_T_panda
                    desired_pos, desired_quat = decompose_transform(desired_world_T_panda)

                    q_panda = _ik_to_pose(robot, desired_pos, desired_quat)
                    set_joint_positions(robot, q_panda)
                    got_pos, got_quat = robot.get_eef()

                    pos_errs.append(float(np.linalg.norm(np.asarray(got_pos, dtype=np.float32) - desired_pos)))
                    ang_errs.append(
                        float(np.degrees(_quat_angle_error_rad(np.asarray(got_quat, dtype=np.float32), desired_quat)))
                    )

                mean_pos = float(np.mean(pos_errs))
                mean_ang_deg = float(np.mean(ang_errs))
                score = mean_pos + 0.002 * np.radians(mean_ang_deg)
                print(
                    f"[calib] z={z:.4f}, yaw={np.degrees(yaw):.1f}deg -> "
                    f"mean_pos_err={mean_pos:.4f}m, mean_ang_err={mean_ang_deg:.2f}deg"
                )

                if score < best_score:
                    best_score = score
                    best_offset = test_offset
    finally:
        set_joint_positions(robot, q_before)

    print(
        f"[calib] Best z={best_offset[2]:.4f} (base z={base_offset[2]:.4f}), "
        f"best yaw={np.degrees(best_offset[5]):.1f}deg (base yaw={np.degrees(base_offset[5]):.1f}deg)"
    )
    return best_offset


def maybe_convert_trajectory_to_panda(
    robot: FrankaPanda,
    positions: np.ndarray,
    convert_tiptop_to_panda: bool,
    tiptop_to_panda_offset_xyzrpy: np.ndarray,
    chain_from_prev: np.ndarray | None = None,
) -> np.ndarray:
    if (not convert_tiptop_to_panda) or len(positions) == 0:
        return positions
    converted: list[np.ndarray] = []
    q_prev_panda: np.ndarray | None = chain_from_prev
    for q in positions:
        qp = convert_tiptop_waypoint_to_panda(
            robot,
            q,
            tiptop_to_panda_offset_xyzrpy,
            rest_panda=q_prev_panda,
        )
        converted.append(qp)
        q_prev_panda = qp
    return np.asarray(converted, dtype=np.float32)


def tcp_xy_from_q_configuration(robot: FrankaPanda, q: np.ndarray) -> np.ndarray:
    q_backup = get_joint_positions(robot)
    try:
        set_joint_positions(robot, np.asarray(q, dtype=np.float32))
        pos, _ = robot.get_eef()
        return np.asarray(pos, dtype=np.float32)[:2].copy()
    finally:
        set_joint_positions(robot, q_backup)


def enforce_min_tcp_height(
    robot: FrankaPanda,
    q_waypoint: np.ndarray,
    min_tcp_z: float,
    release_xy: np.ndarray | None = None,
    release_xy_radius: float = 0.13,
) -> np.ndarray:
    q_waypoint = np.asarray(q_waypoint, dtype=np.float32)
    set_joint_positions(robot, q_waypoint)
    pos, quat = robot.get_eef()
    tcp_pos = np.asarray(pos, dtype=np.float32)
    tcp_quat = np.asarray(quat, dtype=np.float32)
    if release_xy is not None:
        rxy = np.asarray(release_xy, dtype=np.float32).reshape(2)
        horiz = float(np.linalg.norm(tcp_pos[:2] - rxy))
        if horiz <= float(release_xy_radius):
            return q_waypoint
    if float(tcp_pos[2]) >= float(min_tcp_z):
        return q_waypoint
    target_pos = tcp_pos.copy()
    target_pos[2] = float(min_tcp_z)
    return _ik_to_pose(robot, target_pos, tcp_quat, rest_pose=q_waypoint)


def lift_waypoints_tcp_uniform_z(
    robot: FrankaPanda,
    positions: np.ndarray,
    delta_z: float,
    rest_start: np.ndarray | None = None,
) -> np.ndarray:
    """Raise every waypoint TCP by a fixed dz (used for pre-drop carried place path)."""
    positions = np.asarray(positions, dtype=np.float32)
    dz = float(max(0.0, float(delta_z)))
    if len(positions) == 0 or dz <= 0.0:
        return positions
    out: list[np.ndarray] = []
    q_prev: np.ndarray | None = None if rest_start is None else np.asarray(rest_start, dtype=np.float32).reshape(-1)
    for q in positions:
        q_i = np.asarray(q, dtype=np.float32).reshape(-1)
        set_joint_positions(robot, q_i)
        pos, quat = robot.get_eef()
        pos = np.asarray(pos, dtype=np.float32)
        quat = np.asarray(quat, dtype=np.float32)
        target_pos = pos.copy()
        target_pos[2] = float(pos[2] + dz)
        rest = q_prev if q_prev is not None else q_i
        q_new = _ik_to_pose(robot, target_pos, quat, rest_pose=rest)
        out.append(q_new)
        q_prev = q_new
    return np.asarray(out, dtype=np.float32)


def blend_open_pose_height_into_trajectory(
    robot: FrankaPanda,
    positions: np.ndarray,
    q_start_anchor: np.ndarray,
    start_extra_z: float,
) -> np.ndarray:
    """Blend an initial TCP z offset into a trajectory, ending at zero on the last waypoint.

    - First waypoint is forced to the current executed pose (q_start_anchor).
    - Last waypoint remains unchanged in Cartesian z (zero extra offset).
    - Intermediate waypoints use linearly decaying z offset.
    """
    positions = np.asarray(positions, dtype=np.float32)
    q_anchor = np.asarray(q_start_anchor, dtype=np.float32).reshape(-1)
    dz0 = float(max(0.0, float(start_extra_z)))
    if len(positions) == 0:
        return positions
    if len(positions) == 1:
        return np.asarray([q_anchor], dtype=np.float32)
    if dz0 <= 1e-4:
        out = positions.copy()
        out[0] = q_anchor
        return out
    out: list[np.ndarray] = [q_anchor.copy()]
    q_prev = q_anchor.copy()
    n = len(positions)
    for i in range(1, n):
        q_i = np.asarray(positions[i], dtype=np.float32).reshape(-1)
        set_joint_positions(robot, q_i)
        pos, quat = robot.get_eef()
        pos = np.asarray(pos, dtype=np.float32)
        quat = np.asarray(quat, dtype=np.float32)
        alpha = float((n - 1 - i) / (n - 1))  # 1->0 across trajectory, last=0
        target_pos = pos.copy()
        target_pos[2] = float(pos[2] + dz0 * alpha)
        q_new = _ik_to_pose(robot, target_pos, quat, rest_pose=q_prev)
        out.append(q_new)
        q_prev = q_new
    return np.asarray(out, dtype=np.float32)


def trim_leading_descent_while_carrying(
    robot: FrankaPanda,
    positions: np.ndarray,
    gripper_scalar: float,
    z_tolerance: float = 0.022,
) -> np.ndarray:
    """Drop leading waypoints whose TCP z is still below the *current* TCP height.

    After a synthetic post-grasp lift, TiPToP's next segment often still starts with
    low-height grasps; executing those makes the arm dive back down before the planned lift.
    """
    positions = np.asarray(positions, dtype=np.float32)
    if gripper_scalar < 0.5 or len(positions) <= 1:
        return positions
    q_backup = get_joint_positions(robot)
    try:
        p0, _ = robot.get_eef()
        z_now = float(np.asarray(p0, dtype=np.float32)[2])
        i = 0
        while i < len(positions) - 1:
            set_joint_positions(robot, positions[i])
            pi, _ = robot.get_eef()
            zi = float(np.asarray(pi, dtype=np.float32)[2])
            if zi >= z_now - float(z_tolerance):
                break
            i += 1
        if i == 0:
            return positions
        print(
            f"[plan] Trimmed {i} leading waypoint(s): redundant descent vs current TCP z "
            f"(z_now={z_now:.3f}m)"
        )
        return np.asarray(positions[i:], dtype=np.float32).copy()
    finally:
        set_joint_positions(robot, q_backup)


def trim_leading_joint_backtrack(
    positions: np.ndarray,
    q_anchor: np.ndarray,
    max_joint_delta_rad: float,
) -> np.ndarray:
    """Drop early waypoints that are far from the currently executed boundary pose.

    This keeps the first waypoint of the next carried segment consistent with the
    lifted pose that was just executed, instead of replaying stale planner prefix.
    """
    positions = np.asarray(positions, dtype=np.float32)
    if len(positions) <= 1:
        return positions
    q_anchor = np.asarray(q_anchor, dtype=np.float32).reshape(-1)
    th = float(max(0.0, max_joint_delta_rad))
    if th <= 0.0:
        return positions
    i = 0
    while i < len(positions) - 1:
        d = float(np.max(np.abs(np.asarray(positions[i], dtype=np.float32).reshape(-1) - q_anchor)))
        if d <= th:
            break
        i += 1
    if i == 0:
        return positions
    print(
        f"[plan] Trimmed {i} leading waypoint(s): inconsistent with previous boundary "
        f"(joint max-delta threshold={th:.3f} rad)"
    )
    return np.asarray(positions[i:], dtype=np.float32).copy()


def interpolate_joint_path(q_start: np.ndarray, q_end: np.ndarray, steps: int) -> np.ndarray:
    q_start = np.asarray(q_start, dtype=np.float32)
    q_end = np.asarray(q_end, dtype=np.float32)
    steps = int(max(1, steps))
    if steps == 1:
        return q_end[None, :]
    alphas = np.linspace(0.0, 1.0, steps, dtype=np.float32)
    return np.asarray([(1.0 - a) * q_start + a * q_end for a in alphas], dtype=np.float32)


def execute_joint_path(
    env: ManiEnv,
    robot: FrankaPanda,
    q_path: np.ndarray,
    gripper_scalar: float,
    waypoint_sleep_s: float,
) -> None:
    width = open_to_close_scalar_to_width(robot, gripper_scalar)
    for q in np.asarray(q_path, dtype=np.float32):
        action = np.concatenate([q.astype(np.float32), np.array([width], dtype=np.float32)])
        env.step(action, control_method="joint")
        env.step_simulation()
        if waypoint_sleep_s > 0:
            time.sleep(waypoint_sleep_s)


def do_post_grasp_lift(
    env: ManiEnv,
    robot: FrankaPanda,
    gripper_scalar: float,
    lift_delta_z: float,
    lift_steps: int,
    waypoint_sleep_s: float,
) -> None:
    q_curr = get_joint_positions(robot)
    pos, quat = robot.get_eef()
    pos = np.asarray(pos, dtype=np.float32)
    quat = np.asarray(quat, dtype=np.float32)
    target_pos = pos.copy()
    target_pos[2] = float(pos[2] + max(0.0, float(lift_delta_z)))
    q_lift = _ik_to_pose(robot, target_pos, quat, rest_pose=q_curr)
    q_path = interpolate_joint_path(q_curr, q_lift, steps=lift_steps)
    execute_joint_path(env, robot, q_path, gripper_scalar=gripper_scalar, waypoint_sleep_s=waypoint_sleep_s)


def do_pre_release_lift(
    env: ManiEnv,
    robot: FrankaPanda,
    gripper_scalar: float,
    lift_delta_z: float,
    lift_steps: int,
    waypoint_sleep_s: float,
    xy_retreat_m: float = 0.0,
) -> None:
    dz = float(max(0.0, float(lift_delta_z)))
    xy_r = float(max(0.0, float(xy_retreat_m)))
    if dz <= 0.0 and xy_r <= 0.0:
        return
    q_curr = get_joint_positions(robot)
    pos, quat = robot.get_eef()
    pos = np.asarray(pos, dtype=np.float32)
    quat = np.asarray(quat, dtype=np.float32)
    target_pos = pos.copy()
    if xy_r > 0.0:
        base_xy = np.asarray(robot.base_pos[:2], dtype=np.float32)
        tcp_xy = pos[:2].copy()
        away = base_xy - tcp_xy
        n = float(np.linalg.norm(away))
        if n > 1e-6:
            target_pos[:2] = tcp_xy + (away / n) * xy_r
    target_pos[2] = float(pos[2] + dz)
    q_lift = _ik_to_pose(robot, target_pos, quat, rest_pose=q_curr)
    q_path = interpolate_joint_path(q_curr, q_lift, steps=lift_steps)
    execute_joint_path(env, robot, q_path, gripper_scalar=gripper_scalar, waypoint_sleep_s=waypoint_sleep_s)


def estimate_first_tcp_z_of_next_trajectory(
    robot: FrankaPanda,
    plan_steps: list[dict],
    start_idx: int,
    convert_tiptop_to_panda: bool,
    tiptop_to_panda_offset_xyzrpy: np.ndarray,
    skip_step_indices: set[int] | None = None,
) -> float | None:
    if skip_step_indices is None:
        skip_step_indices = set()
    q_backup = get_joint_positions(robot)
    try:
        for j in range(start_idx + 1, len(plan_steps)):
            if j in skip_step_indices:
                continue
            s = plan_steps[j]
            if str(s.get("type", "")) != "trajectory":
                continue
            positions = np.asarray(s.get("positions", []), dtype=np.float32)
            if len(positions) == 0:
                continue
            q0 = positions[0]
            if convert_tiptop_to_panda:
                q0 = convert_tiptop_waypoint_to_panda(
                    robot=robot,
                    q_waypoint=q0,
                    tiptop_to_panda_offset_xyzrpy=tiptop_to_panda_offset_xyzrpy,
                    rest_panda=q_backup,
                )
            set_joint_positions(robot, q0)
            pos, _ = robot.get_eef()
            return float(np.asarray(pos, dtype=np.float32)[2])
        return None
    finally:
        set_joint_positions(robot, q_backup)


def estimate_first_joint_of_next_trajectory(
    robot: FrankaPanda,
    plan_steps: list[dict],
    start_idx: int,
    convert_tiptop_to_panda: bool,
    tiptop_to_panda_offset_xyzrpy: np.ndarray,
    skip_step_indices: set[int] | None = None,
) -> np.ndarray | None:
    if skip_step_indices is None:
        skip_step_indices = set()
    q_backup = get_joint_positions(robot)
    try:
        for j in range(start_idx + 1, len(plan_steps)):
            if j in skip_step_indices:
                continue
            s = plan_steps[j]
            if str(s.get("type", "")) != "trajectory":
                continue
            positions = np.asarray(s.get("positions", []), dtype=np.float32)
            if len(positions) == 0:
                continue
            q0 = np.asarray(positions[0], dtype=np.float32)
            if convert_tiptop_to_panda:
                q0 = convert_tiptop_waypoint_to_panda(
                    robot=robot,
                    q_waypoint=q0,
                    tiptop_to_panda_offset_xyzrpy=tiptop_to_panda_offset_xyzrpy,
                    rest_panda=q_backup,
                )
            return np.asarray(q0, dtype=np.float32)
        return None
    finally:
        set_joint_positions(robot, q_backup)


def _densify_subsample_indices(idxs: np.ndarray, full_q: np.ndarray, max_joint_gap_rad: float) -> np.ndarray:
    """Insert original indices so consecutive frames differ by at most max_joint_gap_rad (approx.)."""
    full_q = np.asarray(full_q, dtype=np.float32)
    idx_list = sorted({int(i) for i in idxs.tolist()})
    if len(idx_list) < 2 or max_joint_gap_rad <= 0.0:
        return np.asarray(idx_list, dtype=int)
    max_gap = float(max_joint_gap_rad)
    while True:
        row: list[int] = [idx_list[0]]
        added = False
        for k in range(1, len(idx_list)):
            ia, ib = row[-1], idx_list[k]
            d = float(np.max(np.abs(full_q[ib] - full_q[ia])))
            if d > max_gap and ib > ia + 1:
                mid = (ia + ib) // 2
                if mid != ia and mid != ib:
                    row.append(mid)
                    added = True
            row.append(ib)
        idx_list = sorted(set(row))
        if not added:
            break
    return np.asarray(idx_list, dtype=int)


def subsample_trajectory(
    positions: np.ndarray,
    stride: int,
    max_joint_gap_rad: float = 0.42,
) -> np.ndarray:
    """Keep every stride-th waypoint (always last). Optionally fill huge joint gaps caused by striding."""
    positions = np.asarray(positions, dtype=np.float32)
    n = len(positions)
    if n == 0:
        return positions
    stride = int(max(1, stride))
    if stride <= 1:
        idxs = np.arange(0, n, dtype=int)
    else:
        idxs = np.arange(0, n, stride, dtype=int)
        if int(idxs[-1]) != n - 1:
            idxs = np.append(idxs, n - 1)
        idxs = np.asarray(idxs, dtype=int)
        if max_joint_gap_rad > 0.0:
            idxs = _densify_subsample_indices(idxs, positions, max_joint_gap_rad)
    return positions[idxs]


def _concat_tiptop_trajectory_chunks(
    chunks: list[np.ndarray],
    joint_dup_eps: float = 1.5e-2,
) -> np.ndarray:
    """Stack TiPToP trajectory segments; drop duplicated boundary configurations.

    cuTAMP often emits one *place* skill as consecutive trajectory steps with the
    same label. (Pick is usually left unmerged — see executor.) Concatenating
    those segments keeps TiPToP joint continuity before TiPToP→Panda conversion.
    """
    if not chunks:
        return np.zeros((0, 0), dtype=np.float32)
    out = np.asarray(chunks[0], dtype=np.float32)
    if out.ndim == 1:
        out = out.reshape(1, -1)
    for ck_raw in chunks[1:]:
        ck = np.asarray(ck_raw, dtype=np.float32)
        if ck.ndim == 1:
            ck = ck.reshape(1, -1)
        if ck.size == 0:
            continue
        if len(out) == 0:
            out = ck
            continue
        if ck.shape[1] != out.shape[1]:
            raise ValueError(f"Trajectory chunk DOF mismatch: {out.shape[1]} vs {ck.shape[1]}")
        if float(np.max(np.abs(ck[0] - out[-1]))) <= float(joint_dup_eps):
            ck_use = ck[1:]
        else:
            ck_use = ck
        if len(ck_use) == 0:
            continue
        out = np.vstack([out, ck_use])
    return out


def smooth_joint_segment(
    q_start: np.ndarray,
    q_end: np.ndarray,
    max_joint_step_rad: float,
    min_steps: int = 1,
) -> np.ndarray:
    """Interpolate in joint space with bounded step size (radians per joint)."""
    q_start = np.asarray(q_start, dtype=np.float32).reshape(-1)
    q_end = np.asarray(q_end, dtype=np.float32).reshape(-1)
    max_step = float(max(1e-4, max_joint_step_rad))
    min_steps = int(max(1, min_steps))
    max_delta = float(np.max(np.abs(q_end - q_start)))
    n_segments = max(min_steps, int(np.ceil(max_delta / max_step)))
    # `interpolate_joint_path(..., steps=N)` returns N points including both
    # endpoints. To keep each executed delta <= max_step, we need N segments and
    # therefore N+1 sampled points; skip the already-current start configuration.
    return interpolate_joint_path(q_start, q_end, steps=n_segments + 1)[1:]


def run_joint_waypoints(
    env: ManiEnv,
    robot: FrankaPanda,
    q_targets: np.ndarray,
    gripper_scalar: float,
    waypoint_sleep_s: float,
    max_joint_step_rad: float,
    auto_bridge_trigger_rad: float = 0.22,
    auto_bridge_step_rad: float = 0.12,
) -> None:
    """Execute joint trajectory, optionally smoothing from current configuration."""
    q_targets = np.asarray(q_targets, dtype=np.float32)
    if len(q_targets) == 0:
        return
    q_prev = get_joint_positions(robot)
    for q_target in q_targets:
        qt = np.asarray(q_target, dtype=np.float32).reshape(-1)
        user_step = float(max_joint_step_rad)
        max_jump = float(np.max(np.abs(qt - q_prev)))
        step_limit = user_step
        if user_step <= 0.0 and max_jump >= float(auto_bridge_trigger_rad):
            step_limit = float(auto_bridge_step_rad)
        if step_limit > 0.0:
            path = smooth_joint_segment(q_prev, qt, max_joint_step_rad=step_limit, min_steps=1)
            for q in path:
                width = open_to_close_scalar_to_width(robot, gripper_scalar)
                action = np.concatenate([q.astype(np.float32), np.array([width], dtype=np.float32)])
                env.step(action, control_method="joint")
                env.step_simulation()
                if waypoint_sleep_s > 0:
                    time.sleep(waypoint_sleep_s)
            q_prev = qt
        else:
            width = open_to_close_scalar_to_width(robot, gripper_scalar)
            action = np.concatenate([qt.astype(np.float32), np.array([width], dtype=np.float32)])
            env.step(action, control_method="joint")
            env.step_simulation()
            if waypoint_sleep_s > 0:
                time.sleep(waypoint_sleep_s)
            q_prev = qt


def wait_for_server(host: str, port: int, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except OSError:
            time.sleep(1.0)
    raise TimeoutError(f"TiPToP server ws://{host}:{port} not reachable after {timeout_s:.1f}s")


def query_tiptop_plan(
    host: str,
    port: int,
    request: dict,
) -> tuple[list[dict], dict]:
    uri = f"ws://{host}:{port}"
    packer = msgpack_numpy.Packer()
    with websockets.sync.client.connect(uri, compression=None, max_size=None) as ws:
        metadata_raw = ws.recv()
        metadata = msgpack_numpy.unpackb(metadata_raw)
        ws.send(packer.pack(request))
        response_text = ws.recv()
        response = json.loads(response_text)

    if not response.get("success", False):
        raise RuntimeError(f"TiPToP planning failed: {response.get('error', 'unknown error')}")

    plan = response["plan"]["steps"]
    return plan, metadata


def _trajectory_same_label_cluster_tag(plan_steps: list[dict], idx: int) -> str:
    """cuTAMP often splits one skill into consecutive trajectory steps with the same label."""
    if idx < 0 or idx >= len(plan_steps) or plan_steps[idx].get("type") != "trajectory":
        return ""
    label = plan_steps[idx].get("label", "")
    lo = idx
    while lo > 0 and plan_steps[lo - 1].get("type") == "trajectory" and plan_steps[lo - 1].get("label") == label:
        lo -= 1
    hi = idx
    while hi + 1 < len(plan_steps) and plan_steps[hi + 1].get("type") == "trajectory" and plan_steps[hi + 1].get("label") == label:
        hi += 1
    total = hi - lo + 1
    if total <= 1:
        return ""
    seg = idx - lo + 1
    return f"same_label_segment={seg}/{total}"


def print_tiptop_plan(plan_steps: list[dict], mode: str = "summary") -> None:
    mode = (mode or "summary").strip().lower()
    # if mode in {"raw", "both"}:
    #     print("\n=== Raw TiPToP Plan (JSON) ===")
    #     print(json.dumps({"steps": plan_steps}, indent=2))
    #     print("=== End Raw TiPToP Plan ===")
    if mode in {"summary", "both"}:
        print("\n=== TiPToP Plan (summary) ===")
        for idx, step in enumerate(plan_steps):
            step_type = str(step.get("type", "unknown"))
            label = step.get("label", "")
            semantic_action = step.get("semantic_action")
            obj = step.get("object")
            dest = step.get("destination")
            details = [f"type={step_type}"]
            if label:
                details.append(f"label={label}")
            if semantic_action:
                details.append(f"semantic_action={semantic_action}")
            if obj:
                details.append(f"object={obj}")
            if dest:
                details.append(f"destination={dest}")
            if step_type == "trajectory":
                pos = np.asarray(step.get("positions", []), dtype=np.float32)
                details.append(f"waypoints={len(pos)}")
                seg_tag = _trajectory_same_label_cluster_tag(plan_steps, idx)
                if seg_tag:
                    details.append(seg_tag)
            elif step_type == "gripper":
                details.append(f"action={step.get('action', 'unknown')}")
            print(f"[{idx:02d}] " + " | ".join(details))
        print("=== End summary ===\n")


def execute_tiptop_plan(
    env: ManiEnv,
    robot: FrankaPanda,
    plan_steps: list[dict],
    hold_steps: int = 15,
    waypoint_sleep_s: float = 0.01,
    trajectory_stride: int = 1,
    convert_tiptop_to_panda: bool = False,
    tiptop_to_panda_offset_xyzrpy: np.ndarray | None = None,
    enforce_carry_lift: bool = True,
    carry_min_tcp_z: float = 0.20,
    post_grasp_lift_delta_z: float = 0.0,
    post_grasp_lift_steps: int = 24,
    pre_release_lift_delta_z: float = 0.0,
    pre_release_lift_steps: int = 16,
    post_release_lift_delta_z: float = 0.0,
    post_release_lift_steps: int = 10,
    post_open_lift_default_dz: float = 0.1,
    adaptive_post_grasp_lift: bool = False,
    post_grasp_skip_if_next_above: float = 0.03,
    skip_step_indices: set[int] | None = None,
    max_joint_step_rad: float = 0.0,
    carry_lift_default_dz: float = 0.06,
    place_pre_release_default_dz: float = 0.05,
    place_pre_release_xy_retreat_m: float = 0.02,
    merge_same_label_trajectories: bool = True,
    max_subsample_joint_gap_rad: float = 0.42,
    carry_approach_xy_radius: float = 0.13,
    carry_lift_tcp_z_slack_m: float = 0.02,
    carry_prev_boundary_joint_consistency_rad: float = 0.35,
    place_carry_lift_delta_z: float = 0.03,
    align_open_to_next_trajectory_start: bool = True,
    open_transition_tcp_z_slack_m: float = 0.02,
) -> None:
    if skip_step_indices is None:
        skip_step_indices = set()
    if tiptop_to_panda_offset_xyzrpy is None:
        tiptop_to_panda_offset_xyzrpy = np.zeros(6, dtype=np.float32)
    gripper_scalar = 0.0
    trim_carry_descent_next_traj = False
    carry_tcp_z_floor: float | None = None  # TCP z after post-grasp lift; enforces transport vs planner dip
    carry_prev_boundary_q: np.ndarray | None = None  # post-lift boundary; next traj should start near this q
    pending_open_blend_start_q: np.ndarray | None = None
    pending_open_blend_extra_z: float = 0.0
    pending_open_tcp_z_floor: float | None = None
    idx = 0
    n_steps = len(plan_steps)
    while idx < n_steps:
        if idx in skip_step_indices:
            print(f"[plan] Step {idx}: skipped (--skip-step-indices)")
            idx += 1
            continue
        step = plan_steps[idx]
        step_type = step.get("type")
        if step_type == "trajectory":
            label = str(step.get("label", ""))
            merged_chunks: list[np.ndarray] = []
            merged_idx_list: list[int] = []
            j = idx
            allow_merge = (
                merge_same_label_trajectories and bool(label) and ("Place(" in label)
            )
            while j < n_steps:
                if j in skip_step_indices:
                    break
                sj = plan_steps[j]
                if sj.get("type") != "trajectory":
                    break
                if str(sj.get("label", "")) != label:
                    break
                pos_j = np.asarray(sj.get("positions", []), dtype=np.float32)
                if len(pos_j) > 0:
                    merged_chunks.append(pos_j)
                merged_idx_list.append(j)
                j += 1
                if not allow_merge:
                    break
            if not merged_chunks:
                print(f"[plan] Step {idx}: empty trajectory, skipping")
                idx += 1
                continue
            positions = _concat_tiptop_trajectory_chunks(merged_chunks)
            q_before_traj_scratch = save_arm_state_for_planning(robot)
            chain_prev = q_before_traj_scratch if convert_tiptop_to_panda else None
            try:
                # IK/FK scratch work calls resetJointState hundreds of times; suppress
                # GUI flashes, then restore the real arm pose before animated playback.
                p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
                positions = maybe_convert_trajectory_to_panda(
                    robot=robot,
                    positions=positions,
                    convert_tiptop_to_panda=convert_tiptop_to_panda,
                    tiptop_to_panda_offset_xyzrpy=tiptop_to_panda_offset_xyzrpy,
                    chain_from_prev=chain_prev,
                )
                original_len = len(positions)
                positions = subsample_trajectory(
                    positions,
                    trajectory_stride,
                    max_joint_gap_rad=float(max_subsample_joint_gap_rad),
                )
                if pending_open_blend_start_q is not None and len(positions) > 0:
                    positions = blend_open_pose_height_into_trajectory(
                        robot,
                        positions,
                        q_start_anchor=pending_open_blend_start_q,
                        start_extra_z=float(pending_open_blend_extra_z),
                    )
                    pending_open_blend_start_q = None
                    pending_open_blend_extra_z = 0.0
                if (
                    enforce_carry_lift
                    and gripper_scalar >= 0.5
                    and ("Place(" in label)
                    and float(place_carry_lift_delta_z) > 0.0
                ):
                    positions = lift_waypoints_tcp_uniform_z(
                        robot,
                        positions,
                        delta_z=float(place_carry_lift_delta_z),
                        rest_start=chain_prev,
                    )
                release_xy: np.ndarray | None = None
                if (
                    enforce_carry_lift
                    and gripper_scalar >= 0.5
                    and ("Place(" in label)
                    and len(positions) > 0
                ):
                    release_xy = tcp_xy_from_q_configuration(robot, positions[-1])
                if enforce_carry_lift and gripper_scalar >= 0.5:
                    r_rad = float(carry_approach_xy_radius)
                    eff_tcp_min_z = float(carry_min_tcp_z)
                    if carry_tcp_z_floor is not None:
                        eff_tcp_min_z = max(
                            eff_tcp_min_z,
                            float(carry_tcp_z_floor) - float(carry_lift_tcp_z_slack_m),
                        )
                    positions = np.asarray(
                        [
                            enforce_min_tcp_height(
                                robot,
                                q,
                                min_tcp_z=eff_tcp_min_z,
                                release_xy=release_xy,
                                release_xy_radius=r_rad,
                            )
                            for q in positions
                        ],
                        dtype=np.float32,
                    )
                if pending_open_tcp_z_floor is not None and gripper_scalar < 0.5:
                    eff_open_min_z = float(pending_open_tcp_z_floor) - float(open_transition_tcp_z_slack_m)
                    positions = np.asarray(
                        [
                            enforce_min_tcp_height(
                                robot,
                                q,
                                min_tcp_z=eff_open_min_z,
                            )
                            for q in positions
                        ],
                        dtype=np.float32,
                    )
                    print(
                        f"[plan] Open transition TCP z floor={pending_open_tcp_z_floor:.3f}m "
                        f"(next trajectory >= ~{eff_open_min_z:.3f}m)"
                    )
                if trim_carry_descent_next_traj and gripper_scalar >= 0.5:
                    positions = trim_leading_descent_while_carrying(
                        robot, positions, gripper_scalar=gripper_scalar
                    )
                    trim_carry_descent_next_traj = False
                if carry_prev_boundary_q is not None and gripper_scalar >= 0.5:
                    positions = trim_leading_joint_backtrack(
                        positions,
                        q_anchor=carry_prev_boundary_q,
                        max_joint_delta_rad=float(carry_prev_boundary_joint_consistency_rad),
                    )
            finally:
                p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
                restore_arm_state_after_planning(robot, q_before_traj_scratch)
            if len(merged_idx_list) > 1:
                span = f"steps {merged_idx_list[0]}-{merged_idx_list[-1]} ({len(merged_idx_list)} chunks)"
            else:
                span = f"step {merged_idx_list[0]}"
            print(
                f"[plan] {span}: trajectory label={label!r} | {original_len} waypoints "
                f"-> {len(positions)} after stride={max(1, int(trajectory_stride))}"
                + (
                    f" (joint-gap cap={max_subsample_joint_gap_rad:.2f} rad)"
                    if max(1, int(trajectory_stride)) > 1 and float(max_subsample_joint_gap_rad) > 0.0
                    else ""
                )
            )
            run_joint_waypoints(
                env=env,
                robot=robot,
                q_targets=positions,
                gripper_scalar=gripper_scalar,
                waypoint_sleep_s=waypoint_sleep_s,
                max_joint_step_rad=max_joint_step_rad,
            )
            if pending_open_tcp_z_floor is not None and gripper_scalar < 0.5:
                pending_open_tcp_z_floor = None
            idx = j
            continue
        elif step_type == "gripper":
            action = str(step.get("action", "")).strip().lower()
            if action not in {"open", "close"}:
                print(f"[plan] Step {idx}: unsupported gripper action '{action}', skipping")
                idx += 1
                continue
            pre_drop_dz = float(pre_release_lift_delta_z)
            if pre_drop_dz <= 0.0 and enforce_carry_lift:
                pre_drop_dz = float(place_pre_release_default_dz)
            pre_drop_xy = float(place_pre_release_xy_retreat_m) if enforce_carry_lift else 0.0
            if action == "open" and gripper_scalar >= 0.5 and (pre_drop_dz > 0.0 or pre_drop_xy > 0.0):
                print(
                    f"[plan] Pre-release hover request ignored for drop interpolation "
                    f"(dz={pre_drop_dz:.3f}m, xy_retreat={pre_drop_xy:.3f}m); "
                    "using current carry boundary snapshot instead."
                )
                curr_pos_before_drop, _ = robot.get_eef()
                carry_tcp_z_floor = float(np.asarray(curr_pos_before_drop, dtype=np.float32)[2])
                carry_prev_boundary_q = get_joint_positions(robot).copy()

            gripper_scalar = 1.0 if action == "close" else 0.0
            if action == "open":
                trim_carry_descent_next_traj = False
            else:
                pending_open_blend_start_q = None
                pending_open_blend_extra_z = 0.0
                pending_open_tcp_z_floor = None
            width = open_to_close_scalar_to_width(robot, gripper_scalar)
            print(f"[plan] Step {idx}: gripper {action} (width={width:.4f}m)")
            hold_action = np.concatenate([get_joint_positions(robot), np.array([width], dtype=np.float32)])
            for _ in range(max(1, hold_steps)):
                env.step(hold_action, control_method="joint")
                env.step_simulation()
                if waypoint_sleep_s > 0:
                    time.sleep(waypoint_sleep_s)
            # Post-grasp lift: either explicit dz, or default when carry-lift mode is on.
            post_dz = float(post_grasp_lift_delta_z)
            if post_dz <= 0.0 and enforce_carry_lift:
                post_dz = float(carry_lift_default_dz)
            if action == "close" and post_dz > 0.0:
                curr_pos, _ = robot.get_eef()
                curr_z = float(np.asarray(curr_pos, dtype=np.float32)[2])
                next_z = estimate_first_tcp_z_of_next_trajectory(
                    robot=robot,
                    plan_steps=plan_steps,
                    start_idx=idx,
                    convert_tiptop_to_panda=convert_tiptop_to_panda,
                    tiptop_to_panda_offset_xyzrpy=tiptop_to_panda_offset_xyzrpy,
                    skip_step_indices=skip_step_indices,
                )
                skip_lift = (
                    adaptive_post_grasp_lift
                    and next_z is not None
                    and (next_z - curr_z) >= float(post_grasp_skip_if_next_above)
                )
                if skip_lift:
                    print(
                        f"[plan] Skip post-grasp lift: next trajectory already goes up "
                        f"(curr_z={curr_z:.3f}, next_z={next_z:.3f})."
                    )
                    carry_tcp_z_floor = curr_z
                    carry_prev_boundary_q = get_joint_positions(robot).copy()
                    trim_carry_descent_next_traj = True
                    print(
                        f"[plan] Carry TCP z floor={carry_tcp_z_floor:.3f}m (from grasp height; planner already clears)"
                    )
                else:
                    print(f"[plan] Post-grasp lift: dz={post_dz:.3f}m")
                    do_post_grasp_lift(
                        env=env,
                        robot=robot,
                        gripper_scalar=gripper_scalar,
                        lift_delta_z=post_dz,
                        lift_steps=post_grasp_lift_steps,
                        waypoint_sleep_s=waypoint_sleep_s,
                    )
                    trim_carry_descent_next_traj = True
                    lift_pos, _ = robot.get_eef()
                    carry_tcp_z_floor = float(np.asarray(lift_pos, dtype=np.float32)[2])
                    carry_prev_boundary_q = get_joint_positions(robot).copy()
                    print(
                        f"[plan] Carry TCP z floor={carry_tcp_z_floor:.3f}m (next trajectories stay at/above "
                        f"~{carry_tcp_z_floor - float(carry_lift_tcp_z_slack_m):.3f} until place approach)"
                    )
            post_open_dz = float(post_release_lift_delta_z)
            if action == "open" and post_open_dz <= 0.0 and enforce_carry_lift:
                post_open_dz = float(post_open_lift_default_dz)
            if action == "open" and post_open_dz > 0.0:
                print(f"[plan] Post-release lift: dz={post_open_dz:.3f}m")
                do_pre_release_lift(
                    env=env,
                    robot=robot,
                    gripper_scalar=gripper_scalar,
                    lift_delta_z=post_open_dz,
                    lift_steps=post_release_lift_steps,
                    waypoint_sleep_s=waypoint_sleep_s,
                )
            if action == "open":
                q_curr_after_open = get_joint_positions(robot).copy()
                pending_open_blend_start_q = None
                pending_open_blend_extra_z = 0.0
                p_open, _ = robot.get_eef()
                pending_open_tcp_z_floor = float(np.asarray(p_open, dtype=np.float32)[2])
                print(
                    f"[plan] Open TCP z floor={pending_open_tcp_z_floor:.3f}m "
                    f"(next trajectory stays at/above ~"
                    f"{pending_open_tcp_z_floor - float(open_transition_tcp_z_slack_m):.3f})"
                )
                q_next = estimate_first_joint_of_next_trajectory(
                    robot=robot,
                    plan_steps=plan_steps,
                    start_idx=idx,
                    convert_tiptop_to_panda=convert_tiptop_to_panda,
                    tiptop_to_panda_offset_xyzrpy=tiptop_to_panda_offset_xyzrpy,
                    skip_step_indices=skip_step_indices,
                )
                if q_next is not None:
                    q_backup = get_joint_positions(robot)
                    try:
                        set_joint_positions(robot, q_curr_after_open)
                        p_curr, _ = robot.get_eef()
                        set_joint_positions(robot, q_next)
                        p_next, _ = robot.get_eef()
                        pending_open_blend_extra_z = max(
                            0.0,
                            float(np.asarray(p_curr, dtype=np.float32)[2] - np.asarray(p_next, dtype=np.float32)[2]),
                        )
                    finally:
                        set_joint_positions(robot, q_backup)
                    if pending_open_blend_extra_z > 1e-4:
                        pending_open_blend_start_q = q_curr_after_open
                        print(
                            f"[plan] Open->next trajectory z-blend: start extra z={pending_open_blend_extra_z:.3f}m, "
                            "decays to 0 at trajectory end (last waypoint unchanged)."
                        )
                    elif align_open_to_next_trajectory_start:
                        pending_open_blend_start_q = q_curr_after_open
                        jump = float(np.max(np.abs(np.asarray(q_next, dtype=np.float32) - q_curr_after_open)))
                        if jump > 1e-3:
                            print(
                                f"[plan] Open->next alignment: bridging to next trajectory start "
                                f"(max joint delta={jump:.3f} rad)"
                            )
                            run_joint_waypoints(
                                env=env,
                                robot=robot,
                                q_targets=np.asarray([q_next], dtype=np.float32),
                                gripper_scalar=gripper_scalar,
                                waypoint_sleep_s=waypoint_sleep_s,
                                max_joint_step_rad=max_joint_step_rad,
                            )
            idx += 1
        else:
            print(f"[plan] Step {idx}: unknown type '{step_type}', skipping")
            idx += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ManiEnv with TiPToP websocket planning")
    parser.add_argument("--prompt", type=str, required=True, help="Task instruction for TiPToP")
    parser.add_argument("--vis", action="store_true", help="Show PyBullet GUI")
    parser.add_argument("--scene-configs", type=str, default=None, help="Scene JSON used by ManiEnv")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fov", type=float, default=70.0)
    parser.add_argument("--camera-near", type=float, default=0.1)
    parser.add_argument("--camera-far", type=float, default=3.0)
    parser.add_argument("--tiptop-host", type=str, default="127.0.0.1")
    parser.add_argument("--tiptop-port", type=int, default=8765)
    parser.add_argument("--wait-server-timeout", type=float, default=120.0)
    parser.add_argument(
        "--plan-print",
        type=str,
        choices=("summary", "raw", "both", "none"),
        default="summary",
        help="How to print the TiPToP plan to stdout.",
    )
    parser.add_argument("--save-plan-json", type=str, default=None, help="Optional path to save returned TiPToP plan")
    parser.add_argument("--hold-steps", type=int, default=8, help="Sim steps to hold each gripper open/close action")
    parser.add_argument("--waypoint-sleep", type=float, default=0.0, help="Sleep seconds after each waypoint")
    parser.add_argument(
        "--trajectory-stride",
        type=int,
        default=2,
        help="Execute every Nth trajectory waypoint (always keeps final waypoint).",
    )
    parser.add_argument("--idle-sim", action="store_true", help="Keep sim running after execution")
    parser.add_argument(
        "--no-carry-clearance",
        action="store_true",
        help="Disable carry helpers: post-grasp lift, min TCP while carrying, and default pre-drop hover.",
    )
    parser.add_argument(
        "--carry-lift-default-dz",
        type=float,
        default=0.06,
        help="After close: vertical lift (m) when carry clearance is on and --post-grasp-lift-delta-z is 0.",
    )
    parser.add_argument(
        "--carry-min-tcp-z",
        type=float,
        default=0.20,
        help="Minimum end-effector world z while carrying an object.",
    )
    parser.add_argument(
        "--post-grasp-lift-delta-z",
        type=float,
        default=0.0,
        help="Extra lift in z (meters) immediately after gripper close.",
    )
    parser.add_argument(
        "--post-grasp-lift-steps",
        type=int,
        default=24,
        help="Interpolation steps for post-grasp lift motion.",
    )
    parser.add_argument(
        "--pre-release-lift-delta-z",
        type=float,
        default=0.0,
        help="Before open: vertical lift (m); if 0 and carry clearance on, uses --place-pre-release-dz.",
    )
    parser.add_argument(
        "--place-pre-release-dz",
        type=float,
        default=0.05,
        help="Default pre-drop vertical hover (m) when carry clearance on and --pre-release-lift-delta-z is 0.",
    )
    parser.add_argument(
        "--place-pre-release-xy-retreat",
        type=float,
        default=0.02,
        help="Before open: horizontal retreat toward robot base (m) while still closed; 0 disables.",
    )
    parser.add_argument(
        "--pre-release-lift-steps",
        type=int,
        default=16,
        help="Interpolation steps for pre-release lift motion.",
    )
    parser.add_argument(
        "--post-release-lift-delta-z",
        type=float,
        default=0.0,
        help="After open: explicit lift in z (m); if 0 and carry clearance on, uses --post-open-lift-default-dz.",
    )
    parser.add_argument(
        "--post-open-lift-default-dz",
        type=float,
        default=0.1,
        help="Default post-open vertical lift (m) when carry clearance is on and --post-release-lift-delta-z is 0.",
    )
    parser.add_argument(
        "--post-release-lift-steps",
        type=int,
        default=10,
        help="Interpolation steps for optional post-release lift motion.",
    )
    parser.add_argument(
        "--adaptive-post-grasp-lift",
        action="store_true",
        help="Skip post-grasp lift if next trajectory already starts sufficiently higher.",
    )
    parser.add_argument(
        "--post-grasp-skip-if-next-above",
        type=float,
        default=0.03,
        help="Skip post-grasp lift when next trajectory start TCP z exceeds current by this threshold (m).",
    )
    parser.add_argument(
        "--skip-step-indices",
        type=int,
        nargs="*",
        default=[],
        help="Plan step indices to skip (e.g. --skip-step-indices 1).",
    )
    parser.add_argument(
        "--no-merge-same-label-traj",
        action="store_true",
        help="Execute each trajectory step separately (default merges consecutive Place(...) chunks only).",
    )
    parser.add_argument(
        "--max-subsample-joint-gap-rad",
        type=float,
        default=0.42,
        help="After --trajectory-stride>1, add waypoints so joint jumps stay below this (0 disables).",
    )
    parser.add_argument(
        "--carry-approach-xy-radius",
        type=float,
        default=0.13,
        help="While carrying on Place: within this xy distance of the final pose, do not force carry-min-tcp-z.",
    )
    parser.add_argument(
        "--carry-lift-tcp-z-slack",
        type=float,
        default=0.02,
        help="After post-grasp lift, allow this much below measured lift TCP z when clamping (numerical slack).",
    )
    parser.add_argument(
        "--carry-prev-boundary-joint-consistency-rad",
        type=float,
        default=0.35,
        help="After a lift, trim next carried trajectory prefix until close to previous boundary joints (0 disables).",
    )
    parser.add_argument(
        "--place-carry-lift-delta-z",
        type=float,
        default=0.08,
        help="Before open on Place while carrying, lift every trajectory waypoint TCP by this z (m).",
    )
    parser.add_argument(
        "--no-open-next-align",
        action="store_true",
        help="Disable alignment move from gripper-open pose to next trajectory start.",
    )
    parser.add_argument(
        "--open-transition-tcp-z-slack",
        type=float,
        default=0.02,
        help="After open, keep first next trajectory above (open TCP z - slack).",
    )
    parser.add_argument(
        "--max-joint-step-rad",
        type=float,
        default=0.0,
        help="If >0, interpolate each joint move so max delta per step is this many radians (smoother motion).",
    )
    parser.add_argument(
        "--convert-tiptop-grasp-to-panda",
        action="store_true",
        help="Apply TiPToP grasp-frame to Panda TCP conversion on trajectory waypoints.",
    )
    parser.add_argument(
        "--tiptop-to-panda-offset-xyzrpy",
        type=float,
        nargs=6,
        default=[0.0, 0.0, 0.105, 0.0, 0.0, 0.785398],
        metavar=("X", "Y", "Z", "R", "P", "Y"),
        help="Static transform TiPToP grasp frame -> Panda TCP (m, rad). Typical Panda z is negative.",
    )
    parser.add_argument(
        "--allow-positive-tiptop-z",
        action="store_true",
        help="Do not auto-flip a positive TiPToP->Panda z offset.",
    )
    parser.add_argument(
        "--auto-calibrate-tiptop-z",
        action="store_true",
        help="Auto-search best TiPToP->Panda z/yaw offset on sampled waypoints before execution.",
    )
    parser.add_argument(
        "--auto-calib-z-half-span",
        type=float,
        default=0.03,
        help="Half span (meters) around initial z offset for auto calibration search.",
    )
    parser.add_argument(
        "--auto-calib-z-num",
        type=int,
        default=7,
        help="Number of z-offset candidates to evaluate.",
    )
    parser.add_argument(
        "--auto-calib-yaw-half-span-deg",
        type=float,
        default=20.0,
        help="Half span (degrees) around initial yaw offset for auto calibration.",
    )
    parser.add_argument(
        "--auto-calib-yaw-num",
        type=int,
        default=5,
        help="Number of yaw-offset candidates to evaluate.",
    )
    parser.add_argument(
        "--auto-calib-max-waypoints",
        type=int,
        default=10,
        help="Max sampled waypoints from first trajectory for calibration.",
    )
    return parser.parse_args()


def build_env(args: argparse.Namespace) -> tuple[ManiEnv, FrankaPanda]:
    table_path = os.path.join(os.path.dirname(__file__), "../manipulation_env/models/urdf/objects/table/table.urdf")
    bin_path = os.path.join(os.path.dirname(__file__), "../manipulation_env/models/urdf/objects/table/bin.urdf")
    banana_path = os.path.join(os.path.dirname(__file__), "../manipulation_env/models/ycb/011_banana/model.urdf")
    plate_path = os.path.join(os.path.dirname(__file__), "../manipulation_env/models/ycb/029_plate/model.urdf")
    apple_path = os.path.join(os.path.dirname(__file__), "../manipulation_env/models/ycb/013_apple/model.urdf")
    orange_path = os.path.join(os.path.dirname(__file__), "../manipulation_env/models/ycb/017_orange/model.urdf")
    pitcher_path = os.path.join(os.path.dirname(__file__), "../manipulation_env/models/ycb/019_pitcher_base/model.urdf")
    e_cups_path = os.path.join(os.path.dirname(__file__), "../manipulation_env/models/ycb/065-e_cups/model.urdf")
    i_cups_path = os.path.join(os.path.dirname(__file__), "../manipulation_env/models/ycb/065-i_cups/model.urdf")
    j_cups_path = os.path.join(os.path.dirname(__file__), "../manipulation_env/models/ycb/065-j_cups/model.urdf")
    sponge_path = os.path.join(os.path.dirname(__file__), "../manipulation_env/models/ycb/026_sponge/model.urdf")
    mug_path = os.path.join(os.path.dirname(__file__), "../manipulation_env/models/ycb/025_mug/model.urdf")
    bowl_path = os.path.join(os.path.dirname(__file__), "../manipulation_env/models/ycb/024_bowl/model.urdf")

    robot = FrankaPanda(model_path=None)
    env = ManiEnv(
        robot,
        block_path=None,
        table_path=table_path,
        bin_path=bin_path,
        banana_path=banana_path,
        plate_path=plate_path,
        apple_path=apple_path,
        orange_path=orange_path,
        pitcher_path=pitcher_path,
        e_cups_path=e_cups_path,
        i_cups_path=i_cups_path,
        j_cups_path=j_cups_path,
        sponge_path=sponge_path,
        mug_path=mug_path,
        bowl_path=bowl_path,
        vis=args.vis,
        object_config_path=args.scene_configs,
    )
    return env, robot


def main() -> None:
    args = parse_args()
    env, robot = build_env(args)
    try:
        print(f"Waiting for TiPToP server at ws://{args.tiptop_host}:{args.tiptop_port} ...")
        wait_for_server(args.tiptop_host, args.tiptop_port, args.wait_server_timeout)

        rgb, depth, _ = env.capture_rgbd(
            width=args.camera_width,
            height=args.camera_height,
            fov=args.camera_fov,
            near=args.camera_near,
            far=args.camera_far,
            return_seg=False,
        )
        intrinsics = compute_intrinsics(args.camera_width, args.camera_height, args.camera_fov)
        world_from_cam = world_from_camera_from_lookat(
            np.asarray(env.camera_eye, dtype=np.float32),
            np.asarray(env.camera_target, dtype=np.float32),
            np.asarray(env.camera_up, dtype=np.float32),
        )
        q_init = get_joint_positions(robot)

        request = {
            "rgb": rgb.astype(np.uint8),
            "depth": depth.astype(np.float32),
            "intrinsics": intrinsics.astype(np.float32),
            "world_from_cam": world_from_cam.astype(np.float32),
            "task": args.prompt,
            "q_init": q_init.astype(np.float32),
        }
        print(f"Requesting TiPToP plan for prompt: {args.prompt!r}")
        plan_steps, metadata = query_tiptop_plan(args.tiptop_host, args.tiptop_port, request)
        print(f"Connected to server metadata: {metadata}")
        print(f"Received {len(plan_steps)} plan steps")
        if args.plan_print != "none":
            print_tiptop_plan(plan_steps, mode=args.plan_print)

        if args.save_plan_json:
            out_path = Path(args.save_plan_json).expanduser().resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"steps": plan_steps}, f, indent=2)
            print(f"Saved plan to: {out_path}")

        exec_offset = np.asarray(args.tiptop_to_panda_offset_xyzrpy, dtype=np.float32)
        if args.convert_tiptop_grasp_to_panda and (not args.allow_positive_tiptop_z) and exec_offset[2] > 0.0:
            print(
                f"[warn] TiPToP->Panda z offset is positive ({exec_offset[2]:.4f}). "
                "For Panda this is usually negative; flipping sign for safety."
            )
            exec_offset[2] = -exec_offset[2]
        if args.convert_tiptop_grasp_to_panda and args.auto_calibrate_tiptop_z:
            exec_offset = auto_calibrate_tiptop_to_panda_zyaw(
                robot=robot,
                plan_steps=plan_steps,
                base_offset_xyzrpy=exec_offset,
                z_search_half_span=float(args.auto_calib_z_half_span),
                z_search_num=int(args.auto_calib_z_num),
                yaw_search_half_span_deg=float(args.auto_calib_yaw_half_span_deg),
                yaw_search_num=int(args.auto_calib_yaw_num),
                max_waypoints=int(args.auto_calib_max_waypoints),
            )

        execute_tiptop_plan(
            env=env,
            robot=robot,
            plan_steps=plan_steps,
            hold_steps=args.hold_steps,
            waypoint_sleep_s=args.waypoint_sleep,
            trajectory_stride=args.trajectory_stride,
            convert_tiptop_to_panda=args.convert_tiptop_grasp_to_panda,
            tiptop_to_panda_offset_xyzrpy=exec_offset,
            enforce_carry_lift=(not args.no_carry_clearance),
            carry_min_tcp_z=float(args.carry_min_tcp_z),
            post_grasp_lift_delta_z=float(args.post_grasp_lift_delta_z),
            post_grasp_lift_steps=int(args.post_grasp_lift_steps),
            pre_release_lift_delta_z=float(args.pre_release_lift_delta_z),
            pre_release_lift_steps=int(args.pre_release_lift_steps),
            post_release_lift_delta_z=float(args.post_release_lift_delta_z),
            post_release_lift_steps=int(args.post_release_lift_steps),
            post_open_lift_default_dz=float(args.post_open_lift_default_dz),
            adaptive_post_grasp_lift=args.adaptive_post_grasp_lift,
            post_grasp_skip_if_next_above=float(args.post_grasp_skip_if_next_above),
            skip_step_indices=set(int(v) for v in args.skip_step_indices),
            max_joint_step_rad=float(args.max_joint_step_rad),
            carry_lift_default_dz=float(args.carry_lift_default_dz),
            place_pre_release_default_dz=float(args.place_pre_release_dz),
            place_pre_release_xy_retreat_m=float(args.place_pre_release_xy_retreat),
            merge_same_label_trajectories=(not args.no_merge_same_label_traj),
            max_subsample_joint_gap_rad=float(args.max_subsample_joint_gap_rad),
            carry_approach_xy_radius=float(args.carry_approach_xy_radius),
            carry_lift_tcp_z_slack_m=float(args.carry_lift_tcp_z_slack),
            carry_prev_boundary_joint_consistency_rad=float(args.carry_prev_boundary_joint_consistency_rad),
            place_carry_lift_delta_z=float(args.place_carry_lift_delta_z),
            align_open_to_next_trajectory_start=(not args.no_open_next_align),
            open_transition_tcp_z_slack_m=float(args.open_transition_tcp_z_slack),
        )
        print("Plan execution complete.")

        if args.idle_sim:
            print("Idle sim enabled. Press Ctrl+C to exit.")
            while True:
                env.step_simulation()
                time.sleep(0.01)
    finally:
        env.close()


if __name__ == "__main__":
    main()


# python llm_mani/test_mani_env_w_tiptop_ws.py --vis --prompt "prepare breakfast with all fruits in plate" --scene-configs llm_mani/scene_1.json --convert-tiptop-grasp-to-panda --auto-calibrate-tiptop-z --auto-calib-yaw-half-span-deg 45  --auto-calibrate-tiptop-z --allow-positive-tiptop-z

# python llm_mani/test_mani_env_w_tiptop_ws.py --vis --prompt "prepare breakfast with all fruits in plate" --convert-tiptop-grasp-to-panda --auto-calibrate-tiptop-z --auto-calib-yaw-half-span-deg 20  --allow-positive-tiptop-z --auto-calibrate-tiptop-z --plan-print both   --max-joint-step-rad 0.04 --trajectory-stride=4 --scene-configs llm_mani/scene_3.json