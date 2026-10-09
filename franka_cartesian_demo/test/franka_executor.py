#!/usr/bin/env python3
import os
import select
import sys
import threading
import time

def _fail_on_mixed_ros_python():
    python_mm = f"{sys.version_info.major}.{sys.version_info.minor}"
    incompatible = []
    for path in sys.path:
        if "site-packages" not in path:
            continue
        if "/opt/ros/jazzy/" in path and f"/python{python_mm}/" not in path:
            incompatible.append(path)
        elif "/franka_ros2_ws/install/" in path and f"/python{python_mm}/" not in path:
            incompatible.append(path)

    if incompatible:
        details = "\n".join(f"  - {path}" for path in incompatible)
        raise RuntimeError(
            "Mixed ROS Python environment detected.\n"
            f"Current interpreter: {sys.executable} (Python {python_mm})\n"
            "Incompatible ROS paths on PYTHONPATH:\n"
            f"{details}\n"
            "Launch this demo from a clean Jazzy shell, or use "
            "'franka_cartesian_demo/test/run_auto_demo.sh'."
        )


_fail_on_mixed_ros_python()

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from std_msgs.msg import Bool, Float64MultiArray
from tf2_ros import Buffer, TransformListener

from pymoveit2 import MoveIt2
from moveit_msgs.srv import GetCartesianPath, GetMotionPlan

from rclpy.action import ActionClient
from franka_msgs.action import Grasp, Homing, Move
import argparse

class StopFlag:
    def __init__(self):
        self._stop = False
        self._lock = threading.Lock()

    def set(self):
        with self._lock:
            self._stop = True

    def is_set(self):
        with self._lock:
            return self._stop


class FrankaZBounce(Node):
    def __init__(self, stop_flag: StopFlag | None = None, gripper_stop_flag: StopFlag | None = None):
        super().__init__("franka_z_bounce_demo")

        self.cb_group = ReentrantCallbackGroup()
        # self.stop_flag = stop_flag if stop_flag is not None else StopFlag()

        self._stopflag_sub = self.create_subscription(
            Bool, "/move_control_flag", self.move_control_flag, 10
        )
        # self.gripper_stop_flag = gripper_stop_flag if gripper_stop_flag is not None else StopFlag()
        self._gripperflag_sub = self.create_subscription(
            Bool, "/gripper_control_flag", self.gripper_control_flag, 10
        )
        self.latest_traj_7dof: list[float] | None = None
        self._traj_sub = self.create_subscription(
            Float64MultiArray, "/traj_7dof", self._on_traj_7dof, 10
        )

        # FR3 naming in franka_fr3_moveit_config
        self.base_link = "fr3_link0"
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

        # Create MoveIt2 interface with an initial guess for EE link.
        self.ee_link = "fr3_hand_tcp"
        self.moveit2 = MoveIt2(
            node=self,
            joint_names=self.joint_names,
            base_link_name=self.base_link,
            end_effector_name=self.ee_link,
            group_name=self.group_name,
            callback_group=self.cb_group,
        )

        self.plan_client = self.create_client(GetMotionPlan, "/plan_kinematic_path")
        self.cart_client = self.create_client(GetCartesianPath, "/compute_cartesian_path")

        self.get_logger().info("Franka Z bounce demo initialized.")

        self.homing_client = ActionClient(self, Homing, "/franka_gripper/homing")
        self.move_client = ActionClient(self, Move, "/franka_gripper/move")
        self.grasp_client = ActionClient(self, Grasp, "/franka_gripper/grasp")


    def move_control_flag(self, msg: Bool):
        self.move_control_flag = msg.data
    def gripper_control_flag(self, msg: Bool):
        self.gripper_control_flag = msg.data

    def _on_traj_7dof(self, msg: Float64MultiArray):
        # Expecting 7 floats: [x, y, z, roll, pitch, yaw, gripper]
        self.latest_traj_7dof = [float(v) for v in msg.data]

    def wait_for_moveit(self, timeout_sec=20.0):
        t0 = time.time()
        while time.time() - t0 < timeout_sec:
            p = self.plan_client.wait_for_service(timeout_sec=0.5)
            c = self.cart_client.wait_for_service(timeout_sec=0.5)
            if p and c:
                self.get_logger().info("MoveIt services are available.")
                return True
        self.get_logger().error("MoveIt services not available in time.")
        return False

    def detect_ee_link(self):
        candidates = [
            "franka_hand_tcp",
            "fr3_hand_tcp",
            "franka_hand",
            "fr3_link8",
        ]
        for frame in candidates:
            if self.tf_buffer.can_transform(
                self.base_link,
                frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.3),
            ):
                self.ee_link = frame
                # Sync detected frame with MoveIt2 target link
                self.moveit2._MoveIt2__end_effector_name = self.ee_link
                self.get_logger().info(f"Using end-effector frame: {self.ee_link}")
                return True

        self.get_logger().error(
            "Could not detect EE frame. Tried: " + ", ".join(candidates)
        )
        return False

    def move_home(self):
        # Conservative home/ready-like posture
        home = [0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.7]
        self.get_logger().info("Moving to home/ready posture...")
        self.moveit2.max_velocity = 0.08
        self.moveit2.max_acceleration = 0.08
        self.moveit2.move_to_configuration(home)
        self.moveit2.wait_until_executed()

    def get_current_pose(self, timeout_sec=2.0):
        tf = self.tf_buffer.lookup_transform(
            self.base_link,
            self.ee_link,
            rclpy.time.Time(),
            timeout=Duration(seconds=timeout_sec),
        )
        p = tf.transform.translation
        q = tf.transform.rotation
        return [p.x, p.y, p.z], [q.x, q.y, q.z, q.w]

    def move_pose(self, pos, quat):
        self.moveit2.max_velocity = 0.05
        self.moveit2.max_acceleration = 0.01
        self.moveit2.move_to_pose(
            position=pos,
            quat_xyzw=quat,
            target_link=self.ee_link,
            cartesian=False,
        )
        self.moveit2.wait_until_executed()


    # grasping related functions
    def _send_goal_and_wait(self, client, goal_msg, name: str, timeout_sec=8.0):
        if not client.wait_for_server(timeout_sec=timeout_sec):
            self.get_logger().error(f"Action server not available: {name}")
            return False, None

        future = client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error(f"Goal rejected: {name}")
            return False, None

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result()
        if result is None:
            self.get_logger().error(f"No result returned: {name}")
            return False, None

        return True, result.result

    def homing(self):
        ok, result = self._send_goal_and_wait(self.homing_client, Homing.Goal(), "homing")
        if not ok:
            return False
        self.get_logger().info(f"Homing success: {result.success}")
        return result.success

    def move(self, width: float, speed: float):
        goal = Move.Goal()
        goal.width = float(width)
        goal.speed = float(speed)
        ok, result = self._send_goal_and_wait(self.move_client, goal, "move")
        if not ok:
            return False
        self.get_logger().info(f"Move success: {result.success}")
        return result.success

    def grasp(self, width: float, speed: float, force: float, eps_inner: float, eps_outer: float):
        goal = Grasp.Goal()
        goal.width = float(width)
        goal.speed = float(speed)
        goal.force = float(force)
        goal.epsilon.inner = float(eps_inner)
        goal.epsilon.outer = float(eps_outer)

        ok, result = self._send_goal_and_wait(self.grasp_client, goal, "grasp")
        if not ok:
            return False
        self.get_logger().info(f"Grasp success: {result.success}")
        return result.success

def keyboard_listener(flag: StopFlag):
    print("\nType 'p' + Enter to stop/pause loop.\n")
    while not flag.is_set():
        rlist, _, _ = select.select([sys.stdin], [], [], 0.2)
        if rlist:
            cmd = sys.stdin.readline().strip().lower()
            if cmd == "p":
                flag.set()
                return

def main():

    # parser = argparse.ArgumentParser(description="Franka gripper test utility")
    # sub = parser.add_subparsers(dest="cmd", required=True)

    # sub.add_parser("homing", help="Run gripper homing")

    # p_open = sub.add_parser("open", help="Open gripper using Move action")
    # p_open.add_argument("--width", type=float, default=0.08)
    # p_open.add_argument("--speed", type=float, default=0.05)

    # p_close = sub.add_parser("close", help="Close/grasp using Grasp action")
    # p_close.add_argument("--width", type=float, default=0.0)
    # p_close.add_argument("--speed", type=float, default=0.03)
    # p_close.add_argument("--force", type=float, default=20.0)
    # p_close.add_argument("--eps-inner", type=float, default=0.01)
    # p_close.add_argument("--eps-outer", type=float, default=0.01)

    # p_seq = sub.add_parser("sequence", help="Homing -> open -> wait Enter -> grasp -> open")
    # p_seq.add_argument("--open-width", type=float, default=0.08)
    # p_seq.add_argument("--open-speed", type=float, default=0.05)
    # p_seq.add_argument("--grasp-width", type=float, default=0.0)
    # p_seq.add_argument("--grasp-speed", type=float, default=0.03)
    # p_seq.add_argument("--grasp-force", type=float, default=20.0)
    # p_seq.add_argument("--eps-inner", type=float, default=0.01)
    # p_seq.add_argument("--eps-outer", type=float, default=0.01)

    # args = parser.parse_args()

    rclpy.init()
    node = FrankaZBounce()

    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    # stop = StopFlag()
    # t = threading.Thread(target=keyboard_listener, args=(stop,), daemon=True)
    # t.start()

    try:
        # Wait for planning services and TF tree
        if not node.wait_for_moveit(timeout_sec=25.0):
            raise RuntimeError("MoveIt not ready (missing planning services).")
        time.sleep(1.0)

        if not node.detect_ee_link():
            raise RuntimeError("EE link not found in TF tree.")

        try:
            node.move_home()
        except Exception as e:
            node.get_logger().warn(
                f"Home move failed/aborted ({e}). Continue from current pose."
            )

        pos, quat = node.get_current_pose()
        node.get_logger().info(
            f"Current pose ({node.ee_link}): x={pos[0]:.4f}, y={pos[1]:.4f}, z={pos[2]:.4f}"
        )

        dz = 0.05
        pause_s = 0.6

        while True:
            move_control_flag = node.move_control_flag
            gripper_control_flag = node.gripper_control_flag
            traj = node.latest_traj_7dof
            
            print(f"gripper_control_flag: {gripper_control_flag}")
            print(f"traj: {traj}")
            print(f"move_control_flag: {move_control_flag}")

            aa = node.get_current_pose()
            aa[0][2] += 0.05
            node.move_pose(aa[0], aa[1])
            import pdb; pdb.set_trace()

            if move_control_flag and traj:
                node.move_pose(traj[:6])
                time.sleep(pause_s)
            if gripper_control_flag: 
                if grap_value==0.08:
                    node.move(0.08, 0.05)  
                elif grap_value==0.0:
                    node.grasp(0.0, 0.03, 200, 0.01,0.01)

    except Exception as e:
        node.get_logger().error(f"Demo failed: {e}")
        raise
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    # try:
    #     # Wait for planning services and TF tree
    #     if not node.wait_for_moveit(timeout_sec=25.0):
    #         raise RuntimeError("MoveIt not ready (missing planning services).")
    #     time.sleep(1.0)

    #     if not node.detect_ee_link():
    #         raise RuntimeError("EE link not found in TF tree.")

    #     try:
    #         node.move_home()
    #     except Exception as e:
    #         node.get_logger().warn(
    #             f"Home move failed/aborted ({e}). Continue from current pose."
    #         )

    #     pos, quat = node.get_current_pose()
    #     node.get_logger().info(
    #         f"Current pose ({node.ee_link}): x={pos[0]:.4f}, y={pos[1]:.4f}, z={pos[2]:.4f}"
    #     )

    #     dz = 0.05
    #     pause_s = 0.6

    #     while not stop.is_set():
    #         pos, quat = node.get_current_pose()
    #         up = [pos[0], pos[1], pos[2] + dz]
    #         node.get_logger().info("Move +5cm in Z")
    #         node.move_pose(up, quat)
    #         time.sleep(pause_s)
    #         if stop.is_set():
    #             break

    #         pos, quat = node.get_current_pose()
    #         down = [pos[0], pos[1], pos[2] - dz]
    #         node.get_logger().info("Move -5cm in Z")
    #         node.move_pose(down, quat)
    #         time.sleep(pause_s)

    #     node.get_logger().info("Stopped by user.")

    # except Exception as e:
    #     node.get_logger().error(f"Demo failed: {e}")
    #     raise
    # finally:
    #     executor.shutdown()
    #     node.destroy_node()
    #     if rclpy.ok():
    #         rclpy.shutdown()

if __name__ == "__main__":
    main()
