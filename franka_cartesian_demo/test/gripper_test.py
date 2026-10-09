#!/usr/bin/env python3
import argparse
import sys
import threading
import time
from collections import deque

import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from franka_msgs.action import Grasp, Homing, Move
from std_msgs.msg import Float64MultiArray


class FrankaGripperTester(Node):
    def __init__(self):
        super().__init__("franka_gripper_tester")
        self.cb_group = ReentrantCallbackGroup()
        self.homing_client = ActionClient(
            self, Homing, "/franka_gripper/homing", callback_group=self.cb_group
        )
        self.move_client = ActionClient(
            self, Move, "/franka_gripper/move", callback_group=self.cb_group
        )
        self.grasp_client = ActionClient(
            self, Grasp, "/franka_gripper/grasp", callback_group=self.cb_group
        )
        self.command_sub = self.create_subscription(
            Float64MultiArray,
            "/gripper_command",
            self._on_gripper_command,
            10,
            callback_group=self.cb_group,
        )
        self._command_queue = deque()
        self._queue_lock = threading.Lock()
        self._queue_event = threading.Event()
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(target=self._command_worker, daemon=True)
        self._worker_thread.start()
        self.get_logger().info(
            "Listening on /gripper_command. Use [width, speed] for move/open, "
            "[width, speed, force, eps_inner, eps_outer] for grasp/close, or [-1] for homing."
        )

    @staticmethod
    def _wait_future(future, timeout_sec: float = 8.0, poll_sec: float = 0.01) -> bool:
        t0 = time.time()
        while not future.done() and (time.time() - t0) < timeout_sec:
            time.sleep(poll_sec)
        return future.done()

    def _send_goal_and_wait(self, client, goal_msg, name: str, timeout_sec=8.0):
        if not client.wait_for_server(timeout_sec=timeout_sec):
            self.get_logger().error(f"Action server not available: {name}")
            return False, None

        future = client.send_goal_async(goal_msg)
        if not self._wait_future(future, timeout_sec=timeout_sec):
            self.get_logger().error(f"Goal send timeout: {name}")
            return False, None
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error(f"Goal rejected: {name}")
            return False, None

        result_future = goal_handle.get_result_async()
        if not self._wait_future(result_future, timeout_sec=timeout_sec):
            self.get_logger().error(f"Result timeout: {name}")
            return False, None
        result = result_future.result()
        if result is None:
            self.get_logger().error(f"No result returned: {name}")
            return False, None

        return True, result.result

    def _on_gripper_command(self, msg: Float64MultiArray):
        data = [float(v) for v in msg.data]
        command = self._parse_command(data)
        if command is None:
            return
        with self._queue_lock:
            self._command_queue.append(command)
            qlen = len(self._command_queue)
        self.get_logger().info(
            f"Queued gripper command '{command['type']}'. queued={qlen}"
        )
        self._queue_event.set()

    def _parse_command(self, data):
        if not data:
            self.get_logger().error(
                "/gripper_command expects [width, speed], [width, speed, force, eps_inner, eps_outer], or [-1]."
            )
            return None
        if len(data) == 1 and int(data[0]) == -1:
            return {"type": "homing"}
        if len(data) == 2:
            return {
                "type": "move",
                "width": float(data[0]),
                "speed": float(data[1]),
            }
        if len(data) >= 5:
            return {
                "type": "grasp",
                "width": float(data[0]),
                "speed": float(data[1]),
                "force": float(data[2]),
                "eps_inner": float(data[3]),
                "eps_outer": float(data[4]),
            }
        self.get_logger().error(
            f"Unsupported /gripper_command payload length {len(data)}: {data}"
        )
        return None

    def _command_worker(self):
        while not self._stop_event.is_set():
            if not self._queue_event.wait(timeout=0.1):
                continue
            while True:
                with self._queue_lock:
                    if not self._command_queue:
                        self._queue_event.clear()
                        break
                    command = self._command_queue.popleft()
                self._execute_command(command)

    def _execute_command(self, command):
        cmd_type = command["type"]
        self.get_logger().info(f"Executing gripper command '{cmd_type}'")
        if cmd_type == "homing":
            self.homing()
        elif cmd_type == "move":
            self.move(command["width"], command["speed"])
        elif cmd_type == "grasp":
            self.grasp(
                command["width"],
                command["speed"],
                command["force"],
                command["eps_inner"],
                command["eps_outer"],
            )

    def destroy_node(self):
        self._stop_event.set()
        self._queue_event.set()
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
        super().destroy_node()

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


def main():
    parser = argparse.ArgumentParser(description="Franka gripper test utility")
    sub = parser.add_subparsers(dest="cmd", required=False)

    sub.add_parser("homing", help="Run gripper homing")
    sub.add_parser("listen", help="Listen on /gripper_command and execute commands")

    p_open = sub.add_parser("open", help="Open gripper using Move action")
    p_open.add_argument("--width", type=float, default=0.08)
    p_open.add_argument("--speed", type=float, default=0.05)

    p_close = sub.add_parser("close", help="Close/grasp using Grasp action")
    p_close.add_argument("--width", type=float, default=0.0)
    p_close.add_argument("--speed", type=float, default=0.03)
    p_close.add_argument("--force", type=float, default=20.0)
    p_close.add_argument("--eps-inner", type=float, default=0.01)
    p_close.add_argument("--eps-outer", type=float, default=0.01)

    p_seq = sub.add_parser("sequence", help="Homing -> open -> wait Enter -> grasp -> open")
    p_seq.add_argument("--open-width", type=float, default=0.08)
    p_seq.add_argument("--open-speed", type=float, default=0.05)
    p_seq.add_argument("--grasp-width", type=float, default=0.0)
    p_seq.add_argument("--grasp-speed", type=float, default=0.03)
    p_seq.add_argument("--grasp-force", type=float, default=20.0)
    p_seq.add_argument("--eps-inner", type=float, default=0.01)
    p_seq.add_argument("--eps-outer", type=float, default=0.01)

    args = parser.parse_args()
    if args.cmd is None:
        args.cmd = "listen"

    rclpy.init()
    node = FrankaGripperTester()
    executor = None

    try:
        if args.cmd == "listen":
            executor = MultiThreadedExecutor(num_threads=2)
            executor.add_node(node)
            executor.spin()

        elif args.cmd == "homing":
            ok = node.homing()
            sys.exit(0 if ok else 1)

        elif args.cmd == "open":
            ok = node.move(args.width, args.speed)
            sys.exit(0 if ok else 1)

        elif args.cmd == "close":
            ok = node.grasp(args.width, args.speed, args.force, args.eps_inner, args.eps_outer)
            sys.exit(0 if ok else 1)

        elif args.cmd == "sequence":
            if not node.homing():
                sys.exit(1)
            if not node.move(args.open_width, args.open_speed):
                sys.exit(1)

            input("Place object between fingers, then press Enter to continue grasp... ")

            if not node.grasp(args.grasp_width, args.grasp_speed, args.grasp_force, args.eps_inner, args.eps_outer):
                sys.exit(1)
            if not node.move(args.open_width, args.open_speed):
                sys.exit(1)

            node.get_logger().info("Sequence complete.")
            sys.exit(0)

    finally:
        if executor is not None:
            executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
