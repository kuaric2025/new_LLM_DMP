#!/usr/bin/env python3
import math
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, MultiArrayDimension


class TrajectoryTestPublisher(Node):
    def __init__(self):
        super().__init__("franka_trajectory_test_publisher")
        self.pub = self.create_publisher(Float64MultiArray, "/demo/trajectory", 10)

    def build_trajectory(self, traj_id: int, steps=20):
        # shape: (steps, 7)
        # [x, y, z, roll, pitch, yaw, gripper]
        data = []

        # Slightly different base offsets for traj1/traj2/traj3
        y_offsets = {1: -0.05, 2: 0.00, 3: 0.05}
        z_offsets = {1: 0.00, 2: 0.02, 3: -0.01}
        y0 = y_offsets.get(traj_id, 0.0)
        z0 = 0.35 + z_offsets.get(traj_id, 0.0)
        x0 = 0.50

        r0, p0, yaw0 = math.pi, 0.0, 0.0

        for i in range(steps):
            s = i / max(1, steps - 1)

            # Different motion shape per trajectory
            if traj_id == 1:
                x = x0
                y = y0 + 0.06 * (s - 0.5)
                z = z0 + 0.02 * math.sin(2 * math.pi * s)
            elif traj_id == 2:
                x = x0 + 0.03 * math.sin(2 * math.pi * s)
                y = y0
                z = z0 + 0.015 * math.cos(2 * math.pi * s)
            else:  # traj_id == 3
                x = x0 + 0.02 * (s - 0.5)
                y = y0 + 0.02 * math.sin(4 * math.pi * s)
                z = z0

            roll = r0
            pitch = p0
            yaw = yaw0 + 0.3 * math.sin(2 * math.pi * s)

            # gripper pattern
            if s < 0.33:
                gripper = 0.08
            elif s < 0.66:
                gripper = 0.03
            else:
                gripper = 0.08

            data.extend([x, y, z, roll, pitch, yaw, gripper])
        msg = Float64MultiArray()
        msg.layout.dim = [
            MultiArrayDimension(label="steps", size=steps, stride=steps * 7),
            MultiArrayDimension(label="n_dim", size=7, stride=7),
        ]
        msg.layout.data_offset = 0
        msg.data = data
        return msg

    def publish_traj(self, traj_id: int, steps=100):
        msg = self.build_trajectory(traj_id=traj_id, steps=steps)
        # publish a few times for reliable reception
        for _ in range(3):
            self.pub.publish(msg)
            time.sleep(0.15)
        self.get_logger().info(
            f"Published traj{traj_id}: shape=({steps},7) on /demo/trajectory"
        )

    def publish_sequence(self):
        for traj_id in [1, 2, 3]:
            self.publish_traj(traj_id=traj_id, steps=20)
            time.sleep(0.8)
        self.get_logger().info("Published sequence: traj1 -> traj2 -> traj3")


def main():
    rclpy.init()
    node = TrajectoryTestPublisher()
    try:
        time.sleep(0.8)  # discovery
        node.publish_sequence()
        time.sleep(0.5)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
