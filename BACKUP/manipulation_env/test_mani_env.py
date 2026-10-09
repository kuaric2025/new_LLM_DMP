"""Quick smoke test for ManiEnv with FrankaPanda.

Runs a few end-effector moves and gripper open/close in GUI.

Usage:
  python manipulation_env/test_mani_env.py
"""

import os
import sys
import time

import numpy as np

# add repo root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manipulation_env.env import ManiEnv
from manipulation_env.robot import FrankaPanda


def main():
    # table_path = "../manipulation_env/models/urdf/objects/table/table.urdf"
    table_path = os.path.join(os.path.dirname(__file__), "models/urdf/objects/table/table.urdf")
    # Use a simple cube from pybullet_data
    block_path = "cube_small.urdf"

    robot = FrankaPanda(model_path=None)
    env = ManiEnv(robot, block_path=block_path, table_path=table_path, vis=True)
    env.reset()

    # Move above the block, close gripper, lift, move aside, open.
    # action in ManiEnv.step (preferred): (x, y, z, roll, pitch, yaw, gripper_width)

    # Point gripper DOWN for this manual test
    roll = np.pi
    pitch = 0.0

    actions = [
        np.array([0.55, 0.0, 0.30, roll, pitch, 0.0, 0.08], dtype=np.float32),  # above, open
        np.array([0.55, 0.0, 0.15, roll, pitch, 0.0, 0.08], dtype=np.float32),  # down
        np.array([0.55, 0.0, 0.15, roll, pitch, 0.0, 0.00], dtype=np.float32),  # close
        np.array([0.55, 0.0, 0.30, roll, pitch, 0.0, 0.00], dtype=np.float32),  # lift
        np.array([0.45, 0.2, 0.30, roll, pitch, 0.6, 0.00], dtype=np.float32),  # move
        np.array([0.45, 0.2, 0.15, roll, pitch, 0.6, 0.00], dtype=np.float32),  # down
        np.array([0.45, 0.2, 0.15, roll, pitch, 0.6, 0.08], dtype=np.float32),  # open
        np.array([0.45, 0.2, 0.30, roll, pitch, 0.6, 0.08], dtype=np.float32),  # retreat
    ]

    print("Running ManiEnv smoke trajectory...")
    for a in actions:
        env.step(a, control_method="end")
        env.step_simulation()
        time.sleep(0.2)

    print("Done. Close the GUI window to exit.")
    while True:
        env.step_simulation()


if __name__ == "__main__":
    main()
