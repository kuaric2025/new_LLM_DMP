#!/usr/bin/env python3
"""Print PyBullet Panda link indices and TCP pose; compare mentally to TiPToP/cuRobo.

TiPToP (robot.type ``panda``) builds motion planning from ``cutamp.robots.franka.franka_curobo_cfg``,
which follows NVIDIA cuRobo: ``ee_link`` is typically ``panda_hand`` (same name as in ROS/Bullet URDFs).

This environment uses ``FrankaPanda.eef_id = 8``. For ``pybullet_data/franka_panda/panda.urdf``,
link index 8 is the ``panda_hand`` link (between ``panda_link8`` at index 7 and fingers at 9–10).

Finger semantics: cuRobo franka.yml often locks ``panda_finger_joint{1,2}`` at 0.04 (open parallel jaw).
ManiEnv calls ``open_gripper()`` so each prismatic finger is near 0.04 — aligned with that convention.

If your TiPToP config uses ``panda_robotiq`` / ``fr3_robotiq``, the planner TCP is a Robotiq frame, not
``panda_hand``; use the TiPToP→sim transform path or match hardware in sim.

Run: ``python llm_mani/panda_tcp_alignment.py``
"""

from __future__ import annotations

import sys


def main() -> int:
    try:
        import pybullet as p
        import pybullet_data
    except ImportError:
        print("pybullet is required.", file=sys.stderr)
        return 1

    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    uid = p.loadURDF("franka_panda/panda.urdf", [0, 0, -0.05], useFixedBase=True)

    print("Joint index -> child link name (same order as controllable arm joints 0..6 = link1..link7)")
    for ji in range(p.getNumJoints(uid)):
        inf = p.getJointInfo(uid, ji)
        print(f"  joint {ji:2d}  {inf[1].decode():22s}  child link {inf[12].decode()}")

    print("\ngetLinkState(linkIndex) -> link name (eef_id in FrankaPanda should match panda_hand)")
    n = p.getNumJoints(uid)
    for li in range(7, min(12, n)):
        inf = p.getJointInfo(uid, li)
        print(f"  linkIndex {li:2d}  {inf[12].decode()}")

    q7 = [0.0, -0.628, 0.0, -2.513, 0.0, 1.885, 0.0]
    for ji, q in enumerate(q7):
        p.resetJointState(uid, ji, q)
    p.resetJointState(uid, 9, 0.04)
    p.resetJointState(uid, 10, 0.04)

    eef = 8
    pos, orn = p.getLinkState(uid, eef, computeForwardKinematics=1)[4:6]
    print(f"\nSample TCP (linkIndex={eef}, open fingers): pos={pos}, quat_xyzw={orn}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
