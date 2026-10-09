import pybullet as p
import numpy as np
from collections import namedtuple


class RobotBase(object):
    """Base class for robots used by ManiEnv."""

    def __init__(self, model_path=None, pos=None, ori=None):
        # Panda URDF root has a +0.05 m visual/collision offset; use -0.05 so base frame aligns to world z=0.
        self.base_pos = [0.0, 0.0, -0.05] if pos is None else list(pos)
        self.base_ori = p.getQuaternionFromEuler([0.0, 0.0, 0.0] if ori is None else list(ori))
        self.model_path = model_path

    def load(self):
        self.__init_robot__()
        self.__parse_joint_info__()
        self.__post_load__()

    def step_simulation(self):
        raise RuntimeError("RobotBase.step_simulation should be hooked by the environment")

    def __parse_joint_info__(self):
        num_joints = p.getNumJoints(self.id)
        joint_info_type = namedtuple(
            "jointInfo",
            [
                "id",
                "name",
                "type",
                "damping",
                "friction",
                "lowerLimit",
                "upperLimit",
                "maxForce",
                "maxVelocity",
                "controllable",
            ],
        )
        self.joints = []
        self.controllable_joints = []
        for i in range(num_joints):
            info = p.getJointInfo(self.id, i)
            joint_id = info[0]
            joint_name = info[1].decode("utf-8")
            joint_type = info[2]
            joint_damping = info[6]
            joint_friction = info[7]
            joint_lower_limit = info[8]
            joint_upper_limit = info[9]
            joint_max_force = info[10]
            joint_max_velocity = info[11]
            controllable = joint_type != p.JOINT_FIXED
            if controllable:
                self.controllable_joints.append(joint_id)
                p.setJointMotorControl2(
                    self.id, joint_id, p.VELOCITY_CONTROL, targetVelocity=0, force=0
                )
            self.joints.append(
                joint_info_type(
                    joint_id,
                    joint_name,
                    joint_type,
                    joint_damping,
                    joint_friction,
                    joint_lower_limit,
                    joint_upper_limit,
                    joint_max_force,
                    joint_max_velocity,
                    controllable,
                )
            )

        assert len(self.controllable_joints) >= self.arm_num_dofs
        self.arm_controllable_joints = self.controllable_joints[: self.arm_num_dofs]
        self.arm_lower_limits = [
            info.lowerLimit for info in self.joints if info.controllable
        ][: self.arm_num_dofs]
        self.arm_upper_limits = [
            info.upperLimit for info in self.joints if info.controllable
        ][: self.arm_num_dofs]
        self.arm_joint_ranges = [
            info.upperLimit - info.lowerLimit for info in self.joints if info.controllable
        ][: self.arm_num_dofs]

    def __init_robot__(self):
        raise NotImplementedError

    def __post_load__(self):
        pass

    def reset(self):
        self.reset_arm()
        self.reset_gripper()

    def reset_arm(self):
        for rest_pose, joint_id in zip(self.arm_rest_poses, self.arm_controllable_joints):
            p.resetJointState(self.id, joint_id, rest_pose)
            p.setJointMotorControl2(
                self.id,
                joint_id,
                p.POSITION_CONTROL,
                rest_pose,
                force=self.joints[joint_id].maxForce,
                maxVelocity=self.joints[joint_id].maxVelocity,
            )
        for _ in range(8):
            self.step_simulation()

    def reset_gripper(self):
        self.open_gripper()

    def open_gripper(self):
        self.move_gripper(self.gripper_range[1])

    def close_gripper(self):
        self.move_gripper(self.gripper_range[0])

    def move_ee(self, action, control_method):
        assert control_method in ("joint", "end")
        if control_method == "end":
            x, y, z, roll, pitch, yaw = action
            pos = (x, y, z)
            orn = p.getQuaternionFromEuler((roll, pitch, yaw))
            joint_poses = p.calculateInverseKinematics(
                self.id,
                self.eef_id,
                pos,
                orn,
                self.arm_lower_limits,
                self.arm_upper_limits,
                self.arm_joint_ranges,
                self.arm_rest_poses,
                maxNumIterations=80,
            )
        else:
            assert len(action) == self.arm_num_dofs
            joint_poses = action

        for i, joint_id in enumerate(self.arm_controllable_joints):
            p.setJointMotorControl2(
                self.id,
                joint_id,
                p.POSITION_CONTROL,
                joint_poses[i],
                force=self.joints[joint_id].maxForce,
                maxVelocity=self.joints[joint_id].maxVelocity,
            )

    def move_gripper(self, open_length):
        raise NotImplementedError

    def get_eef(self):
        state = p.getLinkState(self.id, self.eef_id)
        # Use worldLinkFrame pose (indices 4/5), not COM pose (0/1).
        # COM introduces a fixed offset and breaks TCP/grasp alignment diagnostics.
        return state[4], state[5]


class FrankaPanda(RobotBase):
    """
    Franka Panda model configured to emulate FR3 setup in simulation.

    - Base pose defaults to identity in world frame.
    - Arm reset uses a standard Franka home-like joint configuration.
    """

    def __init_robot__(self):
        if self.model_path is None:
            import pybullet_data

            self.model_path = pybullet_data.getDataPath() + "/franka_panda/panda.urdf"

        # PyBullet getLinkState linkIndex for `pybullet_data/franka_panda/panda.urdf`: 8 -> "panda_hand".
        # TiPToP with robot.type "panda" uses cuRobo franka config with ee_link "panda_hand" (same frame name).
        # URDF files still differ (Bullet vs cuRobo franka_description); run llm_mani/panda_tcp_alignment.py to inspect.
        self.eef_id = 8
        self.arm_num_dofs = 7
        # Null-space rest pose; actual reset is solved with IK in reset_arm().
        self.arm_rest_poses = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785] #[0.0, -0.628, 0.0, -2.513, 0.0, 1.885, 0.0] 
        # Vertical initial joint pose: tool points downward.
        self.vertical_home_joints = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785] #[0.0, -0.628, 0.0, -2.513, 0.0, 1.885, 0.0]

        self.id = p.loadURDF(
            self.model_path,
            self.base_pos,
            self.base_ori,
            useFixedBase=True,
            flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES,
        )

        self.finger_joint_ids = [9, 10]
        self.gripper_range = [0.0, 0.08]

    def __post_load__(self):
        super().__post_load__()
        self.open_gripper()

    def reset_arm(self):
        for i, joint_id in enumerate(self.arm_controllable_joints):
            q = float(self.vertical_home_joints[i])
            p.resetJointState(self.id, joint_id, q)
            p.setJointMotorControl2(
                self.id,
                joint_id,
                p.POSITION_CONTROL,
                q,
                force=max(2500.0, self.joints[joint_id].maxForce),
                maxVelocity=self.joints[joint_id].maxVelocity,
            )
        for _ in range(40):
            self.step_simulation()

    def move_gripper(self, open_length):
        open_length = float(np.clip(open_length, *self.gripper_range))
        target = open_length / 2.0
        # When closing, apply a small squeeze bias and stronger/ slower command so
        # contact is firmer and less likely to slip during transport.
        current = float(self.get_gripper_width())
        closing = open_length < current
        cmd_force = 4800 if closing else 2500
        cmd_vel = 0.1 if closing else 0.1
        # if closing:
        #     # Slightly stronger preload to improve banana pickup success.
        #     target = max(0.0, target - 0.0012)
        #     cmd_force = 3800
        #     cmd_vel = 0.05
        # else:
        #     cmd_force = 2500
        #     cmd_vel = 0.12
        for joint_id in self.finger_joint_ids:
            p.setJointMotorControl2(
                self.id,
                joint_id,
                p.POSITION_CONTROL,
                targetPosition=target,
                force=cmd_force,
                maxVelocity=cmd_vel,
            )

    def get_gripper_width(self):
        width = 0.0
        for joint_id in self.finger_joint_ids:
            width += float(p.getJointState(self.id, joint_id)[0])
        # Panda finger joints are symmetric prismatic joints whose sum equals jaw width.
        return float(np.clip(width, self.gripper_range[0], self.gripper_range[1]))
