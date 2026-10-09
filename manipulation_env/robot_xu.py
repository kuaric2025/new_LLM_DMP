import pybullet as p
import numpy as np
from collections import namedtuple


class RobotBase(object):
    """Base class for robots.

    This repo expects a robot object to provide:
      - load(), reset(), reset_arm(), reset_gripper()
      - move_ee(action, control_method) with control_method in {'joint','end'}
      - move_gripper(open_length)
      - get_gripper_width()
      - get_eef()

    The environment hooks in step_simulation().
    """

    def __init__(self, model_path=None, pos=[0, 0, 0], ori=[0, 0, 0]):
        self.base_pos = pos
        self.base_ori = p.getQuaternionFromEuler(ori)
        self.model_path = model_path

    def load(self):
        self.__init_robot__()
        self.__parse_joint_info__()
        self.__post_load__()

    def step_simulation(self):
        raise RuntimeError("RobotBase.step_simulation should be hooked by the environment")

    def __parse_joint_info__(self):
        numJoints = p.getNumJoints(self.id)
        jointInfo = namedtuple(
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
        for i in range(numJoints):
            info = p.getJointInfo(self.id, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = info[2]
            jointDamping = info[6]
            jointFriction = info[7]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = jointType != p.JOINT_FIXED
            if controllable:
                self.controllable_joints.append(jointID)
                # disable default velocity motors
                p.setJointMotorControl2(self.id, jointID, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
            self.joints.append(
                jointInfo(
                    jointID,
                    jointName,
                    jointType,
                    jointDamping,
                    jointFriction,
                    jointLowerLimit,
                    jointUpperLimit,
                    jointMaxForce,
                    jointMaxVelocity,
                    controllable,
                )
            )

        assert len(self.controllable_joints) >= self.arm_num_dofs
        self.arm_controllable_joints = self.controllable_joints[: self.arm_num_dofs]

        self.arm_lower_limits = [info.lowerLimit for info in self.joints if info.controllable][: self.arm_num_dofs]
        self.arm_upper_limits = [info.upperLimit for info in self.joints if info.controllable][: self.arm_num_dofs]
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
        for _ in range(10):
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
                maxNumIterations=50,
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
        position = p.getLinkState(self.id, self.eef_id)[0]
        orientation = p.getLinkState(self.id, self.eef_id)[1]
        return position, orientation


class FrankaPanda(RobotBase):
    """Franka Panda robot (pybullet_data) with simple gripper width control."""

    def __init_robot__(self):
        if self.model_path is None:
            import pybullet_data

            self.model_path = pybullet_data.getDataPath() + "/franka_panda/panda.urdf"

        # panda_hand link index for IK
        self.eef_id = 8
        self.arm_num_dofs = 7
        self.arm_rest_poses = [0.0, -0.6, 0.0, -2.2, 0.0, 2.0, 0.8]

        self.id = p.loadURDF(
            self.model_path,
            self.base_pos,
            self.base_ori,
            useFixedBase=True,
            flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES,
        )

        # finger joints (prismatic)
        self.finger_joint_ids = [9, 10]
        self.gripper_range = [0.0, 0.08]

    def move_gripper(self, open_length):
        open_length = float(np.clip(open_length, *self.gripper_range))
        target = open_length / 2.0
        for jid in self.finger_joint_ids:
            p.setJointMotorControl2(
                self.id,
                jid,
                p.POSITION_CONTROL,
                targetPosition=target,
                force=100,
                maxVelocity=1.0,
            )

    def get_gripper_width(self):
        w = 0.0
        for jid in self.finger_joint_ids:
            w += float(p.getJointState(self.id, jid)[0])
        return 2.0 * w
