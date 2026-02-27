import time
import math
import random
import os

import numpy as np
import pybullet as p
import pybullet_data
from collections import namedtuple
from tqdm import tqdm
from PIL import Image


def _normalize(vec):
    arr = np.asarray(vec, dtype=np.float64)
    norm = np.linalg.norm(arr)
    if norm < 1e-9:
        raise ValueError('Cannot normalize near-zero vector')
    return arr / norm


def _rotation_matrix_to_quaternion(matrix):
    m = np.asarray(matrix, dtype=np.float64)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    quat = np.array([qx, qy, qz, qw], dtype=np.float64)
    quat /= np.linalg.norm(quat)
    return quat


class Env: 
    
    SIMULATION_STEP_DELAY = 1 / 960.

    def __init__(self, robot, block_path=None, table_path=None, bin_path=None, vis=False) -> None:
        """
        robot: robot object
        models: models object

        """
        self.robot = robot
        self.block_path = block_path
        self.table_path = table_path
        self.bin_path = bin_path

        # default camera configuration (used for RGB-D capture + TF broadcasting)
        self.camera_eye = [0.6, 0.0, 1.2]
        self.camera_target = [0.6, 0.0, 0.0]
        self.camera_up = [0.0, 1.0, 0.0]

        self.vis = vis
        if self.vis:
            self.p_bar = tqdm(ncols=0, disable=False)

        # define environment
        self.physicsClient = p.connect(p.GUI if self.vis else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        
        self.planeID = p.loadURDF("plane.urdf", [0, 0, -0.72], useFixedBase=True)

        self.robot.load()
        self.robot.step_simulation = self.step_simulation
        
        self.tableID = p.loadURDF(self.table_path, [0.0, 0, -0.74],
                                # p.getQuaternionFromEuler([0, 1.5706453, 0]),
                                p.getQuaternionFromEuler([0, 0, 0]),
                                useFixedBase=True,
                                flags=p.URDF_MERGE_FIXED_LINKS | p.URDF_USE_SELF_COLLISION)
        
        # self.boxID = p.loadURDF(self.block_path, [0.6, 0, 0.05],
        #                         # p.getQuaternionFromEuler([0, 1.5706453, 0]),
        #                         p.getQuaternionFromEuler([0, 0, 0]),
        #                         useFixedBase=False,
        #                         flags=p.URDF_MERGE_FIXED_LINKS | p.URDF_USE_SELF_COLLISION)
        self.binID = p.loadURDF(self.bin_path, [-0.2, -0.5, 0.05],
                                # p.getQuaternionFromEuler([0, 1.5706453, 0]),
                                p.getQuaternionFromEuler([0, 0, 0]),
                                useFixedBase=False,
                                flags=p.URDF_MERGE_FIXED_LINKS | p.URDF_USE_SELF_COLLISION)

        self.ycb_objects = []
        self._load_ycb_props()

    def _load_ycb_props(self):
        """Spawn a small set of tabletop YCB proxies for visual context."""
        base_dir = os.path.join(os.path.dirname(__file__), "models", "ycb")
        
        layout = [

            # # serve breakfast with all fruits
            # ("011_banana", [0.55, -0.24, 0.06], [0, 0, 1.0]),
            # ("029_plate", [0.92, -0.1, 0.055], [0, 0, 0]),
            # # ("032_knife", [0.64, 0.06, 0.045], [0, 0, -0.35]),
            # ("013_apple", [0.4, -0.24, 0.06], [0, 0, 1.0]),
            # ("017_orange", [0.69, 0.12, 0.06], [0, 0, 1.0]),
            

            # serve multiple cups of drink
            ("065-h_cups", [0.55, -0.24, 0.06], [0, 0, 1.0]),
            ("065-i_cups", [0.55, 0.00, 0.06], [0, 0, 1.0]),
            ("065-j_cups", [0.55, 0.24, 0.06], [0, 0, 1.0]),
            ("019_pitcher_base", [-0.2, -0.5, 0.06], [0, 0, 1.0]),

            # # clear the table into bin and clean the table
            # ("011_banana", [0.55, -0.24, 0.06], [0, 0, 1.0]),
            # ("025_mug", [0.65, 0.24, 0.07], [0, 0, 0]),
            # # ("010_potted_meat_can", [0.4, 0.2, 0.06], [0, 0, 1.0]),
            # # ("065-h_cups", [0.55, -0.24, 0.06], [0, 0, 1.0]),
            # ("013_apple", [0.4, -0.24, 0.06], [0, 0, 1.0]),
            # ("026_sponge", [0.2, 0.2, 0.06], [0, 0, 1.0]),
        ]
        if not os.path.isdir(base_dir):
            return
        try:
            block_aabb = p.getAABB(self.boxID)
            table_surface_z = block_aabb[0][2]
        except Exception:
            table_surface_z = 0.0
        for name, target_pos, euler in layout:
            urdf_path = os.path.join(base_dir, name, "model.urdf")
            if not os.path.isfile(urdf_path):
                continue
            try:
                quat = p.getQuaternionFromEuler(euler)
                obj_id = p.loadURDF(
                    urdf_path,
                    basePosition=target_pos,
                    baseOrientation=quat,
                    useFixedBase=True,
                    flags=p.URDF_MERGE_FIXED_LINKS | p.URDF_USE_SELF_COLLISION,
                )
                aabb_min, _ = p.getAABB(obj_id)
                delta = table_surface_z - aabb_min[2]
                if abs(delta) > 1e-4:
                    corrected = list(target_pos)
                    corrected[2] += delta
                    p.resetBasePositionAndOrientation(obj_id, corrected, quat)
                self.ycb_objects.append(obj_id)
            except Exception as exc:
                print(f"[YCB] Failed to load {name}: {exc}")
            # import pdb; pdb.set_trace()

    def step_simulation(self):
        """
        Hook p.stepSimulation()
        """
        p.stepSimulation()
        if self.vis:
            time.sleep(self.SIMULATION_STEP_DELAY)
            self.p_bar.update(1)

    def step(self, action, control_method='joint'):
        """
        action: (x, y, z, roll, pitch, yaw, gripper_opening_length) for End Effector Position Control
                (a1, a2, a3, a4, a5, a6, a7, gripper_opening_length) for Joint Position Control
        control_method:  'end' for end effector position control
                         'joint' for joint position control
        """
        assert control_method in ('joint', 'end')
        self.robot.move_ee(action[:-1], control_method)
        self.robot.move_gripper(action[-1])
        for _ in range(20):  # Wait for a few steps
            self.step_simulation()

        reward = 0
        done = True if reward == 1 else False
        info = None
        return reward, done, info

    def reset(self):
        self.robot.reset()

    def close(self):        
        p.disconnect(self.physicsClient)
            

class ManiEnv(Env):
    def __init__(self, robot, block_path=None, table_path=None, bin_path=None, vis=False):
        super().__init__(robot, block_path=block_path, table_path=table_path, bin_path=bin_path, vis=vis) 

        # define workspace
        self.min_pose = [0.4, -0.4, 0] 
        self.max_pose = [1, 0.4, 0.8]          

    def reset_block(self, pos, euler_angle):
        p.resetBasePositionAndOrientation(self.boxID, pos, p.getQuaternionFromEuler(euler_angle))

    def execute_trajectory(self, trajectory):
        # action = (x, y, z, roll, pitch, yaw, gripper_width)
        for action in trajectory:
            self.step(action, control_method='end')
        return self.check_success()

    def check_success(self):
        return True 
    
    def step(self, action, control_method='end'):
        """Step the environment.

        Action (7D):
          (x, y, z, roll, pitch, yaw, gripper_width)

        - gripper_width is in meters and will be clipped to robot.gripper_range
        """

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        assert action.shape[0] == 7, f"expected action dim 7, got {action.shape}"

        # update position
        pose = action[:3]
        pose = np.clip(pose, self.min_pose, self.max_pose)

        roll, pitch, yaw = float(action[3]), float(action[4]), float(action[5])
        gripper_width = float(action[6])

        rotation = np.array([roll, pitch, yaw], dtype=np.float32)
        gripper_width = float(np.clip(gripper_width, self.robot.gripper_range[0], self.robot.gripper_range[1]))

        self.robot.move_ee(np.concatenate([pose, rotation]), control_method)
        self.robot.move_gripper(gripper_width)

        for _ in range(120):  # Wait for a few steps
            self.step_simulation()

    def get_block_position(self):
        return p.getBasePositionAndOrientation(self.boxID)[0]

    def capture_rgbd(self, width=640, height=480, fov=60.0, near=0.1, far=3.0, return_seg=False):
        """Render an RGB-D frame from a fixed overhead camera view.

        Returns:
            rgb: (H, W, 3) uint8 array
            depth: (H, W) float32 array in meters
            seg: (H, W) int array (optional)
        """
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)

        view_matrix = p.computeViewMatrix(self.camera_eye, self.camera_target, self.camera_up)
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=fov,
            aspect=float(width) / height,
            nearVal=near,
            farVal=far,
        )

        _, _, rgba, depth_buffer, seg_index = p.getCameraImage(
            width,
            height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )

        rgb = np.reshape(rgba, (height, width, 4))[:, :, :3].astype(np.uint8)
        depth_buffer = np.reshape(depth_buffer, (height, width))
        depth = (far * near) / (far - (far - near) * depth_buffer)
        depth = depth.astype(np.float32)

        if return_seg:
            seg_index = np.reshape(seg_index, (height, width))
            return rgb, depth, seg_index
        return rgb, depth, None

    def get_camera_pose(self):
        eye = np.asarray(self.camera_eye, dtype=np.float64)
        target = np.asarray(self.camera_target, dtype=np.float64)
        up = np.asarray(self.camera_up, dtype=np.float64)
        forward = _normalize(target - eye)
        right = _normalize(np.cross(forward, up))
        true_up = _normalize(np.cross(right, forward))
        rotation = np.column_stack((right, -true_up, forward))
        quat = _rotation_matrix_to_quaternion(rotation)
        return tuple(eye.tolist()), tuple(quat.tolist())

    def get_robot_base_pose(self):
        return p.getBasePositionAndOrientation(self.robot.id)

    def get_robot_to_camera_transform(self):
        base_pos, base_quat = self.get_robot_base_pose()
        cam_pos, cam_quat = self.get_camera_pose()
        inv_pos, inv_quat = p.invertTransform(base_pos, base_quat)
        rel_pos, rel_quat = p.multiplyTransforms(inv_pos, inv_quat, cam_pos, cam_quat)
        return rel_pos, rel_quat

    def render(self, width=1920, height=1080):
        """
        Renders a high-quality image from a PyBullet simulation and saves it to a file.

        Parameters:
        - filename: str, the path to the file where the image will be saved.
        - width: int, the width of the rendered image.
        - height: int, the height of the rendered image.
        """
        rgb, _, seg_index = self.capture_rgbd(width=width, height=height, return_seg=True)
        return rgb, seg_index
    
    def traj_plot(self, traj, color=None):
        """
        plot the trajectory
        """
        if color is None:
            # generate a random
            color = np.random.rand(3)
                        
        for i in range(len(traj) - 1):
            start_point = traj[i]
            end_point = traj[i + 1]
            p.addUserDebugLine(start_point, end_point, color, 4)

        # add the last point
        p.addUserDebugLine(traj[1], traj[1], [0, 0, 1], lineWidth=10)
        p.addUserDebugLine(traj[-1], traj[-1], [1, 0, 0], lineWidth=10)

    def plot_dot(self, pos, text, color=None):
        """
        plot a dot
        """
        if color is None:
            color = np.random.rand(3)
        p.addUserDebugLine(pos, pos, color, lineWidth=50)
        # add text
        p.addUserDebugText(text, pos, textColorRGB=[0, 0, 0], textSize=1)

    def clean_traj_plot(self):
        """
        clean the trajectory plot
        """
        p.removeAllUserDebugItems()

    def check_touching(self):
        """
        check if the robot gripper is touching the block
        """        
        contact_points = p.getContactPoints(self.robot.id, self.boxID)
        return len(contact_points) > 0
    
    def check_grasping(self):
        """
        check if both robot gripper touch the block
        """
        finger_id_a = 12
        finger_id_b = 17

        left_contact = p.getContactPoints(self.robot.id, self.boxID, finger_id_a)
        right_contact = p.getContactPoints(self.robot.id, self.boxID, finger_id_b)    
        grasped = (left_contact != () or right_contact != ())
        return grasped
    
    def check_arrived(self, target_pos, tol=0.01):
        """
        check if the block has arrived at the target position
        """
        ee_pos = self.get_block_position()
        return abs(ee_pos[0] - target_pos[0]) < tol   
