import os
import json
import time

import numpy as np
import pybullet as p
import pybullet_data
from tqdm import tqdm
from PIL import Image


class Env:
    SIMULATION_STEP_DELAY = 1.0 / 240.0

    def __init__(
        self,
        robot,
        table_path=None,
        vis=False,
        object_config_path=None,
        ycb_root=None,
    ) -> None:
        """
        Base environment wrapper around PyBullet + a robot.

        Args:
            robot: robot object (e.g., FrankaPanda)
            table_path: URDF path for the table
            vis: if True, open GUI
            object_config_path: JSON scene config for additional objects (YCB, etc.)
            ycb_root: root folder for YCB models; defaults to ./models/ycb
        """
        self.robot = robot
        self.table_path = table_path
        self.vis = vis

        # default camera configuration (used for RGB-D capture + TF broadcasting)
        self.camera_eye = [0.6, 0.0, 1.2]
        self.camera_target = [0.6, 0.0, 0.0]
        self.camera_up = [0.0, 1.0, 0.0]
        
        if self.vis:
            self.p_bar = tqdm(ncols=0, disable=False)

        # ------------------------------------------------------------------
        # Bullet setup
        # ------------------------------------------------------------------
        self.physicsClient = p.connect(p.GUI if self.vis else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)

        # Plane
        self.planeID = p.loadURDF("plane.urdf", [0, 0, -0.72], useFixedBase=True)

        # Robot
        self.robot.load()
        self.robot.step_simulation = self.step_simulation

        # Table
        self.tableID = None
        if self.table_path is not None:
            self.tableID = p.loadURDF(
                self.table_path,
                [0.0, 0.0, -0.74],
                p.getQuaternionFromEuler([0, 0, 0]),
                useFixedBase=True,
                flags=p.URDF_MERGE_FIXED_LINKS | p.URDF_USE_SELF_COLLISION,
            )

        # ------------------------------------------------------------------
        # YCB / named object management via config
        # ------------------------------------------------------------------
        self.ycb_root = (
            ycb_root
            if ycb_root is not None
            else os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "models", "ycb"
            )
        )

        # Mapping from logical name -> PyBullet body id
        self.objects = {}

        if object_config_path is not None:
            print(f"[Env] Loading objects from config: {object_config_path}")
            self._load_objects_from_config(object_config_path)

    # ----------------------------------------------------------------------
    # Core simulation loop
    # ----------------------------------------------------------------------
    def step_simulation(self):
        p.stepSimulation()
        if self.vis:
            time.sleep(self.SIMULATION_STEP_DELAY)
            self.p_bar.update(1)

    def reset(self):
        """Reset robot joint state (objects keep their positions)."""
        self.robot.reset()

    def close(self):
        p.disconnect(self.physicsClient)

    # ----------------------------------------------------------------------
    # YCB / named object helpers
    # ----------------------------------------------------------------------
    def _find_ycb_urdf(self, name: str) -> str:
        """Search self.ycb_root for a URDF matching the given YCB name."""
        direct_dir = os.path.join(self.ycb_root, name)
        if os.path.isdir(direct_dir):
            for candidate in ("model.urdf", "textured.urdf"):
                cand_path = os.path.join(direct_dir, candidate)
                if os.path.isfile(cand_path):
                    return cand_path
            for root, _, files in os.walk(direct_dir):
                for f in files:
                    if f.endswith(".urdf"):
                        return os.path.join(root, f)

        lower_name = name.lower()
        for root, _, files in os.walk(self.ycb_root):
            for f in files:
                if not f.endswith(".urdf"):
                    continue
                full_path = os.path.join(root, f)
                if lower_name in full_path.lower():
                    return full_path

        raise FileNotFoundError(
            f"Could not find a URDF for '{name}' under '{self.ycb_root}'"
        )

    def _load_objects_from_config(self, config_path: str):
        """
        Load objects defined in a JSON config.

        Expected format:
          {
            "objects": [
              {
                "name": "banana",
                "urdf_path": "...",        # preferred
                "ycb_name": "011_banana",  # fallback if no urdf_path
                "position": [x, y, z],
                "orientation_euler": [r, p, y],
                "fixed_base": false
              },
              ...
            ]
          }
        """
        with open(config_path, "r") as f:
            cfg = json.load(f)

        for obj in cfg.get("objects", []):
            name = obj["name"]

            urdf_path = obj.get("urdf_path", None)
            if urdf_path is not None:
                model_path = urdf_path
            else:
                ycb_name = obj.get("ycb_name", name)
                model_path = self._find_ycb_urdf(ycb_name)

            pos = obj.get("position", [0.6, 0.0, 0.05])
            euler = obj.get("orientation_euler", [0.0, 0.0, 0.0])
            orn = p.getQuaternionFromEuler(euler)
            fixed = bool(obj.get("fixed_base", False))

            print(f"[Env] Loading object '{name}' from {model_path}")
            print(f"      config position: {pos}, euler: {euler}, fixed_base={fixed}")

            body_id = p.loadURDF(
                model_path,
                pos,
                orn,
                useFixedBase=fixed,
                flags=p.URDF_MERGE_FIXED_LINKS | p.URDF_USE_SELF_COLLISION,
            )

            self.objects[name] = body_id

            # Print actual pose so you can see if URDF has an internal offset
            final_pos, final_orn = p.getBasePositionAndOrientation(body_id)
            print(f"      spawned at world pos: {final_pos}")

    def get_object_pose(self, name: str, euler: bool = False):
        """Get pose of a named object."""
        if name not in self.objects:
            raise KeyError(f"Unknown object name '{name}'")
        body_id = self.objects[name]
        pos, orn = p.getBasePositionAndOrientation(body_id)
        if euler:
            return pos, p.getEulerFromQuaternion(orn)
        return pos, orn

    def set_object_pose(
        self,
        name: str,
        position=None,
        orn_quat=None,
        orn_euler=None,
    ):
        """Set pose of a named object at runtime."""
        if name not in self.objects:
            raise KeyError(f"Unknown object name '{name}'")
        body_id = self.objects[name]
        cur_pos, cur_orn = p.getBasePositionAndOrientation(body_id)

        if position is None:
            position = cur_pos
        if orn_quat is None:
            if orn_euler is not None:
                orn_quat = p.getQuaternionFromEuler(orn_euler)
            else:
                orn_quat = cur_orn

        p.resetBasePositionAndOrientation(body_id, position, orn_quat)


class ManiEnv(Env):
    def __init__(
        self,
        robot,
        table_path=None,
        vis=False,
        object_config_path=None,
        ycb_root=None,
    ):
        super().__init__(
            robot,
            table_path=table_path,
            vis=vis,
            object_config_path=object_config_path,
            ycb_root=ycb_root,
        )

        # Workspace bounds for EE (used to clip actions)
        self.min_pose = np.array([0.4, -0.4, 0.0], dtype=np.float32)
        self.max_pose = np.array([1.0, 0.4, 0.8], dtype=np.float32)

    def reset(self):
        """Reset robot only (objects stay where they are)."""
        super().reset()

    # ------------------------------------------------------------------
    # High-level 7D action interface used in your pick/place tests
    # ------------------------------------------------------------------
    def step(self, action, control_method="end"):
        """7D EE action: (x, y, z, roll, pitch, yaw, gripper_width)."""
        assert control_method in ("end", "joint")
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        assert action.shape[0] == 7, f"expected action dim 7, got {action.shape}"

        pose = action[:3]
        pose = np.clip(pose, self.min_pose, self.max_pose)

        roll = float(action[3])
        pitch = float(action[4])
        yaw = float(action[5])
        gripper_width = float(action[6])

        rotation = np.array([roll, pitch, yaw], dtype=np.float32)
        gripper_width = float(
            np.clip(
                gripper_width, self.robot.gripper_range[0], self.robot.gripper_range[1]
            )
        )

        self.robot.move_ee(np.concatenate([pose, rotation]), control_method)
        self.robot.move_gripper(gripper_width)

        for _ in range(120):
            self.step_simulation()

    def capture_rgbd(
        self,
        width=640,
        height=480,
        fov=70.0,
        near=0.1,
        far=3.0,
        return_seg=True,
        eye_position=None,
        target_position=None,
        up_vector=None,
    ):
        if eye_position is None:
            eye_position = self.camera_eye
        if target_position is None:
            target_position = self.camera_target
        if up_vector is None:
            up_vector = self.camera_up
        view_matrix = p.computeViewMatrix(eye_position, target_position, up_vector)
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=fov, aspect=float(width) / float(height), nearVal=near, farVal=far
        )
        _, _, rgba, depth_buffer, seg = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )

        rgba = np.asarray(rgba, dtype=np.uint8).reshape(height, width, 4)
        rgb = rgba[:, :, :3]
        depth_buffer = np.asarray(depth_buffer, dtype=np.float32).reshape(height, width)
        depth = (2.0 * near * far) / (far + near - (2.0 * depth_buffer - 1.0) * (far - near))
        seg = np.asarray(seg, dtype=np.int32).reshape(height, width)
        if return_seg:
            return rgb, depth, seg
        return rgb, depth, None

    def render(self, width=1920, height=1080):
        rgb, _, seg = self.capture_rgbd(
            width=width,
            height=height,
            fov=60.0,
            near=0.1,
            far=3.0,
            return_seg=True,
        )
        return rgb, seg

    def traj_plot(self, traj, color=None):
        if color is None:
            color = np.random.rand(3)
        for i in range(len(traj) - 1):
            p.addUserDebugLine(traj[i], traj[i + 1], color, 4)
        if len(traj) > 1:
            p.addUserDebugLine(traj[0], traj[0], [0, 0, 1], lineWidth=10)
            p.addUserDebugLine(traj[-1], traj[-1], [1, 0, 0], lineWidth=10)

    def plot_dot(self, pos, text, color=None):
        if color is None:
            color = np.random.rand(3)
        p.addUserDebugLine(pos, pos, color, lineWidth=50)
        p.addUserDebugText(text, pos, textColorRGB=[0, 0, 0], textSize=1)

    def clean_traj_plot(self):
        p.removeAllUserDebugItems()

    def check_touching(self):
        if self.boxID is None:
            return False
        return len(p.getContactPoints(self.robot.id, self.boxID)) > 0

    def check_grasping(self):
        if self.boxID is None:
            return False
        finger_id_a = self.robot.finger_joint_ids[0]
        finger_id_b = self.robot.finger_joint_ids[1]
        left_contact = p.getContactPoints(self.robot.id, self.boxID, finger_id_a)
        right_contact = p.getContactPoints(self.robot.id, self.boxID, finger_id_b)
        return (left_contact != ()) or (right_contact != ())

    def check_arrived(self, target_pos, tol=0.01):
        ee_pos = self.get_block_position()
        if ee_pos is None:
            return False
        return abs(ee_pos[0] - target_pos[0]) < tol
