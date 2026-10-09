import os
import time
import numpy as np
import pybullet as p
import pybullet_data
from tqdm import tqdm


class Env:
    SIMULATION_STEP_DELAY = 1.0 / 240.0

    def __init__(
        self,
        robot,
        block_path=None,
        table_path=None,
        bin_path=None,
        banana_path=None,
        plate_path=None,
        apple_path=None,
        orange_path=None,
        plum_path=None,
        pitcher_path=None,
        h_cups_path=None,
        i_cups_path=None,
        j_cups_path=None,
        sponge_path=None,
        mug_path=None,
        bowl_path=None,
        vis=False,
    ) -> None:
        self.robot = robot
        self.block_path = block_path
        self.table_path = table_path
        self.bin_path = bin_path
        self.banana_path = banana_path
        self.plate_path = plate_path
        self.apple_path = apple_path
        self.orange_path = orange_path
        self.plum_path = plum_path
        self.pitcher_path = pitcher_path
        self.h_cups_path = h_cups_path
        self.i_cups_path = i_cups_path
        self.j_cups_path = j_cups_path
        self.sponge_path = sponge_path
        self.mug_path = mug_path
        self.bowl_path = bowl_path
        self.vis = vis
        self.p_bar = tqdm(ncols=0, disable=not self.vis)
        # Zero clearance so objects start exactly on the table surface.
        self.object_table_clearance = 0.0

        # default camera configuration (used for RGB-D capture + TF broadcasting)
        self.camera_eye = [0.6, 0.0, 1.2]
        self.camera_target = [0.6, 0.0, 0.0]
        self.camera_up = [0.0, 1.0, 0.0]

        self.physics_client = p.connect(p.GUI if self.vis else p.DIRECT)
        if self.vis:
            # Hide built-in GUI overlays (including the XYZ axis widget).
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.SIMULATION_STEP_DELAY)
        p.setRealTimeSimulation(0)
        # Improve contact robustness to reduce interpenetration.
        p.setPhysicsEngineParameter(numSolverIterations=240, numSubSteps=4)

        self.plane_id = p.loadURDF("plane.urdf", [0, 0, -0.74], useFixedBase=True)
        self.robot.load()
        self.robot.step_simulation = self.step_simulation
        self._set_robot_finger_dynamics()

        self.table_id = self._load_table()
        self.tableID = self.table_id
        self.table_surface_z = p.getAABB(self.table_id)[1][2]
        self._place_robot_on_table(clearance=0.0005)

        self.bin_id = self._load_bin()
        self.binID = self.bin_id
        self.boxID = self._load_block()

        self.object_spawn_specs = {
            # "plate": {"xy": [0.66, -0.30], "yaw": 0.0},
            # "banana": {"xy": [0.6, 0.02], "yaw": 1.2},
            # "apple": {"xy": [0.5, 0.2], "yaw": 0.0},
            # "orange": {"xy": [0.7, 0.3], "yaw": 0.0},
            # "plum": {"xy": [0.79, 0.3], "yaw": 0.0}
            # Arrange pouring objects in a clear row on tabletop.
            # Keep pitcher and cups upright (face up); yaw only for handle direction.
            "pitcher": {"xy": [0.70, -0.28], "yaw": 1.5708},
            "h_cups": {"xy": [0.58, -0.06], "yaw": 0.0},
            "i_cups": {"xy": [0.58, 0.12], "yaw": 0.0},
            "j_cups": {"xy": [0.58, 0.30], "yaw": 0.0},
            # "sponge": {"xy": [0.2, 0.2], "yaw": 0.0},
            # "mug": {"xy": [0.65, 0.24], "yaw": 0.0},
            # "bowl": {"xy": [0.3, 0.0], "yaw": 0.0},
        }

        self.object_ids = {}
        # self.object_ids["plate"] = self._load_ycb(
        #     "plate", self.plate_path
        # )
        # self.object_ids["banana"] = self._load_ycb(
        #     "banana", self.banana_path
        # )
        # self.object_ids["apple"] = self._load_ycb(
        #     "apple", self.apple_path
        # )
        # self.object_ids["orange"] = self._load_ycb(
        #     "orange", self.orange_path
        # )
        self.object_ids["pitcher"] = self._load_ycb(
            "pitcher", self.pitcher_path
        )
        self.object_ids["h_cups"] = self._load_ycb(
            "h_cups", self.h_cups_path
        )
        self.object_ids["i_cups"] = self._load_ycb(
            "i_cups", self.i_cups_path
        )
        self.object_ids["j_cups"] = self._load_ycb(
            "j_cups", self.j_cups_path
        )
        # self.object_ids["plum"] = self._load_ycb(
        #     "plum", self.plum_path
        # )
        # self.plateID = self.object_ids["plate"]
        # self.bananaID = self.object_ids["banana"]
        # self.appleID = self.object_ids["apple"]
        # self.orangeID = self.object_ids["orange"]
        # self.plumID = self.object_ids["plum"]
        self.pitcherID = self.object_ids["pitcher"]
        self.h_cupsID = self.object_ids["h_cups"]
        self.i_cupsID = self.object_ids["i_cups"]
        self.j_cupsID = self.object_ids["j_cups"]

        self.reset()
        self.object_spawn_poses = {
            name: p.getBasePositionAndOrientation(obj_id)
            for name, obj_id in self.object_ids.items()
        }
        if self.boxID is not None:
            self.box_spawn_pose = p.getBasePositionAndOrientation(self.boxID)

    def _repo_path(self, rel_path):
        return os.path.join(os.path.dirname(__file__), rel_path)

    def _load_table(self):
        if self.table_path is None:
            self.table_path = self._repo_path("models/urdf/objects/table/table.urdf")
        return p.loadURDF(
            self.table_path,
            [0.0, 0.0, -0.74],
            p.getQuaternionFromEuler([0.0, 0.0, 0.0]),
            useFixedBase=True,
            flags=p.URDF_MERGE_FIXED_LINKS | p.URDF_USE_SELF_COLLISION,
        )

    def _set_robot_finger_dynamics(self):
        """Set gripper finger friction to stabilize grasp contact."""
        finger_ids = getattr(self.robot, "finger_joint_ids", [])
        for link_id in finger_ids:
            p.changeDynamics(
                self.robot.id,
                link_id,
                lateralFriction=2.5,
                spinningFriction=0.20,
                rollingFriction=0.05,
                restitution=0.0,
                contactStiffness=6000.0,
                contactDamping=180.0,
            )

    def _load_bin(self):
        if self.bin_path is None:
            return None
        bin_id = p.loadURDF(
            self.bin_path,
            [0.15, -9.55, self.table_surface_z + 0.2],
            # [0.15, -0.55, self.table_surface_z + 0.2],
            p.getQuaternionFromEuler([0.0, 0.0, 0.0]),
            useFixedBase=True,
            flags=p.URDF_MERGE_FIXED_LINKS | p.URDF_USE_SELF_COLLISION,
        )
        # Ensure the full bin geometry sits on the tabletop (allow up/down alignment).
        aabb_min, _ = p.getAABB(bin_id)
        dz = self.table_surface_z - aabb_min[2]
        if abs(dz) > 1e-9:
            pos, quat = p.getBasePositionAndOrientation(bin_id)
            p.resetBasePositionAndOrientation(bin_id, [pos[0], pos[1], pos[2] + dz], quat)
        return bin_id

    def _place_robot_on_table(self, clearance=0.0005):
        """Lift robot base so its full body starts on/above tabletop."""
        aabb_min, _ = p.getAABB(self.robot.id)
        dz = (self.table_surface_z + clearance) - aabb_min[2]
        if dz > 1e-6:
            pos, quat = p.getBasePositionAndOrientation(self.robot.id)
            p.resetBasePositionAndOrientation(
                self.robot.id, [pos[0], pos[1], pos[2] + dz], quat
            )

    def _place_on_table(self, body_id, xy, yaw=0.0):
        # Drop above tabletop, settle, then snap to a small clearance.
        quat = p.getQuaternionFromEuler([0.0, 0.0, yaw])
        p.resetBasePositionAndOrientation(body_id, [xy[0], xy[1], self.table_surface_z + 0.25], quat)
        p.resetBaseVelocity(body_id, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        for _ in range(180):
            p.stepSimulation()

        self._snap_body_to_table(body_id, clearance=self.object_table_clearance)
        for _ in range(120):
            p.stepSimulation()
        self._snap_body_to_table(body_id, clearance=self.object_table_clearance)
        p.resetBaseVelocity(body_id, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])


    def _snap_body_to_table(self, body_id, clearance=0.003):
        aabb_min, _ = p.getAABB(body_id)
        dz = (self.table_surface_z + clearance) - aabb_min[2]
        # Only lift if object penetrates/undershoots table surface; never force it downward.
        if dz > 1e-6:
            pos, quat = p.getBasePositionAndOrientation(body_id)
            p.resetBasePositionAndOrientation(body_id, [pos[0], pos[1], pos[2] + dz], quat)
            p.resetBaseVelocity(body_id, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

    def _wait_until_stable(self, body_ids, max_steps=720, lin_eps=0.01, ang_eps=0.12, stable_steps=80):
        stable_count = 0
        for _ in range(max_steps):
            p.stepSimulation()
            all_stable = True
            for body_id in body_ids:
                lin_v, ang_v = p.getBaseVelocity(body_id)
                if np.linalg.norm(lin_v) > lin_eps or np.linalg.norm(ang_v) > ang_eps:
                    all_stable = False
                    break
            if all_stable:
                stable_count += 1
                if stable_count >= stable_steps:
                    return
            else:
                stable_count = 0

    def _candidate_grasp_object_ids(self):
        ids = list(self.object_ids.values())
        if self.boxID is not None:
            ids.append(self.boxID)
        return ids

    def _has_bilateral_finger_contact(self):
        finger_ids = getattr(self.robot, "finger_joint_ids", [9, 10])
        if len(finger_ids) < 2:
            return False
        left_id, right_id = finger_ids[0], finger_ids[1]
        for obj_id in self._candidate_grasp_object_ids():
            c_left = p.getContactPoints(self.robot.id, obj_id, left_id)
            c_right = p.getContactPoints(self.robot.id, obj_id, right_id)
            if c_left and c_right:
                return True
        return False

    def _has_any_finger_contact(self):
        finger_ids = getattr(self.robot, "finger_joint_ids", [9, 10])
        if len(finger_ids) < 2:
            return False
        left_id, right_id = finger_ids[0], finger_ids[1]
        for obj_id in self._candidate_grasp_object_ids():
            c_left = p.getContactPoints(self.robot.id, obj_id, left_id)
            c_right = p.getContactPoints(self.robot.id, obj_id, right_id)
            if c_left or c_right:
                return True
        return False

    def _move_gripper_with_surface_stop(self, target_width, close_substeps=30, sim_steps_per_substep=6):
        target_width = float(np.clip(target_width, *self.robot.gripper_range))
        current_width = float(self.robot.get_gripper_width())
        # Opening: use progressive release so residual contacts can separate.
        if target_width >= current_width:
            open_steps = 16
            for i in range(open_steps):
                alpha = float(i + 1) / float(open_steps)
                cmd_width = current_width + (target_width - current_width) * alpha
                self.robot.move_gripper(cmd_width)
                for _ in range(max(2, sim_steps_per_substep)):
                    self.step_simulation()
                # Once fingers are sufficiently open and no finger contact remains,
                # keep the final command and exit.
                if cmd_width >= 0.06 and not self._has_any_finger_contact():
                    self.robot.move_gripper(target_width)
                    for _ in range(max(2, sim_steps_per_substep)):
                        self.step_simulation()
                    return
            return

        # Keep only a tiny non-zero floor to avoid hard collapse.
        # A large floor (e.g. 0.018) leaves a visible gap on apple.
        min_close_width = 0.004
        target_width = max(target_width, min_close_width)

        # Phase 1: coarse close until first finger contact (or target reached).
        for i in range(close_substeps):
            if self._has_any_finger_contact():
                break
            alpha = float(i + 1) / float(close_substeps)
            cmd_width = current_width + (target_width - current_width) * alpha
            self.robot.move_gripper(cmd_width)
            for _ in range(sim_steps_per_substep):
                self.step_simulation()

        # Phase 2: fine close with contact-aware increments.
        # - If one finger has contact, use tiny steps to avoid tunneling.
        # - If both fingers have contact, add a short squeeze for stable grasp.
        cmd_width = float(self.robot.get_gripper_width())
        for _ in range(40):
            bilateral = self._has_bilateral_finger_contact()
            any_contact = bilateral or self._has_any_finger_contact()

            if bilateral:
                squeeze_target = max(target_width, cmd_width - 0.008)
                for j in range(10):
                    beta = float(j + 1) / 10.0
                    squeeze_width = cmd_width + (squeeze_target - cmd_width) * beta
                    self.robot.move_gripper(squeeze_width)
                    for _ in range(sim_steps_per_substep):
                        self.step_simulation()
                return

            if cmd_width <= target_width + 1e-4:
                return

            step_down = 0.0005 if any_contact else 0.0015
            cmd_width = max(target_width, cmd_width - step_down)
            self.robot.move_gripper(cmd_width)
            for _ in range(sim_steps_per_substep):
                self.step_simulation()

    def _load_block(self):
        if self.block_path is None:
            return None
        body_id = p.loadURDF(
            self.block_path,
            [0.50, 1.25, 0.08],
            p.getQuaternionFromEuler([0.0, 0.0, 0.0]),
            useFixedBase=False,
            flags=p.URDF_MERGE_FIXED_LINKS | p.URDF_USE_SELF_COLLISION,
        )
        self._place_on_table(body_id, [0.50, 0.25], yaw=0.0)
        return body_id

    def _load_ycb(self, name, path):
        if path is None:
            default_paths = {
                "plate": self._repo_path("models/ycb/029_plate/model.urdf"),
                "banana": self._repo_path("models/ycb/011_banana/model.urdf"),
                "apple": self._repo_path("models/ycb/013_apple/model.urdf"),
                "orange": self._repo_path("models/ycb/017_orange/model.urdf"),
                "plum": self._repo_path("models/ycb/018_plum/model.urdf"),
                "pitcher": self._repo_path("models/ycb/019_pitcher_base/model.urdf"),
                "h_cups": self._repo_path("models/ycb/065-h_cups/model.urdf"),
                "i_cups": self._repo_path("models/ycb/065-i_cups/model.urdf"),
                "j_cups": self._repo_path("models/ycb/065-j_cups/model.urdf"),
                "sponge": self._repo_path("models/ycb/026_sponge/model.urdf"),
                "mug": self._repo_path("models/ycb/025_mug/model.urdf"),
                "bowl": self._repo_path("models/ycb/024_bowl/model.urdf"),
            }
            if name not in default_paths:
                raise KeyError(
                    f"Unknown object '{name}' in _load_ycb; add it to default_paths or pass an explicit path."
                )
            path = default_paths[name]

        spec = self.object_spawn_specs[name]
        xy = spec["xy"]
        yaw = spec["yaw"]
        obj_id = p.loadURDF(
            path,
            [xy[0], xy[1], self.table_surface_z + 0.25],
            p.getQuaternionFromEuler([0.0, 0.0, yaw]),
            useFixedBase=False,
            flags=p.URDF_MERGE_FIXED_LINKS | p.URDF_USE_SELF_COLLISION,
        )
        dyn = {
            "lateralFriction": 3.0,
            "spinningFriction": 0.20,
            "rollingFriction": 0.08,
            "restitution": 0.0,
            "linearDamping": 0.05,
            "angularDamping": 0.1,
            # Improve collision robustness while objects move quickly.
            "ccdSweptSphereRadius": 0.004,
            "contactProcessingThreshold": 0.0,
            # Stiffer contact response so gripper stops on the surface.
            "contactStiffness": 6000.0,
            "contactDamping": 180.0,
        }
        if name in ("apple", "orange"):
            dyn.update(
                {
                    "lateralFriction": 4.8,
                    "spinningFriction": 0.35,
                    "rollingFriction": 0.14,
                    "linearDamping": 0.08,
                    "angularDamping": 0.14,
                }
            )
        if name == "apple":
            dyn.update(
                {
                    "lateralFriction": 4.8,
                    "spinningFriction": 0.35,
                    "rollingFriction": 0.14,
                }
            )
        if name == "orange":
            dyn.update({"rollingFriction": 0.08, "linearDamping": 0.18, "angularDamping": 0.3})
        if name == "banana":
            dyn.update(
                {
                    # Banana should be graspable but not "glued" to finger pads.
                    "lateralFriction": 2.2,
                    "spinningFriction": 0.10,
                    "rollingFriction": 0.03,
                    "linearDamping": 0.14,
                    "angularDamping": 0.24,
                    "contactStiffness": 2800.0,
                    "contactDamping": 90.0,
                }
            )
        if name == "plate":
            # Lower plate contact resistance to avoid temporary "sticking" on finger touch.
            dyn.update(
                {
                    "lateralFriction": 1.2,
                    "spinningFriction": 0.03,
                    "rollingFriction": 0.005,
                    "linearDamping": 0.04,
                    "angularDamping": 0.08,
                }
            )
        p.changeDynamics(obj_id, -1, **dyn)
        self._place_on_table(obj_id, xy, yaw=yaw)
        return obj_id

    def step_simulation(self):
        p.stepSimulation()
        if getattr(self, "_startup_stabilize_steps", 0) > 0:
            if self.boxID is not None:
                self._snap_body_to_table(self.boxID, clearance=self.object_table_clearance)
            for obj_id in self.object_ids.values():
                self._snap_body_to_table(obj_id, clearance=self.object_table_clearance)
            self._startup_stabilize_steps -= 1
        if self.vis:
            time.sleep(self.SIMULATION_STEP_DELAY)
            self.p_bar.update(1)

    def step(self, action, control_method="joint"):
        assert control_method in ("joint", "end")
        self.robot.move_ee(action[:-1], control_method)
        self._move_gripper_with_surface_stop(action[-1])
        for _ in range(20):
            self.step_simulation()
        return 0, False, None

    def reset(self):
        self.robot.reset()
        self._place_robot_on_table(clearance=0.0005)
        if hasattr(self, "object_spawn_poses"):
            if self.boxID is not None and hasattr(self, "box_spawn_pose"):
                box_pos, box_quat = self.box_spawn_pose
                p.resetBasePositionAndOrientation(self.boxID, box_pos, box_quat)
                p.resetBaseVelocity(self.boxID, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
            for name, obj_id in self.object_ids.items():
                pos, quat = self.object_spawn_poses[name]
                p.resetBasePositionAndOrientation(obj_id, pos, quat)
                p.resetBaseVelocity(obj_id, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        else:
            if self.boxID is not None:
                self._place_on_table(self.boxID, [0.50, 5.25], yaw=0.0)
            for name, obj_id in self.object_ids.items():
                spec = self.object_spawn_specs[name]
                self._place_on_table(obj_id, spec["xy"], yaw=spec["yaw"])
        settle_ids = list(self.object_ids.values())
        if self.boxID is not None:
            settle_ids.append(self.boxID)
        self._wait_until_stable(settle_ids)
        # Keep all dynamic objects resting on top of the table after settling.
        if self.boxID is not None:
            self._snap_body_to_table(self.boxID, clearance=self.object_table_clearance)
        for obj_id in self.object_ids.values():
            self._snap_body_to_table(obj_id, clearance=self.object_table_clearance)
        # No startup hold: let gravity/contact settle naturally from the beginning.
        self._startup_stabilize_steps = 0

    def close(self):
        p.disconnect(self.physics_client)


class ManiEnv(Env):
    def __init__(
        self,
        robot,
        block_path=None,
        table_path=None,
        bin_path=None,
        banana_path=None,
        plate_path=None,
        apple_path=None,
        orange_path=None,
        pitcher_path=None,
        h_cups_path=None,
        i_cups_path=None,
        j_cups_path=None,
        sponge_path=None,
        mug_path=None,
        bowl_path=None,
        vis=False,
    ):
        super().__init__(
            robot,
            block_path=block_path,
            table_path=table_path,
            bin_path=bin_path,
            banana_path=banana_path,
            plate_path=plate_path,
            apple_path=apple_path,
            orange_path=orange_path,
            vis=vis,
        )
        self.min_pose = np.array([0.25, -0.45, 0.02], dtype=np.float32)
        self.max_pose = np.array([0.90, 0.45, 0.75], dtype=np.float32)

    def reset_block(self, pos, euler_angle):
        if self.boxID is not None:
            p.resetBasePositionAndOrientation(self.boxID, pos, p.getQuaternionFromEuler(euler_angle))

    def execute_trajectory(self, trajectory):
        for action in trajectory:
            self.step(action, control_method="end")
        return self.check_success()

    def check_success(self):
        return True

    def step(self, action, control_method="end"):
        assert control_method in ("joint", "end")
        if control_method == "joint":
            return super().step(action, control_method=control_method)

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] == 5:
            pose = np.clip(action[:3], self.min_pose, self.max_pose)
            rotation = np.array([np.pi, 0.0, action[3]], dtype=np.float32)
            gripper_width = float(
                action[4] * (self.robot.gripper_range[1] - self.robot.gripper_range[0])
                + self.robot.gripper_range[0]
            )
        elif action.shape[0] == 7:
            pose = np.clip(action[:3], self.min_pose, self.max_pose)
            rotation = action[3:6]
            gripper_width = float(action[6])
        else:
            raise ValueError("Expected 5D or 7D end-effector action.")

        self.robot.move_ee(np.concatenate([pose, rotation]), control_method="end")
        self._move_gripper_with_surface_stop(gripper_width)
        for _ in range(120):
            self.step_simulation()
        return 0, False, None

    def get_block_position(self):
        if self.boxID is None:
            return None
        return p.getBasePositionAndOrientation(self.boxID)[0]

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
