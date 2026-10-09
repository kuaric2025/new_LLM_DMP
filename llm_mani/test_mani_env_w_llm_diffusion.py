"""Quick smoke test for ManiEnv with FrankaPanda.

Runs a few end-effector moves and gripper open/close in GUI.

Usage:
  python manipulation_env/test_mani_env.py
"""

import argparse
import json
import os
import sys
import threading
import time
from typing import Optional
from pathlib import Path

import cv2
import numpy as np

import torch
# if torch.cuda.is_available():
#     torch.cuda.init()  # Force early CUDA init

DISPLAY_AVAILABLE = bool(os.environ.get("DISPLAY"))
_WINDOW_CREATED = False
_DISPLAY_FAILED = False

# add repo root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manipulation_env.env import ManiEnv
from manipulation_env.robot import FrankaPanda

# ROS 2 imports (available via rebuilt Jazzy bindings for Python 3.10)
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from builtin_interfaces.msg import Time as TimeMsg
from sensor_msgs.msg import CameraInfo, Image
from message_filters import ApproximateTimeSynchronizer, Subscriber as MFSubscriber
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage

import pybullet as p
import math

# cwd = os.getcwd()
# print(cwd)
from utils.llm_utils import load_llm_config, apply_action_ids, ask_gpt4o
from utils.manipulation_utils import (
    sam3_inference,
    DiffusionPolicy,
    plate_drop_offset_xy_grid_4,
    pixel_depth_to_camera_frame,
    world_to_camera_frame,
    world_to_robot_frame,
    pose4x4_to_xyzrpy,
    get_robot_pose_from_tcp,
    anygrasp_demo,
    grasp_to_xyzrpy,
    euler_closest_to_ref,
    shortest_yaw
)

LLM_CONFIG, ACTION_ID_MAPPING, SYSTEM_PROMPT = load_llm_config()

def compute_intrinsics(width: int, height: int, fov_deg: float) -> tuple[float, float, float, float]:
    """Return (fx, fy, cx, cy) for a pinhole camera given FOV."""
    fov_rad = np.deg2rad(fov_deg)
    fx = (width / 2.0) / np.tan(fov_rad / 2.0)
    fy = (height / 2.0) / np.tan(fov_rad / 2.0)
    cx = width / 2.0
    cy = height / 2.0
    return fx, fy, cx, cy


def rotation_axes_summary(label: str, R: np.ndarray) -> None:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    print(f"{label}:")
    print(f"    x-axis {np.round(R[:, 0], 4)}")
    print(f"    y-axis {np.round(R[:, 1], 4)}")
    print(f"    z-axis {np.round(R[:, 2], 4)}")


def tcp_tracking_summary(label: str, commanded_xyzrpy: np.ndarray, robot) -> np.ndarray:
    actual_pos, actual_quat = robot.get_eef()
    actual_pos = np.asarray(actual_pos, dtype=np.float64)
    actual_quat = np.asarray(actual_quat, dtype=np.float64)

    commanded_xyzrpy = np.asarray(commanded_xyzrpy, dtype=np.float64).reshape(-1)
    target_pos = commanded_xyzrpy[:3]
    target_quat = np.asarray(
        p.getQuaternionFromEuler(tuple(float(v) for v in commanded_xyzrpy[3:6])),
        dtype=np.float64,
    )

    pos_err = actual_pos - target_pos
    q_err = p.getDifferenceQuaternion(actual_quat.tolist(), target_quat.tolist())
    ang_err = 2.0 * np.arctan2(np.linalg.norm(q_err[:3]), abs(q_err[3]))
    print(
        f"{label}: pos_err={np.round(pos_err, 4)} m, "
        f"pos_norm={np.linalg.norm(pos_err):.4f} m, "
        f"ang_err={np.degrees(ang_err):.2f} deg"
    )
    return np.concatenate([actual_pos, p.getEulerFromQuaternion(actual_quat)]).astype(np.float32)


class SimulatedD435Publisher(Node):
    def __init__(
        self,
        env: ManiEnv,
        width: int = 640,
        height: int = 480,
        fov: float = 70.0,
        near: float = 0.1,
        far: float = 3.0,
        frame_id: str = "camera_color_optical_frame",
        publish_rate_hz: float = 10.0,
        # robot_frame: str = "panda_link0",   # add
        # world_frame: str = "world",          # add
        # publish_tf: bool = True,             # add: optional toggle
    ) -> None:
        super().__init__("simulated_d435")
        self.env = env
        self.width = width
        self.height = height
        self.fov = fov
        self.near = near
        self.far = far
        self.frame_id = frame_id
        self.latest_rgb: Optional[np.ndarray] = None
        self.latest_depth: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()

        self.color_pub = self.create_publisher(Image, "/camera/color/image_raw", 10)
        self.depth_pub = self.create_publisher(Image, "/camera/depth/image_rect_raw", 10)
        self.info_pub = self.create_publisher(CameraInfo, "/camera/color/camera_info", 10)

        self.fx, self.fy, self.cx, self.cy = compute_intrinsics(width, height, fov)
        self.timer = self.create_timer(1.0 / publish_rate_hz, self._publish)

        # # TF broadcaster (static - publish once)
        # if publish_tf:
        #     self._tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        #     self._publish_tf()

    def capture_from_env(self):
        """Capture camera frame from the main thread; PyBullet is not thread-safe."""
        rgb, depth, _ = self.env.capture_rgbd(
            width=self.width,
            height=self.height,
            fov=self.fov,
            near=self.near,
            far=self.far,
            return_seg=False,
        )
        with self._frame_lock:
            self.latest_rgb = rgb.copy()
            self.latest_depth = depth.copy()

    def _publish(self):
        with self._frame_lock:
            if self.latest_rgb is None or self.latest_depth is None:
                return
            rgb = self.latest_rgb.copy()
            depth = self.latest_depth.copy()
        now = self.get_clock().now().to_msg()
        color_msg = self._make_color_msg(rgb, now)
        depth_msg = self._make_depth_msg(depth, now)
        info_msg = self._make_camera_info(now)

        self.color_pub.publish(color_msg)
        self.depth_pub.publish(depth_msg)
        self.info_pub.publish(info_msg)

    def _make_color_msg(self, rgb: np.ndarray, stamp: TimeMsg) -> Image:
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        msg.height = self.height
        msg.width = self.width
        msg.encoding = "rgb8"
        msg.step = self.width * 3
        msg.data = rgb.astype(np.uint8).tobytes()
        return msg

    def _make_depth_msg(self, depth: np.ndarray, stamp: TimeMsg) -> Image:
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id.replace("color", "depth")
        msg.height = self.height
        msg.width = self.width
        msg.encoding = "32FC1"
        msg.step = self.width * 4
        msg.data = depth.astype(np.float32).tobytes()
        return msg

    def _make_camera_info(self, stamp: TimeMsg) -> CameraInfo:
        msg = CameraInfo()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        msg.height = self.height
        msg.width = self.width
        msg.k = [self.fx, 0.0, self.cx, 0.0, self.fy, self.cy, 0.0, 0.0, 1.0]
        msg.p = [self.fx, 0.0, self.cx, 0.0, 0.0, self.fy, self.cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        msg.distortion_model = "plumb_bob"
        return msg

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

from VAE_DMP_mani.data.manipulation_data.read_npy import read_first_last

class CameraTripleSubscriber(Node):
    def __init__(
        self,
        color_topic: str = "/camera/color/image_raw",
        depth_topic: str = "/camera/depth/image_rect_raw",
        info_topic: str = "/camera/color/camera_info",
        log_interval: float = 5.0,
    ) -> None:
        super().__init__("camera_triple_subscriber")
        qos = qos_profile_sensor_data
        info_qos = QoSProfile(depth=10)
        self.color_sub = MFSubscriber(self, Image, color_topic, qos_profile=qos)
        self.depth_sub = MFSubscriber(self, Image, depth_topic, qos_profile=qos)
        self.info_sub = MFSubscriber(self, CameraInfo, info_topic, qos_profile=info_qos)

        self.sync = ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub, self.info_sub], queue_size=10, slop=0.1
        )
        self.sync.registerCallback(self._synced_callback)
        self.latest_rgb: Optional[np.ndarray] = None
        self.latest_depth: Optional[np.ndarray] = None
        self.latest_info: Optional[CameraInfo] = None
        self._lock = threading.Lock()
        self._last_log = 0.0
        self._log_interval = max(0.1, log_interval)

    def _synced_callback(self, rgb_msg: Image, depth_msg: Image, info_msg: CameraInfo):
        rgb = self._image_to_numpy(rgb_msg)
        depth = self._depth_to_numpy(depth_msg)
        with self._lock:
            self.latest_rgb = rgb
            self.latest_depth = depth
            self.latest_info = info_msg

    def _image_to_numpy(self, msg: Image) -> Optional[np.ndarray]:
        if msg.encoding not in ("rgb8", "bgr8"):
            self.get_logger().warn(f"Unsupported RGB encoding: {msg.encoding}")
            return None
        data = np.frombuffer(msg.data, dtype=np.uint8)
        row_stride = msg.step
        expected = msg.width * 3
        if row_stride == expected:
            arr = data.reshape((msg.height, msg.width, 3))
        else:
            arr = data.reshape((msg.height, row_stride))[:, :expected].reshape((msg.height, msg.width, 3))
        if msg.encoding == "bgr8":
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        return arr

    def _depth_to_numpy(self, msg: Image) -> Optional[np.ndarray]:
        if msg.encoding == "32FC1":
            arr = np.frombuffer(msg.data, dtype=np.float32).reshape((msg.height, msg.width))
            return arr
        if msg.encoding == "16UC1":
            arr = np.frombuffer(msg.data, dtype=np.uint16).reshape((msg.height, msg.width)).astype(np.float32)
            arr /= 1000.0
            return arr
        self.get_logger().warn(f"Unsupported depth encoding: {msg.encoding}")
        return None

    def get_latest_images(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[CameraInfo]]:
        """Thread-safe access to latest RGB, depth, and camera info."""
        with self._lock:
            return (
                self.latest_rgb.copy() if self.latest_rgb is not None else None,
                self.latest_depth.copy() if self.latest_depth is not None else None,
                self.latest_info,
            )

    def maybe_log_status(self):
        now = time.time()
        if now - self._last_log < self._log_interval:
            return
        self._last_log = now
        with self._lock:
            if self.latest_rgb is None or self.latest_depth is None or self.latest_info is None:
                self.get_logger().info("Waiting for synchronized RGB/depth/CameraInfo...")
                return
            rgb_shape = self.latest_rgb.shape
            depth_shape = self.latest_depth.shape
            fx = self.latest_info.k[0]
            fy = self.latest_info.k[4]
        self.get_logger().info(
            f"Synced frames: RGB {rgb_shape}, Depth {depth_shape}, fx={fx:.2f}, fy={fy:.2f}"
        )

def run_simulation(
    env: ManiEnv,
    actions: list[np.ndarray],
    control_method: str = "end",
    camera_node: Optional[SimulatedD435Publisher] = None,
    show_camera: bool = False,
    triple_subscriber: Optional[CameraTripleSubscriber] = None,
):
    print("Running ManiEnv smoke trajectory...")
    for a in actions:
        env.step(a, control_method=control_method)
        env.step_simulation()
        if show_camera and camera_node is not None:
            maybe_show_camera_frame(camera_node)
        if triple_subscriber is not None:
            triple_subscriber.maybe_log_status()
        time.sleep(0.01) # 0.05

    print("Done. Close the GUI window to exit.")
    while True:
        env.step_simulation()
        if show_camera and camera_node is not None:
            maybe_show_camera_frame(camera_node)
        if triple_subscriber is not None:
            triple_subscriber.maybe_log_status()
        time.sleep(0.005) # 0.01


def start_ros_nodes(nodes: list[Node]) -> tuple[rclpy.executors.Executor, threading.Thread]:
    executor = rclpy.executors.MultiThreadedExecutor()
    for node in nodes:
        executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    return executor, thread


def maybe_show_camera_frame(node: SimulatedD435Publisher, window_name: str = "Simulated D435"):
    global _WINDOW_CREATED, _DISPLAY_FAILED
    if not DISPLAY_AVAILABLE:
        if not _DISPLAY_FAILED:
            print("[camera] DISPLAY not available; skipping on-screen preview")
            _DISPLAY_FAILED = True
        return
    if _DISPLAY_FAILED:
        return
    if node.latest_rgb is None:
        return
    try:
        if not _WINDOW_CREATED:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            _WINDOW_CREATED = True
        rgb = node.latest_rgb
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imshow(window_name, rgb_bgr)
        cv2.waitKey(1)
    except cv2.error as exc:
        print(f"[camera] OpenCV display failed: {exc}")
        _DISPLAY_FAILED = True


def shutdown_ros(executor: Optional[rclpy.executors.Executor]):
    if executor is not None:
        executor.shutdown()
    if rclpy.ok():
        rclpy.shutdown()


def parse_args():
    parser = argparse.ArgumentParser(description="ManiEnv test with simulated D435 publisher")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fov", type=float, default=70.0)
    parser.add_argument("--camera-rate", type=float, default=10.0, help="Publish rate in Hz")
    parser.add_argument("--vis", action="store_true", help="Show PyBullet GUI")
    parser.add_argument("--show-camera", action="store_true", help="Display realtime camera view via OpenCV")
    parser.add_argument("--subscribe-camera", action="store_true", help="Subscribe to RGB+depth+CameraInfo topics for monitoring")
    parser.add_argument("--subscriber-log-interval", type=float, default=5.0, help="Seconds between subscriber status logs")
    parser.add_argument("--prompt", type=str, default="prepare breakfast")

    parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
    parser.add_argument('--max_gripper_width', type=float, default=0.08, help='Maximum gripper width (<=0.1m)')
    parser.add_argument('--gripper_height', type=float, default=0.01, help='Gripper height')
    parser.add_argument(
        '--grasp-center-offset-z',
        type=float,
        default=0.105,
        help='Hand-frame Z offset from panda_hand to the intended grasp center in meters.',
    )
    parser.add_argument('--top_down_grasp', action='store_true', help='Output top-down grasps.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--num-runs', type=int, default=1, help='Number of trials for success rate evaluation (default: 1)')
    parser.add_argument('--seed', type=int, default=None, help='Base seed for object pose randomization; each trial uses seed+trial_idx')
    parser.add_argument('--no-randomize', action='store_true', help='Disable object pose randomization even when num-runs > 1')
    parser.add_argument('--scene-configs', type=str, default=None,
        help='Path to pre-generated scene_configs.json; use same 10 scenes across pipeline tests for fair comparison')
    parser.add_argument(
        "--dp-checkpoint",
        type=str,
        default=os.environ.get("DP_CHECKPOINT"),
        help="Path to trained real-stanford/diffusion_policy checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--dp-repo-path",
        type=str,
        default=os.environ.get("DP_REPO_PATH"),
        help="Path to local clone of real-stanford/diffusion_policy repository.",
    )
    parser.add_argument("--dp-obs-steps", type=int, default=2, help="Observation horizon To for diffusion policy.")
    parser.add_argument("--dp-action-steps", type=int, default=16, help="Action horizon Ta for diffusion policy.")
    parser.add_argument("--dp-inference-steps", type=int, default=16, help="DDIM denoising steps per policy call.")
    return parser.parse_args()



def main():
    args = parse_args()

    table_path = os.path.join(os.path.dirname(__file__), "../manipulation_env/models/urdf/objects/table/table.urdf")
    block_path = None

    # Task 1
    bin_path = os.path.join(os.path.dirname(__file__), "../manipulation_env/models/urdf/objects/table/bin.urdf")
    banana_path = os.path.join(os.path.dirname(__file__), "../manipulation_env/models/ycb/011_banana/model.urdf")
    plate_path = os.path.join(os.path.dirname(__file__), "../manipulation_env/models/ycb/029_plate/model.urdf")
    apple_path = os.path.join(os.path.dirname(__file__), "../manipulation_env/models/ycb/013_apple/model.urdf")
    orange_path = os.path.join(os.path.dirname(__file__), "../manipulation_env/models/ycb/017_orange/model.urdf")
    
    # Task 2
    pitcher_path = os.path.join(os.path.dirname(__file__), "../manipulation_env/models/ycb/019_pitcher_base/model.urdf")
    e_cups_path = os.path.join(os.path.dirname(__file__), "../manipulation_env/models/ycb/065-e_cups/model.urdf")
    i_cups_path = os.path.join(os.path.dirname(__file__), "../manipulation_env/models/ycb/065-i_cups/model.urdf")
    j_cups_path = os.path.join(os.path.dirname(__file__), "../manipulation_env/models/ycb/065-j_cups/model.urdf")
    g_cups_path = os.path.join(os.path.dirname(__file__), "../manipulation_env/models/ycb/065-g_cups/model.urdf")
    h_cups_path = os.path.join(os.path.dirname(__file__), "../manipulation_env/models/ycb/065-h_cups/model.urdf")

    # Task 3
    sponge_path = os.path.join(os.path.dirname(__file__), "../manipulation_env/models/ycb/026_sponge/model.urdf")
    mug_path = os.path.join(os.path.dirname(__file__), "../manipulation_env/models/ycb/025_mug/model.urdf")
    bowl_path = os.path.join(os.path.dirname(__file__), "../manipulation_env/models/ycb/024_plate/model.urdf")
    # banana, bowl

    # # serve multiple cups of drink
            # ("065-h_cups", [0.55, -0.24, 0.06], [0, 0, 1.0]),
            # ("065-i_cups", [0.55, 0.00, 0.06], [0, 0, 1.0]),
            # ("065-j_cups", [0.55, 0.24, 0.06], [0, 0, 1.0]),
            # ("019_pitcher_base", [-0.2, -0.5, 0.06], [0, 0, 1.0]),

            # # clear the table into bin and clean the table
            # ("011_banana", [0.55, -0.24, 0.06], [0, 0, 1.0]),
            # ("025_mug", [0.65, 0.24, 0.07], [0, 0, 0]),
            # # ("010_potted_meat_can", [0.4, 0.2, 0.06], [0, 0, 1.0]),
            # # ("065-h_cups", [0.55, -0.24, 0.06], [0, 0, 1.0]),
            # ("013_apple", [0.4, -0.24, 0.06], [0, 0, 1.0]),
            # ("026_sponge", [0.2, 0.2, 0.06], [0, 0, 1.0]),

    robot = FrankaPanda(model_path=None)
    # env = ManiEnv(robot, block_path=block_path, table_path=table_path, bin_path=bin_path, banana_path=banana_path, plate_path=plate_path, apple_path=apple_path, orange_path=orange_path, pitcher_path=pitcher_path, e_cups_path=e_cups_path, i_cups_path=i_cups_path, j_cups_path=j_cups_path, sponge_path=sponge_path, mug_path=mug_path, bowl_path=bowl_path, vis=args.vis)
    # config_path = os.path.join(os.path.dirname(__file__), "../manipulation_env/configs/task_1.json")
    env = ManiEnv(
        robot,
        block_path=block_path, table_path=table_path, bin_path=bin_path, banana_path=banana_path, plate_path=plate_path, apple_path=apple_path, orange_path=orange_path, pitcher_path=pitcher_path, e_cups_path=e_cups_path, i_cups_path=i_cups_path, j_cups_path=j_cups_path, sponge_path=sponge_path, mug_path=mug_path, bowl_path=bowl_path, vis=args.vis,
        object_config_path=args.scene_configs,
    )
    rclpy.init()
    nodes: list[Node] = []
    camera_node = SimulatedD435Publisher(
        env,
        width=args.camera_width,
        height=args.camera_height,
        fov=args.camera_fov,
        near=0.1,
        far=3.0,
        publish_rate_hz=args.camera_rate,
    )
    nodes.append(camera_node)

    triple_sub = None
    if args.subscribe_camera:
        triple_sub = CameraTripleSubscriber(
            log_interval=args.subscriber_log_interval,
        )
        nodes.append(triple_sub)
    
    executor, exec_thread = start_ros_nodes(nodes)
    # Capture PyBullet camera frames on the main thread. The ROS timer only
    # publishes this cached frame to avoid OpenGL/PyBullet thread crashes.
    camera_node.capture_from_env()
    time.sleep(0.2)

    scene_configs = None
    if args.scene_configs is not None:
        with open(args.scene_configs, "r", encoding="utf-8") as f:
            data = json.load(f)
        scene_configs = data.get("scenes", data) if isinstance(data, dict) else data
        if not isinstance(scene_configs, list):
            scene_configs = [scene_configs]
        print(f"Loaded {len(scene_configs)} pre-generated scene configs from {args.scene_configs}")

    use_randomize = args.num_runs > 1 and not args.no_randomize and scene_configs is None
    results: list[bool] = []
    
    for trial_idx in range(args.num_runs):
        scene_config = None
        if scene_configs is not None:
            scene_config = scene_configs[trial_idx % len(scene_configs)]
            if args.num_runs > 1:
                sid = int(
                    np.random.default_rng(
                        (args.seed + trial_idx) if args.seed is not None else None
                    ).integers(0, 2**31 - 1)
                )
                print(f"\n\033[36m=== Trial {trial_idx + 1}/{args.num_runs} (scene seed={sid}) ===\033[0m")
        else:
            if args.num_runs > 1:
                if use_randomize:
                    print(f"\n\033[36m=== Trial {trial_idx + 1}/{args.num_runs} (randomized object poses) ===\033[0m")
                else:
                    print(f"\n\033[36m=== Trial {trial_idx + 1}/{args.num_runs} ===\033[0m")

        try:
            # if scene_config is not None:
            #     print(f"Scene config: {scene_config}=====================")
            #     env.reset(scene_config=scene_config)
            # else:
            #     print(f"Scene config: None=====================")
            #     trial_rng = np.random.default_rng((args.seed + trial_idx) if args.seed is not None else None)
            #     env.reset(randomize_poses=use_randomize, rng=trial_rng if use_randomize else None)
            # time.sleep(0.5)  # Wait for camera to capture new scene
            if triple_sub is not None:
                rgb, depth, cam_info = triple_sub.get_latest_images()
                if rgb is not None:
                    print(f"RGB shape: {rgb.shape}, dtype: {rgb.dtype}")
                    print(f"RGB min: {rgb.min()}, max: {rgb.max()}, mean: {rgb.mean()}")
                    print(f"RGB pixel at (y=100, x=200): R={rgb[100, 200, 0]}, G={rgb[100, 200, 1]}, B={rgb[100, 200, 2]}")
                if depth is not None:
                    print(f"Depth shape: {depth.shape}, dtype: {depth.dtype}")
                    print(f"Depth min: {depth.min()}, max: {depth.max()}, mean: {depth.mean()}")
                    print(f"Depth value at (100, 200): {depth[100, 200]} meters")
            #===========================MLLMs======================================
        
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            # Get RGB image from subscriber or camera node
            rgb_image = None
            depth_image = None
            if triple_sub is not None:
                rgb, depth, _ = triple_sub.get_latest_images()
                if rgb is not None:
                    rgb_image = rgb
                    depth_image = depth
                else:
                    print("Warning: triple_sub has no RGB image yet, waiting...")
                    time.sleep(1.0)  # Wait a bit more
                    rgb, depth, _ = triple_sub.get_latest_images()
                    if rgb is not None:
                        rgb_image = rgb
                        depth_image = depth
            elif camera_node is not None and camera_node.latest_rgb is not None:
                rgb_image = camera_node.latest_rgb.copy()
                _, depth_image, _ = env.capture_rgbd(
                    width=args.camera_width, height=args.camera_height, fov=args.camera_fov,
                    near=0.1, far=3.0, return_seg=False,
                )
            
            # Fallback to file path if no RGB available
            if rgb_image is None:
                image_path = os.environ.get('LLM_IMAGE_PATH', '/home/mhumais/Downloads/55.png')
                image_bgr = cv2.imread(str(image_path))
                if image_bgr is not None:
                    print(f"Warning: No RGB image available, using file path: {image_path}")
                    rgb_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                else:
                    print(
                        f"Warning: No RGB image available and fallback file is missing: {image_path}. "
                        "Capturing directly from env."
                    )
                    rgb_image, depth_image, _ = env.capture_rgbd(
                        width=args.camera_width, height=args.camera_height, fov=args.camera_fov,
                        near=0.1, far=3.0, return_seg=False,
                    )
                if depth_image is None:
                    _, depth_image, _ = env.capture_rgbd(
                    width=args.camera_width, height=args.camera_height, fov=args.camera_fov,
                    near=0.1, far=3.0, return_seg=False,
                    )
                prompt = os.environ.get('LLM_PROMPT', args.prompt)
                reply = ask_gpt4o(
                    prompt=prompt,
                    system_prompt=SYSTEM_PROMPT,
                    image_arrays=[rgb_image],
                )
            else:
                print(f"Using RGB image from camera: shape={rgb_image.shape}, dtype={rgb_image.dtype}")
                prompt = os.environ.get('LLM_PROMPT', args.prompt)
                reply = ask_gpt4o(
                    prompt=prompt,
                    system_prompt=SYSTEM_PROMPT,
                    image_arrays=[rgb_image],
                )
            
            # # build lib for segmentation masks
            # object_list = ['banana', 'orange', 'apple', 'plate']
            # # object_list = ['apple']
            # segmap_list = {}
            # for obj in object_list:
            #     # import pdb; pdb.set_trace()
            #     print(f"  Segmenting {obj}...")
            #     segmap = sam3_inference(rgb_image, obj)
            #     segmap_list[obj] = segmap
            #     # import pdb; pdb.set_trace()
            # torch.save(segmap_list, "task_12.pt")
            # # breakfast_preparation: task_1.pt
            # import pdb; pdb.set_trace()
            
            
            # loaded = torch.load("task_1.pt")
            # np.array(loaded['banana'][0]).shape
            
            # f = open("breakfast_segmap.json", "w", encoding="utf-8")
            # try:
            #     json.dump(segmap_list, f, ensure_ascii=False, indent=4)
            # finally:
            #     f.close()
        
            # segmap_list = torch.load("task_12.pt")
        
            # Add action_id to each step from config mapping
            reply = apply_action_ids(reply, ACTION_ID_MAPPING)
        
            if not args.dp_checkpoint:
                raise ValueError("Missing --dp-checkpoint (or DP_CHECKPOINT env var) for diffusion policy.")
            if not args.dp_repo_path:
                raise ValueError("Missing --dp-repo-path (or DP_REPO_PATH env var) for diffusion policy.")

            ## build diffusion policy generators
            traj_gen_cache = {}
            for i in range(0,3):
                traj_gen_cache[i] = DiffusionPolicy(
                    task_id=i,
                    checkpoint_path=args.dp_checkpoint,
                    repo_path=args.dp_repo_path,
                    device=device,
                    n_obs_steps=args.dp_obs_steps,
                    n_action_steps=args.dp_action_steps,
                    num_inference_steps=args.dp_inference_steps,
                )
            
        
            # Print formatted JSON response
            print("\n=== GPT-4o Response (JSON) ===")
            print(json.dumps(reply, indent=2))
            num_steps = len(reply)
        
            gripper_val = 0.15
            drop_id = 0
        
            robot_current_pose = get_robot_pose_from_tcp(env.robot)
            
            print("\n=== Parsed Action Plan ===")
            for step in reply:
                env.clean_traj_plot()
                print(f"\033[32mStep {step.get('step_id', 'N/A')}: {step.get('action', 'N/A')} (action_id: {step.get('action_id', 'N/A')})\033[0m")
                if 'target_object' in step and 'action_id' in step:
                    color = step['target_object'].get('attributes')
                    if color == 'white':
                        color = 'blue'
                    if color is not None:
                        color = str(color).strip()
                    # import pdb; pdb.set_trace()
                    obj_name = step['target_object'].get('name')
                    
                    obj_name = '' if obj_name is None else str(obj_name).strip()
                    obj = f"{color} {obj_name}".strip() if color else obj_name
                    if obj_name == 'kettle':
                        obj = 'pitcher handle'
                    if obj_name == 'bowl':
                        obj_name = 'plate'
                    # import pdb; pdb.set_trace()
                    # if obj == 'bowl':
                    #     obj = 'plate'
                    print(f"  Target Object: {obj}")
                    action_id = step['action_id']
                    if action_id is not None:
                        action_id = int(action_id)
                    print(f"  Action ID: {action_id}")
                    action_name = step['action']
                    print(f"  Action Name: {action_name}")
                    step_id = step.get('step_id', 'N/A')
                    # import pdb; pdb.set_trace()
                    
                    
                    # gripper_val = float(robot_current_pose[-1])
                    # msg = JointTrajectory()
                    # msg.header.stamp = camera_node.get_clock().now().to_msg()
                    # msg.header.frame_id = "panda_link0"
                    # msg.joint_names = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
                    # action_id = 3
                    dt = 0.05
                    if action_id == 0:
                        print(f"  SAM3 prompt: {obj}")
                        is_bin_target = obj.lower() in {"gray bin", "grey bin", "bin"} or obj_name.lower() == "bin"
                        bin_id = None
                        if is_bin_target:
                            bin_id = env.object_ids.get("bin") if hasattr(env, "object_ids") else None
                            if bin_id is None:
                                bin_id = getattr(env, "bin_id", None)
                            if bin_id is None:
                                bin_id = getattr(env, "binID", None)
                        if is_bin_target and bin_id is not None:
                            _, bin_depth, bin_seg = env.capture_rgbd(
                                width=args.camera_width,
                                height=args.camera_height,
                                fov=args.camera_fov,
                                near=0.1,
                                far=3.0,
                                return_seg=True,
                            )
                            bin_id = int(bin_id)
                            bin_seg = np.asarray(bin_seg, dtype=np.int32)
                            bin_mask = (bin_seg & ((1 << 24) - 1)) == bin_id
                            if np.any(bin_mask):
                                depth_image = bin_depth
                                segmap = torch.from_numpy(bin_mask).to(device)
                                print(
                                    f"  Using PyBullet bin mask: pixels={int(bin_mask.sum())}, "
                                    f"object_id={bin_id}"
                                )
                            else:
                                unique_seg = np.unique(bin_seg)
                                print(
                                    f"  PyBullet bin mask is empty for object_id={bin_id}; "
                                    f"seg unique sample={unique_seg[:10]}. Falling back to SAM3."
                                )
                                segmap = sam3_inference(rgb_image, obj)
                        else:
                            if is_bin_target:
                                print("  No PyBullet bin id found; falling back to SAM3.")
                            segmap = sam3_inference(rgb_image, obj)
                        # segmap = torch.tensor(segmap_list[obj][0]).to(device)
                        # segmap =segmap_list[obj][0]
                        if segmap is not None:
                        # aa = True
                        # if aa:
                            # import pdb; pdb.set_trace()
                            # segmap = segmap.permute(2, 1, 0).squeeze(2) # real-time segmentation
                            segmap = segmap.squeeze(0)
                            print(f"  Segmap: {segmap.shape}")
                            ys, xs = torch.where(segmap)
                            # xs, ys = torch.where(segmap)
                            # # import pdb; pdb.set_trace()
                            # center_y = ys.float().mean()   # row coordinate (v)
                            # center_x = xs.float().mean()   # col coordinate (u)
                            # # If you want integer pixel center:
                            # center_y_int = int(center_y.round().item())
                            # center_x_int = int(center_x.round().item())
        
                            center_y = ys.float().median().item()   # more robust to asymmetric masks
                            center_x = xs.float().median().item()
                            # import pdb; pdb.set_trace()
                            center_y_int = int(center_y)
                            center_x_int = int(center_x)
        
                            
                            print(f"  Center (y, x): ({center_y_int}, {center_x_int})") # 329, 294
                            
                            center_z_m = depth_image[center_y_int, center_x_int] # 1.138107180595398
                            print(f"  Center Z (m): {center_z_m}")
                            
                            #==============get pose in robot frame======================================
                            T_c2r = np.array([[ 1.0, 0.0, 0.0, 0.6000 ],
                                            [ 0.0, -1.0, 0.0, 0.0 ],
                                            [ 0.0, 0.0, -1.0, 1.200 ],
                                            [ 0.0, 0.0, 0.0, 1.0 ]])
                            T_r2c = np.linalg.inv(T_c2r)
        
                            T_o2c = np.array([
                                    [1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0,-1, 0],
                                    [0, 0, 0, 1],
                                ])
        
                            # cam_info = np.array([457.0073621574767,  0, 320.0,
                            #             0, 342.75552161810754, 240.0,
                            #                                 0,  0,  1])
                            cam_info = np.array([342.7555,  0, 320.0,
                                        0, 342.7555, 240.0,
                                                            0,  0,  1])
                            # # import pdb; pdb.set_trace()
                            # pos_camera = pixel_depth_to_camera_frame(
                            #             center_x_int, center_y_int, center_z_m, cam_info
                            #         )
        
                            # Backproject all segmented pixels to 3D
                            pos_camera = pixel_depth_to_camera_frame(depth_image, segmap.cpu().numpy(), cam_info)
        
                            # Empirical correction: camera-frame offset (X, Y, Z) from ground-truth comparison.
                            # Tune per object if needed; default derived from apple.
                            # OBJECT_OFFSETS_CAM = {
                            #     "apple": np.array([-0.019, -0.022, 0.060]),
                            #     "banana": np.array([-0.019, -0.022, 0.060]),  # same as apple; validate & tune
                            #     "orange": np.array([-0.019, -0.022, 0.060]),
                            #     "plate": np.array([-0.019, -0.022, 0.060]),
                            # }
                            # offset_cam = OBJECT_OFFSETS_CAM.get(obj_name, np.array([-0.019, -0.022, 0.060]))
                            # pos_camera = np.asarray(pos_camera, dtype=np.float64) + offset_cam
        
                            # # Debug: compare with ground truth (optional)
                            # # obj_id_key = {"kettle": "pitcher"}.get(obj_name, obj_name)
                            # obj_id = env.object_ids.get('apple')
                            # if obj_id is not None:
                            #     # Better ground truth: AABB center (geometric center) instead of link origin
                            #     aabb_min, aabb_max = p.getAABB(obj_id)
                            #     center_world = 0.5 * (np.array(aabb_min) + np.array(aabb_max))
                            #     pos_world_h = np.append(center_world, 1.0)
                            #     pos_cam_gt = (T_r2c @ pos_world_h)[:3]
                            #     print(f"Ground truth in camera:", pos_cam_gt)
                            #     print("Predicted from segmap (after offset):", pos_camera)
        
                            # pos_camera = pixel_depth_to_camera_frame(center_y_int, center_x_int, center_z_m, cam_info_2)
                            pos_camera_ = np.eye(4)
                            pos_camera_[:3, 3] = pos_camera.reshape(3)
                            # import pdb; pdb.set_trace()
        
                            # # anygrasp pose in camera frame
                            if step_id < num_steps and reply[step_id]['action'] == "grasp":
                            #     print("AnyGrasp Predicting=========")
                                # grasp = anygrasp_demo(rgb_image, depth_image, np.array(segmap.permute(1,0).cpu()), step_id, args) # real-time segmentation
                                grasp = anygrasp_demo(rgb_image, depth_image, np.array(segmap.cpu()), step_id, args) # pre-computed segmentation
                                rotation_axes_summary("  AnyGrasp camera-frame axes", grasp.rotation_matrix)
                                # Map AnyGrasp's gripper frame to the Panda hand frame.
                                # This remaps the nearly vertical AnyGrasp x-axis onto the
                                # Panda hand's local z-axis, which is the axis IK uses.
                                R_grasp_to_panda = np.array([
                                    [0.0, 0.0, 1.0],
                                    [0.0, 1.0, 0.0],
                                    [-1.0, 0.0, 0.0],
                                ], dtype=np.float64)
                                grasp_rotation_panda = np.asarray(grasp.rotation_matrix, dtype=np.float64) @ R_grasp_to_panda
                                pos_camera_[:3, :3] = grasp_rotation_panda
                                pos_camera_[:3, 3] = grasp.translation
                                rotation_axes_summary("  Panda-aligned camera-frame axes", grasp_rotation_panda)
        
                            pos_robot = T_c2r @ pos_camera_
                            print(f"  Pos robot: {pos_robot}")
                            rotation_axes_summary("  Robot-frame grasp axes", pos_robot[:3, :3])
        
                            # Offset is expressed in the Panda hand frame. The default
                            # 0.105 m matches panda_hand -> panda_grasptarget in the URDF,
                            # but this can be tuned if AnyGrasp's translation is already
                            # closer to the true contact center.
                            GRASP_CENTER_OFFSET = np.array(
                                [0.0, -0.0, -args.grasp_center_offset_z], #World-frame offset delta: [ 0.014  -0.03    0.0837]
                                dtype=np.float64,
                            )
                            FINGER_LENGTH =  0 # 0.1034  # m
        
                            # 根据你的手爪，在这里选对轴：
                            # GRIPPER_AXIS = np.array([1.0, 0.0, 0.0])  # 如果手指沿 +X
                            GRIPPER_AXIS = np.array([0.0, 0.0, 1.0])  # 如果手指沿 +Z
        
                            TCP_OFFSET_WITH_FINGER = GRASP_CENTER_OFFSET + FINGER_LENGTH * GRIPPER_AXIS
                            print(f"  Hand-frame TCP offset: {TCP_OFFSET_WITH_FINGER}")
                            offset_world_delta = pos_robot[:3, :3] @ (-TCP_OFFSET_WITH_FINGER)
                            print(f"  World-frame offset delta: {np.round(offset_world_delta, 4)}")
        
                            T_offset_inv = np.eye(4)
                            T_offset_inv[:3, 3] = -TCP_OFFSET_WITH_FINGER
                            pos_robot_tcp = pos_robot @ T_offset_inv  # TCP pose in robot frame
                            # pos_robot_tcp = T_offset_inv @ pos_robot
                            rotation_axes_summary("  Robot-frame TCP axes", pos_robot_tcp[:3, :3])
                            
                            
                            
                            #==================call diffusion policy==========================
                            # import pdb; pdb.set_trace()
                            if step_id == 1:
                                robot_initial_pose = get_robot_pose_from_tcp(env.robot) # in world frame    
                            else:
                                robot_initial_pose = robot_current_pose
                            # robot_initial_pose = get_robot_pose_from_tcp(env.robot)
                            
                            robot_initial_pose[6] = gripper_val #0.08
                            print(f"  Robot initial pose: {robot_initial_pose}")
        
                            # env.step(robot_initial_pose, control_method="end")
                            # robot_end_pose = pos_robot  
                            # robot_end_pose[3:6] = 0
                            robot_end_pose = pos_robot_tcp
                            # robot_end_pose[3:6] = 0
                            
        
                            ## check the closest euler representation
                            # import pdb; pdb.set_trace()
                            xyzrpy_end = pose4x4_to_xyzrpy(robot_end_pose, gripper_val)
                            # # Prefer Euler representation that is close to current pose
                            xyzrpy_end[3:6] = euler_closest_to_ref(
                                xyzrpy_end[3:6],
                                robot_initial_pose[3:6]
                            )
                            target_quat = p.getQuaternionFromEuler(tuple(float(v) for v in xyzrpy_end[3:6]))
                            target_R = np.array(p.getMatrixFromQuaternion(target_quat), dtype=np.float64).reshape(3, 3)
                            current_quat = env.robot.get_eef()[1]
                            current_R = np.array(p.getMatrixFromQuaternion(current_quat), dtype=np.float64).reshape(3, 3)
                            rotation_axes_summary("  Current Panda hand axes", current_R)
                            rotation_axes_summary("  Commanded Panda hand axes", target_R)
                            # import pdb; pdb.set_trace()
                            # array([ 0.30517235,  0.2956315 ,  0.0627137 ,  3.1415927 , -0.        ,0.        ,  0.15      ])  image 
                            #array([ 0.29596588,  0.32353005,  0.06697686,  3.1415927 ,  1.2753307 ,-1.3081839 ,  0.15      ], dtype=float32) anygrasp
        
                            # if step_id == 1:
                            #     xyzrpy_end[2] -= 0.018
                            if reply[step_id]['action'] == "drop":
                                drop_id += 1
                                xyzrpy_end[2] += 0.18
                                xyzrpy_end[2] += 0.05 * drop_id
                                # Avoid overlap: place objects on predefined grid
                                if obj_name == 'plate':
                                    offset_xy = plate_drop_offset_xy_grid_4(drop_id, radius=0.06)
                                    xyzrpy_end[0] += offset_xy[0]
                                    xyzrpy_end[1] += offset_xy[1]
                                    print(f"  Plate drop offset: {offset_xy}")
                            if reply[step_id]['action'] == "pour":
                                xyzrpy_end[2] += 0.25
                                # xyzrpy_end[3] = 0
                                # xyzrpy_end[4] = math.pi

                                ref_yaw = float(robot_initial_pose[5])
                                yaw_candidates = [
                                    float(xyzrpy_end[5]),
                                    float(xyzrpy_end[5] + np.pi),
                                    float(xyzrpy_end[5] - np.pi),
                                ]
                                best_yaw = min(
                                    yaw_candidates,
                                    key=lambda y: abs(np.arctan2(np.sin(y - ref_yaw), np.cos(y - ref_yaw))),
                                )
                                xyzrpy_end[5] = shortest_yaw(ref_yaw, best_yaw)

                            # if obj == "orange":
                            #     xyzrpy_end[0] += 0.02
                            #     xyzrpy_end[1] += 0.02
                            # if obj == "orange" and reply[step_id]['action'] == "drop":
                            #     xyzrpy_end[0] -= 0.05
                            # if obj == "apple":
                            #     xyzrpy_end[0] -= 0.02
                            #     xyzrpy_end[1] += 0.01
                        
                            # begin_yaw = robot_initial_pose[5]   # current yaw
                            # target_yaw = xyzrpy_end[5]   # target yaw
                            # xyzrpy_end[5] = shortest_yaw(begin_yaw, target_yaw)
                            # # Use xyzrpy_end for trajectory goal
                            # torch_robot_end_pose = torch.tensor(xyzrpy_end, dtype=torch.float32).to(device)
        
                            # traj_gen = DiffusionPolicy(task_id=action_id)
                            traj_gen = traj_gen_cache[action_id]
                        
                            # generate trajectory from diffusion policy
                            traj_gen.eval()
                            torch_robot_intial_pose = torch.tensor(robot_initial_pose, dtype=torch.float32).to(device)
                            # torch_robot_end_pose = torch.tensor(pose4x4_to_xyzrpy(robot_end_pose,0.0), dtype=torch.float32).to(device)
                            torch_robot_end_pose = torch.tensor(xyzrpy_end, dtype=torch.float32).to(device)
                            # import pdb; pdb.set_trace()
                            # torch_robot_end_pose[3] = math.pi # make the grasp vertically facing down
                            # torch_robot_end_pose[4] = 0
                            print(f"  Torch robot end pose: {torch_robot_end_pose}")
                            
                            with torch.no_grad():
                                traj = traj_gen(class_idx=int(action_id), x0=torch_robot_intial_pose[0:6], goal=torch_robot_end_pose[0:6])  
        
                            # traj[0,:,-1] = torch_robot_end_pose[0:6]   
                        
                            traj = traj.detach().cpu().numpy()[0]
                            
                            # # plot the trajectory
                            env.traj_plot(traj[:3,:].T)
                            # import pdb; pdb.set_trace()
                            # Use dense waypoint execution while carrying an object to reduce slip.
                            # step_stride = 1 if gripper_val < 0.08 else 5
                            step_stride = 1
                            executed_prev = np.asarray(get_robot_pose_from_tcp(env.robot)[:3], dtype=np.float64)
                            for i in range(traj.shape[1]):
                                if i % step_stride == 0 or i == traj.shape[1] - 1:
                                    # traj[:, i][3] = math.pi
                                    # traj[:, i][4] = 0
                                    traj_7DoF = np.append(traj[:, i], gripper_val)
                                    env.step(traj_7DoF)
                                    actual_pose = tcp_tracking_summary("  Executed waypoint tracking", traj_7DoF[:6], env.robot)
                                    executed_curr = np.asarray(actual_pose[:3], dtype=np.float64)
                                    p.addUserDebugLine(executed_prev, executed_curr, [0.0, 1.0, 1.0], 3)
                                    executed_prev = executed_curr
                                    robot_current_pose = np.append(actual_pose, gripper_val)
                            # import pdb; pdb.set_trace()
                    elif action_id == 1: # wipe
                        print("wipe")
                        # wipe: move to target and wipe
                        # traj_7DoF = np.append(traj[:, i], gripper_val)
                        # env.step(traj_7DoF)
                        # robot_current_pose = traj_7DoF
                        # gripper_val = 0.08
                        # import pdb; pdb.set_trace()
                        traj_gen = traj_gen_cache[action_id]
                        
                        # generate trajectory from diffusion policy
                        traj_gen.eval()
                        # torch_robot_intial_pose = torch.tensor(robot_current_pose, dtype=torch.float32).to(device)
                        # # torch_robot_end_pose = torch.tensor(pose4x4_to_xyzrpy(robot_end_pose,0.0), dtype=torch.float32).to(device)
                        # torch_robot_end_pose = torch.tensor(xyzrpy_end, dtype=torch.float32).to(device)
                        # import pdb; pdb.set_trace()
                        # torch_robot_end_pose[3] = math.pi # make the grasp vertically facing down
                        # torch_robot_end_pose[4] = 0
                        print(f"  Torch robot end pose: {torch_robot_end_pose}")
                        robot_pour_pose = robot_current_pose.copy()
                        # robot_pour_pose[3] += 1.75
                        
                        with torch.no_grad():
                            traj = traj_gen(class_idx=int(action_id), x0=robot_current_pose[0:6], goal=robot_pour_pose[0:6])
                        # traj[0,:,-1] = torch_robot_end_pose[0:6]   
                    
                        traj = traj.detach().cpu().numpy()[0]
        
                        # # plot the trajectory
                        env.traj_plot(traj[:3,:].T)
                        # Use dense waypoint execution while carrying an object to reduce slip.
                        step_stride = 5#  if gripper_val <= 0.07 else 5
                        
                        for i in range(traj.shape[1]):
                            if i % step_stride == 0 or i == traj.shape[1] - 1:
                                # traj_r[:, i][3] = math.pi
                                # traj_r[:, i][4] = 0
                                traj_7DoF = np.append(traj[:, i], gripper_val)
                                env.step(traj_7DoF)
                                robot_current_pose = traj_7DoF
                    elif action_id == 2: # pour
                        print("pour")
        
                        traj_gen = traj_gen_cache[action_id]
                        
                        # generate trajectory from diffusion policy
                        traj_gen.eval()
                        # torch_robot_intial_pose = torch.tensor(robot_current_pose, dtype=torch.float32).to(device)
                        # # torch_robot_end_pose = torch.tensor(pose4x4_to_xyzrpy(robot_end_pose,0.0), dtype=torch.float32).to(device)
                        # torch_robot_end_pose = torch.tensor(xyzrpy_end, dtype=torch.float32).to(device)
                        # import pdb; pdb.set_trace()
                        # torch_robot_end_pose[3] = math.pi # make the grasp vertically facing down
                        # torch_robot_end_pose[4] = 0
                        print(f"  Torch robot end pose: {torch_robot_end_pose}")
                        robot_pour_pose = robot_current_pose.copy()
                        # robot_pour_pose[3] += 1.75
                        
                        with torch.no_grad():
                            traj = traj_gen(class_idx=int(action_id), x0=robot_current_pose[0:6], goal=robot_pour_pose[0:6])
                            traj_r = traj_gen(class_idx=int(action_id), x0=robot_pour_pose[0:6], goal=robot_current_pose[0:6]) 
                        # traj[0,:,-1] = torch_robot_end_pose[0:6]   
                    
                        traj = traj.detach().cpu().numpy()[0]
                        traj_r = traj_r.detach().cpu().numpy()[0]
        
                        # # plot the trajectory
                        env.traj_plot(traj[:3,:].T)
                        # Use dense waypoint execution while carrying an object to reduce slip.
                        step_stride = 5#  if gripper_val <= 0.07 else 5
                        
                        for i in range(traj.shape[1]):
                            if i % step_stride == 0 or i == traj.shape[1] - 1:
                                # traj_r[:, i][3] = math.pi
                                # traj_r[:, i][4] = 0
                                traj_7DoF = np.append(traj[:, i], gripper_val)
                                env.step(traj_7DoF)
                                robot_current_pose = traj_7DoF
                        # import pdb; pdb.set_trace()
                    elif action_id == 4: # drop
                        # Release in place first so the object can separate under gravity
                        # before the arm retreats upward.
                        keep_pose = np.asarray(robot_current_pose[:6], dtype=np.float32)
                        env.robot.move_ee(keep_pose, control_method="end")
                        released = False
                        for _ in range(240):
                            env.robot.move_gripper(0.08)
                            env.step_simulation()
                            if not env._has_any_finger_contact():
                                released = True
                                # Give the object a short extra settling window after contact breaks.
                                for _ in range(24):
                                    env.robot.move_gripper(0.08)
                                    env.step_simulation()
                                break
        
                        retreat_pose = robot_current_pose.copy()
                        retreat_pose[2] += 0.06
                        retreat_pose[6] = 0.08
                        env.robot.move_ee(np.asarray(retreat_pose[:6], dtype=np.float32), control_method="end")
                        for _ in range(80 if released else 120):
                            env.robot.move_gripper(0.08)
                            env.step_simulation()
                        robot_current_pose = retreat_pose
                        gripper_val = 0.08
        
                        # env.robot.move_gripper(0.08)
                        # for _ in range(480):  # ~4× more time
                        #     env.step_simulation()
        
                        # gripper_val= 0.08
                        # traj_7DoF = np.append(traj[:, i], gripper_val)
                        # pt = JointTrajectoryPoint()
                        # pt.positions = traj_7DoF.tolist()
                        # pt.time_from_start = Duration(sec=int(i * dt), nanosec=int((i * dt % 1) * 1e9))
                        # msg.points.append(pt)
                    elif action_id == 3: # grasp
                        print("grasp")
                        # # grasp: move to target and close gripper
        
        
                        # robot_current_pose[6] = 0.00
                        # env.step(robot_current_pose, control_method="end")
                        # gripper_val = 0.00
                        # # Briefly keep clamping after grasp to stabilize contact before transport.
                        # for _ in range(180):
                        #     env.robot.move_gripper(0.00)
                        #     env.step_simulation()
        
                        robot_current_pose[6] = 0.02
                        env.step(robot_current_pose, control_method="end")
        
                        # gripper_val = float(env.robot.get_gripper_width())
                        # import pdb; pdb.set_trace()
                        gripper_val = 0.00
                        gripper_val = max(0.008, float(env.robot.get_gripper_width()) - 0.0045)
                        # if obj_name == "cup" or obj_name == "sponge":
                        #     gripper_val = 0.00
                        # else:
                        #     gripper_val = max(0.008, float(env.robot.get_gripper_width()) - 0.0045)
                        print(f" float(env.robot.get_gripper_width()): {float(env.robot.get_gripper_width())}")
                        robot_current_pose[6] = gripper_val
        
                        for _ in range(180):
                            env.robot.move_gripper(gripper_val)
                            env.step_simulation()
        
                        # env.robot.move_gripper(0.00)
                        # import pdb; pdb.set_trace()
                        # for _ in range(480):  # ~4× more time
                        #     env.step_simulation()
                        # gripper_val= 0.00
                        # traj_7DoF = np.append(traj[:, i], gripper_val)
                    #     pt = JointTrajectoryPoint()
                    #     pt.positions = traj_7DoF.tolist()
                    #     pt.time_from_start = Duration(sec=int(i * dt), nanosec=int((i * dt % 1) * 1e9))
                    #     msg.points.append(pt)
            # camera_node.traj_pub.publish(msg)
            #===============================================================

            success = env.check_success()
            results.append(success)
            if args.num_runs > 1 and success:
                print(f"\033[32mTrial {trial_idx + 1} completed successfully.\033[0m")
        except Exception as e:
            if args.num_runs > 1:
                print(f"\033[31mTrial {trial_idx + 1} failed: {e}\033[0m")
            else:
                print(f"\033[31mFailed: {e}\033[0m")
            import traceback
            traceback.print_exc()
            results.append(False)

    if args.num_runs > 1:
        n_ok = sum(results)
        print(f"\n\033[1m=== Success Rate: {n_ok}/{args.num_runs} ({100.0 * n_ok / args.num_runs:.1f}%) ===\033[0m")

    shutdown_ros(executor)


if __name__ == "__main__":
    main()
    import pdb; pdb.set_trace()

# python test_mani_env_w_llm.py --vis --show-camera --subscribe-camera --subscriber-log-interval 5
# --prompt "prepare breakfast with all fruits" YES
# --prompt "clear the table: put all objects into bin" YES
# --prompt "serve multiple cups of drink" YES
# --prompt "clear the table: put all objects into bin, and clean the table" 
# --prompt "serve three plates of breakfast"
# --prompt "clean the table"


# python test_mani_env_w_llm.py --vis --show-camera --subscribe-camera --subscriber-log-interval 5 --prompt "prepare breakfast with all fruits in plate" --checkpoint_path /home/mhumais/Huang/anygrasp_sdk/grasp_detection/log/checkpoint_detection.tar --top_down_grasp --grasp-center-offset-z -0.105 --num-runs 1  --scene-configs scene_4.json