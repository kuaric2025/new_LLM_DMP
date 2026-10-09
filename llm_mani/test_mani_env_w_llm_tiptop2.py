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
    DMP,
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

        self.color_pub = self.create_publisher(Image, "/camera/color/image_raw", 10)
        self.depth_pub = self.create_publisher(Image, "/camera/depth/image_rect_raw", 10)
        self.info_pub = self.create_publisher(CameraInfo, "/camera/color/camera_info", 10)

        self.fx, self.fy, self.cx, self.cy = compute_intrinsics(width, height, fov)
        self.timer = self.create_timer(1.0 / publish_rate_hz, self._publish)

        # # TF broadcaster (static - publish once)
        # if publish_tf:
        #     self._tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        #     self._publish_tf()

    def _publish(self):
        rgb, depth, _ = self.env.capture_rgbd(
            width=self.width,
            height=self.height,
            fov=self.fov,
            near=self.near,
            far=self.far,
            return_seg=False,
        )
        self.latest_rgb = rgb.copy()
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