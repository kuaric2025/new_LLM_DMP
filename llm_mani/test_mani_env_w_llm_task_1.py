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
        time.sleep(0.05)

    print("Done. Close the GUI window to exit.")
    while True:
        env.step_simulation()
        if show_camera and camera_node is not None:
            maybe_show_camera_frame(camera_node)
        if triple_subscriber is not None:
            triple_subscriber.maybe_log_status()
        time.sleep(0.01)


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
    parser.add_argument('--top_down_grasp', action='store_true', help='Output top-down grasps.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    return parser.parse_args()



def main():
    args = parse_args()

    table_path = os.path.join(os.path.dirname(__file__), "../manipulation_env/models/urdf/objects/table/table.urdf")
    block_path = None
    bin_path = os.path.join(os.path.dirname(__file__), "../manipulation_env/models/urdf/objects/table/bin.urdf")
    banana_path = os.path.join(os.path.dirname(__file__), "../manipulation_env/models/ycb/011_banana/model.urdf")
    plate_path = os.path.join(os.path.dirname(__file__), "../manipulation_env/models/ycb/029_plate/model.urdf")
    apple_path = os.path.join(os.path.dirname(__file__), "../manipulation_env/models/ycb/013_apple/model.urdf")
    orange_path = os.path.join(os.path.dirname(__file__), "../manipulation_env/models/ycb/017_orange/model.urdf")
    # plum_path = os.path.join(os.path.dirname(__file__), "../manipulation_env/models/ycb/018_plum/model.urdf")
    # banana_path = os.path.join(os.path.dirname(__file__), "../manipulation_env/models/ycb/011_banana/model.urdf")
    # plate_path = os.path.join(os.path.dirname(__file__), "../manipulation_env/models/ycb/029_plate/model.urdf")
    # apple_path = os.path.join(os.path.dirname(__file__), "../manipulation_env/models/YCB_Dataset/ycb/apple.urdf")
    # orange_path = os.path.join(os.path.dirname(__file__), "../manipulation_env/models/ycb/017_orange/model.urdf")
    robot = FrankaPanda(model_path=None)
    env = ManiEnv(robot, block_path=block_path, table_path=table_path, bin_path=bin_path, banana_path=banana_path, plate_path=plate_path, apple_path=apple_path, orange_path=orange_path, vis=args.vis)
    config_path = os.path.join(os.path.dirname(__file__), "../manipulation_env/configs/task_1.json")
    # env = ManiEnv(
    #     robot,
    #     table_path=table_path,
    #     vis=True,
    #     object_config_path=config_path,
    # )
    env.reset()

    # import pdb; pdb.set_trace()

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
    
    import time
    time.sleep(0.5)  # Wait for messages
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
    
    # Fallback to file path if no RGB available
    if rgb_image is None:
        image_path = os.environ.get('LLM_IMAGE_PATH', '/home/mhumais/Downloads/55.png')
        print(f"Warning: No RGB image available, using file path: {image_path}")
        prompt = os.environ.get('LLM_PROMPT', args.prompt)
        reply = ask_gpt4o(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
            image_paths=[Path(image_path)],
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

    segmap_list = torch.load("task_12.pt")

    # Add action_id to each step from config mapping
    reply = apply_action_ids(reply, ACTION_ID_MAPPING)

    ## generate DMP data
    # Before the "for step in reply:" loop (around line 422):
    traj_gen_cache = {}  # or load once: traj_gen = DMP(task_id=0)
    for i in range(0,3):
        traj_gen_cache[i] = DMP(task_id=i)
    

    # Print formatted JSON response
    print("\n=== GPT-4o Response (JSON) ===")
    print(json.dumps(reply, indent=2))
    num_steps = len(reply)

    gripper_val = 0.15
    
    robot_current_pose = get_robot_pose_from_tcp(env.robot)
    
    print("\n=== Parsed Action Plan ===")
    for step in reply:
        env.clean_traj_plot()
        print(f"\033[32mStep {step.get('step_id', 'N/A')}: {step.get('action', 'N/A')} (action_id: {step.get('action_id', 'N/A')})\033[0m")
        if 'target_object' in step and 'action_id' in step:
            obj = step['target_object'].get('name')
            if obj == 'bowl':
                obj = 'plate'
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
            if action_id in (0, 1, 2):
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
                    xs, ys = torch.where(segmap)

                    center_y = ys.float().mean()   # row coordinate (v)
                    center_x = xs.float().mean()   # col coordinate (u)

                    # If you want integer pixel center:
                    center_y_int = int(center_y.round().item())
                    center_x_int = int(center_x.round().item())
                    print(f"  Center (y, x): ({center_y_int}, {center_x_int})") # 329, 294
                    # import pdb; pdb.set_trace()
                    center_z_m = depth_image[center_x_int, center_y_int] # 1.138107180595398
                    print(f"  Center Z (m): {center_z_m}")
                    
                    #==============get pose in robot frame======================================
                    T_c2r = np.array([[ 1.0, 0.0, 0.0, 0.6000 ],
                                    [ 0.0, -1.0, 0.0, 0.0 ],
                                    [ 0.0, 0.0, -1.0, 1.200 ],
                                    [ 0.0, 0.0, 0.0, 1.0 ]])

                    # T_c2r = np.array([[ 1.0, 0.0, 0.0, 0.6000 ],
                    #                 [ 0.0, 1.0, 0.0, 0.0 ],
                    #                 [ 0.0, 0.0, 1.0, 1.200 ],
                    #                 [ 0.0, 0.0, 0.0, 1.0 ]])

                    # T_c2r = np.array([[ 1.0, 0.0, 0.0, 0.5 ],
                    #                 [ 0.0, -1.0, 0.0, 0.0 ],
                    #                 [ 0.0, 0.0, -1.0, 1.050 ],
                    #                 [ 0.0, 0.0, 0.0, 1.0 ]])

                    cam_info = np.array([457.0073621574767,  0, 320.0,
                                0, 342.75552161810754, 240.0,
                                                    0,  0,  1])

                    # center_x_int = 329
                    # center_y_int = 294
                    # center_z_m = 1.14
                    
                    pos_camera = pixel_depth_to_camera_frame(
                                center_x_int, center_y_int, center_z_m, cam_info
                            )

                    pos_camera_ = np.eye(4)
                    pos_camera_[:3, 3] = pos_camera.reshape(3)
                    # import pdb; pdb.set_trace()

                    # # anygrasp pose in camera frame
                    if step_id < num_steps and reply[step_id]['action'] == "grasp":
                    #     print("AnyGrasp Predicting=========")
                        # grasp = anygrasp_demo(rgb_image, depth_image, np.array(segmap.permute(1,0).cpu()), step_id, args) # real-time segmentation
                        grasp = anygrasp_demo(rgb_image, depth_image, np.array(segmap.cpu()), step_id, args) # pre-computed segmentation
                        pos_camera_[:3, :3] = grasp.rotation_matrix
                        pos_camera_[:3, 3] = grasp.translation
                        # import pdb; pdb.set_trace()

                    pos_robot = T_c2r @ pos_camera_
                    print(f"  Pos robot: {pos_robot}")

                    # GRASP_CENTER_OFFSET = np.array([-0.01, -0.01, -0.105])  # TCP → grasp center in grasp frame # [0.01495, 0, -0.03537]
                    GRASP_CENTER_OFFSET = np.array([0.0, 0.0, -0.09]) # [-0.01, 0.0, -0.09] # TCP → grasp center in grasp frame # [0.01495, 0, -0.03537]
                    # [-0.01, -0.05, 0.03537] apple
                    FINGER_LENGTH =  0 # 0.1034  # m

                    # 根据你的手爪，在这里选对轴：
                    # GRIPPER_AXIS = np.array([1.0, 0.0, 0.0])  # 如果手指沿 +X
                    GRIPPER_AXIS = np.array([0.0, 0.0, 1.0])  # 如果手指沿 +Z

                    TCP_OFFSET_WITH_FINGER = GRASP_CENTER_OFFSET + FINGER_LENGTH * GRIPPER_AXIS
                    print(TCP_OFFSET_WITH_FINGER)

                    T_offset_inv = np.eye(4)
                    T_offset_inv[:3, 3] = -TCP_OFFSET_WITH_FINGER
                    # pos_robot_tcp = pos_robot @ T_offset_inv  # TCP pose in robot frame
                    pos_robot_tcp = T_offset_inv @ pos_robot 
                    
                    
                    
                    #==================call DMP======================================
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
                    # Prefer Euler representation that is close to current pose
                    # xyzrpy_end[3:6] = euler_closest_to_ref(
                    #     xyzrpy_end[3:6],
                    #     robot_initial_pose[3:6]
                    # )
                    if step_id == 1:
                        xyzrpy_end[2] -= 0.018
                    if reply[step_id]['action'] == "drop":
                        xyzrpy_end[2] += 0.11
                    if obj == "orange":
                        xyzrpy_end[0] += 0.02
                        xyzrpy_end[1] += 0.02
                    if obj == "orange" and reply[step_id]['action'] == "drop":
                        xyzrpy_end[0] -= 0.05
                    if obj == "apple":
                        xyzrpy_end[0] -= 0.02
                        xyzrpy_end[1] += 0.01
                
                    # begin_yaw = robot_initial_pose[5]   # current yaw
                    # target_yaw = xyzrpy_end[5]   # target yaw
                    # xyzrpy_end[5] = shortest_yaw(begin_yaw, target_yaw)
                    # # Use xyzrpy_end for DMP goal
                    # torch_robot_end_pose = torch.tensor(xyzrpy_end, dtype=torch.float32).to(device)

                    # traj_gen = DMP(task_id=action_id)
                    traj_gen = traj_gen_cache[action_id]
                
                    # generate trajectory from cVAE-dmp    
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

                    traj[0,:,-1] = torch_robot_end_pose[0:6]   
                
                    traj = traj.detach().cpu().numpy()[0]
                    
                    # # plot the trajectory
                    env.traj_plot(traj[:3,:].T)
                    # Use dense waypoint execution while carrying an object to reduce slip.
                    step_stride = 1 if gripper_val <= 0.04 else 5
                    for i in range(traj.shape[1]):
                        if i % step_stride == 0 or i == traj.shape[1] - 1:
                            traj[:, i][3] = math.pi
                            traj[:, i][4] = 0
                            traj_7DoF = np.append(traj[:, i], gripper_val)
                            env.step(traj_7DoF)
                            robot_current_pose = traj_7DoF
                    # import pdb; pdb.set_trace()

            elif action_id == 4: # drop
                # drop: fast release (bypass gradual open) + immediate retreat
                keep_pose = np.asarray(robot_current_pose[:6], dtype=np.float32)
                env.robot.move_ee(keep_pose, control_method="end")
                for _ in range(6):
                    env.robot.move_gripper(0.08)
                    env.step_simulation()

                retreat_pose = robot_current_pose.copy()
                retreat_pose[2] += 0.06
                retreat_pose[6] = 0.08
                env.robot.move_ee(np.asarray(retreat_pose[:6], dtype=np.float32), control_method="end")
                for _ in range(80):
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
                # traj_7DoF = np.append(traj[:, i], gripper_val)
                # robot_current_pose = get_robot_pose_from_tcp(env.robot)
                robot_current_pose[6] = 0.00
                env.step(robot_current_pose, control_method="end")
                gripper_val = 0.00
                # Briefly keep clamping after grasp to stabilize contact before transport.
                for _ in range(180):
                    env.robot.move_gripper(0.00)
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


if __name__ == "__main__":
    main()

# python test_mani_env_w_llm.py --vis --show-camera --subscribe-camera --subscriber-log-interval 5
# --prompt "prepare breakfast with all fruits" YES
# --prompt "clear the table: put all objects into bin" YES
# --prompt "serve multiple cups of drink" YES
# --prompt "clear the table: put all objects into bin, and clean the table" 
# --prompt "serve three plates of breakfast"
# --prompt "clean the table"
