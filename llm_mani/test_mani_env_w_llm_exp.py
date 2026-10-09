"""Quick smoke test for ManiEnv with FrankaPanda.

Runs a few end-effector moves and gripper open/close in GUI.

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

# ROS 2 imports (available via rebuilt Jazzy bindings for Python 3.10)
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from builtin_interfaces.msg import Time as TimeMsg
from sensor_msgs.msg import CameraInfo, Image
from message_filters import ApproximateTimeSynchronizer, Subscriber as MFSubscriber
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage
from std_msgs.msg import Bool, Float64MultiArray
from std_msgs.msg import Float64MultiArray
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
    anygrasp_demo_exp,
    grasp_to_xyzrpy,
    euler_closest_to_ref,
    shortest_yaw,
    matrix_to_transform_stamped,
    matrix_to_pose,
    pose_to_matrix,
    shortest_rotation_path_xyzrpy,
    choose_equivalent_grasp_rotation,
    make_pour_pose_from_current,
    check_rpy
)

from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from tf2_ros import Buffer, TransformListener
from tf2_ros import ConnectivityException, ExtrapolationException, LookupException
from tf2_geometry_msgs import do_transform_pose

from moveit_msgs.srv import GetCartesianPath, GetMotionPlan

LLM_CONFIG, ACTION_ID_MAPPING, SYSTEM_PROMPT = load_llm_config(config_path="/home/mhumais/Huang/DMP/llm_mani/config_exp.json")

DEFAULT_TCP_TO_BASE = np.array([
    [0.594824, 0.000000, 0.803856, 0.606883],
    [-0.000000, -1.000000, 0.000000, 0.000000],
    [0.803856, -0.000000, -0.594824, 0.728523],
    [0.000000, 0.000000, 0.000000, 1.000000],
], dtype=np.float64)


def transform_stamped_to_matrix(transform_stamped: TransformStamped) -> np.ndarray:
    t = transform_stamped.transform.translation
    q = transform_stamped.transform.rotation
    rotation = np.array(
        p.getMatrixFromQuaternion([q.x, q.y, q.z, q.w]),
        dtype=np.float64,
    ).reshape(3, 3)

    transform_matrix = np.eye(4, dtype=np.float64)
    transform_matrix[:3, :3] = rotation
    transform_matrix[:3, 3] = [t.x, t.y, t.z]
    return transform_matrix

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

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

from VAE_DMP_mani.data.manipulation_data.read_npy import read_first_last

class CameraTripleSubscriber(Node):
    def __init__(
        self,
        color_topic: str = "/camera/camera/color/image_rect_raw",
        depth_topic: str = "/camera/camera/depth/image_rect_raw",
        info_topic: str = "/camera/camera/color/camera_info",
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
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
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

    def lookup_transform_matrix(
        self,
        target_frame: str,
        source_frame: str,
        timeout_sec: float = 2.0,
    ) -> np.ndarray:
        transform = self.tf_buffer.lookup_transform(
            target_frame,
            source_frame,
            rclpy.time.Time(),
            timeout=rclpy.duration.Duration(seconds=float(timeout_sec)),
        )
        return transform_stamped_to_matrix(transform)

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

class DataPublisher(Node):
    def __init__(
        self,
        publish_rate_hz: float = 10.0,

    ) -> None:
        super().__init__("simulated_d435")
        self.last_traj = None
        self.last_move_flag = None
        self.last_gripper_flag = None
        self.traj_pub = self.create_publisher(
            Float64MultiArray, "/demo/trajectory", 10
        )
        # self.pub = self.create_publisher(Float64MultiArray, "/demo/trajectory", 10)
        self.move_flag_pub = self.create_publisher(
            Bool, "/move_control_flag", 10
        )
        self.gripper_flag_pub = self.create_publisher(
            Bool, "/gripper_control_flag", 10
        )
        self.gripper_cmd_pub = self.create_publisher(
            Float64MultiArray, "/gripper_command", 10
        )


        # self.timer = self.create_timer(1.0 / publish_rate_hz, self._publish)

    def publish_gripper_move(self, width: float, speed: float):
        msg = Float64MultiArray()
        msg.data = [float(width), float(speed)]
        self.gripper_cmd_pub.publish(msg)
        self.get_logger().info(
            f"Published gripper move command: width={width:.3f}, speed={speed:.3f}"
        )

    def publish_gripper_grasp(
        self,
        width: float,
        speed: float,
        force: float,
        eps_inner: float,
        eps_outer: float,
    ):
        msg = Float64MultiArray()
        msg.data = [
            float(width),
            float(speed),
            float(force),
            float(eps_inner),
            float(eps_outer),
        ]
        self.gripper_cmd_pub.publish(msg)
        self.get_logger().info(
            "Published gripper grasp command: "
            f"width={width:.3f}, speed={speed:.3f}, force={force:.1f}"
        )



    # def _publish(self):
    #     rgb, depth, _ = self.env.capture_rgbd(
    #         width=self.width,
    #         height=self.height,
    #         fov=self.fov,
    #         near=self.near,
    #         far=self.far,
    #         return_seg=False,
    #     )
    #     self.latest_rgb = rgb.copy()
    #     now = self.get_clock().now().to_msg()
    #     color_msg = self._make_color_msg(rgb, now)
    #     depth_msg = self._make_depth_msg(depth, now)
    #     info_msg = self._make_camera_info(now)

    #     self.color_pub.publish(color_msg)
    #     self.depth_pub.publish(depth_msg)
    #     self.info_pub.publish(info_msg)

    # def _make_color_msg(self, rgb: np.ndarray, stamp: TimeMsg) -> Image:
    #     msg = Image()
    #     msg.header.stamp = stamp
    #     msg.header.frame_id = self.frame_id
    #     msg.height = self.height
    #     msg.width = self.width
    #     msg.encoding = "rgb8"
    #     msg.step = self.width * 3
    #     msg.data = rgb.astype(np.uint8).tobytes()
    #     return msg

    # def _make_depth_msg(self, depth: np.ndarray, stamp: TimeMsg) -> Image:
    #     msg = Image()
    #     msg.header.stamp = stamp
    #     msg.header.frame_id = self.frame_id.replace("color", "depth")
    #     msg.height = self.height
    #     msg.width = self.width
    #     msg.encoding = "32FC1"
    #     msg.step = self.width * 4
    #     msg.data = depth.astype(np.float32).tobytes()
    #     return msg

    # def _make_camera_info(self, stamp: TimeMsg) -> CameraInfo:
    #     msg = CameraInfo()
    #     msg.header.stamp = stamp
    #     msg.header.frame_id = self.frame_id
    #     msg.height = self.height
    #     msg.width = self.width
    #     msg.k = [self.fx, 0.0, self.cx, 0.0, self.fy, self.cy, 0.0, 0.0, 1.0]
    #     msg.p = [self.fx, 0.0, self.cx, 0.0, 0.0, self.fy, self.cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    #     msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
    #     msg.distortion_model = "plumb_bob"
    #     return msg

# class TrajectoryTestPublisher(Node):
#     def __init__(self):
#         super().__init__("franka_trajectory_test_publisher")
#         self.pub = self.create_publisher(Float64MultiArray, "/demo/trajectory", 10)

#     def publish_traj(self, traj_id: int, steps=20):
#         msg = self.build_trajectory(traj_id=traj_id, steps=steps)

#         self.pub.publish(msg)
#             time.sleep(0.15)
#         self.get_logger().info(
#             f"Published traj{traj_id}: shape=({steps},7) on /demo/trajectory"
#         )

#     def publish_sequence(self):
#         for traj_id in [1, 2, 3]:
#             self.publish_traj(traj_id=traj_id, steps=20)
#             time.sleep(0.8)
#         self.get_logger().info("Published sequence: traj1 -> traj2 -> traj3")
        
def start_ros_nodes(nodes: list[Node]) -> tuple[rclpy.executors.Executor, threading.Thread]:
    executor = rclpy.executors.MultiThreadedExecutor()
    for node in nodes:
        executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    return executor, thread


def shutdown_ros(executor: Optional[rclpy.executors.Executor]):
    if executor is not None:
        executor.shutdown()
    if rclpy.ok():
        rclpy.shutdown()

from std_msgs.msg import Float64MultiArray, MultiArrayDimension
class TrajectoryTestPublisher(Node):
    def __init__(self):
        super().__init__("franka_trajectory_test_publisher")
        self.pub = self.create_publisher(Float64MultiArray, "/demo/trajectory", 10)

    def build_trajectory(self, data: list[float], steps=20):
        # shape: (steps, 7)
        # [x, y, z, roll, pitch, yaw, gripper]
        msg = Float64MultiArray()
        msg.layout.dim = [
            MultiArrayDimension(label="steps", size=steps, stride=steps * 7),
            MultiArrayDimension(label="n_dim", size=7, stride=7),
        ]
        msg.layout.data_offset = 0
        msg.data = data
        return msg

    def publish_traj(self, data: list[float], steps=20):
        msg = self.build_trajectory(data, steps=steps)
        self.pub.publish(msg)
        self.get_logger().info(
            f"Published traj: shape=({steps},7) on /demo/trajectory"
        )

    def publish_sequence(self):
        for traj_id in [1, 2, 3]:
            self.publish_traj(traj_id=traj_id, steps=100)
            time.sleep(0.8)
        self.get_logger().info("Published sequence: traj1 -> traj2 -> traj3")

def parse_args():
    parser = argparse.ArgumentParser(description="ManiEnv test with simulated D435 publisher")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fov", type=float, default=70.0)
    parser.add_argument("--camera-rate", type=float, default=10.0, help="Publish rate in Hz")
    parser.add_argument("--show-camera", action="store_true", help="Display realtime camera view via OpenCV")
    parser.add_argument("--subscribe-camera", action="store_true", help="Subscribe to RGB+depth+CameraInfo topics for monitoring")
    parser.add_argument("--subscriber-log-interval", type=float, default=5.0, help="Seconds between subscriber status logs")
    parser.add_argument("--prompt", type=str, default="prepare breakfast")

    parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
    parser.add_argument('--max_gripper_width', type=float, default=0.08, help='Maximum gripper width (<=0.1m)')
    parser.add_argument('--gripper_height', type=float, default=0.045, help='Gripper height')
    parser.add_argument(
        '--grasp-center-offset-z',
        type=float,
        default=0.105,
        help='Hand-frame Z offset from panda_hand to the intended grasp center in meters.',
    )
    parser.add_argument(
        '--base-frame',
        type=str,
        default='fr3_link0',
        help='TF target frame used as the robot base when looking up TCP pose.',
    )
    parser.add_argument(
        '--tcp-frame',
        type=str,
        default='fr3_link8',
        help='TF source frame for the robot TCP/end-effector.',
    )
    parser.add_argument(
        '--tf-timeout',
        type=float,
        default=2.0,
        help='Seconds to wait for the TF lookup from TCP to base.',
    )
    parser.add_argument('--top_down_grasp', action='store_true', help='Output top-down grasps.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    return parser.parse_args()


def main():
    args = parse_args()

    # robot = FrankaPanda(model_path=None)
    # env = ManiEnv(robot, block_path=block_path, table_path=table_path, bin_path=bin_path, banana_path=banana_path, plate_path=plate_path, apple_path=apple_path, orange_path=orange_path, pitcher_path=pitcher_path, e_cups_path=e_cups_path, i_cups_path=i_cups_path, j_cups_path=j_cups_path, sponge_path=sponge_path, mug_path=mug_path, bowl_path=bowl_path, vis=args.vis)
    # config_path = os.path.join(os.path.dirname(__file__), "../manipulation_env/configs/task_1.json")

    rclpy.init()
    nodes: list[Node] = []

    triple_sub = None
    if args.subscribe_camera:
        triple_sub = CameraTripleSubscriber(
            log_interval=args.subscriber_log_interval,
        )
        nodes.append(triple_sub)
 
    flag_node = DataPublisher(
        publish_rate_hz=args.camera_rate,
    )
    nodes.append(flag_node)

    traj_pub = TrajectoryTestPublisher()
    nodes.append(traj_pub)

    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(nodes[0])
    executor.add_node(nodes[1])
    executor.add_node(nodes[2])
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()


    
    
    executor, exec_thread = start_ros_nodes(nodes)

    # home_pose = np.array([-0.398, 0.340, 0.518, -3.138, -0.091, -0.713, 0.08])
    home_pose = np.array([0.535, -0.002, 0.654, -3.098, -0.086, -0.794, 0.08])

    # import pdb; pdb.set_trace()
    try:
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
            rgb, depth, _ = triple_sub.get_latest_images() # (480, 848)
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

        else:
            print("Warning: No RGB image available, waiting...")

        print(f"Using RGB image from camera: shape={rgb_image.shape}, dtype={rgb_image.dtype}")

        prompt = os.environ.get('LLM_PROMPT', args.prompt)
        reply = ask_gpt4o(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
            image_arrays=[rgb_image],
        )
    
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
        drop_id = 0

        rgb_image_s1 = None
        depth_image_s1 = None



        object_memory = {}

        for step in reply:
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
                    obj = 'mug' #handle'
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

                if step_id == 1:
                    rgb_image_s1 = rgb_image.copy()
                    depth_image_s1 = depth_image.copy()


                if obj not in object_memory and action_id !=3 and action_id !=4 and action_id !=1:
                    print(f'extracting {obj} mask and pose')
                    object_memory[obj] = {}
                    segmap = sam3_inference(rgb_image_s1, obj)
                    if segmap is not None:
                        # aa = True
                        # if aa:
                        
                        # segmap = segmap.permute(2, 1, 0).squeeze(2) # real-time segmentation
                        segmap = segmap.squeeze(0)
                        print(f"  Segmap: {segmap.shape}")
                        ys, xs = torch.where(segmap)

                        center_y = ys.float().median().item()   # more robust to asymmetric masks
                        center_x = xs.float().median().item()
                        # import pdb; pdb.set_trace()
                        center_y_int = int(center_y)
                        center_x_int = int(center_x)

                        
                        print(f"  Center (y, x): ({center_y_int}, {center_x_int})") # 329, 294
                        
                        depth_image_s1[depth_image_s1 > 3.0] = 0
                        center_z_m = depth_image_s1[center_y_int, center_x_int] # 1.138107180595398
                        print(f"  Center Z (m): {center_z_m}")
                        
                        #==============get pose in robot frame======================================
                        # T_c2r = np.array([[ 1.0, 0.0, 0.0, 0.6000 ],
                        #                 [ 0.0, -1.0, 0.0, 0.0 ],
                        #                 [ 0.0, 0.0, -1.0, 1.200 ],
                        #                 [ 0.0, 0.0, 0.0, 1.0 ]])
                        # T_cam2tcp = np.array([[ -0.997 , 0.071, -0.015,  0.004],
                        #                     [-0.066, -0.815, 0.576, -0.102],
                        #                     [0.029, 0.576, 0.817, 0.031],
                        #                     [0.000, 0.000, 0.000, 1.000]])

                        # T_cam2tcp = np.array([
                        #                     [-0.9993, -0.0342,  0.0135,  0.004747],
                        #                     [ 0.0332, -0.8177,  0.5747, -0.106023],
                        #                     [-0.0164,  0.5746,  0.8183,  0.024061],
                        #                     [ 0.0   ,  0.0   ,  0.0   ,  1.0     ]
                        #                     ])

                        T_cam2tcp = np.array([
                                            [-0.9992, -0.0184,  0.0350,  0.008101],
                                            [ 0.0166, -0.8178,  0.5753, -0.106187],
                                            [-0.0360,  0.5752,  0.8172,  0.025851],
                                            [ 0.0   ,  0.0   ,  0.0   ,  1.0     ]
                                        ])

                        T_tcp2base = DEFAULT_TCP_TO_BASE.copy()
                        if triple_sub is not None:
                            try:
                                T_tcp2base = triple_sub.lookup_transform_matrix(
                                    target_frame=args.base_frame,
                                    source_frame=args.tcp_frame,
                                    timeout_sec=args.tf_timeout,
                                )
                                print(
                                    f"  TF matrix {args.base_frame} <- {args.tcp_frame}:\n"
                                    f"{np.array2string(T_tcp2base, precision=6, suppress_small=True)}"
                                )
                            except (LookupException, ConnectivityException, ExtrapolationException) as exc:
                                print(
                                    f"  Warning: TF lookup {args.base_frame} <- {args.tcp_frame} failed: {exc}. "
                                    "Using fallback T_tcp2base."
                                )
                        # T_cam2tcp = np.array([[-0.99626,   -0.06635,    0.05404,   -0.00333202],
                        #                     [ 0.07081 ,  -0.81294 ,   0.57797  , -0.101635  ],
                        #                     [ 0.01804 ,   0.57923 ,   0.81496 ,   0.0336044 ],
                        #                     [ 0         ,   0         ,   0         ,   1         ]])
                        T_cam2base = T_tcp2base @ T_cam2tcp

                        # T_cam2tcp = np.linalg.inv(T_tcp2cam)

                        T_o2c = np.array([
                                [1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0,-1, 0],
                                [0, 0, 0, 1],
                            ])

                        cam_info = np.array([430.3013610839844,  0, 417.06976318359375,
                                    0, 429.9326171875, 245.400390625,
                                                        0,  0,  1])

                        # Backproject all segmented pixels to 3D
                        pos_camera = pixel_depth_to_camera_frame(depth_image_s1, segmap.cpu().numpy(), cam_info)

                        # pos_camera = pixel_depth_to_camera_frame(center_y_int, center_x_int, center_z_m, cam_info_2)
                        pos_camera_ = np.eye(4)
                        pos_camera_[:3, 3] = pos_camera.reshape(3)
                        # import pdb; pdb.set_trace()

                        # # anygrasp pose in camera frame
                        if step_id < num_steps and reply[step_id]['action'] == "grasp":
                        #     print("AnyGrasp Predicting=========")
                            # grasp = anygrasp_demo(rgb_image, depth_image, np.array(segmap.permute(1,0).cpu()), step_id, args) # real-time segmentation
                            grasp = anygrasp_demo_exp(rgb_image_s1, depth_image_s1, np.array(segmap.cpu()), step_id, args) # pre-computed segmentation
                            rotation_axes_summary("  AnyGrasp camera-frame axes", grasp.rotation_matrix)
                            # Map AnyGrasp's gripper frame to the Panda hand frame.
                            # This remaps the nearly vertical AnyGrasp x-axis onto the
                            # Panda hand's local z-axis, which is the axis IK uses.
                            R_grasp_to_panda = np.array([
                                [0.0, 0.0, 1.0],
                                [0.0, 1.0, 0.0],
                                [-1.0, 0.0, 0.0],
                            ], dtype=np.float64)

                            # T_optical_to_cameralink = np.array([
                            #     [0.,  0.,  1.],
                            #     [-1., 0.,  0.],
                            #     [0., -1.,  0.],
                            # ])
                            # T_cameralink_to_optical = np.array([
                            #     [0.,  0.,  1.],
                            #     [-1., 0.,  0.],
                            #     [0., -1.,  0.],
                            # ])

                            R = np.array([
                                [-1, 0, 0],
                                [0, 1, 0],
                                [0, 0, 1]
                            ])
                            grasp_rotation_panda = np.asarray(grasp.rotation_matrix, dtype=np.float64) @ R_grasp_to_panda
                            pos_camera_[:3, :3] = grasp_rotation_panda
                            pos_camera_[:3, 3] = grasp.translation
                            rotation_axes_summary("  Panda-aligned camera-frame axes", grasp_rotation_panda)
                        # import pdb; pdb.set_trace()
                        T_cam2tcp_tf = matrix_to_transform_stamped(T_cam2tcp, "fr3_link8", "camera_color_optical_frame")
                        pose_tcp = do_transform_pose(matrix_to_pose(pos_camera_), T_cam2tcp_tf)
                        T_tcp2base_tf = matrix_to_transform_stamped(T_tcp2base, "fr3_link0", "fr3_link8")
                        pose_base = do_transform_pose(pose_tcp, T_tcp2base_tf)
                        # pos_robot = T_cam2base @ pos_camera_
                        pos_robot = pose_to_matrix(pose_base)
                        print(f"  Pos robot: {pos_robot}")
                        rotation_axes_summary("  Robot-frame grasp axes", pos_robot[:3, :3])


                        GRASP_CENTER_OFFSET = np.array(
                            [0,0, -args.grasp_center_offset_z], #World-frame offset delta: [ 0.014  -0.03    0.0837]
                            dtype=np.float64,
                        )
                        FINGER_LENGTH =  0 # 0.1034  # m

                        # 根据你的手爪，在这里选对轴：
                        # GRIPPER_AXIS = np.array([1.0, 0.0, 0.0])  # 如果手指沿 +X
                        GRIPPER_AXIS = np.array([1.0, 1.0, 1.0])  # 如果手指沿 +Z

                        TCP_OFFSET_WITH_FINGER = GRASP_CENTER_OFFSET + FINGER_LENGTH * GRIPPER_AXIS
                        print(f"  Hand-frame TCP offset: {TCP_OFFSET_WITH_FINGER}")
                        offset_world_delta = pos_robot[:3, :3] @ (-TCP_OFFSET_WITH_FINGER)
                        print(f"  World-frame offset delta: {np.round(offset_world_delta, 4)}")

                        T_offset_inv = np.eye(4)
                        T_offset_inv[:3, 3] = -TCP_OFFSET_WITH_FINGER
                        pos_robot_tcp = pos_robot @ T_offset_inv  # TCP pose in robot frame
                        rotation_axes_summary("  Robot-frame TCP axes", pos_robot_tcp[:3, :3])
                        # import pdb; pdb.set_trace()
                        
                        object_memory[obj]["mask"] = segmap.cpu().numpy()
                        object_memory[obj]["pose_base"] = pos_robot_tcp  # 4x4 matrix
        # import pdb; pdb.set_trace()

        print("\n=== Parsed Action Plan ===")
        for step in reply:
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
                    obj = 'mug'# handle'
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
                    # segmap = sam3_inference(rgb_image_s1, obj)
                    # # segmap = torch.tensor(segmap_list[obj][0]).to(device)
                    # # segmap =segmap_list[obj][0]
                    # if segmap is not None:
                    aa = True
                    if aa:
                    #     # import pdb; pdb.set_trace()
                    #     # segmap = segmap.permute(2, 1, 0).squeeze(2) # real-time segmentation
                    #     segmap = segmap.squeeze(0)
                    #     print(f"  Segmap: {segmap.shape}")
                    #     ys, xs = torch.where(segmap)
    
                    #     center_y = ys.float().median().item()   # more robust to asymmetric masks
                    #     center_x = xs.float().median().item()
                    #     # import pdb; pdb.set_trace()
                    #     center_y_int = int(center_y)
                    #     center_x_int = int(center_x)
    
                        
                    #     print(f"  Center (y, x): ({center_y_int}, {center_x_int})") # 329, 294
                        
                    #     depth_image_s1[depth_image_s1 > 3.0] = 0
                    #     center_z_m = depth_image_s1[center_y_int, center_x_int] # 1.138107180595398
                    #     print(f"  Center Z (m): {center_z_m}")
                        
                    #     #==============get pose in robot frame======================================
                    #     # T_c2r = np.array([[ 1.0, 0.0, 0.0, 0.6000 ],
                    #     #                 [ 0.0, -1.0, 0.0, 0.0 ],
                    #     #                 [ 0.0, 0.0, -1.0, 1.200 ],
                    #     #                 [ 0.0, 0.0, 0.0, 1.0 ]])
                    #     T_cam2tcp = np.array([[ -0.997 , 0.071, -0.015,  0.004],
                    #                       [-0.066, -0.815, 0.576, -0.102],
                    #                       [0.029, 0.576, 0.817, 0.031],
                    #                       [0.000, 0.000, 0.000, 1.000]])

                    #     T_tcp2base = DEFAULT_TCP_TO_BASE.copy()
                    #     if triple_sub is not None:
                    #         try:
                    #             T_tcp2base = triple_sub.lookup_transform_matrix(
                    #                 target_frame=args.base_frame,
                    #                 source_frame=args.tcp_frame,
                    #                 timeout_sec=args.tf_timeout,
                    #             )
                    #             print(
                    #                 f"  TF matrix {args.base_frame} <- {args.tcp_frame}:\n"
                    #                 f"{np.array2string(T_tcp2base, precision=6, suppress_small=True)}"
                    #             )
                    #         except (LookupException, ConnectivityException, ExtrapolationException) as exc:
                    #             print(
                    #                 f"  Warning: TF lookup {args.base_frame} <- {args.tcp_frame} failed: {exc}. "
                    #                 "Using fallback T_tcp2base."
                    #             )
                    #     # T_cam2tcp = np.array([[-0.99626,   -0.06635,    0.05404,   -0.00333202],
                    #     #                     [ 0.07081 ,  -0.81294 ,   0.57797  , -0.101635  ],
                    #     #                     [ 0.01804 ,   0.57923 ,   0.81496 ,   0.0336044 ],
                    #     #                     [ 0         ,   0         ,   0         ,   1         ]])
                    #     T_cam2base = T_tcp2base @ T_cam2tcp

                    #     # T_cam2tcp = np.linalg.inv(T_tcp2cam)
    
                    #     T_o2c = np.array([
                    #             [1, 0, 0, 0],
                    #             [0, 1, 0, 0],
                    #             [0, 0,-1, 0],
                    #             [0, 0, 0, 1],
                    #         ])

                    #     cam_info = np.array([430.3013610839844,  0, 417.06976318359375,
                    #                 0, 429.9326171875, 245.400390625,
                    #                                     0,  0,  1])
    
                    #     # Backproject all segmented pixels to 3D
                    #     pos_camera = pixel_depth_to_camera_frame(depth_image_s1, segmap.cpu().numpy(), cam_info)
    
                    #     # pos_camera = pixel_depth_to_camera_frame(center_y_int, center_x_int, center_z_m, cam_info_2)
                    #     pos_camera_ = np.eye(4)
                    #     pos_camera_[:3, 3] = pos_camera.reshape(3)
                    #     # import pdb; pdb.set_trace()
    
                    #     # # anygrasp pose in camera frame
                    #     if step_id < num_steps and reply[step_id]['action'] == "grasp":
                    #     #     print("AnyGrasp Predicting=========")
                    #         # grasp = anygrasp_demo(rgb_image, depth_image, np.array(segmap.permute(1,0).cpu()), step_id, args) # real-time segmentation
                    #         grasp = anygrasp_demo_exp(rgb_image_s1, depth_image_s1, np.array(segmap.cpu()), step_id, args) # pre-computed segmentation
                    #         rotation_axes_summary("  AnyGrasp camera-frame axes", grasp.rotation_matrix)
                    #         # Map AnyGrasp's gripper frame to the Panda hand frame.
                    #         # This remaps the nearly vertical AnyGrasp x-axis onto the
                    #         # Panda hand's local z-axis, which is the axis IK uses.
                    #         # R_grasp_to_panda = np.array([
                    #         #     [0.0, 0.0, 1.0],
                    #         #     [0.0, 1.0, 0.0],
                    #         #     [-1.0, 0.0, 0.0],
                    #         # ], dtype=np.float64)

                    #         # T_optical_to_cameralink = np.array([
                    #         #     [0.,  0.,  1.],
                    #         #     [-1., 0.,  0.],
                    #         #     [0., -1.,  0.],
                    #         # ])
                    #         T_cameralink_to_optical = np.array([
                    #             [0.,  0.,  1.],
                    #             [-1., 0.,  0.],
                    #             [0., -1.,  0.],
                    #         ])
                    #         grasp_rotation_panda = np.asarray(grasp.rotation_matrix, dtype=np.float64)  @ T_cameralink_to_optical
                    #         pos_camera_[:3, :3] = grasp_rotation_panda
                    #         pos_camera_[:3, 3] = grasp.translation
                    #         rotation_axes_summary("  Panda-aligned camera-frame axes", grasp_rotation_panda)
                    #     # import pdb; pdb.set_trace()
                    #     T_cam2tcp_tf = matrix_to_transform_stamped(T_cam2tcp, "fr3_link8", "camera_color_optical_frame")
                    #     pose_tcp = do_transform_pose(matrix_to_pose(pos_camera_), T_cam2tcp_tf)
                    #     T_tcp2base_tf = matrix_to_transform_stamped(T_tcp2base, "fr3_link0", "fr3_link8")
                    #     pose_base = do_transform_pose(pose_tcp, T_tcp2base_tf)
                    #     # pos_robot = T_cam2base @ pos_camera_
                    #     pos_robot = pose_to_matrix(pose_base)
                    #     print(f"  Pos robot: {pos_robot}")
                    #     rotation_axes_summary("  Robot-frame grasp axes", pos_robot[:3, :3])
    

                    #     GRASP_CENTER_OFFSET = np.array(
                    #         [0.0, -0.0, -args.grasp_center_offset_z], #World-frame offset delta: [ 0.014  -0.03    0.0837]
                    #         dtype=np.float64,
                    #     )
                    #     FINGER_LENGTH =  0 # 0.1034  # m
    
                    #     # 根据你的手爪，在这里选对轴：
                    #     # GRIPPER_AXIS = np.array([1.0, 0.0, 0.0])  # 如果手指沿 +X
                    #     GRIPPER_AXIS = np.array([0.0, 0.0, 1.0])  # 如果手指沿 +Z
    
                    #     TCP_OFFSET_WITH_FINGER = GRASP_CENTER_OFFSET + FINGER_LENGTH * GRIPPER_AXIS
                    #     print(f"  Hand-frame TCP offset: {TCP_OFFSET_WITH_FINGER}")
                    #     offset_world_delta = pos_robot[:3, :3] @ (-TCP_OFFSET_WITH_FINGER)
                    #     print(f"  World-frame offset delta: {np.round(offset_world_delta, 4)}")
    
                    #     T_offset_inv = np.eye(4)
                    #     T_offset_inv[:3, 3] = -TCP_OFFSET_WITH_FINGER
                    #     pos_robot_tcp = pos_robot @ T_offset_inv  # TCP pose in robot frame
                    #     rotation_axes_summary("  Robot-frame TCP axes", pos_robot_tcp[:3, :3])
                        
                        
                        
                        #==================call DMP======================================
                        pos_robot_tcp = object_memory[obj]["pose_base"]   # 4x4 matrix
                        if step_id == 1:
                            robot_initial_pose = home_pose# in world frame    
                        else:
                            robot_initial_pose = robot_current_pose
                        robot_initial_pose[6] = gripper_val #0.08
                        print(f"  Robot initial pose: {robot_initial_pose}")
    
                        robot_end_pose = pos_robot_tcp.copy()
                        current_pose_matrix = None
                        if triple_sub is not None:
                            try:
                                current_pose_matrix = triple_sub.lookup_transform_matrix(
                                    target_frame=args.base_frame,
                                    source_frame=args.tcp_frame,
                                    timeout_sec=args.tf_timeout,
                                )
                                print(
                                    f"  Current TF pose {args.base_frame} <- {args.tcp_frame}:\n"
                                    f"{np.array2string(current_pose_matrix, precision=6, suppress_small=True)}"
                                )
                            except (LookupException, ConnectivityException, ExtrapolationException) as exc:
                                print(
                                    f"  Warning: current TF lookup {args.base_frame} <- {args.tcp_frame} failed: {exc}. "
                                    "Falling back to robot_initial_pose orientation."
                                )

                        current_R = current_pose_matrix[:3, :3]
                        adjusted_R, adjusted_angle, within_limit = choose_equivalent_grasp_rotation(
                            current_R,
                            robot_end_pose[:3, :3],
                            symmetry_axis="z",
                            candidate_angles=(0.0, np.pi),
                            max_angle=np.pi / 2.0,
                        )
                        robot_end_pose[:3, :3] = adjusted_R
                        print(
                            "  Selected target orientation angle from current: "
                            f"{np.degrees(adjusted_angle):.2f} deg "
                            f"(within_90={within_limit})"
                        )

                        ## check the closest euler representation
                        # import pdb; pdb.set_trace()
                        xyzrpy_end = pose4x4_to_xyzrpy(robot_end_pose, gripper_val)
                        if reply[step_id]['action'] == "pour":
                            xyzrpy_end[2] += 0.2
                            xyzrpy_end[3] = 0
                            xyzrpy_end[4] = math.pi
                            # xyzrpy_end[5] = 0
                        # Prefer Euler representation that is close to current pose
                        current_pose_xyzrpy = pose4x4_to_xyzrpy(current_pose_matrix, gripper_val)
                        xyzrpy_end[3:6] = euler_closest_to_ref(
                            xyzrpy_end[3:6],
                            # current_pose_xyzrpy[3:6]
                            robot_initial_pose[3:6]
                        )

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
                        # # import pdb; pdb.set_trace()

                        if reply[step_id]['action'] == "drop":
                            drop_id += 1
                            xyzrpy_end[2] += 0.1 
                            xyzrpy_end[2] += 0.02 * drop_id
                            # Avoid overlap: place objects on predefined grid
                            if obj_name == 'plate':
                                offset_xy = plate_drop_offset_xy_grid_4(drop_id, radius=0.02)
                                xyzrpy_end[0] += offset_xy[0]
                                xyzrpy_end[1] += offset_xy[1]
                                # xyzrpy_end[3] = math.pi
                                # xyzrpy_end[4] = 0
                                # xyzrpy_end[5] = 0
                                print(f"  Plate drop offset: {offset_xy}")
                        # import pdb; pdb.set_trace()
                        

    
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
                        # import pdb; pdb.set_trace()
                        traj = traj.detach().cpu().numpy()[0]
                        
                        # Use dense waypoint execution while carrying an object to reduce slip.
                        step_stride = 1 if gripper_val < 0.078 else 5
                        # step_stride = 5
                        # flag_node.move_flag_pub.publish(Bool(data=True))
                        
                        # msg = Float64MultiArray()
                        # msg.data = traj #[float(v) for v in np.asarray(traj_7DoF).reshape(-1)]
                        # flag_node.traj_pub.publish(msg)

                        # traj: shape (6, N) from model => [x,y,z,roll,pitch,yaw] across columns
                        N = traj.shape[1]
                        gripper_col = np.full((1, N), gripper_val, dtype=traj.dtype)   # or zeros if intentional
                        traj7 = np.vstack([traj, gripper_col])                         # (7, N)

                        # Convert to (N, 7): each row = one waypoint expected by trajectory_listener_executor
                        traj_steps = traj7.T                                           # (N, 7)

                        # Sanity checks
                        assert traj_steps.shape == (N, 7)
                        # print("first waypoint:", traj_steps[0])
                        # print("last waypoint:", traj_steps[-1])

                        print("robot_initial_pose:", robot_initial_pose)
                        print("torch_robot_end_pose:", torch_robot_end_pose)
                        print("first waypoint:", traj_steps[0])
                        print("last waypoint:", traj_steps[-1])
                        print("z min/max:", traj_steps[:, 2].min(), traj_steps[:, 2].max())

                        import pdb; pdb.set_trace()
                        # Flatten in row-major so data is [wp0(7), wp1(7), ...]
                        # N=20
                        traj_pub.publish_traj(traj_steps[:, :].reshape(-1).tolist(), steps=N)

                        robot_current_pose = torch_robot_end_pose.cpu()
                        # time.sleep(20)

                elif action_id == 1: # wipe
                    print("wipe")
                    # # wipe: move to target and wipe
                    # traj_7DoF = np.append(traj[:, i], gripper_val)
                    # robot_current_pose = traj_7DoF
                    # gripper_val = 0.08
                    # # import pdb; pdb.set_trace()

                    # traj_gen = DMP(task_id=action_id)
                    traj_gen = traj_gen_cache[action_id]
                
                    # generate trajectory from cVAE-dmp    
                    traj_gen.eval()
                    torch_robot_intial_pose = torch.tensor(robot_current_pose, dtype=torch.float32).to(device)
                    xyzrpy_end = robot_current_pose
                    xyzrpy_end[2] -= 0.001
                    # torch_robot_end_pose = torch.tensor(pose4x4_to_xyzrpy(robot_end_pose,0.0), dtype=torch.float32).to(device)
                    torch_robot_end_pose = torch.tensor(xyzrpy_end, dtype=torch.float32).to(device)
                    # import pdb; pdb.set_trace()
                    # torch_robot_end_pose[3] = math.pi # make the grasp vertically facing down
                    # torch_robot_end_pose[4] = 0
                    print(f"  Torch robot end pose: {torch_robot_end_pose}")
                    
                    with torch.no_grad():
                        traj = traj_gen(class_idx=int(action_id), x0=torch_robot_intial_pose[0:6], goal=torch_robot_end_pose[0:6])  

                    traj[0,:,-1] = torch_robot_end_pose[0:6]   
                    # import pdb; pdb.set_trace()
                    traj = traj.detach().cpu().numpy()[0]

                    N = traj.shape[1]
                    gripper_col = np.full((1, N), gripper_val, dtype=traj.dtype)   # or zeros if intentional
                    traj7 = np.vstack([traj, gripper_col])                         # (7, N)

                    # Convert to (N, 7): each row = one waypoint expected by trajectory_listener_executor
                    traj_steps = traj7.T                                           # (N, 7)

                    # Sanity checks
                    assert traj_steps.shape == (N, 7)
                    # print("first waypoint:", traj_steps[0])
                    # print("last waypoint:", traj_steps[-1])

                    print("robot_initial_pose:", robot_initial_pose)
                    print("torch_robot_end_pose:", torch_robot_end_pose)
                    print("first waypoint:", traj_steps[0])
                    print("last waypoint:", traj_steps[-1])
                    print("z min/max:", traj_steps[:, 2].min(), traj_steps[:, 2].max())

                    import pdb; pdb.set_trace()
                    # Flatten in row-major so data is [wp0(7), wp1(7), ...]
                    # N=20
                    traj_pub.publish_traj(traj_steps[:, :].reshape(-1).tolist(), steps=N)

                    robot_current_pose = torch_robot_end_pose.cpu()

                elif action_id == 2: # pour
                    print("pour")
    
                    traj_gen = traj_gen_cache[action_id]
                    
                    # generate trajectory from cVAE-dmp    
                    traj_gen.eval()
                    print(f"  Torch robot end pose: {torch_robot_end_pose}")
                    robot_pour_pose = robot_current_pose

                    # robot_pour_pose = make_pour_pose_from_current(robot_current_pose, robot_current_pose[0:3])
                    # robot_pour_pose[3] += 0.05
                    
                    with torch.no_grad():
                        traj = traj_gen(class_idx=int(action_id), x0=robot_current_pose[0:6], goal=robot_pour_pose[0:6])  
                        traj_r = traj_gen(class_idx=int(action_id), x0=robot_pour_pose[0:6], goal=robot_current_pose[0:6]) 
                    # traj[0,:,-1] = torch_robot_end_pose[0:6]   
                
                    traj = traj.detach().cpu().numpy()[0]
                    traj_r = traj_r.detach().cpu().numpy()[0]
    

                    # Use dense waypoint execution while carrying an object to reduce slip.
                    step_stride = 1  if gripper_val <= 0.07 else 5
                    
                    # flag_node.move_flag_pub.publish(Bool(data=True))
                    # for i in range(traj_r.shape[1]):
                    #     if i % step_stride == 0 or i == traj_r.shape[1] - 1:
                    #         traj_7DoF = np.append(traj_r[:, i], gripper_val)
                    #         msg = Float64MultiArray()
                    #         msg.data = [float(v) for v in np.asarray(traj_7DoF).reshape(-1)]
                    #         flag_node.traj_pub.publish(msg)
                    #         robot_current_pose = traj_7DoF
                    # flag_node.move_flag_pub.publish(Bool(data=False))

                    # robot_current_pose = torch_robot_end_pose.cpu()
                    N = traj.shape[1]
                    gripper_col = np.full((1, N), gripper_val, dtype=traj.dtype)   # or zeros if intentional
                    traj7 = np.vstack([traj, gripper_col])                         # (7, N)

                    # Convert to (N, 7): each row = one waypoint expected by trajectory_listener_executor
                    traj_steps = traj7.T                                           # (N, 7)

                    # Sanity checks
                    assert traj_steps.shape == (N, 7)
                    # print("first waypoint:", traj_steps[0])
                    # print("last waypoint:", traj_steps[-1])

                    print("robot_initial_pose:", robot_initial_pose)
                    print("torch_robot_end_pose:", torch_robot_end_pose)
                    print("first waypoint:", traj_steps[0])
                    print("last waypoint:", traj_steps[-1])
                    print("z min/max:", traj_steps[:, 2].min(), traj_steps[:, 2].max())

                    import pdb; pdb.set_trace()
                    # Flatten in row-major so data is [wp0(7), wp1(7), ...]
                    # N=20
                    traj_pub.publish_traj(traj_steps[:, :].reshape(-1).tolist(), steps=N)

                    robot_current_pose = torch_robot_end_pose.cpu()
                    # import pdb; pdb.set_trace()
                elif action_id == 4: # drop
                    # Release in place first so the object can separate under gravity
                    # before the arm retreats upward.
                    # keep_pose = np.asarray(robot_current_pose[:6], dtype=np.float32)
                    # # env.robot.move_ee(keep_pose, control_method="end")
                    # gripper_val = 0.08
                    # released = False

                    # retreat_pose = robot_current_pose.copy()
                    # retreat_pose[2] += 0.06
                    # retreat_pose[6] = 0.08

                    # robot_current_pose = retreat_pose
                    gripper_val = 0.08
                    # flag_node.gripper_flag_pub.publish(Bool(data=True))

                    # flag_node.gripper_flag_pub.publish(Bool(data=False))

                    flag_node.publish_gripper_move(width=gripper_val, speed=0.05)

                    time.sleep(2)

    
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
    
                    gripper_val = 0.0
                    # nodes.move_pose(keep_pose, gripper_val)

                    # # gripper_val = float(env.robot.get_gripper_width())
                    # robot_current_pose[6] = gripper_val

                    # traj = robot_current_pose  
                    # traj_steps = traj.T                                           # (N, 7)
                    # traj_pub.publish_traj(traj_steps.reshape(-1).tolist(), steps=N)

                    flag_node.publish_gripper_grasp(
                        width=gripper_val,
                        speed=0.05,
                        force=20.0,
                        eps_inner=0.01,
                        eps_outer=0.01,
                    )
                    time.sleep(2)
        #===============================================================
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
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


# python test_mani_env_w_llm_exp.py --show-camera --subscribe-camera --subscriber-log-interval 5 --prompt "prepare breakfast with all fruits in plate" --checkpoint_path /home/mhumais/Huang/anygrasp_sdk/grasp_detection/log/checkpoint_detection.tar --top_down_grasp --grasp-center-offset-z -0.105 --num-runs 1  --scene-configs scene_4.json