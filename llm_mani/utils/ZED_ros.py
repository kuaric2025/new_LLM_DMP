"""ROS2 subscriber for ZED camera: depth, RGB, and camera info.

Subscribes to ZED ROS2 wrapper topics (e.g. from zed_wrapper zed_camera.launch.py).
Default topics match the ZED2/ZED2i wrapper naming:
  - RGB:  <namespace>/left/image_rect_color
  - Depth: <namespace>/depth/depth_registered
  - Info:  <namespace>/left/camera_info
"""

from __future__ import annotations

import threading
from typing import Optional

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from message_filters import ApproximateTimeSynchronizer, Subscriber as MFSubscriber
from sensor_msgs.msg import CameraInfo, Image


# Default ZED topic base (e.g. /zed2i/zed_node or /zed_node)
DEFAULT_ZED_NAMESPACE = "/zed_node"


class ZEDCameraSubscriber(Node):
    """ROS2 subscriber for ZED camera RGB, depth, and camera info.

    Uses ApproximateTimeSynchronizer to receive synchronized RGB, depth,
    and camera_info messages. Access latest data via get_latest_images().
    """

    def __init__(
        self,
        rgb_topic: str | None = None,
        depth_topic: str | None = None,
        info_topic: str | None = None,
        namespace: str = DEFAULT_ZED_NAMESPACE,
        queue_size: int = 10,
        slop: float = 0.1,
    ) -> None:
        super().__init__("zed_camera_subscriber")

        base = namespace.rstrip("/")
        self._rgb_topic = rgb_topic or f"{base}/left/image_rect_color"
        self._depth_topic = depth_topic or f"{base}/depth/depth_registered"
        self._info_topic = info_topic or f"{base}/left/camera_info"

        qos = qos_profile_sensor_data
        info_qos = QoSProfile(depth=10)

        self._rgb_sub = MFSubscriber(self, Image, self._rgb_topic, qos_profile=qos)
        self._depth_sub = MFSubscriber(self, Image, self._depth_topic, qos_profile=qos)
        self._info_sub = MFSubscriber(
            self, CameraInfo, self._info_topic, qos_profile=info_qos
        )

        self._sync = ApproximateTimeSynchronizer(
            [self._rgb_sub, self._depth_sub, self._info_sub],
            queue_size=queue_size,
            slop=slop,
        )
        self._sync.registerCallback(self._synced_callback)

        self._latest_rgb: Optional[np.ndarray] = None
        self._latest_depth: Optional[np.ndarray] = None
        self._latest_info: Optional[CameraInfo] = None
        self._lock = threading.Lock()

    def _synced_callback(
        self, rgb_msg: Image, depth_msg: Image, info_msg: CameraInfo
    ) -> None:
        rgb = self._image_to_numpy(rgb_msg)
        depth = self._depth_to_numpy(depth_msg)
        with self._lock:
            self._latest_rgb = rgb
            self._latest_depth = depth
            self._latest_info = info_msg

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
            arr = (
                data.reshape((msg.height, row_stride))[:, :expected]
                .reshape((msg.height, msg.width, 3))
            )
        if msg.encoding == "bgr8":
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        return arr

    def _depth_to_numpy(self, msg: Image) -> Optional[np.ndarray]:
        if msg.encoding == "32FC1":
            arr = np.frombuffer(msg.data, dtype=np.float32).reshape(
                (msg.height, msg.width)
            )
            return arr
        if msg.encoding == "16UC1":
            arr = (
                np.frombuffer(msg.data, dtype=np.uint16)
                .reshape((msg.height, msg.width))
                .astype(np.float32)
            )
            arr /= 1000.0
            return arr
        self.get_logger().warn(f"Unsupported depth encoding: {msg.encoding}")
        return None

    def get_latest_images(
        self,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[CameraInfo]]:
        """Thread-safe access to latest RGB, depth, and camera info."""
        with self._lock:
            rgb = self._latest_rgb.copy() if self._latest_rgb is not None else None
            depth = (
                self._latest_depth.copy() if self._latest_depth is not None else None
            )
            return rgb, depth, self._latest_info

    @property
    def rgb_topic(self) -> str:
        return self._rgb_topic

    @property
    def depth_topic(self) -> str:
        return self._depth_topic

    @property
    def info_topic(self) -> str:
        return self._info_topic
