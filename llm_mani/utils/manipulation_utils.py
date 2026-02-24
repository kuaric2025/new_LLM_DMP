from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pybullet as p
import torch


def sam3_inference(image: np.ndarray, object_name: str):
    """Run SAM3 inference and return the best segmentation map (torch tensor)."""
    import matplotlib.pyplot as plt  # noqa: F401  # imported for potential debugging
    import sam3
    from PIL import Image
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    print("sam3 starting...")
    sam3_root = Path(sam3.__file__).resolve().parent
    bpe_path = sam3_root.parent / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    model = build_sam3_image_model(bpe_path=str(bpe_path))

    pil_image = Image.fromarray(image)
    processor = Sam3Processor(model, confidence_threshold=0.5)
    inference_state = processor.set_image(pil_image)
    processor.reset_all_prompts(inference_state)
    inference_state = processor.set_text_prompt(state=inference_state, prompt=object_name)

    scores = inference_state["scores"]
    if scores.shape[0] < 1:
        print("❌ No valid masks or scores from VLM")
        return None

    segmap_all = inference_state["masks"]
    idx = torch.argmax(scores).item()
    return segmap_all[idx]


def DMP(task_id: int):
    """Load manipulation demonstrations and the trajectory generator for a task."""
    task_id = int(task_id)
    parent_dir = Path(os.getcwd()).resolve().parent / "VAE_DMP_mani"
    import sys

    sys.path.append(str(parent_dir))

    from VAE_DMP_mani.models.vae import TrajGen
    from VAE_DMP_mani.models.dmp import CanonicalSystem, SingleDMP
    from VAE_DMP_mani.utils.data_loader import TorqueLoader as Torque_dataset
    from VAE_DMP_mani.utils.early_stop import EarlyStop  # noqa: F401
    from collections import OrderedDict

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cs = CanonicalSystem(dt=0.01, ax=1)
    dmp = SingleDMP(n_bfs=50, cs=cs, run_time=1.0, dt=0.01)
    train_dataset = Torque_dataset(run_time=1, dmp=dmp, dt=0.01, dof=2)
    train_dataset.load_data("../VAE_DMP_mani/data/manipulation_data/train_torque.npz", device=device)
    train_dataset.torque = train_dataset.normalize_data(device=device)
    max_vals = train_dataset.max.cpu().numpy()
    min_vals = train_dataset.min.cpu().numpy()

    checkpoint = torch.load("../VAE_DMP_mani/models/cVAE_torque_manipulation.pt", map_location=device)
    decoder_param = {k.replace("decoder.", ""): v for k, v in checkpoint["net"].items() if "decoder." in k}
    torch.save(decoder_param, "../VAE_DMP_mani/models/decoder.pt")

    label_encoder_param = {k.replace("label_embedding.", ""): v for k, v in checkpoint["net"].items() if "label_embedding." in k}
    torch.save(label_encoder_param, "../VAE_DMP_mani/models/label_encoder.pt")

    shape = (6, 100)
    nclass = 2
    nhid = 8
    ncond = 8
    traj_gen = TrajGen(shape=shape, nclass=nclass, nhid=nhid, ncond=ncond, min=min_vals, max=max_vals, device=device)
    traj_gen.decoder_o.load_state_dict(torch.load('../VAE_DMP_mani/models/decoder.pt'))
    traj_gen.decoder_n.load_state_dict(torch.load('../VAE_DMP_mani/models/decoder.pt'))
    traj_gen.label_embedding.load_state_dict(torch.load('../VAE_DMP_mani/models/label_encoder.pt'))

    demon_path = Path("../VAE_DMP_mani/data/manipulation_data")
    demons_data = {}
    for file in demon_path.iterdir():
        if file.suffix == ".npy":
            data = np.load(file, allow_pickle=True)
            demons_data[f"task_id_{task_id}"] = data
    return demons_data, traj_gen


def pixel_depth_to_camera_frame(center_x: int, center_y: int, center_z_m: float, camera_info) -> np.ndarray:
    fx = camera_info[0]
    fy = camera_info[4]
    cx = camera_info[2]
    cy = camera_info[5]

    u, v, z = float(center_x), float(center_y), center_z_m
    x_cam = (u - cx) * z / fx
    y_cam = (v - cy) * z / fy
    return np.array([x_cam, y_cam, z], dtype=np.float32)


def world_to_camera_frame(env, p_world):
    base_pos, base_quat = env.get_robot_base_pose()
    cam_pos, cam_quat = env.get_camera_pose()
    inv_cam_pos, inv_cam_quat = p.invertTransform(cam_pos, cam_quat)
    p_cam, _ = p.multiplyTransforms(inv_cam_pos, inv_cam_quat, p_world, [0, 0, 0, 1])
    return np.array(p_cam)


def world_to_robot_frame(env, p_world):
    base_pos, base_quat = env.get_robot_base_pose()
    inv_pos, inv_quat = p.invertTransform(base_pos, base_quat)
    p_robot, _ = p.multiplyTransforms(inv_pos, inv_quat, p_world, [0, 0, 0, 1])
    return np.array(p_robot)


def pose4x4_to_rpy(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0
    return np.array([roll, pitch, yaw])


def pose4x4_to_xyzrpy(T: np.ndarray, gripper_width: float = 0.0) -> np.ndarray:
    x, y, z = T[0, 3], T[1, 3], T[2, 3]
    R = T[:3, :3]
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0
    return np.array([x, y, z, roll, pitch, yaw, gripper_width], dtype=np.float32)


def get_robot_pose_from_tcp(robot) -> np.ndarray:
    pos, quat = robot.get_eef()
    euler = p.getEulerFromQuaternion(quat)
    gripper_width = robot.get_gripper_width()
    return np.array([pos[0], pos[1], pos[2], euler[0], euler[1], euler[2], gripper_width], dtype=np.float32)


__all__ = [
    "sam3_inference",
    "DMP",
    "pixel_depth_to_camera_frame",
    "world_to_camera_frame",
    "world_to_robot_frame",
    "pose4x4_to_rpy",
    "pose4x4_to_xyzrpy",
    "get_robot_pose_from_tcp",
]
