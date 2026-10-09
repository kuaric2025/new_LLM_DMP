from __future__ import annotations

import atexit
import json
import os
import pickle
import select
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pybullet as p
import torch

import sys
sys.path.append("/home/mhumais/Huang/contact_graspnet_pytorch/contact_graspnet_pytorch")
sys.path.append("/home/mhumais/Huang/anygrasp_sdk/grasp_detection")


from PIL import Image

_SAM3_CACHE: dict[str, object] = {}
_SAM3_WORKER_PROC: subprocess.Popen | None = None
_SAM3_WORKER_LOCK = threading.Lock()


def _load_sam3_cached() -> tuple[object, object]:
    """Load SAM3 model once and reuse it across calls."""
    global _SAM3_CACHE
    if "model" in _SAM3_CACHE and "processor_cls" in _SAM3_CACHE:
        return _SAM3_CACHE["model"], _SAM3_CACHE["processor_cls"]

    # SAM3 (or deps) may import wandb; skew between wandb and protobuf can raise:
    # AttributeError: module 'wandb.proto.wandb_internal_pb2' has no attribute 'Result'
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("WANDB_SILENT", "true")

    try:
        import sam3
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
    except AttributeError as e:
        err = str(e)
        if "wandb_internal_pb2" in err and "Result" in err:
            raise RuntimeError(
                "wandb/protobuf mismatch while importing SAM3 (AttributeError on "
                "wandb_internal_pb2.Result). Align versions in this env, e.g. "
                "`pip install -U wandb 'protobuf>=6.32.1'` or upgrade wandb alone; "
                "if another package pins protobuf, match wandb to that stack."
            ) from e
        raise

    print("sam3 loading model...")
    sam3_root = Path(sam3.__file__).resolve().parent
    bpe_path = sam3_root.parent / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    model = build_sam3_image_model(bpe_path=str(bpe_path))
    _SAM3_CACHE = {
        "model": model,
        "processor_cls": Sam3Processor,
    }
    return model, Sam3Processor

def sam3_inference(image: np.ndarray, object_name: str):
    """Run SAM3 inference and return the best segmentation map (torch tensor)."""
    if os.environ.get("SAM3_IN_PROCESS", "0") != "1":
        return sam3_inference_subprocess(image, object_name)
    return sam3_inference_in_process(image, object_name)


def sam3_inference_subprocess(
    image: np.ndarray,
    object_name: str,
    timeout_s: float = 120.0,
):
    """Run SAM3 in an isolated child process.

    A fresh process per request is slower than a persistent worker, but it keeps
    SAM3/CUDA state out of the long-running PyBullet GUI + ROS process.
    """
    image = np.asarray(image, dtype=np.uint8)

    with tempfile.TemporaryDirectory(prefix="sam3_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        image_path = tmpdir_path / "image.npy"
        mask_path = tmpdir_path / "mask.npy"
        np.save(image_path, image)

        request_id = uuid.uuid4().hex
        request = {
            "id": request_id,
            "image_npy": str(image_path),
            "object_name": str(object_name),
            "mask_npy": str(mask_path),
        }
        print(f"sam3 worker inference for: {object_name}", flush=True)
        result = _sam3_worker_request_once(request, timeout_s=timeout_s)
        if result is None:
            return None
        returncode = int(result.get("returncode", 1))
        if returncode == 2:
            print(f"SAM3 worker found no mask for {object_name}", flush=True)
            return None
        if returncode != 0:
            err = result.get("error", "")
            print(
                f"SAM3 worker failed for {object_name} with code "
                f"{result.get('returncode')}: {err}",
                flush=True,
            )
            return None

        if not mask_path.exists():
            print(f"SAM3 worker did not produce mask for {object_name}", flush=True)
            return None

        mask = np.load(mask_path).astype(bool)
        if not np.any(mask):
            print(f"SAM3 worker returned empty mask for {object_name}", flush=True)
            return None
        return torch.from_numpy(mask)


def _sam3_worker_request_once(request: dict[str, object], timeout_s: float) -> dict[str, object] | None:
    cmd = [
        sys.executable,
        str(_sam3_worker_path()),
        "--image-npy",
        str(request["image_npy"]),
        "--object-name",
        str(request["object_name"]),
        "--mask-npy",
        str(request["mask_npy"]),
    ]
    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "disabled")
    env.setdefault("WANDB_SILENT", "true")
    try:
        completed = subprocess.run(
            cmd,
            timeout=timeout_s,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired:
        print(f"SAM3 worker timed out after {timeout_s:.1f}s", flush=True)
        return None
    return {"id": request["id"], "returncode": completed.returncode}


def _sam3_worker_path() -> Path:
    worker_path = Path(__file__).resolve().parents[1] / "sam3_worker.py"
    if not worker_path.exists():
        raise FileNotFoundError(f"SAM3 worker not found: {worker_path}")
    return worker_path


def _start_sam3_worker() -> subprocess.Popen:
    global _SAM3_WORKER_PROC
    if _SAM3_WORKER_PROC is not None and _SAM3_WORKER_PROC.poll() is None:
        return _SAM3_WORKER_PROC

    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "disabled")
    env.setdefault("WANDB_SILENT", "true")
    cmd = [sys.executable, str(_sam3_worker_path()), "--server"]
    print("starting persistent SAM3 worker", flush=True)
    _SAM3_WORKER_PROC = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
        bufsize=1,
        env=env,
    )
    return _SAM3_WORKER_PROC


def _stop_sam3_worker() -> None:
    global _SAM3_WORKER_PROC
    proc = _SAM3_WORKER_PROC
    if proc is None or proc.poll() is not None:
        _SAM3_WORKER_PROC = None
        return
    try:
        if proc.stdin is not None:
            proc.stdin.write(json.dumps({"id": "shutdown", "cmd": "shutdown"}) + "\n")
            proc.stdin.flush()
        proc.wait(timeout=5)
    except Exception:
        proc.kill()
    finally:
        _SAM3_WORKER_PROC = None


atexit.register(_stop_sam3_worker)


def _sam3_worker_request(request: dict[str, object], timeout_s: float) -> dict[str, object] | None:
    global _SAM3_WORKER_PROC
    with _SAM3_WORKER_LOCK:
        proc = _start_sam3_worker()
        if proc.stdin is None or proc.stdout is None:
            print("SAM3 worker pipes are unavailable", flush=True)
            return None

        request_id = str(request["id"])
        proc.stdin.write(json.dumps(request) + "\n")
        proc.stdin.flush()

        deadline = time.monotonic() + timeout_s
        while True:
            if proc.poll() is not None:
                print(f"SAM3 worker exited with code {proc.returncode}", flush=True)
                _SAM3_WORKER_PROC = None
                return None

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                print(f"SAM3 worker timed out after {timeout_s:.1f}s", flush=True)
                proc.kill()
                _SAM3_WORKER_PROC = None
                return None

            readable, _, _ = select.select([proc.stdout], [], [], min(remaining, 1.0))
            if not readable:
                continue

            line = proc.stdout.readline()
            if not line:
                continue
            try:
                response = json.loads(line)
            except json.JSONDecodeError:
                # Third-party libraries sometimes print during model load.
                print(line.rstrip(), flush=True)
                continue
            if response.get("id") == request_id:
                return response


def sam3_inference_in_process(image: np.ndarray, object_name: str):
    """Run SAM3 inference and return the best segmentation map (torch tensor)."""
    import matplotlib.pyplot as plt  # noqa: F401  # imported for potential debugging
    model, Sam3Processor = _load_sam3_cached()

    pil_image = Image.fromarray(image)
    processor = Sam3Processor(model, confidence_threshold=0.1)
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

# def load_sam3_model():
#     import sam3
#     from sam3 import build_sam3_image_model
#     from sam3.model.sam3_image_processor import Sam3Processor
#     sam3_root = Path(sam3.__file__).resolve().parent
#     bpe_path = sam3_root.parent / "assets" / "bpe_simple_vocab_16e6.txt.gz"
#     model = build_sam3_image_model(bpe_path=str(bpe_path))
#     return model, Sam3Processor

# def sam3_inference(image: np.ndarray, object_name: str):
#     """Run SAM3 inference and return the best segmentation map (torch tensor)."""
#     import matplotlib.pyplot as plt  # noqa: F401  # imported for potential debugging
#     # import sam3
#     from PIL import Image
#     # from sam3 import build_sam3_image_model
#     # from sam3.model.sam3_image_processor import Sam3Processor

#     # print("sam3 starting...")
#     # sam3_root = Path(sam3.__file__).resolve().parent
#     # bpe_path = sam3_root.parent / "assets" / "bpe_simple_vocab_16e6.txt.gz"
#     # model = build_sam3_image_model(bpe_path=str(bpe_path))
#     model, Sam3Processor = load_sam3_model()

#     pil_image = Image.fromarray(image)
#     processor = Sam3Processor(model, confidence_threshold=0.5)
#     inference_state = processor.set_image(pil_image)
#     processor.reset_all_prompts(inference_state)
#     inference_state = processor.set_text_prompt(state=inference_state, prompt=object_name)

#     scores = inference_state["scores"]
#     if scores.shape[0] < 1:
#         print("❌ No valid masks or scores from VLM")
#         return None

#     segmap_all = inference_state["masks"]
#     idx = torch.argmax(scores).item()
#     return segmap_all[idx]

from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation as R
from tf2_geometry_msgs import do_transform_pose
def matrix_to_transform_stamped(T, parent_frame="base", child_frame="tcp"):
    tf_msg = TransformStamped()
    tf_msg.header.frame_id = parent_frame
    tf_msg.child_frame_id = child_frame

    tf_msg.transform.translation.x = float(T[0, 3])
    tf_msg.transform.translation.y = float(T[1, 3])
    tf_msg.transform.translation.z = float(T[2, 3])

    quat = R.from_matrix(T[:3, :3]).as_quat()   # [x, y, z, w]
    tf_msg.transform.rotation.x = float(quat[0])
    tf_msg.transform.rotation.y = float(quat[1])
    tf_msg.transform.rotation.z = float(quat[2])
    tf_msg.transform.rotation.w = float(quat[3])

    return tf_msg

import numpy as np
from geometry_msgs.msg import Pose
from scipy.spatial.transform import Rotation as R

def matrix_to_pose(T):
    pose = Pose()
    pose.position.x = float(T[0, 3])
    pose.position.y = float(T[1, 3])
    pose.position.z = float(T[2, 3])

    quat = R.from_matrix(T[:3, :3]).as_quat()  # x, y, z, w
    pose.orientation.x = float(quat[0])
    pose.orientation.y = float(quat[1])
    pose.orientation.z = float(quat[2])
    pose.orientation.w = float(quat[3])

    return pose

from scipy.spatial.transform import Rotation as R

def pose_to_matrix(pose):
    T = np.eye(4)
    T[:3, 3] = [
        pose.position.x,
        pose.position.y,
        pose.position.z,
    ]
    T[:3, :3] = R.from_quat([
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    ]).as_matrix()
    return T

def DMP_v1(task_id: int):
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

from VAE_DMP_mani.models.vae import TrajGen
from VAE_DMP_mani.models.dmp import CanonicalSystem, SingleDMP
from VAE_DMP_mani.utils.data_loader import TorqueLoader as Torque_dataset
from VAE_DMP_mani.utils.early_stop import EarlyStop  # noqa: F401
from collections import OrderedDict

from llm_mani.utils.paths import add_root_to_path, infer_dof_from_npz, infer_steps_from_npz
from llm_mani.utils.data_loader import TorqueLoader as TorqueDataset
ROOT = add_root_to_path()


class DiffusionTrajectoryPolicy(torch.nn.Module):
    """Adapter for real-stanford/diffusion_policy with DMP-compatible API."""

    def __init__(
        self,
        checkpoint_path: str,
        repo_path: str,
        device: torch.device,
        n_obs_steps: int = 2,
        n_action_steps: int = 16,
        num_inference_steps: int = 16,
    ) -> None:
        super().__init__()
        self.device = device
        self.n_obs_steps = int(n_obs_steps)
        self.n_action_steps = int(n_action_steps)
        self.num_inference_steps = int(num_inference_steps)
        self.policy = self._load_policy(checkpoint_path=checkpoint_path, repo_path=repo_path)
        self._last_obs: Optional[torch.Tensor] = None

    def _load_policy(self, checkpoint_path: str, repo_path: str):
        try:
            import dill
            import hydra
        except ImportError as exc:
            raise ImportError(
                "Missing dependency for real-stanford/diffusion_policy adapter. "
                "Install `dill` and `hydra-core` in this environment."
            ) from exc

        repo_path = str(Path(repo_path).resolve())
        if repo_path not in sys.path:
            sys.path.append(repo_path)

        checkpoint_path = str(Path(checkpoint_path).resolve())
        payload = torch.load(open(checkpoint_path, "rb"), map_location=self.device, pickle_module=dill)
        cfg = payload["cfg"]
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg)
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        policy = workspace.model
        if bool(getattr(cfg.training, "use_ema", False)):
            policy = workspace.ema_model
        policy.to(self.device).eval()

        # Keep inference params configurable as in upstream eval_real_robot.py
        if hasattr(policy, "num_inference_steps"):
            policy.num_inference_steps = self.num_inference_steps
        if hasattr(policy, "n_action_steps"):
            policy.n_action_steps = self.n_action_steps
        return policy

    def _build_obs_dict(self, x0: torch.Tensor, goal: torch.Tensor) -> dict:
        state = torch.cat([x0, goal], dim=0).to(self.device)
        if self._last_obs is None:
            obs_seq = state.unsqueeze(0).repeat(self.n_obs_steps, 1)
        else:
            prev = self._last_obs
            if prev.shape[-1] != state.shape[-1]:
                prev = state.unsqueeze(0).repeat(self.n_obs_steps, 1)
            obs_seq = torch.cat([prev[1:], state.unsqueeze(0)], dim=0)
        self._last_obs = obs_seq.detach()
        return {"obs": obs_seq.unsqueeze(0)}

    def forward(self, class_idx: int, x0: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        del class_idx
        x0 = x0.to(dtype=torch.float32).reshape(-1)
        goal = goal.to(dtype=torch.float32).reshape(-1)
        obs_dict = self._build_obs_dict(x0, goal)
        with torch.no_grad():
            result = self.policy.predict_action(obs_dict)

        if "action" not in result:
            raise RuntimeError("diffusion_policy.predict_action() did not return 'action'.")
        action = result["action"][0]
        if action.ndim != 2:
            raise RuntimeError(f"Expected action shape (Ta, Da), got {tuple(action.shape)}")
        if action.shape[-1] < 6:
            raise RuntimeError(
                f"Diffusion policy action dim must be >=6 for xyzrpy control, got {action.shape[-1]}"
            )
        if action.shape[-1] > 6:
            action = action[:, :6]
        traj = action.transpose(0, 1).to(dtype=torch.float32)
        return traj.unsqueeze(0)


def DiffusionPolicy(
    task_id: int,
    checkpoint_path: str,
    repo_path: str,
    device: torch.device,
    n_obs_steps: int = 2,
    n_action_steps: int = 16,
    num_inference_steps: int = 16,
) -> DiffusionTrajectoryPolicy:
    """Return a real diffusion_policy adapter with DMP-compatible call signature."""
    del task_id
    return DiffusionTrajectoryPolicy(
        checkpoint_path=checkpoint_path,
        repo_path=repo_path,
        device=device,
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        num_inference_steps=num_inference_steps,
    )


import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class DMPConfig:
    n_dims: int = 6
    n_basis: int = 30
    dt: float = 0.01
    T: float = 2.0
    alpha_z: float = 25.0
    beta_z: float = 25.0 / 4.0
    alpha_x: float = 1.0


class MultiDMP:
    """
    A simple multi-dimensional DMP for xyzrpy trajectory generation.

    This implementation uses:
      - 1 canonical system
      - 1 transformation system per dimension
      - Gaussian basis functions
      - zero forcing term by default (smooth point-to-point motion)

    You can later extend it by fitting weights from demonstrations.
    """

    def __init__(self, config: DMPConfig):
        self.cfg = config
        self.n_dims = config.n_dims
        self.n_basis = config.n_basis

        # Canonical system centers in phase space
        self.centers = np.exp(
            -self.cfg.alpha_x * np.linspace(0, 1, self.n_basis)
        )

        # Basis widths
        # Chosen so neighboring basis functions overlap reasonably
        self.widths = np.ones(self.n_basis) * self.n_basis ** 1.5 / self.centers / self.cfg.alpha_x

        # Forcing term weights for each dimension
        # Shape: (n_dims, n_basis)
        self.weights = np.zeros((self.n_dims, self.n_basis), dtype=np.float64)

    def _basis_function(self, x: float) -> np.ndarray:
        """Compute Gaussian basis activations for a scalar phase x."""
        return np.exp(-self.widths * (x - self.centers) ** 2)

    def forcing_term(self, x: float, g: np.ndarray, y0: np.ndarray) -> np.ndarray:
        """
        Compute forcing term for each dimension.
        With zero weights, this becomes zero and yields a smooth point attractor trajectory.
        """
        psi = self._basis_function(x)
        psi_sum = np.sum(psi) + 1e-10

        f = np.zeros(self.n_dims, dtype=np.float64)
        for d in range(self.n_dims):
            f[d] = (psi @ self.weights[d]) / psi_sum * x * (g[d] - y0[d])
        return f

    def rollout(self, y0, g):
        """
        Generate a DMP trajectory from start y0 to goal g.

        Args:
            y0: array-like, shape (6,), [x, y, z, roll, pitch, yaw]
            g:  array-like, shape (6,), [x, y, z, roll, pitch, yaw]

        Returns:
            traj: positions, shape (N, 6)
            vel: velocities, shape (N, 6)
            acc: accelerations, shape (N, 6)
            time: shape (N,)
        """
        y0 = np.asarray(y0, dtype=np.float64)
        g = np.asarray(g, dtype=np.float64)

        assert y0.shape == (self.n_dims,)
        assert g.shape == (self.n_dims,)

        n_steps = int(self.cfg.T / self.cfg.dt)
        time = np.arange(n_steps) * self.cfg.dt

        y = y0.copy()
        dy = np.zeros(self.n_dims, dtype=np.float64)
        ddy = np.zeros(self.n_dims, dtype=np.float64)

        x = 1.0  # canonical state

        traj = np.zeros((n_steps, self.n_dims), dtype=np.float64)
        vel = np.zeros((n_steps, self.n_dims), dtype=np.float64)
        acc = np.zeros((n_steps, self.n_dims), dtype=np.float64)

        tau = self.cfg.T

        for t in range(n_steps):
            f = self.forcing_term(x, g, y0)

            # Transformation system:
            # tau * z_dot = alpha_z * (beta_z * (g - y) - z) + f
            # tau * y_dot = z
            #
            # Here dy plays role of y_dot, and ddy = y_ddot
            ddy = (
                self.cfg.alpha_z * (self.cfg.beta_z * (g - y) - tau * dy) + f
            ) / (tau ** 2)

            dy = dy + ddy * self.cfg.dt
            y = y + dy * self.cfg.dt

            # Canonical system:
            # tau * x_dot = -alpha_x * x
            dx = -self.cfg.alpha_x * x / tau
            x = x + dx * self.cfg.dt

            traj[t] = y
            vel[t] = dy
            acc[t] = ddy

        return traj, vel, acc, time

    def fit_from_demo(self, demo_traj: np.ndarray, regularization: float = 1e-6) -> None:
        """Fit DMP forcing weights from a single demonstration of shape (T, n_dims)."""
        demo_traj = np.asarray(demo_traj, dtype=np.float64)
        if demo_traj.ndim != 2 or demo_traj.shape[1] != self.n_dims:
            raise ValueError(
                f"`demo_traj` must have shape (T, {self.n_dims}), got {demo_traj.shape}"
            )
        if demo_traj.shape[0] < 3:
            raise ValueError("`demo_traj` must contain at least 3 timesteps.")

        y = demo_traj
        y0 = y[0]
        g = y[-1]
        n_steps = y.shape[0]
        tau = self.cfg.T

        dy = np.gradient(y, self.cfg.dt, axis=0)
        ddy = np.gradient(dy, self.cfg.dt, axis=0)

        # Invert transformation dynamics to recover target forcing term.
        denom = (g - y0)
        safe_denom = np.where(np.abs(denom) < 1e-8, 1.0, denom)
        f_target = (
            (tau ** 2) * ddy - self.cfg.alpha_z * (self.cfg.beta_z * (g - y) - tau * dy)
        ) / safe_denom
        f_target *= (np.abs(denom) >= 1e-8)

        x_track = np.zeros(n_steps, dtype=np.float64)
        x = 1.0
        for t in range(n_steps):
            x_track[t] = x
            dx = -self.cfg.alpha_x * x / tau
            x = x + dx * self.cfg.dt

        psi_track = np.zeros((n_steps, self.n_basis), dtype=np.float64)
        for t in range(n_steps):
            psi_track[t] = self._basis_function(x_track[t])

        # Linear least squares per dimension:
        # f_target[:, d] ~ ((psi @ w_d)/sum(psi)) * x
        for d in range(self.n_dims):
            basis_scale = (x_track * (g[d] - y0[d])).reshape(-1, 1)
            phi = (psi_track / (np.sum(psi_track, axis=1, keepdims=True) + 1e-10)) * basis_scale
            ata = phi.T @ phi
            self.weights[d] = np.linalg.solve(
                ata + regularization * np.eye(self.n_basis),
                phi.T @ f_target[:, d],
            )

    def fit_from_demos(self, demos: list[np.ndarray], regularization: float = 1e-6) -> None:
        """Fit weights from multiple demonstrations by averaging per-demo weights."""
        if len(demos) == 0:
            raise ValueError("`demos` must not be empty.")
        all_weights = []
        for demo in demos:
            tmp = MultiDMP(self.cfg)
            tmp.fit_from_demo(demo, regularization=regularization)
            all_weights.append(tmp.weights.copy())
        self.weights = np.mean(np.stack(all_weights, axis=0), axis=0)


def unwrap_rpy(start_rpy, goal_rpy):
    """
    Optional helper to avoid large angle jumps.
    Adjusts goal RPY so each angle takes the shortest path from start.
    """
    start_rpy = np.asarray(start_rpy, dtype=np.float64)
    goal_rpy = np.asarray(goal_rpy, dtype=np.float64)

    adjusted = goal_rpy.copy()
    for i in range(3):
        diff = adjusted[i] - start_rpy[i]
        while diff > np.pi:
            adjusted[i] -= 2 * np.pi
            diff = adjusted[i] - start_rpy[i]
        while diff < -np.pi:
            adjusted[i] += 2 * np.pi
            diff = adjusted[i] - start_rpy[i]
    return adjusted


def standard_dmp_traj(
    start,
    goal,
    *,
    dt: float = 0.01,
    horizon_s: float = 2.0,
    n_basis: int = 30,
    unwrap_orientation: bool = True,
    save_csv_path: str | None = None,
    show_plot: bool = False,
):
    """Generate a standard 6-DoF DMP trajectory from start to goal."""
    start = np.asarray(start, dtype=np.float64).reshape(-1).copy()
    goal = np.asarray(goal, dtype=np.float64).reshape(-1).copy()
    if start.shape[0] != 6 or goal.shape[0] != 6:
        raise ValueError(
            f"`start` and `goal` must be shape (6,), got {start.shape} and {goal.shape}"
        )

    # Avoid unnecessary large angle jumps across +/-pi boundaries.
    if unwrap_orientation:
        goal[3:] = unwrap_rpy(start[3:], goal[3:])

    cfg = DMPConfig(
        n_dims=6,
        n_basis=int(n_basis),
        dt=float(dt),
        T=float(horizon_s),
        alpha_z=25.0,
        beta_z=25.0 / 4.0,
        alpha_x=1.0,
    )

    dmp = MultiDMP(cfg)
    traj, vel, acc, time = dmp.rollout(start, goal)

    if save_csv_path:
        np.savetxt(
            save_csv_path,
            np.hstack([time[:, None], traj, vel, acc]),
            delimiter=",",
            header="t,x,y,z,roll,pitch,yaw,vx,vy,vz,vroll,vpitch,vyaw,ax,ay,az,aroll,apitch,ayaw",
            comments="",
        )

    if show_plot:
        labels = ["x", "y", "z", "roll", "pitch", "yaw"]
        fig, axes = plt.subplots(3, 2, figsize=(10, 8))
        axes = axes.flatten()
        for i in range(6):
            axes[i].plot(time, traj[:, i])
            axes[i].set_title(labels[i])
            axes[i].set_xlabel("time (s)")
            axes[i].set_ylabel(labels[i])
        plt.tight_layout()
        plt.show()

    return traj


def _resample_traj_to_steps(traj: np.ndarray, n_steps: int) -> np.ndarray:
    """Linearly resample trajectory from (T, D) to (n_steps, D)."""
    traj = np.asarray(traj, dtype=np.float64)
    if traj.ndim != 2:
        raise ValueError(f"`traj` must be 2D, got shape {traj.shape}")
    if traj.shape[0] == n_steps:
        return traj.copy()
    src_t = np.linspace(0.0, 1.0, traj.shape[0])
    dst_t = np.linspace(0.0, 1.0, n_steps)
    out = np.zeros((n_steps, traj.shape[1]), dtype=np.float64)
    for d in range(traj.shape[1]):
        out[:, d] = np.interp(dst_t, src_t, traj[:, d])
    return out


def _safe_unit(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        out = np.zeros_like(vec)
        if out.shape[0] > 0:
            out[0] = 1.0
        return out
    return vec / norm


def _rotation_align_vectors(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Return a 3x3 rotation matrix that aligns src to dst."""
    src_u = _safe_unit(src)
    dst_u = _safe_unit(dst)
    v = np.cross(src_u, dst_u)
    c = float(np.clip(np.dot(src_u, dst_u), -1.0, 1.0))
    s = float(np.linalg.norm(v))

    if s < 1e-8:
        if c > 0.0:
            return np.eye(3, dtype=np.float64)
        # 180-degree case: rotate around any axis orthogonal to src.
        axis_guess = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(src_u[0]) > 0.9:
            axis_guess = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = _safe_unit(np.cross(src_u, axis_guess))
        return -np.eye(3, dtype=np.float64) + 2.0 * np.outer(axis, axis)

    vx = np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + vx + vx @ vx * ((1.0 - c) / (s ** 2))


def _demo_similarity_score(demo: np.ndarray, start: np.ndarray, goal: np.ndarray) -> float:
    """Lower is better: prioritize demos with similar start/end geometry."""
    demo = np.asarray(demo, dtype=np.float64)
    start_xyz = np.asarray(start[:3], dtype=np.float64)
    goal_xyz = np.asarray(goal[:3], dtype=np.float64)
    demo_start = demo[0, :3]
    demo_goal = demo[-1, :3]
    target_seg = goal_xyz - start_xyz
    demo_seg = demo_goal - demo_start
    target_len = float(np.linalg.norm(target_seg))
    demo_len = float(np.linalg.norm(demo_seg))
    dir_penalty = 1.0 - float(np.clip(np.dot(_safe_unit(target_seg), _safe_unit(demo_seg)), -1.0, 1.0))
    return (
        float(np.linalg.norm(demo_start - start_xyz))
        + float(np.linalg.norm(demo_goal - goal_xyz))
        + 0.25 * abs(demo_len - target_len)
        + 0.05 * dir_penalty
    )


def _align_demo_to_task(demo: np.ndarray, start: np.ndarray, goal: np.ndarray) -> np.ndarray:
    """
    Align a demo to the current task by mapping its straight start-goal segment
    to the new start-goal segment and rotating/scaling the residual bend.
    """
    demo = np.asarray(demo, dtype=np.float64)
    aligned = demo.copy()
    n_steps = demo.shape[0]

    demo_start_xyz = demo[0, :3]
    demo_goal_xyz = demo[-1, :3]
    target_start_xyz = np.asarray(start[:3], dtype=np.float64)
    target_goal_xyz = np.asarray(goal[:3], dtype=np.float64)

    demo_line_xyz = np.linspace(demo_start_xyz, demo_goal_xyz, n_steps, dtype=np.float64)
    target_line_xyz = np.linspace(target_start_xyz, target_goal_xyz, n_steps, dtype=np.float64)
    residual_xyz = demo[:, :3] - demo_line_xyz

    demo_seg = demo_goal_xyz - demo_start_xyz
    target_seg = target_goal_xyz - target_start_xyz
    rot = _rotation_align_vectors(demo_seg, target_seg)
    demo_len = float(np.linalg.norm(demo_seg))
    target_len = float(np.linalg.norm(target_seg))
    residual_scale = float(np.clip(target_len / max(demo_len, 1e-8), 0.85, 1.15))
    aligned[:, :3] = target_line_xyz + (residual_xyz @ rot.T) * residual_scale

    # Orientation is not demo-guided in the current policy; keep it as shortest-path interpolation.
    start_rpy = np.asarray(start[3:6], dtype=np.float64)
    end_rpy = unwrap_rpy(start_rpy, np.asarray(goal[3:6], dtype=np.float64))
    aligned[:, 3:6] = np.linspace(start_rpy, end_rpy, n_steps, dtype=np.float64)

    aligned[0, :6] = np.asarray(start[:6], dtype=np.float64)
    aligned[-1, :6] = np.asarray(goal[:6], dtype=np.float64)
    return aligned


def demo_guided_dmp_traj(
    start,
    goal,
    demos,
    *,
    dt: float = 0.01,
    horizon_s: float = 2.0,
    n_basis: int = 30,
    unwrap_orientation: bool = True,
    regularization: float = 1e-6,
    top_k: int | None = 16,
    align_to_task: bool = True,
):
    """
    Fit DMP from sample trajectories and retarget to a new start/goal.

    Args:
        start, goal: (6,) xyzrpy.
        demos: one demo (T, 6) or list of demos with shape (Ti, 6).
    Returns:
        traj: (N, 6)
    """
    start = np.asarray(start, dtype=np.float64).reshape(-1).copy()
    goal = np.asarray(goal, dtype=np.float64).reshape(-1).copy()
    if start.shape[0] != 6 or goal.shape[0] != 6:
        raise ValueError(
            f"`start` and `goal` must be shape (6,), got {start.shape} and {goal.shape}"
        )
    if unwrap_orientation:
        goal[3:] = unwrap_rpy(start[3:], goal[3:])

    cfg = DMPConfig(
        n_dims=6,
        n_basis=int(n_basis),
        dt=float(dt),
        T=float(horizon_s),
        alpha_z=25.0,
        beta_z=25.0 / 4.0,
        alpha_x=1.0,
    )
    dmp = MultiDMP(cfg)
    n_steps = int(cfg.T / cfg.dt)

    if isinstance(demos, np.ndarray):
        demo_list = [demos]
    else:
        demo_list = list(demos)
    if len(demo_list) == 0:
        raise ValueError("`demos` must not be empty.")

    processed = []
    for demo in demo_list:
        demo = np.asarray(demo, dtype=np.float64)
        if demo.ndim != 2 or demo.shape[1] != 6:
            raise ValueError(f"Each demo must have shape (T, 6), got {demo.shape}")
        processed.append(_resample_traj_to_steps(demo, n_steps))

    if top_k is not None and top_k > 0 and len(processed) > top_k:
        scored = sorted(
            processed,
            key=lambda demo: _demo_similarity_score(demo, start, goal),
        )
        processed = scored[: int(top_k)]

    if align_to_task:
        processed = [_align_demo_to_task(demo, start, goal) for demo in processed]

    if len(processed) == 1:
        dmp.fit_from_demo(processed[0], regularization=regularization)
    else:
        dmp.fit_from_demos(processed, regularization=regularization)

    traj, _, _, _ = dmp.rollout(start, goal)
    return traj



def build_dmp(run_time: float, dt: float) -> SingleDMP:
    cs = CanonicalSystem(dt=dt, ax=1)
    return SingleDMP(n_bfs=50, cs=cs, run_time=run_time, dt=dt)


def load_decoder_and_label(checkpoint: dict) -> tuple[dict, dict]:
    decoder_param = OrderedDict()
    label_encoder_param = OrderedDict()
    for layer_name, param in checkpoint["net"].items():
        if "decoder." in layer_name:
            decoder_param[layer_name.replace("decoder.", "")] = param
        if "label_embedding." in layer_name:
            label_encoder_param[layer_name.replace("label_embedding.", "")] = param
    return decoder_param, label_encoder_param

def DMP(task_id: int):
    """Load manipulation demonstrations and the trajectory generator for a task."""
    task_id = int(task_id)
    parent_dir = Path(os.getcwd()).resolve().parent / "VAE_DMP_mani"
    import sys

    sys.path.append(str(parent_dir))

    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_time = 1.0
    dt = 0.01
    nclass, nhid, ncond = 3, 8, 8

    dmp = build_dmp(run_time, dt)
    torque_path = "/home/mhumais/Huang/DMP/VAE_DMP_mani/data/manipulation_data/train_torque.npz"

    dof = infer_dof_from_npz(torque_path, key="torque")
    steps = infer_steps_from_npz(torque_path, key="torque")
    shape = (dof, steps)

    train_dataset = TorqueDataset(run_time=run_time, dmp=dmp, dt=dt, dof=dof)
    train_dataset.load_data(str(torque_path), device=device)
    train_dataset.torque = train_dataset.normalize_data(device=device)

    data_max = train_dataset.max.cpu().numpy()
    data_min = train_dataset.min.cpu().numpy()

    # label mapping (same as training)
    label_min = int(train_dataset.torque_labels[:, 0].min().item())
    label_max = int(train_dataset.torque_labels[:, 0].max().item())
    nclass = label_max - label_min + 1


    checkpoint = torch.load('/home/mhumais/Huang/DMP/VAE_DMP_mani/models/cVAE_torque_manipulation.pt', map_location=device)
    decoder_param, label_encoder_param = load_decoder_and_label(checkpoint)

    traj_gen = TrajGen(shape=shape, nclass=nclass, nhid=nhid, ncond=ncond, min=data_min, max=data_max, device=device)
    traj_gen.decoder_o.load_state_dict(decoder_param)
    traj_gen.decoder_n.load_state_dict(decoder_param)
    traj_gen.label_embedding.load_state_dict(label_encoder_param)

    return traj_gen


def plate_drop_offset_xy(drop_index: int, radius: float = 0.04, max_slots: int = 6) -> np.ndarray:
    """
    Compute XY offset for the n-th object dropped on a plate to avoid overlap.
    Uses a ring layout: first at center, then subsequent objects in a circle.
    Returns (dx, dy) in meters, in the plate's local XY plane (world/robot frame).
    """
    if drop_index <= 1:
        return np.array([0.0, 0.0], dtype=np.float64)
    angle = 2.0 * np.pi * (drop_index - 2) / max(1, max_slots - 1)
    dx = radius * np.cos(angle)
    dy = radius * np.sin(angle)
    return np.array([dx, dy], dtype=np.float64)


def plate_drop_offset_xy_grid(
    drop_index: int, cell_size: float = 0.04
) -> np.ndarray:
    """
    Compute XY offset using a predefined 3x3 grid. Center first, then cardinal
    directions, then corners. Returns (dx, dy) in meters.
    """
    # Order: center, right, left, up, down, top-right, top-left, bottom-right, bottom-left
    grid = [
        (0.0, 0.0),           # 1: center
        (cell_size, 0.0),     # 2: right
        (-cell_size, 0.0),    # 3: left
        (0.0, cell_size),     # 4: up (y+)
        (0.0, -cell_size),    # 5: down (y-)
        (cell_size, cell_size),
        (-cell_size, cell_size),
        (cell_size, -cell_size),
        (-cell_size, -cell_size),
    ]
    idx = min(drop_index - 1, len(grid) - 1)
    return np.array(grid[idx], dtype=np.float64)


def plate_drop_offset_xy_grid_4(
    drop_index: int, radius: float = 0.04
) -> np.ndarray:
    """
    Compute XY offset using a 2x2 grid (4 quadrants). Each drop goes to one quadrant.
    Order: top-right, top-left, bottom-left, bottom-right.
    Returns (dx, dy) in meters.
    """
    grid = [
        (radius, radius),    # 1: top-right
        (-radius, radius),   # 2: top-left
        (-radius, -radius),  # 3: bottom-left
        (radius, -radius),   # 4: bottom-right
    ]
    idx = min(drop_index - 1, len(grid) - 1)
    return np.array(grid[idx], dtype=np.float64)


def pixel_depth_to_camera_frame(depth_image, segmap, camera_info) -> np.ndarray:
    fx = camera_info[0]
    fy = camera_info[4]
    cx = camera_info[2]
    cy = camera_info[5]

    # u, v, z = float(center_x), float(center_y), center_z_m
    # x_cam = (u - cx) * z / fx
    # y_cam = (v - cy) * z / fy
    # return np.array([x_cam, y_cam, z], dtype=np.float32)
    # fx, fy = 457.0073621574767, 342.75552161810754
    # cx, cy = 320.0, 240.0
    depths_at_mask = depth_image[segmap]
    rows, cols = np.where(segmap)

    # After: rows, cols = np.where(segmap)
    height = depth_image.shape[0]
    # rows_flipped = height - 1 - rows  # convert to top-left origin

    z = depths_at_mask
    x_cam = (cols - cx) * z / fx
    y_cam = (rows - cy) * z / fy
    # y_cam = (rows_flipped - cy) * z / fy  # use rows_flipped instead of rows
    points_cam = np.stack([x_cam, y_cam, z], axis=-1)
    # Use median for robustness
    center_cam = np.median(points_cam, axis=0)
    pos_camera = center_cam
    return pos_camera

def pixel_depth_to_camera_frame_llm(depth_image, location_y, location_x, camera_info) -> np.ndarray:
    fx = camera_info[0]
    fy = camera_info[4]
    cx = camera_info[2]
    cy = camera_info[5]

    # u, v, z = float(center_x), float(center_y), center_z_m
    # x_cam = (u - cx) * z / fx
    # y_cam = (v - cy) * z / fy
    # return np.array([x_cam, y_cam, z], dtype=np.float32)
    # fx, fy = 457.0073621574767, 342.75552161810754
    # cx, cy = 320.0, 240.0
    depths_at_mask = depth_image[location_y, location_x]
    rows, cols = location_y, location_x

    # After: rows, cols = np.where(segmap)
    height = depth_image.shape[0]
    # rows_flipped = height - 1 - rows  # convert to top-left origin

    z = depths_at_mask
    x_cam = (cols - cx) * z / fx
    y_cam = (rows - cy) * z / fy
    # y_cam = (rows_flipped - cy) * z / fy  # use rows_flipped instead of rows
    # import pdb; pdb.set_trace()
    points_cam = np.stack([x_cam, y_cam, z], axis=-1)
    # Use median for robustness
    # center_cam = np.median(points_cam, axis=0)
    pos_camera = points_cam
    return pos_camera


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


def euler_closest_to_ref(euler_target, euler_ref):
    """
    Return an Euler triple equivalent to ``euler_target`` that produces the
    shortest physical orientation path when linearly interpolating XYZ-Euler
    angles from ``euler_ref``.

    A pure SO(3) start-to-end angular distance cannot distinguish equivalent
    Euler branches because they represent the same final orientation. Instead,
    we evaluate nearby equivalent Euler representations of the target and score
    each by the cumulative physical rotation along the interpolated Euler path.
    This favors the branch that moves through the smallest overall 3D rotation,
    while still preferring numerically close roll/pitch/yaw changes.
    """
    target = np.asarray(euler_target, dtype=np.float64)
    ref = np.asarray(euler_ref, dtype=np.float64)
    target_q = np.asarray(p.getQuaternionFromEuler(target.tolist()), dtype=np.float64)
    two_pi = 2.0 * np.pi

    branches = (
        target,
        np.array([target[0] + np.pi, np.pi - target[1], target[2] + np.pi], dtype=np.float64),
    )

    best = None
    best_key = None

    def quat_angle(q_a, q_b):
        q_err = p.getDifferenceQuaternion(q_a.tolist(), q_b.tolist())
        return 2.0 * np.arctan2(np.linalg.norm(q_err[:3]), abs(q_err[3]))

    def path_metrics(candidate):
        prev_q = np.asarray(p.getQuaternionFromEuler(ref.tolist()), dtype=np.float64)
        path_len = 0.0
        max_step = 0.0
        for alpha in np.linspace(0.0, 1.0, 33)[1:]:
            e = ref + alpha * (candidate - ref)
            q = np.asarray(p.getQuaternionFromEuler(e.tolist()), dtype=np.float64)
            step = quat_angle(prev_q, q)
            path_len += step
            max_step = max(max_step, step)
            prev_q = q
        return path_len, max_step

    for branch in branches:
        centered = np.array([
            branch[i] + two_pi * np.round((ref[i] - branch[i]) / two_pi)
            for i in range(3)
        ], dtype=np.float64)

        for dr in (-two_pi, 0.0, two_pi):
            for dp in (-two_pi, 0.0, two_pi):
                for dy in (-two_pi, 0.0, two_pi):
                    candidate = centered + np.array([dr, dp, dy], dtype=np.float64)
                    candidate_q = np.asarray(
                        p.getQuaternionFromEuler(candidate.tolist()),
                        dtype=np.float64,
                    )

                    # Keep only candidates representing the same physical target
                    # orientation; branch generation near singularities can
                    # otherwise produce numerically close but inequivalent poses.
                    if quat_angle(candidate_q, target_q) > 1e-6:
                        continue

                    delta = candidate - ref
                    abs_delta = np.abs(delta)
                    path_len, max_step = path_metrics(candidate)
                    key = (
                        float(path_len),        # shortest physical interpolated path
                        float(max_step),        # avoid large sudden rotation jumps
                        float(np.max(abs_delta)),
                        float(np.sum(abs_delta)),
                        float(np.sum(delta ** 2)),
                    )
                    if best_key is None or key < best_key:
                        best = candidate
                        best_key = key

    if best is None:
        best = ref + np.arctan2(np.sin(target - ref), np.cos(target - ref))

    return best.astype(np.float32)


import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp


def wrap_to_pi(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def xyzrpy_to_matrix(pose_xyzrpy):
    pose_xyzrpy = np.asarray(pose_xyzrpy, dtype=np.float64).reshape(6)
    T = np.eye(4)
    T[:3, :3] = R.from_euler("xyz", pose_xyzrpy[3:6]).as_matrix()
    T[:3, 3] = pose_xyzrpy[:3]
    return T


def matrix_to_xyzrpy(T):
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    xyz = T[:3, 3]
    rpy = R.from_matrix(T[:3, :3]).as_euler("xyz")
    rpy = np.array([wrap_to_pi(a) for a in rpy], dtype=np.float64)
    return np.concatenate([xyz, rpy])


def rotation_angle_between_matrices(R_a, R_b):
    R_a = np.asarray(R_a, dtype=np.float64).reshape(3, 3)
    R_b = np.asarray(R_b, dtype=np.float64).reshape(3, 3)
    R_rel = R_a.T @ R_b
    trace_val = np.clip((np.trace(R_rel) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.arccos(trace_val))


def choose_equivalent_grasp_rotation(
    ref_rotation,
    target_rotation,
    symmetry_axis="z",
    candidate_angles=(0.0, np.pi),
    max_angle=np.pi / 2.0,
):
    """
    Choose the symmetry-equivalent target rotation that is closest to the
    reference rotation.

    The symmetry is applied in the target's local frame, which is appropriate
    for parallel-jaw grasps where a 180 deg roll around the approach axis is
    physically equivalent.

    Returns:
        best_rotation: (3, 3) target rotation after local symmetry adjustment
        best_angle: angular distance from ref_rotation in radians
        within_limit: True if a candidate was found with angle <= max_angle
    """
    ref_rotation = np.asarray(ref_rotation, dtype=np.float64).reshape(3, 3)
    target_rotation = np.asarray(target_rotation, dtype=np.float64).reshape(3, 3)

    axis_map = {
        "x": np.array([1.0, 0.0, 0.0], dtype=np.float64),
        "y": np.array([0.0, 1.0, 0.0], dtype=np.float64),
        "z": np.array([0.0, 0.0, 1.0], dtype=np.float64),
    }
    if symmetry_axis not in axis_map:
        raise ValueError(f"Invalid symmetry_axis: {symmetry_axis}")

    local_axis = axis_map[symmetry_axis]
    candidates = []
    for angle in candidate_angles:
        R_sym = R.from_rotvec(local_axis * float(angle)).as_matrix()
        R_candidate = target_rotation @ R_sym
        ang = rotation_angle_between_matrices(ref_rotation, R_candidate)
        candidates.append((ang, float(angle), R_candidate))

    within_limit = [item for item in candidates if item[0] <= float(max_angle)]
    best_pool = within_limit if within_limit else candidates
    best_angle, _, best_rotation = min(best_pool, key=lambda item: item[0])
    return best_rotation, float(best_angle), bool(within_limit)


def shortest_rotation_path_xyzrpy(PA, PB, num_steps=50):
    """
    Generate a path from PA to PB with shortest physical rotation.

    Args:
        PA: [x, y, z, roll, pitch, yaw]
        PB: [x, y, z, roll, pitch, yaw]
        num_steps: number of waypoints

    Returns:
        path: (num_steps, 6) array, each row is [x, y, z, roll, pitch, yaw]
    """
    import pdb; pdb.set_trace()
    PA = np.asarray(PA, dtype=np.float64).reshape(6)
    PB = np.asarray(PB, dtype=np.float64).reshape(6)

    # Position: linear interpolation
    pos_path = np.linspace(PA[:3], PB[:3], num_steps)

    # Orientation: shortest-path quaternion interpolation
    rA = R.from_euler("xyz", PA[3:6])
    rB = R.from_euler("xyz", PB[3:6])

    qA = rA.as_quat()  # [x, y, z, w]
    qB = rB.as_quat()

    # Force same hemisphere so SLERP uses shortest rotation
    if np.dot(qA, qB) < 0.0:
        qB = -qB
        rB = R.from_quat(qB)

    key_times = [0.0, 1.0]
    key_rots = R.concatenate([rA, rB])
    slerp = Slerp(key_times, key_rots)

    ts = np.linspace(0.0, 1.0, num_steps)
    rot_path = slerp(ts)

    # Convert back to xyzrpy
    path = np.zeros((num_steps, 6), dtype=np.float64)
    path[:, :3] = pos_path
    path[:, 3:6] = rot_path.as_euler("xyz")

    # Wrap angles for readability
    path[:, 3:6] = np.vectorize(wrap_to_pi)(path[:, 3:6])

    return path

def shortest_yaw(begin_yaw: float, target_yaw: float) -> float:
    """
    Return an equivalent target_yaw that minimizes the rotation from begin_yaw.
    Wraps the angular difference to (-π, π].
    """
    diff = target_yaw - begin_yaw
    diff_wrapped = np.arctan2(np.sin(diff), np.cos(diff))
    return float(begin_yaw + diff_wrapped)

def get_robot_pose_from_tcp(robot) -> np.ndarray:
    pos, quat = robot.get_eef()
    euler = p.getEulerFromQuaternion(quat)
    gripper_width = robot.get_gripper_width()
    return np.array([pos[0], pos[1], pos[2], euler[0], euler[1], euler[2], gripper_width], dtype=np.float32)

def is_vertical_grasp(grasp, table_normal=(0, 0, 1), min_alignment=0.9):
    """
    Check if grasp approach is vertical to table (top-down).
    table_normal: world Z (0,0,1) if Z-up; (0,0,-1) if Z-down.
    min_alignment: |dot(approach, table_normal)| must be >= this (0.9 ≈ 25°).
    """
    R = grasp.rotation_matrix  # shape (3, 3)
    # Approach direction: typically -R[:, 2] or R[:, 2] depending on convention
    approach = -R[:, 2]  # gripper approaches along -Z in grasp frame
    table_n = np.array(table_normal, dtype=np.float64)
    table_n = table_n / np.linalg.norm(table_n)
    alignment = np.abs(np.dot(approach, table_n))
    return alignment >= min_alignment

def is_vertical_facing_down_grasp(
    grasp,
    table_normal=(0, 0, 1),
    min_alignment=0.9,
    approach_axis="x",
):
    """
    Check if grasp is vertical to table AND facing down (toward table).

    For AnyGrasp, the approach direction is typically the gripper +X axis.
    If your model uses another convention, set approach_axis to one of:
      "x", "-x", "y", "-y", "z", "-z".
    """
    R = grasp.rotation_matrix  # shape (3, 3)
    axis_map = {
        "x": R[:, 0],
        "-x": -R[:, 0],
        "y": R[:, 1],
        "-y": -R[:, 1],
        "z": R[:, 2],
        "-z": -R[:, 2],
    }
    if approach_axis not in axis_map:
        raise ValueError(f"Invalid approach_axis: {approach_axis}")
    approach = axis_map[approach_axis]

    table_n = np.array(table_normal, dtype=np.float64)
    table_n = table_n / np.linalg.norm(table_n)

    # Downward direction points into the table.
    down = -table_n

    # Vertical: approach parallel to table normal.
    alignment_vertical = np.abs(np.dot(approach, table_n))
    # Facing down: approach aligned with downward direction.
    alignment_down = np.dot(approach, down)

    return alignment_vertical >= min_alignment and alignment_down >= min_alignment


def is_parallel_to_table_grasp(
    grasp,
    table_normal=(0, 0, 1),
    max_normal_alignment=0.25,
    approach_axis="x",
):
    """
    Check if grasp approach is parallel to table surface (side grasp).

    Condition:
      abs(dot(approach, table_normal)) <= max_normal_alignment

    Smaller max_normal_alignment means stricter "parallel to table":
      0.25 -> within ~14.5 deg from table plane
      0.35 -> within ~20.5 deg from table plane
    """
    R = np.asarray(grasp.rotation_matrix, dtype=np.float64).reshape(3, 3)
    axis_map = {
        "x": R[:, 0],
        "-x": -R[:, 0],
        "y": R[:, 1],
        "-y": -R[:, 1],
        "z": R[:, 2],
        "-z": -R[:, 2],
    }
    if approach_axis not in axis_map:
        raise ValueError(f"Invalid approach_axis: {approach_axis}")
    approach = np.asarray(axis_map[approach_axis], dtype=np.float64).reshape(3)
    approach_norm = float(np.linalg.norm(approach))
    if approach_norm < 1e-9:
        return False
    approach = approach / approach_norm

    table_n = np.array(table_normal, dtype=np.float64)
    table_norm = float(np.linalg.norm(table_n))
    if table_norm < 1e-9:
        raise ValueError("table_normal must be non-zero.")
    table_n = table_n / table_norm

    thr = float(np.clip(max_normal_alignment, 0.0, 1.0))

    normal_alignment = np.abs(np.dot(approach, table_n))
    return float(normal_alignment) <= thr


def select_center_grasp(
    grasps,
    points=None,
    center=None,
    top_k=None,
    use_median_center=True,
    return_index=False,
):
    """
    Select the grasp whose translation is closest to object center.

    Args:
        grasps: AnyGrasp grasp collection (supports len/indexing) or list.
        points: (N, 3) object point cloud in same frame as grasp.translation.
        center: Optional explicit center (3,). If set, overrides points.
        top_k: If set, search only first top_k grasps (after score sorting).
        use_median_center: Use median(points) for robust center; else mean(points).
        return_index: If True, return (best_grasp, best_index, center_vec).
    """
    if len(grasps) == 0:
        if return_index:
            return None, -1, None
        return None

    if center is None:
        if points is None:
            raise ValueError("Provide either `center` or `points`.")
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if pts.shape[0] == 0:
            raise ValueError("`points` is empty.")
        center_vec = np.median(pts, axis=0) if use_median_center else np.mean(pts, axis=0)
    else:
        center_vec = np.asarray(center, dtype=np.float64).reshape(3)

    n = len(grasps) if top_k is None else min(int(top_k), len(grasps))
    best_idx = 0
    best_dist = float("inf")
    for i in range(n):
        g = grasps[i]
        t = np.asarray(g.translation, dtype=np.float64).reshape(3)
        d = float(np.linalg.norm(t - center_vec))
        if d < best_dist:
            best_dist = d
            best_idx = i

    best_grasp = grasps[best_idx]
    if return_index:
        return best_grasp, best_idx, center_vec
    return best_grasp

def is_top_down(grasp, threshold_deg=20):
    approach = grasp.rotation_matrix[:, 2]  # 取 approach 方向
    z_axis = np.array([0, 0, -1])  # 向下

    cos = np.dot(approach, z_axis)
    angle = np.arccos(cos) * 180 / np.pi

    return angle < threshold_deg


class SimpleGrasp:
    def __init__(self, translation, rotation_matrix, score=0.0):
        self.translation = np.asarray(translation, dtype=np.float64).reshape(3)
        self.rotation_matrix = np.asarray(rotation_matrix, dtype=np.float64).reshape(3, 3)
        self.score = float(score)


def anygrasp_demo(colors, depths, segmap, step_id, cfgs): #camera_intrinsics, scale,
    if os.environ.get("ANYGRASP_IN_PROCESS", "0") != "1":
        return anygrasp_demo_subprocess(colors, depths, segmap, cfgs)
    return anygrasp_demo_in_process(colors, depths, segmap, step_id, cfgs)


def anygrasp_demo_subprocess(colors, depths, segmap, cfgs, timeout_s: float = 180.0):
    worker_path = Path(__file__).resolve().parents[1] / "anygrasp_worker.py"
    if not worker_path.exists():
        raise FileNotFoundError(f"AnyGrasp worker not found: {worker_path}")

    with tempfile.TemporaryDirectory(prefix="anygrasp_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        input_path = tmpdir_path / "input.npz"
        cfg_path = tmpdir_path / "cfg.pkl"
        output_path = tmpdir_path / "grasp.npz"

        np.savez(
            input_path,
            colors=np.asarray(colors),
            depths=np.asarray(depths),
            segmap=np.asarray(segmap, dtype=bool),
        )
        with open(cfg_path, "wb") as f:
            pickle.dump(cfgs, f)

        cmd = [
            sys.executable,
            str(worker_path),
            "--input-npz",
            str(input_path),
            "--cfg-pkl",
            str(cfg_path),
            "--output-npz",
            str(output_path),
        ]
        env = os.environ.copy()
        env.setdefault("WANDB_MODE", "disabled")
        env.setdefault("WANDB_SILENT", "true")

        print("anygrasp subprocess starting", flush=True)
        try:
            result = subprocess.run(
                cmd,
                env=env,
                text=True,
                capture_output=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            print(f"AnyGrasp subprocess timed out after {timeout_s:.1f}s", flush=True)
            return None

        if result.stdout:
            print(result.stdout, flush=True)
        if result.returncode != 0:
            print(f"AnyGrasp subprocess failed with code {result.returncode}", flush=True)
            if result.stderr:
                print(result.stderr, flush=True)
            return None

        if not output_path.exists():
            print("AnyGrasp subprocess did not produce a grasp.", flush=True)
            return None

        payload = np.load(output_path)
        return SimpleGrasp(
            translation=payload["translation"],
            rotation_matrix=payload["rotation_matrix"],
            score=float(payload["score"]),
        )


def anygrasp_demo_in_process(colors, depths, segmap, step_id, cfgs): #camera_intrinsics, scale,
    from gsnet import AnyGrasp
    import open3d as o3d

    anygrasp = AnyGrasp(cfgs)
    anygrasp.load_net()

    # # get data
    # colors = np.array(Image.open(rgb_path), dtype=np.float32) / 255.0
    # depths = np.array(Image.open(depth_path))

    # get camera intrinsics
    fx, fy = 342.7555, 342.7555
    cx, cy = 320, 240
    scale = 1.0

    # # get camera intrinsics
    # fx, fy = camera_intrinsics[0][0], camera_intrinsics[1][1]
    # cx, cy = camera_intrinsics[0][2], camera_intrinsics[1][2]
    # scale = 1000.0 / scale

    # # set workspace to filter output grasps
    # xmin, xmax = -0.6, 0.6
    # ymin, ymax = -0.3, 0.3
    # zmin, zmax = 0.0, 1.0
    # lims = [xmin, xmax, ymin, ymax, zmin, zmax]
    # import pdb; pdb.set_trace()
    # get point cloud
    xmap, ymap = np.arange(depths.shape[1]), np.arange(depths.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depths / scale
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z

    # set your workspace to crop point cloud
    # mask_z = (points_z > 0) & (points_z < 1)
    # mask = mask_z & segmap
    mask = segmap
    points = np.stack([points_x, points_y, points_z], axis=-1)
    # import pdb; pdb.set_trace()
    points = points[mask].astype(np.float32)
    colors = colors[mask].astype(np.float32)
    print(points.min(axis=0), points.max(axis=0))

    xmin, ymin, zmin = np.min(points, axis=0)
    xmax, ymax, zmax = np.max(points, axis=0)
    margin = 0.01  # 1 cm
    lims = [xmin - margin, xmax + margin, ymin - margin, ymax + margin, zmin - margin, zmax + margin]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    xmin, xmax, ymin, ymax, zmin, zmax = lims
    min_bound = (xmin, ymin, zmin)
    max_bound = (xmax, ymax, zmax)
    
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)
    bbox.color = (1.0, 0.0, 0.0)  # red box
    # o3d.visualization.draw_geometries([pcd, bbox],window_name="Point Cloud with Lims Box")

    # import pdb; pdb.set_trace()
    gg, cloud = anygrasp.get_grasp(points, colors, lims=lims, apply_object_mask=True, dense_grasp=False, collision_detection=False)

    if len(gg) == 0:
        print('No Grasp detected after collision detection!')

    gg = gg.nms().sort_by_score()
    gg_pick = gg[0:30]
    print(gg_pick.scores)
    print('best grasp score:', gg_pick[0].score)
    print('grasp score:', gg_pick[0].score)
    # import pdb; pdb.set_trace()

    # # Keep only side grasps (approach parallel to table surface).
    # parallel_mask = [is_vertical_facing_down_grasp(g) for g in gg_pick]
    # gg_parallel = gg_pick[parallel_mask]
    # print(f"  Parallel grasps found: {len(gg_parallel)}")

    # Prefer a top-down grasp whose translation is also closest to the object
    # center estimated from the segmented point cloud. This reduces sideways
    # bias compared with picking only the highest-scoring grasp.
    # candidate_grasps = gg_parallel if len(gg_parallel) > 0 else gg_pick
    # best_grasp = select_center_grasp(candidate_grasps, points=points)
    best_grasp = select_center_grasp(gg_pick, points=points)

    # import pdb; pdb.set_trace()

    # visualization
    if cfgs.debug:
        # Visualize the raw AnyGrasp pose, because this is the exact pose that is
        # later transformed into the robot frame and commanded to IK.
        colors = np.tile(np.array([[0.0, 1.0, 0.0]]), (np.asarray(cloud.points).shape[0], 1))
        cloud.colors = o3d.utility.Vector3dVector(colors)

        best_geom = best_grasp.to_open3d_geometry()
        best_geom.paint_uniform_color([0.0, 0.0, 1.0])
        o3d.visualization.draw_geometries([best_geom, cloud])

        o3d.io.write_point_cloud(str(step_id)+"_cloud_colored.ply", cloud)
    return best_grasp #gg_pick[0] #best_grasp

def anygrasp_demo_exp(colors, depths, segmap, step_id, cfgs): #camera_intrinsics, scale, 
    from gsnet import AnyGrasp
    import open3d as o3d

    anygrasp = AnyGrasp(cfgs)
    anygrasp.load_net()

    # # get data
    # colors = np.array(Image.open(rgb_path), dtype=np.float32) / 255.0
    # depths = np.array(Image.open(depth_path))

    # get camera intrinsics
    fx, fy = 430.3013610839844, 429.9326171875
    cx, cy = 417.06976318359375, 245.400390625
    scale = 1.0

    # # get camera intrinsics
    # fx, fy = camera_intrinsics[0][0], camera_intrinsics[1][1]
    # cx, cy = camera_intrinsics[0][2], camera_intrinsics[1][2]
    # scale = 1000.0 / scale

    # # set workspace to filter output grasps
    # xmin, xmax = -0.6, 0.6
    # ymin, ymax = -0.3, 0.3
    # zmin, zmax = 0.0, 1.0
    # lims = [xmin, xmax, ymin, ymax, zmin, zmax]
    # import pdb; pdb.set_trace()
    # get point cloud
    xmap, ymap = np.arange(depths.shape[1]), np.arange(depths.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depths / scale
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z

    # set your workspace to crop point cloud
    # mask_z = (points_z > 0) & (points_z < 1)
    # mask = mask_z & segmap
    mask = segmap
    points = np.stack([points_x, points_y, points_z], axis=-1)
    # import pdb; pdb.set_trace()
    points = points[mask].astype(np.float32)
    colors = colors[mask].astype(np.float32)
    print(points.min(axis=0), points.max(axis=0))

    xmin, ymin, zmin = np.min(points, axis=0)
    xmax, ymax, zmax = np.max(points, axis=0)
    margin = 0.01  # 1 cm
    lims = [xmin - margin, xmax + margin, ymin - margin, ymax + margin, zmin - margin, zmax + margin]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    xmin, xmax, ymin, ymax, zmin, zmax = lims
    min_bound = (xmin, ymin, zmin)
    max_bound = (xmax, ymax, zmax)
    
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)
    bbox.color = (1.0, 0.0, 0.0)  # red box
    # o3d.visualization.draw_geometries([pcd, bbox],window_name="Point Cloud with Lims Box")

    # import pdb; pdb.set_trace()
    gg, cloud = anygrasp.get_grasp(points, colors, lims=lims, apply_object_mask=True, dense_grasp=False, collision_detection=True)

    if len(gg) == 0:
        print('No Grasp detected after collision detection!')

    gg = gg.nms().sort_by_score()
    gg_pick = gg[0:30]
    print(gg_pick.scores)
    print('best grasp score:', gg_pick[0].score)
    print('grasp score:', gg_pick[0].score)
    # import pdb; pdb.set_trace()

    # # Keep only side grasps (approach parallel to table surface).
    # parallel_mask = [is_vertical_facing_down_grasp(g) for g in gg_pick]
    # gg_parallel = gg_pick[parallel_mask]
    # print(f"  Parallel grasps found: {len(gg_parallel)}")

    # Prefer a top-down grasp whose translation is also closest to the object
    # center estimated from the segmented point cloud. This reduces sideways
    # bias compared with picking only the highest-scoring grasp.
    # candidate_grasps = gg_parallel if len(gg_parallel) > 0 else gg_pick
    # best_grasp = select_center_grasp(candidate_grasps, points=points)
    # best_grasp = select_center_grasp(gg_pick, points=points)
    best_grasp = gg_pick[0]
    # candidate_grasps = [g for g in gg_pick if is_top_down(g)]
    # if len(candidate_grasps) == 0:
    #     best_grasp = gg_pick[0]
    # else:
    #     best_grasp = select_center_grasp(candidate_grasps, points=points)

    # import pdb; pdb.set_trace()

    # visualization
    if cfgs.debug:
        # Visualize the raw AnyGrasp pose, because this is the exact pose that is
        # later transformed into the robot frame and commanded to IK.
        colors = np.tile(np.array([[0.0, 1.0, 0.0]]), (np.asarray(cloud.points).shape[0], 1))
        cloud.colors = o3d.utility.Vector3dVector(colors)
        grippers = gg.to_open3d_geometry_list()
        

        best_geom = best_grasp.to_open3d_geometry()
        best_geom.paint_uniform_color([0.0, 0.0, 1.0])
        o3d.visualization.draw_geometries([best_geom, cloud])
        # o3d.visualization.draw_geometries([*grippers, cloud])


        o3d.io.write_point_cloud(str(step_id)+"_cloud_colored.ply", cloud)
    return best_grasp #gg_pick[0] #best_grasp

def grasp_to_xyzrpy(grasp):
    t = grasp.translation    # shape (3,)
    R = grasp.rotation_matrix       # shape (3, 3)
    T = np.eye(4)          # 4x4 identity
    T[:3, :3] = R          # top-left: rotation
    T[:3, 3]  = t          # top-right: translation
    return pose4x4_to_xyzrpy(T)

def make_pour_pose_from_current(
    current_pose,
    target_xyz,
    height_offset=0.12,
    tilt_angle_deg=50.0,
    tilt_axis="pitch",   # "roll" or "pitch"
):
    """
    current_pose: [x, y, z, roll, pitch, yaw]
    target_xyz: [x_t, y_t, z_t] 目标容器中心位置
    """
    current_pose = np.asarray(current_pose, dtype=np.float64).copy()
    target_xyz = np.asarray(target_xyz, dtype=np.float64).reshape(3)

    pour_pose = current_pose.copy()

    # 1) 先移动到目标上方
    pour_pose[0] = target_xyz[0]
    pour_pose[1] = target_xyz[1]
    pour_pose[2] = target_xyz[2] + height_offset

    # 2) 在当前姿态基础上增加倾倒角
    tilt = np.deg2rad(tilt_angle_deg)

    if tilt_axis == "roll":
        pour_pose[3] = current_pose[3] + tilt
    elif tilt_axis == "pitch":
        pour_pose[4] = current_pose[4] + tilt
    else:
        raise ValueError("tilt_axis must be 'roll' or 'pitch'")

    return pour_pose

def check_rpy(roll, pitch, yaw):
    Rot = R.from_euler("xyz", [roll, pitch, yaw]).as_matrix()
    print("x-axis:", np.round(Rot[:, 0], 4))
    print("y-axis:", np.round(Rot[:, 1], 4))
    print("z-axis:", np.round(Rot[:, 2], 4))
    
__all__ = [
    "sam3_inference",
    "DMP",
    "DiffusionPolicy",
    "pixel_depth_to_camera_frame",
    "world_to_camera_frame",
    "world_to_robot_frame",
    "pose4x4_to_rpy",
    "pose4x4_to_xyzrpy",
    "get_robot_pose_from_tcp",
    "is_parallel_to_table_grasp",
    "select_center_grasp",
]
