import argparse
from pathlib import Path
from typing import List

import numpy as np


def load_trajectories(data_path: str) -> np.ndarray:
    """
    Load trajectories as (N, T, 6).
    Supports:
    - .npy single demo (T,6)
    - .npz with key 'trajectories' (N,T,6)
    - directory with .npy/.csv files, each (T,6)
    """
    p = Path(data_path)
    if p.is_file() and p.suffix == ".npy":
        arr = np.asarray(np.load(p, allow_pickle=True), dtype=np.float32)
        if arr.ndim != 2 or arr.shape[-1] != 6:
            raise ValueError(f"Expected (T,6), got {arr.shape}")
        return arr[None, ...]
    if p.is_file() and p.suffix == ".npz":
        data = np.load(p, allow_pickle=True)
        if "trajectories" not in data:
            raise KeyError("NPZ must contain key 'trajectories' with shape (N,T,6)")
        arr = np.asarray(data["trajectories"], dtype=np.float32)
        if arr.ndim != 3 or arr.shape[-1] != 6:
            raise ValueError(f"Expected (N,T,6), got {arr.shape}")
        return arr
    if p.is_dir():
        items: List[np.ndarray] = []
        for f in sorted(p.iterdir()):
            if f.suffix == ".npy":
                arr = np.asarray(np.load(f, allow_pickle=True), dtype=np.float32)
            elif f.suffix == ".csv":
                arr = np.asarray(np.loadtxt(f, delimiter=","), dtype=np.float32)
            else:
                continue
            if arr.ndim == 2 and arr.shape[-1] == 6:
                items.append(arr)
        if not items:
            raise ValueError("No valid (T,6) trajectory files found.")
        t_min = min(x.shape[0] for x in items)
        items = [x[:t_min] for x in items]
        return np.stack(items, axis=0).astype(np.float32)
    raise ValueError("Unsupported data_path format.")


def make_gaussian_basis(T: int, n_basis: int = 50) -> np.ndarray:
    t = np.linspace(0.0, 1.0, T, dtype=np.float32)
    centers = np.linspace(0.0, 1.0, n_basis, dtype=np.float32)
    spacing = 1.0 / max(1, n_basis - 1)
    sigma = 0.8 * spacing
    phi = np.exp(-0.5 * ((t[:, None] - centers[None, :]) / max(1e-6, sigma)) ** 2)
    phi /= (phi.sum(axis=1, keepdims=True) + 1e-8)
    return phi.astype(np.float32)


def augment_single_demo(
    demo: np.ndarray,
    n_augmented: int = 10000,
    n_basis: int = 50,
    coeff_noise_std: float = 0.10,
    endpoint_noise_std: float = 0.002,
    seed: int = 42,
) -> np.ndarray:
    if demo.ndim != 2 or demo.shape[-1] != 6:
        raise ValueError(f"demo must be (T,6), got {demo.shape}")
    T, dof = demo.shape
    phi = make_gaussian_basis(T, n_basis=n_basis)
    pinv = np.linalg.pinv(phi)
    W = pinv @ demo

    rng = np.random.default_rng(seed)
    out = [demo.astype(np.float32)]
    for _ in range(n_augmented):
        W_noise = rng.normal(0.0, coeff_noise_std, size=W.shape).astype(np.float32)
        W_new = W * (1.0 + W_noise)
        traj = (phi @ W_new).astype(np.float32)
        traj[0] = demo[0] + rng.normal(0.0, endpoint_noise_std, size=(dof,)).astype(np.float32)
        traj[-1] = demo[-1] + rng.normal(0.0, endpoint_noise_std, size=(dof,)).astype(np.float32)
        out.append(traj)
    return np.stack(out, axis=0).astype(np.float32)


def write_zarr_for_pusht_dataset(trajectories: np.ndarray, zarr_path: Path) -> None:
    """
    Create zarr compatible with diffusion_policy.dataset.pusht_dataset.PushTLowdimDataset.
    Required arrays:
    - data/keypoint: (total_steps, 9, 2)
    - data/state: (total_steps, 2)
    - data/action: (total_steps, 6)
    - meta/episode_ends: (N,)
    """
    try:
        import zarr
    except ImportError as exc:
        raise ImportError("Please install zarr first: pip install zarr") from exc

    N, T, _ = trajectories.shape
    total = N * T

    action = trajectories.reshape(total, 6).astype(np.float32)
    state = trajectories[:, :, :2].reshape(total, 2).astype(np.float32)

    # Build 18-dim keypoint feature (9x2), placing xyzrpy in the first 6 slots.
    keypoint_flat = np.zeros((total, 18), dtype=np.float32)
    keypoint_flat[:, :6] = action
    keypoint = keypoint_flat.reshape(total, 9, 2)

    episode_ends = (np.arange(1, N + 1, dtype=np.int64) * T).astype(np.int64)

    zarr_path.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(zarr_path), mode="w")
    data_grp = root.create_group("data")
    meta_grp = root.create_group("meta")
    data_grp.create_dataset("keypoint", shape=keypoint.shape, dtype="f4", data=keypoint)
    data_grp.create_dataset("state", shape=state.shape, dtype="f4", data=state)
    data_grp.create_dataset("action", shape=action.shape, dtype="f4", data=action)
    meta_grp.create_dataset("episode_ends", shape=episode_ends.shape, dtype="i8", data=episode_ends)


def write_dummy_runner(dummy_runner_path: Path) -> None:
    content = """from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner


class DummyLowdimRunner(BaseLowdimRunner):
    def __init__(self, **kwargs):
        super().__init__(output_dir=kwargs.get("output_dir"))
        self.kwargs = kwargs

    def run(self, policy):
        _ = policy
        return {
            "test/mean_score": 0.0,
            "test_mean_score": 0.0
        }
"""
    dummy_runner_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_runner_path.write_text(content, encoding="utf-8")


def write_task_yaml(task_yaml_path: Path, zarr_path: Path, horizon: int) -> None:
    content = f"""name: custom_lowdim_xyzrpy

obs_dim: 20
action_dim: 6
keypoint_dim: 2

env_runner:
  _target_: llm_mani.dp_dummy_runner.DummyLowdimRunner

dataset:
  _target_: diffusion_policy.dataset.pusht_dataset.PushTLowdimDataset
  zarr_path: {zarr_path}
  horizon: ${{horizon}}
  pad_before: ${{eval:'${{n_obs_steps}}-1+${{n_latency_steps}}'}}
  pad_after: ${{eval:'${{n_action_steps}}-1'}}
  seed: 42
  val_ratio: 0.2
  max_train_episodes: null
"""
    task_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    task_yaml_path.write_text(content, encoding="utf-8")


def write_command_file(
    command_path: Path,
    dp_repo_path: Path,
    workspace_root: Path,
    task_name: str,
    horizon: int,
    n_obs_steps: int,
    n_action_steps: int,
    epochs: int,
) -> None:
    # diffusion_policy (original repo) often expects old diffusers/hf-hub APIs.
    # This pin avoids: ImportError: cannot import name 'cached_download' from huggingface_hub
    fix_cmd = (
        f'cd "{dp_repo_path}" && '
        "python -m pip install 'huggingface_hub<0.26' 'diffusers==0.11.1' 'numba>=0.59,<0.61'"
    )
    # Avoid xformers/flash-attn ABI/version conflicts in this environment.
    disable_xformers_cmd = (
        f'cd "{dp_repo_path}" && '
        "python -m pip uninstall -y xformers flash-attn || true"
    )
    cmd = (
        f'cd "{dp_repo_path}" && '
        f'PYTHONPATH="{workspace_root}:$PYTHONPATH" '
        "XFORMERS_DISABLED=1 "
        "HYDRA_FULL_ERROR=1 "
        f'python train.py '
        f'--config-name=train_diffusion_unet_lowdim_workspace '
        f'task={task_name} '
        f'horizon={horizon} '
        f'n_obs_steps={n_obs_steps} '
        f'n_action_steps={n_action_steps} '
        f'training.num_epochs={epochs} '
        f'training.device=cuda:0 '
        f'logging.mode=offline '
        f'checkpoint.topk.monitor_key=val_loss '
        f'checkpoint.topk.mode=min'
    )
    command_path.parent.mkdir(parents=True, exist_ok=True)
    command_path.write_text(
        "#!/usr/bin/env bash\nset -e\n\n"
        "# 1) fix common dependency mismatch for diffusion_policy\n"
        f"{fix_cmd}\n\n"
        "# 2) disable xformers/flash-attn to avoid import/runtime conflicts\n"
        f"{disable_xformers_cmd}\n\n"
        "# 3) launch training\n"
        f"{cmd}\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description="Prepare Stanford diffusion_policy lowdim training assets.")
    parser.add_argument("--data-path", type=str, required=True, help="Single demo .npy, trajectories .npz, or directory.")
    parser.add_argument("--dp-repo-path", type=str, required=True, help="Path to cloned real-stanford/diffusion_policy repo.")
    parser.add_argument("--out-dir", type=str, default="llm_mani/dp_prepare_out")
    parser.add_argument("--augment-count", type=int, default=10000)
    parser.add_argument("--n-basis", type=int, default=50)
    parser.add_argument("--coeff-noise-std", type=float, default=0.10)
    parser.add_argument("--endpoint-noise-std", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-obs-steps", type=int, default=2)
    parser.add_argument("--n-action-steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument(
        "--horizon-multiple",
        type=int,
        default=8,
        help="Trim trajectory length T to be divisible by this value for UNet down/up sampling.",
    )
    args = parser.parse_args()

    workspace_root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out_dir).resolve()
    dp_repo = Path(args.dp_repo_path).resolve()

    traj = load_trajectories(args.data_path)
    if traj.shape[0] == 1 and args.augment_count > 0:
        traj = augment_single_demo(
            demo=traj[0],
            n_augmented=args.augment_count,
            n_basis=args.n_basis,
            coeff_noise_std=args.coeff_noise_std,
            endpoint_noise_std=args.endpoint_noise_std,
            seed=args.seed,
        )

    # diffusion_policy's 1D UNet expects horizon compatible with downsample factors.
    # With default down dims, 8-divisible horizons avoid skip-connection mismatches.
    if args.horizon_multiple > 1:
        T_raw = traj.shape[1]
        T_trim = (T_raw // args.horizon_multiple) * args.horizon_multiple
        if T_trim < args.horizon_multiple:
            raise ValueError(
                f"Trajectory length {T_raw} is too short for horizon_multiple={args.horizon_multiple}."
            )
        if T_trim != T_raw:
            traj = traj[:, :T_trim, :]
            print(f"Trimmed horizon from {T_raw} to {T_trim} (multiple of {args.horizon_multiple}).")

    N, T, _ = traj.shape
    zarr_path = out_dir / "custom_lowdim_replay.zarr"
    write_zarr_for_pusht_dataset(traj, zarr_path)

    # Write runner module in this workspace
    dummy_runner_path = workspace_root / "llm_mani" / "dp_dummy_runner.py"
    write_dummy_runner(dummy_runner_path)

    # Write task config into diffusion_policy repo
    task_name = "custom_lowdim_xyzrpy"
    task_yaml = dp_repo / "diffusion_policy" / "config" / "task" / f"{task_name}.yaml"
    write_task_yaml(task_yaml, zarr_path, horizon=T)

    # Write convenient training command
    cmd_path = out_dir / "train_command.sh"
    write_command_file(
        command_path=cmd_path,
        dp_repo_path=dp_repo,
        workspace_root=workspace_root,
        task_name=task_name,
        horizon=T,
        n_obs_steps=args.n_obs_steps,
        n_action_steps=args.n_action_steps,
        epochs=args.epochs,
    )

    print("Preparation completed.")
    print(f"Total trajectories: {N}, horizon: {T}")
    print(f"Zarr dataset: {zarr_path}")
    print(f"Task config: {task_yaml}")
    print(f"Dummy runner: {dummy_runner_path}")
    print(f"Train command: {cmd_path}")


if __name__ == "__main__":
    main()


# python prepare_stanford_lowdim_training.py   --data-path /home/mhumais/Huang/DMP/VAE_DMP_mani/dataset/1_wiping_6d.npy   --dp-repo-path /home/mhumais/Huang/diffusion_policy   --augment-count 10000   --n-basis 50   --
# out-dir llm_mani/dp_prepare_out

# bash llm_mani/dp_prepare_out/train_command.sh