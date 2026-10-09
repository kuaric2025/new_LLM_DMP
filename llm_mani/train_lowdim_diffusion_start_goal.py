import argparse
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

def load_trajectories(data_path: str) -> np.ndarray:
    """
    Return trajectories with shape (N, T, 6).
    Supported:
    1) .npz file with key 'trajectories'
    2) directory containing .npy or .csv files, each one (T,6)
    """
    p = Path(data_path)
    if p.is_file() and p.suffix == ".npz":
        data = np.load(p, allow_pickle=True)
        if "trajectories" not in data:
            raise KeyError("NPZ must contain key 'trajectories' with shape (N,T,6).")
        traj = np.asarray(data["trajectories"], dtype=np.float32)
        if traj.ndim != 3 or traj.shape[-1] != 6:
            raise ValueError(f"Expected (N,T,6), got {traj.shape}")
        return traj
    if p.is_file() and p.suffix == ".npy":
        arr = np.asarray(np.load(p, allow_pickle=True), dtype=np.float32)
        if arr.ndim != 2 or arr.shape[-1] != 6:
            raise ValueError(f"Expected single trajectory (T,6), got {arr.shape}")
        return arr[None, ...]

    if p.is_dir():
        items: List[np.ndarray] = []
        for f in sorted(p.iterdir()):
            if f.suffix == ".npy":
                arr = np.asarray(np.load(f, allow_pickle=True), dtype=np.float32)
            elif f.suffix == ".csv":
                arr = np.asarray(np.loadtxt(f, delimiter=","), dtype=np.float32)
            else:
                continue
            if arr.ndim != 2 or arr.shape[-1] != 6:
                continue
            items.append(arr)
        if not items:
            raise ValueError("No valid (T,6) trajectory files found in directory.")
        t_min = min(x.shape[0] for x in items)
        items = [x[:t_min] for x in items]
        return np.stack(items, axis=0).astype(np.float32)

    raise ValueError("data_path must be .npz file or a directory.")


def make_gaussian_basis(T: int, n_basis: int = 50) -> np.ndarray:
    """Return Gaussian basis matrix Phi with shape (T, n_basis)."""
    t = np.linspace(0.0, 1.0, T, dtype=np.float32)
    centers = np.linspace(0.0, 1.0, n_basis, dtype=np.float32)
    spacing = 1.0 / max(1, n_basis - 1)
    sigma = 0.8 * spacing
    phi = np.exp(-0.5 * ((t[:, None] - centers[None, :]) / max(1e-6, sigma)) ** 2)
    phi /= (phi.sum(axis=1, keepdims=True) + 1e-8)
    return phi.astype(np.float32)


def augment_single_demo_with_rbf(
    demo: np.ndarray,
    n_augmented: int = 10_000,
    n_basis: int = 50,
    coeff_noise_std: float = 0.10,
    endpoint_noise_std: float = 0.002,
    seed: int = 42,
) -> np.ndarray:
    """
    From one (T,6) demo, generate n_augmented trajectories with Gaussian basis perturbation.
    Returns shape (n_augmented + 1, T, 6), including original demo as first sample.
    """
    if demo.ndim != 2 or demo.shape[-1] != 6:
        raise ValueError(f"demo must be (T,6), got {demo.shape}")
    T, dof = demo.shape
    phi = make_gaussian_basis(T=T, n_basis=n_basis)  # (T,K)
    pinv = np.linalg.pinv(phi)  # (K,T)

    # Fit basis coefficients for each dof: W shape (K,6)
    W = pinv @ demo

    rng = np.random.default_rng(seed)
    out = [demo.astype(np.float32)]
    for _ in range(n_augmented):
        W_noise = rng.normal(loc=0.0, scale=coeff_noise_std, size=W.shape).astype(np.float32)
        W_new = W * (1.0 + W_noise)
        traj = (phi @ W_new).astype(np.float32)  # (T,6)

        # Keep start/goal close to real demo while allowing small variation.
        traj[0] = demo[0] + rng.normal(0.0, endpoint_noise_std, size=(dof,)).astype(np.float32)
        traj[-1] = demo[-1] + rng.normal(0.0, endpoint_noise_std, size=(dof,)).astype(np.float32)
        out.append(traj)
    return np.stack(out, axis=0).astype(np.float32)


def split_train_infer(
    trajectories: np.ndarray, train_ratio: float = 0.8, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Split trajectories into train/infer sets with deterministic shuffle."""
    N = trajectories.shape[0]
    rng = np.random.default_rng(seed)
    idx = np.arange(N)
    rng.shuffle(idx)
    n_train = int(round(N * train_ratio))
    n_train = min(max(n_train, 1), N - 1)
    train = trajectories[idx[:n_train]]
    infer = trajectories[idx[n_train:]]
    return train.astype(np.float32), infer.astype(np.float32)


class TrajectoryDataset(Dataset):
    def __init__(self, trajectories: np.ndarray, mean: np.ndarray = None, std: np.ndarray = None):
        self.traj = trajectories
        if mean is None or std is None:
            flat = self.traj.reshape(-1, self.traj.shape[-1])
            self.mean = flat.mean(axis=0, keepdims=True).astype(np.float32)
            self.std = flat.std(axis=0, keepdims=True).astype(np.float32) + 1e-6
        else:
            self.mean = np.asarray(mean, dtype=np.float32).reshape(1, self.traj.shape[-1])
            self.std = np.asarray(std, dtype=np.float32).reshape(1, self.traj.shape[-1])
        self.traj_norm = ((self.traj - self.mean) / self.std).astype(np.float32)

    def __len__(self) -> int:
        return self.traj.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.traj_norm[idx]  # (T,6)
        cond = np.concatenate([x[0], x[-1]], axis=0)  # (12,)
        return torch.from_numpy(x), torch.from_numpy(cond)


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            torch.linspace(math.log(1.0), math.log(10000.0), half, device=t.device)
        )
        angles = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return self.mlp(emb)


class NoisePredictor(nn.Module):
    def __init__(self, traj_dim: int, cond_dim: int, t_dim: int = 128, hidden: int = 512):
        super().__init__()
        self.t_embed = TimeEmbedding(t_dim)
        self.net = nn.Sequential(
            nn.Linear(traj_dim + cond_dim + t_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, traj_dim),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        te = self.t_embed(t)
        h = torch.cat([x_t, cond, te], dim=-1)
        return self.net(h)


class DiffusionModel(nn.Module):
    def __init__(self, T: int, dof: int = 6, timesteps: int = 1000):
        super().__init__()
        self.T = T
        self.dof = dof
        self.dim = T * dof
        self.timesteps = timesteps

        self.model = NoisePredictor(traj_dim=self.dim, cond_dim=12)
        betas = torch.linspace(1e-4, 2e-2, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def q_sample(self, x0: torch.Tensor, t_idx: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        s1 = self.sqrt_alphas_cumprod[t_idx].unsqueeze(-1)
        s2 = self.sqrt_one_minus_alphas_cumprod[t_idx].unsqueeze(-1)
        return s1 * x0 + s2 * noise

    def training_loss(self, x0: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        b = x0.shape[0]
        x0 = x0.reshape(b, -1)
        t_idx = torch.randint(0, self.timesteps, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t_idx, noise)
        t = t_idx.float() / float(self.timesteps - 1)
        pred_noise = self.model(xt, t, cond)
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, steps: int = 1000) -> torch.Tensor:
        self.eval()
        b = cond.shape[0]
        x = torch.randn((b, self.dim), device=cond.device)
        steps = min(steps, self.timesteps)
        for i in reversed(range(steps)):
            t_idx = torch.full((b,), i, device=cond.device, dtype=torch.long)
            t = t_idx.float() / float(self.timesteps - 1)
            eps = self.model(x, t, cond)

            alpha = self.alphas[t_idx].unsqueeze(-1)
            alpha_bar = self.alphas_cumprod[t_idx].unsqueeze(-1)
            beta = self.betas[t_idx].unsqueeze(-1)

            mean = (1.0 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * eps)
            if i > 0:
                z = torch.randn_like(x)
                x = mean + torch.sqrt(beta) * z
            else:
                x = mean
        return x.reshape(b, self.T, self.dof)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    traj = load_trajectories(args.data_path)
    if traj.shape[0] == 1 and args.augment_count > 0:
        traj = augment_single_demo_with_rbf(
            demo=traj[0],
            n_augmented=args.augment_count,
            n_basis=args.n_basis,
            coeff_noise_std=args.coeff_noise_std,
            endpoint_noise_std=args.endpoint_noise_std,
            seed=args.seed,
        )

    train_traj, infer_traj = split_train_infer(traj, train_ratio=args.train_ratio, seed=args.seed)
    train_dataset = TrajectoryDataset(train_traj)
    infer_dataset = TrajectoryDataset(infer_traj, mean=train_dataset.mean, std=train_dataset.std)
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    infer_loader = DataLoader(infer_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = DiffusionModel(T=traj.shape[1], dof=6, timesteps=args.diffusion_steps).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    print(
        f"Loaded data: total={traj.shape[0]}, train={train_traj.shape[0]}, infer={infer_traj.shape[0]}, "
        f"shape=(T={traj.shape[1]},6), device={device}"
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for x, cond in loader:
            x = x.to(device)
            cond = cond.to(device)
            loss = model.training_loss(x, cond)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_losses.append(loss.item())

        model.eval()
        infer_losses = []
        with torch.no_grad():
            for x, cond in infer_loader:
                x = x.to(device)
                cond = cond.to(device)
                infer_losses.append(model.training_loss(x, cond).item())
        print(
            f"Epoch {epoch:04d} | train_loss={np.mean(train_losses):.6f} | "
            f"infer_loss={np.mean(infer_losses):.6f}"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "lowdim_diffusion_start_goal.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "mean": train_dataset.mean,
            "std": train_dataset.std,
            "T": traj.shape[1],
            "dof": 6,
            "diffusion_steps": args.diffusion_steps,
            "n_total": int(traj.shape[0]),
            "n_train": int(train_traj.shape[0]),
            "n_infer": int(infer_traj.shape[0]),
        },
        ckpt_path,
    )
    split_path = out_dir / "train_infer_split.npz"
    np.savez_compressed(
        split_path,
        train_trajectories=train_traj.astype(np.float32),
        infer_trajectories=infer_traj.astype(np.float32),
    )
    print(f"Saved checkpoint to: {ckpt_path}")
    print(f"Saved split dataset to: {split_path}")


@torch.no_grad()
def infer(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device)
    T = int(ckpt["T"])
    dof = int(ckpt["dof"])
    model = DiffusionModel(T=T, dof=dof, timesteps=int(ckpt["diffusion_steps"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    start = np.asarray(args.start, dtype=np.float32).reshape(6)
    goal = np.asarray(args.goal, dtype=np.float32).reshape(6)
    mean = np.asarray(ckpt["mean"], dtype=np.float32).reshape(1, 6)
    std = np.asarray(ckpt["std"], dtype=np.float32).reshape(1, 6)
    start_n = (start - mean[0]) / std[0]
    goal_n = (goal - mean[0]) / std[0]

    cond = torch.from_numpy(np.concatenate([start_n, goal_n], axis=0)).unsqueeze(0).to(device)
    sample_n = model.sample(cond=cond, steps=args.sample_steps)[0].cpu().numpy()  # (T,6)
    sample = sample_n * std + mean

    out_path = Path(args.sample_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, sample.astype(np.float32))
    print(f"Saved sampled trajectory to: {out_path}, shape={sample.shape}")


def parse_args():
    p = argparse.ArgumentParser(description="Train low-dim diffusion from start+goal(6D) to trajectory.")
    sub = p.add_subparsers(dest="mode", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--data-path", type=str, required=True, help="NPZ with trajectories(N,T,6) or folder of per-demo files.")
    p_train.add_argument("--out-dir", type=str, default="llm_mani/checkpoints")
    p_train.add_argument("--epochs", type=int, default=300)
    p_train.add_argument("--batch-size", type=int, default=64)
    p_train.add_argument("--lr", type=float, default=1e-4)
    p_train.add_argument("--diffusion-steps", type=int, default=1000)
    p_train.add_argument("--augment-count", type=int, default=10000, help="When only one demo exists, generate this many augmented trajectories.")
    p_train.add_argument("--n-basis", type=int, default=50, help="Number of Gaussian basis functions.")
    p_train.add_argument("--coeff-noise-std", type=float, default=0.10, help="RBF coefficient perturbation std.")
    p_train.add_argument("--endpoint-noise-std", type=float, default=0.002, help="Start/goal perturbation std.")
    p_train.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio.")
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--cpu", action="store_true")

    p_inf = sub.add_parser("infer")
    p_inf.add_argument("--ckpt", type=str, required=True)
    p_inf.add_argument("--start", type=float, nargs=6, required=True, help="x y z r p y")
    p_inf.add_argument("--goal", type=float, nargs=6, required=True, help="x y z r p y")
    p_inf.add_argument("--sample-steps", type=int, default=200)
    p_inf.add_argument("--sample-out", type=str, default="llm_mani/output/sample_traj.npy")
    p_inf.add_argument("--cpu", action="store_true")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "train":
        train(args)
    elif args.mode == "infer":
        infer(args)

# python train_lowdim_diffusion_start_goal.py train   --data-path /home/mhumais/Huang/DMP/VAE_DMP_mani/dataset/1_wiping_6d.npy   --augment-count 10000   --n-basis 50   --train-ratio 0.8   --out-dir llm_mani/checkpoints_wiping
