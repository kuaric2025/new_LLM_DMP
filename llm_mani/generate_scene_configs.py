"""Pre-generate fixed scene configurations for fair pipeline evaluation.

Each scene has fixed object poses (plate, banana, apple, orange, box) so that
different pipelines can be tested on the same 10 scenes for comparable results.

Usage:
  python generate_scene_configs.py --num-scenes 10 --seed 0 --output scene_configs.json
"""

import argparse
import json
import numpy as np
from pathlib import Path

# Default object spawn specs (must match manipulation_env/env.py)
DEFAULT_OBJECT_SPECS = {
    "plate": {"xy": [0.66, -0.30], "yaw": 0.0},
    "banana": {"xy": [0.6, -0.05], "yaw": 1.2},
    "apple": {"xy": [0.5, 0.1], "yaw": 0.0},
    "orange": {"xy": [0.4, 0.3], "yaw": 0.0},
}
DEFAULT_BOX_XY = [0.50, 0.25]

# Table bounds for clamping (aligned with ManiEnv workspace)
# x: [0.25, 0.90], y: [-0.45, 0.45] with margin 0.04
TABLE_BOUNDS = {"x": (0.25 + 0.04, 0.90 - 0.04), "y": (-0.45 + 0.04, 0.45 - 0.04)}

XY_OFFSET_RANGE = (-0.06, 0.06)
YAW_OFFSET_RANGE = (-0.35, 0.35)


def _clamp_xy(xy: list[float]) -> list[float]:
    x = float(np.clip(xy[0], TABLE_BOUNDS["x"][0], TABLE_BOUNDS["x"][1]))
    y = float(np.clip(xy[1], TABLE_BOUNDS["y"][0], TABLE_BOUNDS["y"][1]))
    return [x, y]


def sample_scene_config(seed: int) -> dict:
    """Sample one scene configuration with fixed object poses."""
    rng = np.random.default_rng(seed)
    config = {"seed": seed, "objects": {}}

    # Box (if present)
    box_xy = [
        DEFAULT_BOX_XY[0] + float(rng.uniform(*XY_OFFSET_RANGE)),
        DEFAULT_BOX_XY[1] + float(rng.uniform(*XY_OFFSET_RANGE)),
    ]
    config["box"] = {"xy": _clamp_xy(box_xy)}

    # Objects
    for name, spec in DEFAULT_OBJECT_SPECS.items():
        xy = [
            spec["xy"][0] + float(rng.uniform(*XY_OFFSET_RANGE)),
            spec["xy"][1] + float(rng.uniform(*XY_OFFSET_RANGE)),
        ]
        yaw = spec.get("yaw", 0.0) + float(rng.uniform(*YAW_OFFSET_RANGE))
        config["objects"][name] = {
            "xy": _clamp_xy(xy),
            "yaw": float(yaw),
        }

    return config


def main():
    parser = argparse.ArgumentParser(description="Generate fixed scene configs for evaluation")
    parser.add_argument("--num-scenes", type=int, default=10, help="Number of scenes to generate")
    parser.add_argument("--seed", type=int, default=0, help="Base seed; scene i uses seed+i")
    parser.add_argument("--output", type=str, default="scene_configs.json", help="Output JSON path")
    args = parser.parse_args()

    scenes = [sample_scene_config(args.seed + i) for i in range(args.num_scenes)]
    output = {"base_seed": args.seed, "num_scenes": args.num_scenes, "scenes": scenes}

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Generated {args.num_scenes} scene configs -> {out_path}")
    for i, s in enumerate(scenes):
        print(f"  Scene {i}: seed={s['seed']}, plate xy={s['objects']['plate']['xy']}, ...")


if __name__ == "__main__":
    main()
