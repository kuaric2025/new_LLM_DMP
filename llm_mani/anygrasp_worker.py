from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np


sys.path.append("/home/mhumais/Huang/anygrasp_sdk/grasp_detection")


def parse_args():
    parser = argparse.ArgumentParser(description="Run AnyGrasp in an isolated process.")
    parser.add_argument("--input-npz", required=True)
    parser.add_argument("--cfg-pkl", required=True)
    parser.add_argument("--output-npz", required=True)
    return parser.parse_args()


def select_center_grasp(grasps, points):
    if len(grasps) == 0:
        return None
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    center_vec = np.median(pts, axis=0)
    best_idx = 0
    best_dist = float("inf")
    for i in range(len(grasps)):
        g = grasps[i]
        t = np.asarray(g.translation, dtype=np.float64).reshape(3)
        d = float(np.linalg.norm(t - center_vec))
        if d < best_dist:
            best_dist = d
            best_idx = i
    return grasps[best_idx]


def main() -> int:
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("WANDB_SILENT", "true")
    args = parse_args()

    from gsnet import AnyGrasp

    with open(args.cfg_pkl, "rb") as f:
        cfgs = pickle.load(f)

    payload = np.load(args.input_npz)
    colors = payload["colors"]
    depths = payload["depths"]
    segmap = payload["segmap"].astype(bool)

    anygrasp = AnyGrasp(cfgs)
    anygrasp.load_net()

    fx, fy = 342.7555, 342.7555
    cx, cy = 320, 240
    scale = 1.0

    xmap, ymap = np.arange(depths.shape[1]), np.arange(depths.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depths / scale
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z

    points = np.stack([points_x, points_y, points_z], axis=-1)
    points = points[segmap].astype(np.float32)
    colors = colors[segmap].astype(np.float32)

    if points.shape[0] == 0:
        print("No points left after applying segmentation mask.", file=sys.stderr, flush=True)
        return 2

    print(points.min(axis=0), points.max(axis=0), flush=True)
    xmin, ymin, zmin = np.min(points, axis=0)
    xmax, ymax, zmax = np.max(points, axis=0)
    margin = 0.01
    lims = [xmin - margin, xmax + margin, ymin - margin, ymax + margin, zmin - margin, zmax + margin]

    gg, cloud = anygrasp.get_grasp(
        points,
        colors,
        lims=lims,
        apply_object_mask=True,
        dense_grasp=False,
        collision_detection=True,
    )

    if len(gg) == 0:
        print("No Grasp detected after collision detection!", file=sys.stderr, flush=True)
        return 3

    gg = gg.nms().sort_by_score()
    gg_pick = gg[0:30]
    print(gg_pick.scores, flush=True)
    print("best grasp score:", gg_pick[0].score, flush=True)
    print("grasp score:", gg_pick[0].score, flush=True)

    best_grasp = select_center_grasp(gg_pick, points=points)
    if best_grasp is None:
        return 4

    if getattr(cfgs, "debug", False):
        import open3d as o3d

        colors_dbg = np.tile(np.array([[0.0, 1.0, 0.0]]), (np.asarray(cloud.points).shape[0], 1))
        cloud.colors = o3d.utility.Vector3dVector(colors_dbg)
        best_geom = best_grasp.to_open3d_geometry()
        best_geom.paint_uniform_color([0.0, 0.0, 1.0])
        o3d.visualization.draw_geometries([best_geom, cloud])

    np.savez(
        args.output_npz,
        translation=np.asarray(best_grasp.translation, dtype=np.float64).reshape(3),
        rotation_matrix=np.asarray(best_grasp.rotation_matrix, dtype=np.float64).reshape(3, 3),
        score=np.asarray(float(best_grasp.score), dtype=np.float64),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
