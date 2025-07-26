import os
import h5py
import numpy as np
import torch
from glob import glob
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree
import random

def load_hdf5_pointcloud(filepath):
    with h5py.File(filepath, 'r') as f:
        data = f['data'][:]  # shape: (B, N, 3)
    return data

def random_rotate(pc):
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    angle = np.random.rand() * 2 * np.pi
    rot = R.from_rotvec(axis * angle)
    return rot.apply(pc), np.append(axis, angle)

def pointcloud_to_voxel(pc, grid_size=32):
    voxel = np.zeros((grid_size, grid_size, grid_size), dtype=np.uint8)
    coords = ((pc + 0.5) * grid_size).astype(int)
    coords = np.clip(coords, 0, grid_size - 1)
    voxel[coords[:, 0], coords[:, 1], coords[:, 2]] = 1
    return voxel

def compute_closest_points(pc, grid_size=32):
    half_step = 1.0 / grid_size / 2
    coords = np.linspace(-0.5 + half_step, 0.5 - half_step, grid_size)
    X, Y, Z = np.meshgrid(coords, coords, coords)
    grid_pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    tree = cKDTree(pc)
    _, idx = tree.query(grid_pts)
    return pc[idx].reshape((grid_size, grid_size, grid_size, 3))

def save_pt(out_path, Volume, surfaceSamples, vertices, faces, axisangle, closestPoints):
    torch.save({
        'Volume': torch.from_numpy(Volume).float(),                  # (32,32,32)
        'surfaceSamples': torch.from_numpy(surfaceSamples).float(), # (P,3)
        'vertices': torch.from_numpy(vertices).float(),             # (P,3)
        'faces': torch.from_numpy(faces).long(),                    # (F,3) or (0,3)
        'axisangle': torch.from_numpy(axisangle).float(),           # (4,)
        'closestPoints': torch.from_numpy(closestPoints).float()    # (32,32,32,3)
    }, out_path)

def process_h5_to_pt_all(h5_dir, out_dir, grid_size=32):
    os.makedirs(out_dir, exist_ok=True)
    h5_files = glob(os.path.join(h5_dir, '*.h5'))
    for h5_file in h5_files:
        name_prefix = os.path.splitext(os.path.basename(h5_file))[0]
        pcs = load_hdf5_pointcloud(h5_file)
        for i, pc in enumerate(pcs):
            rotated, axisangle = random_rotate(pc)
            Volume = pointcloud_to_voxel(rotated, grid_size)
            closestPoints = compute_closest_points(rotated, grid_size)
            fname = f"{name_prefix}_pc_{i:05d}.pt"
            fpath = os.path.join(out_dir, fname)
            save_pt(fpath, Volume, rotated, rotated, np.zeros((0, 3), dtype=np.int32), axisangle, closestPoints)
            print(f"Saved: {fpath}")

# === 示例调用 ===
if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)

    h5_dir = './ShapeNetCoreV2'           # 输入：.h5 点云数据目录
    out_dir = './SNC_pt'                  # 输出：.pt 结构化数据
    process_h5_to_pt_all(h5_dir, out_dir, grid_size=32)
