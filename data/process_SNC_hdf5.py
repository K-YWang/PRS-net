import os
import json
import math
import h5py
import numpy as np
import torch
from glob import glob
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree
from collections import defaultdict
from typing import Dict, List, Tuple
import random


def load_hdf5_pointcloud(filepath: str) -> np.ndarray:
    """
    读取 HDF5 文件中的 'data' 数据集，返回 (B, N, 3) 的点云数组。
    """
    with h5py.File(filepath, 'r') as f:
        if 'data' not in f:
            raise KeyError(f"'data' dataset not found in HDF5 file: {filepath}")
        data = f['data'][:]  # shape: (B, N, 3)
    return data


def random_rotate(pc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    对点云做一次随机旋转。
    返回：
      - 旋转后的点云 (N,3)
      - axis-angle 向量 (4,) = [axis_x, axis_y, axis_z, angle_rad]
    """
    axis = np.random.randn(3)
    axis /= (np.linalg.norm(axis) + 1e-12)
    angle = np.random.rand() * 2 * np.pi
    rot = R.from_rotvec(axis * angle)
    return rot.apply(pc), np.append(axis, angle)


def pointcloud_to_voxel(pc: np.ndarray, grid_size: int = 32) -> np.ndarray:
    """
    将 [-0.5, 0.5] 范围内的点云栅格化为占据体素 (gs, gs, gs)。
    """
    voxel = np.zeros((grid_size, grid_size, grid_size), dtype=np.uint8)
    coords = ((pc + 0.5) * grid_size).astype(int)
    coords = np.clip(coords, 0, grid_size - 1)
    voxel[coords[:, 0], coords[:, 1], coords[:, 2]] = 1
    return voxel


def compute_closest_points(pc: np.ndarray, grid_size: int = 32) -> np.ndarray:
    """
    为每个体素中心点在点云中查询最近点。
    返回形状 (gs, gs, gs, 3)。
    """
    half_step = 1.0 / grid_size / 2
    coords = np.linspace(-0.5 + half_step, 0.5 - half_step, grid_size)
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")
    grid_pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)  # (gs^3, 3)
    tree = cKDTree(pc)
    _, idx = tree.query(grid_pts, k=1)
    return pc[idx].reshape((grid_size, grid_size, grid_size, 3))


def save_pt(out_path: str,
            Volume: np.ndarray,
            surfaceSamples: np.ndarray,
            vertices: np.ndarray,
            faces: np.ndarray,
            axisangle: np.ndarray,
            closestPoints: np.ndarray,
            category: str,
            category_id: str,
            model_id: str,
            meta: Dict = None):
    """
    保存 .pt，包含必要字段与类别信息。
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = {
        'Volume': torch.from_numpy(Volume).float(),                  # (gs,gs,gs)
        'surfaceSamples': torch.from_numpy(surfaceSamples).float(),  # (P,3)
        'vertices': torch.from_numpy(vertices).float(),              # 兼容字段：与 surfaceSamples 一致
        'faces': torch.from_numpy(faces).long(),                     # (F,3)，无面信息时写空 (0,3)
        'axisangle': torch.from_numpy(axisangle).float(),            # (4,)
        'closestPoints': torch.from_numpy(closestPoints).float(),    # (gs,gs,gs,3)
        # 新增类别元数据：
        'category': category,           # 英文类别名（来自 id2name list）
        'category_id': category_id,     # synset id（来自 id2file 第一段，如 03001627）
        'model_id': model_id,           # 具体模型 id（来自 id2file 第二段）
    }
    if meta is not None:
        payload['meta'] = meta          # 源信息（来源文件、样本索引等）

    torch.save(payload, out_path)


# =========================
# FPS：最远点采样（NumPy 版）
# =========================
def farthest_point_sampling_np(points: np.ndarray, k: int, rng: np.random.Generator = None) -> np.ndarray:
    """
    最远点采样（Farthest Point Sampling, FPS）。
    points: (N,3) float32/float64
    k:      目标数量
    返回:   选中点的下标 indices，shape=(k,)
    复杂度: O(N*k)，本任务 N≈2048, k=1000，CPU 足够
    """
    N = points.shape[0]
    if k >= N or N == 0:
        return np.arange(N, dtype=np.int64)
    if rng is None:
        rng = np.random.default_rng()

    # 随机起点（也可以改为极值点）
    start = int(rng.integers(0, N))
    sel = np.empty(k, dtype=np.int64)
    sel[0] = start

    # 与起点的距离平方
    diff = points - points[start]
    dmin = np.einsum('ij,ij->i', diff, diff)   # (N,)

    for i in range(1, k):
        nxt = int(np.argmax(dmin))
        sel[i] = nxt
        diff = points - points[nxt]
        dnew = np.einsum('ij,ij->i', diff, diff)
        dmin = np.minimum(dmin, dnew)

    return sel


# =========================
# 解析 id2file / id2name（两者均为 list）
# =========================
def load_id_lists(json_id2file_path: str, json_id2name_path: str) -> Tuple[List[str], List[str]]:
    """
    读取两份 JSON（都为 list）。
      - id2file_list: list[str]，按样本顺序的路径字符串。
      - id2name_list: list[str]，按样本顺序的英文类别名。
    """
    with open(json_id2file_path, 'r', encoding='utf-8') as f:
        id2file_list = json.load(f)
    with open(json_id2name_path, 'r', encoding='utf-8') as f:
        id2name_list = json.load(f)

    if not isinstance(id2file_list, list):
        raise TypeError(f"id2file must be a list, got: {type(id2file_list)}")
    if not isinstance(id2name_list, list):
        raise TypeError(f"id2name must be a list, got: {type(id2name_list)}")

    return id2file_list, id2name_list


def parse_cat_and_model(path_str: str) -> Tuple[str, str]:
    """
    从 id2file 的值中解析 category_id (synset) 与 model_id。
    兼容 Windows 路径分隔。
    常见形态: "03001627/1a04e3e.../xxx.npy"
    """
    p = path_str.replace("\\", "/").split("/")
    if len(p) >= 2:
        category_id = p[0]
        model_id = p[1]
    else:
        category_id = "unknown"
        model_id = path_str.replace("/", "_")
    return category_id, model_id


# =========================
# 主流程：聚合、增强、划分
# =========================
def process_all(
    h5_dir: str,
    out_root: str,
    grid_size: int = 32,
    per_class_target: int = 4000,
    train_ratio: float = 0.8,
    sample_k: int = 1000,       # NEW: 先均匀下采样到 K 个点（默认 1000）
    seed: int = 42
):
    """
    读取 h5 + 对应 list 结构的 id 列表（id2file, id2name），聚合到类别维度；
    对每个样本先用 FPS 均匀选取 sample_k 个点，
    为每个类别生成 per_class_target 个增强样本（随机旋转），
    再按 train_ratio 划分到 train/test 目录。
    """
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # 1) 收集文件对：每个 *.h5 需要对应 *_id2file.json 与 *_id2name.json
    h5_files = sorted(glob(os.path.join(h5_dir, "*.h5")))
    if not h5_files:
        print(f"[WARN] No .h5 files under: {h5_dir}")
        return

    # 每个类别（按 category_id）收集样本引用：
    #   (h5_path, sample_index, cat_id, model_id, category_name)
    cat_to_refs: Dict[str, List[Tuple[str, int, str, str, str]]] = defaultdict(list)

    # 2) 遍历每个 h5，读取对应映射并登记
    for h5_path in h5_files:
        stem = os.path.splitext(os.path.basename(h5_path))[0]  # e.g., "train9"
        id2file_path = os.path.join(h5_dir, f"{stem}_id2file.json")
        id2name_path = os.path.join(h5_dir, f"{stem}_id2name.json")

        if not os.path.isfile(id2file_path) or not os.path.isfile(id2name_path):
            print(f"[WARN] Missing mapping json for {h5_path}, skip.")
            continue

        try:
            id2file_list, id2name_list = load_id_lists(id2file_path, id2name_path)
        except Exception as e:
            print(f"[WARN] Failed to load id-lists for {h5_path}: {e}")
            continue

        # 确定该 h5 的样本数
        try:
            with h5py.File(h5_path, 'r') as f:
                if 'data' not in f:
                    print(f"[WARN] 'data' dataset missing in {h5_path}, skip.")
                    continue
                B = f['data'].shape[0]
        except Exception as e:
            print(f"[WARN] Failed to open {h5_path}: {e}")
            continue

        # 将每个样本登记到类别映射
        for i in range(B):
            if i >= len(id2file_list):
                continue
            cat_id, model_id = parse_cat_and_model(id2file_list[i])
            # category（英文名）来自 id2name_list[i]；若不存在则用 cat_id 兜底
            category_name = id2name_list[i] if i < len(id2name_list) else cat_id
            if not isinstance(category_name, str):
                category_name = str(category_name)
            cat_to_refs[cat_id].append((h5_path, i, cat_id, model_id, category_name))

    if not cat_to_refs:
        print("[WARN] No samples collected. Check your id2file/id2name jsons.")
        return

    # 3) 针对每个类别进行增强并保存，同时 8:2 划分
    # 为减少 I/O 开销，做简单的文件缓存
    h5_cache: Dict[str, np.ndarray] = {}

    # 输出结构：out_root/train/  & out_root/test/
    train_dir = os.path.join(out_root, "train")
    test_dir = os.path.join(out_root, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for cat_id, refs in cat_to_refs.items():
        # 保持同一 cat_id 下的 category_name 一致（万一有差异取第一个）
        category_name = refs[0][4] if refs else cat_id
        cat_name_safe = category_name.strip().replace(" ", "_")

        N = len(refs)
        if N == 0:
            continue

        target = per_class_target

        # 若原始模型数 >= 4000，则随机采样 4000；否则每模型做重复增强
        if N >= target:
            selected_refs = random.sample(refs, target)
            per_ref_reps = 1
        else:
            per_ref_reps = math.ceil(target / N)
            selected_refs = refs

        # 先决定划分：8:2（对“生成序号”分割，这样总量严格 4000）
        total_needed = target
        test_count = int(round(total_needed * (1.0 - train_ratio)))
        test_indices = set(random.sample(range(total_needed), test_count))

        gen_count = 0  # 已生成的样本数（用于判定 train/test）

        print(f"[CATEGORY] {cat_id} ({category_name}) | models={N} | reps/model={per_ref_reps} | target={target}")
        for (h5_path, sample_idx, cat_id_, model_id, category_name_) in selected_refs:
            # 载入该 h5 的点云数组（缓存）
            if h5_path not in h5_cache:
                h5_cache[h5_path] = load_hdf5_pointcloud(h5_path)  # (B,N,3)

            pc_orig = h5_cache[h5_path][sample_idx]  # (N,3)   —— 原始 2048 点

            # 先做 FPS 下采样到 sample_k（默认 1000）
            if pc_orig.shape[0] > sample_k:
                idx_sel = farthest_point_sampling_np(pc_orig, k=sample_k, rng=rng)  # (K,)
                pc_sub = pc_orig[idx_sel]                                           # (K,3)
            else:
                pc_sub = pc_orig                                                     # (≤K,3)

            # 对下采样后的点云做随机旋转
            rotated, axisangle = random_rotate(pc_sub)                                # (K,3), (4,)

            # 体素化 & 最近点
            Volume = pointcloud_to_voxel(rotated, grid_size)
            closestPoints = compute_closest_points(rotated, grid_size)

            # 决定落盘目录（按照本类别的全局生成索引划分）
            split_dir = test_dir if gen_count in test_indices else train_dir

            # 文件名：cg_cgid_idx.pt（cg=英文类别名，cgid=synset，idx=四位序号）
            fname = f"{cat_name_safe}_{cat_id}_{gen_count:04d}.pt"
            fpath = os.path.join(split_dir, fname)

            # 保存（faces 不可得时保存空）
            save_pt(
                fpath,
                Volume=Volume,
                surfaceSamples=rotated,              # (K,3)
                vertices=rotated,                    # 兼容字段
                faces=np.zeros((0, 3), dtype=np.int64),
                axisangle=axisangle,
                closestPoints=closestPoints,         # (gs,gs,gs,3)
                category=category_name_,             # 该条样本自身的英文类名
                category_id=cat_id,                  # synset
                model_id=model_id,                   # 模型 id
                meta={
                    "source_h5": os.path.basename(h5_path),
                    "source_index": int(sample_idx),
                }
            )
            gen_count += 1

            if gen_count >= target:
                break  # 达到目标后提前结束该类的生成

        print(f"  -> generated {gen_count} samples for category {cat_id} ({category_name})")


# =========================
# 示例调用
# =========================
if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)

    # 输入：包含 trainX.h5 / trainX_id2file.json / trainX_id2name.json 等文件的目录
    h5_dir = './ShapeNetCoreV2'
    out_root = './SNC_valid'

    process_all(
        h5_dir=h5_dir,
        out_root=out_root,
        grid_size=32,
        per_class_target=4000,    # 每类 4000
        train_ratio=0.8,          # 8:2 划分
        sample_k=1000,            # NEW：先 FPS 到 1000 点
        seed=42
    )
