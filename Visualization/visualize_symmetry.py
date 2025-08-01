import os
import random
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
from mpl_toolkits.mplot3d import Axes3D

import sys, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]   # ../  -> 项目根（与 src 同级）
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model.PrsNet_model import PRSNet


# ---------- 基础工具 ----------
def to_ns(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: to_ns(v) for k, v in obj.items()})
    return obj

def load_cfg(path="config.yaml"):
    with open(path, "r") as f:
        return to_ns(yaml.safe_load(f))

def select_device(s: str) -> torch.device:
    if s and s.lower() == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def build_model_from_cfg(cfg, device):
    model = PRSNet(
        input_nc=cfg.model.input_nc,
        base_channels=cfg.model.output_nc,
        conv_layers=cfg.model.conv_layers,
        num_planes=cfg.model.num_plane,
        num_quats=cfg.model.num_quat,
        use_bn=bool(cfg.model.use_bn),
        activation=cfg.model.activation,
        dropout=0.0,
    ).to(device)
    return model

def get_checkpoint_path(cfg) -> str:
    ckpt = getattr(getattr(cfg, "testing", SimpleNamespace()), "checkpoint", None)
    if ckpt is None:
        ckpt = getattr(getattr(cfg, "output", SimpleNamespace()), "checkpoint", None)
    if ckpt is None or not os.path.isfile(ckpt):
        raise FileNotFoundError(
            "请在 config.yaml 中设置 testing.checkpoint 指向模型权重。例如：\n"
            "testing:\n  checkpoint: ./checkpoints/<run_ts>/00099_net_PRSNet.pth"
        )
    return ckpt

def pick_random_pt(cfg) -> str:
    root = cfg.dataset.dataroot
    split = cfg.dataset.test_split  # 在 test 里挑
    base = os.path.join(root, split)
    pool = [os.path.join(r, f) for r,_,fs in os.walk(base) for f in fs if f.endswith(".pt")]
    if not pool:
        raise RuntimeError(f"No .pt files found under {base}")
    return random.choice(pool)


# ---------- 可视化相关 ----------
def voxel_to_points(vox: np.ndarray) -> np.ndarray:
    """把占据栅格 (gs,gs,gs) 转为坐标点（每个占据格中心，范围 ~[-0.5,0.5]）"""
    gs = vox.shape[0]
    idx = np.argwhere(vox > 0)     # (M,3) 索引
    centers = (idx + 0.5) / gs - 0.5
    return centers

def plane_patch_from_normal_d(n: np.ndarray, d: float, size: float = 0.7, res: int = 25):
    """生成平面 n·x + d = 0 的矩形面片网格坐标 (X,Y,Z)"""
    n = n / (np.linalg.norm(n) + 1e-12)
    p0 = -d * n  # 距原点最近点
    a = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(a, n)) > 0.9:
        a = np.array([0.0, 1.0, 0.0])
    u = a - np.dot(a, n) * n
    u /= (np.linalg.norm(u) + 1e-12)
    v = np.cross(n, u); v /= (np.linalg.norm(v) + 1e-12)
    s = np.linspace(-size, size, res)
    t = np.linspace(-size, size, res)
    S, T = np.meshgrid(s, t)
    P = p0[None,None,:] + S[...,None]*u[None,None,:] + T[...,None]*v[None,None,:]
    return P[...,0], P[...,1], P[...,2]

def quat_to_axis(q: np.ndarray) -> np.ndarray:
    """四元数 (w,x,y,z) → 旋转轴方向向量（单位长度）。若退化，返回 (0,0,1)。"""
    axis = q[1:4]
    n = np.linalg.norm(axis)
    if n < 1e-8:
        return np.array([0.0, 0.0, 1.0])
    return axis / n

def set_axes_equal(ax):
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(-0.5, 0.5)
    try:
        ax.set_box_aspect([1,1,1])
    except Exception:
        pass
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

def draw_voxels(ax, pts: np.ndarray, s=3):
    if pts.size:
        ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=s, depthshade=False)

def show_plane(voxel_np: np.ndarray, plane: np.ndarray, title: str, save_path: str):
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    pts = voxel_to_points(voxel_np)
    draw_voxels(ax, pts)
    X,Y,Z = plane_patch_from_normal_d(plane[:3], float(plane[3]), size=0.7, res=25)
    ax.plot_surface(X, Y, Z, alpha=0.35, linewidth=0, antialiased=False)
    set_axes_equal(ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.show()

def show_axis(voxel_np: np.ndarray, axis_dir: np.ndarray, title: str, save_path: str):
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    pts = voxel_to_points(voxel_np)
    draw_voxels(ax, pts)
    L = 0.6
    p1 =  L * axis_dir
    p2 = -L * axis_dir
    ax.plot([p2[0], p1[0]], [p2[1], p1[1]], [p2[2], p1[2]], linewidth=3)
    set_axes_equal(ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.show()


# ---------- 主流程 ----------
def main():
    cfg = load_cfg("config.yaml")
    device = select_device(cfg.training.device)

    # 随机选取一个 .pt
    pt_path = pick_random_pt(cfg)
    sample = torch.load(pt_path)
    voxel_np = sample["Volume"].cpu().numpy()                  # (gs,gs,gs)
    voxel = sample["Volume"].float().unsqueeze(0).unsqueeze(0) # (1,1,gs,gs,gs)

    # 模型 & 权重
    model = build_model_from_cfg(cfg, device)
    ckpt = get_checkpoint_path(cfg)
    state = torch.load(ckpt, map_location=device)
    try:
        model.load_state_dict(state, strict=True)
    except Exception:
        if isinstance(state, dict) and "state_dict" in state:
            model.load_state_dict(state["state_dict"], strict=False)
        else:
            model.load_state_dict(state, strict=False)
    model.eval()

    # 推理
    with torch.no_grad():
        quats, planes = model(voxel.to(device))    # quats:(1,Kq,4), planes:(1,Kp,4)

    planes_np = planes[0].detach().cpu().numpy() if planes is not None else np.zeros((0,4))
    quats_np  = quats[0].detach().cpu().numpy()  if quats  is not None else np.zeros((0,4))

    Kp = min(3, planes_np.shape[0])
    Kq = min(3, quats_np.shape[0])

    out_dir = os.path.join("./Visualization/vis_sym")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Random sample: {pt_path}")
    # 画平面
    for i in range(Kp):
        p = planes_np[i]
        title = f"{os.path.basename(pt_path)} | Plane {i+1}: n=({p[0]:.2f},{p[1]:.2f},{p[2]:.2f}), d={p[3]:.2f}"
        save_path = os.path.join(out_dir, f"plane_{i+1}.png")
        show_plane(voxel_np, p, title, save_path)
        print("Saved:", save_path)

    # 画轴（由四元数得到）
    for i in range(Kq):
        q = quats_np[i]
        axis = quat_to_axis(q)
        title = f"{os.path.basename(pt_path)} | Axis {i+1}: ({axis[0]:.2f},{axis[1]:.2f},{axis[2]:.2f})"
        save_path = os.path.join(out_dir, f"axis_{i+1}.png")
        show_axis(voxel_np, axis, title, save_path)
        print("Saved:", save_path)

    if Kp < 3 or Kq < 3:
        print(f"[Note] Only had {Kp} planes and {Kq} axes available from the model outputs.")

if __name__ == "__main__":
    main()
