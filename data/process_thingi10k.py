import os
import random
import numpy as np
import torch
from glob import glob
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree

# ------------ Hard-coded params ------------
CACHE_DIR   = "./data/Thingi10K/extracted/bf3a2acdcf4021a5543b75162094d8973c5ddbf80302c92408404fdb11abb423/npz"         # Thingi10K 数据所在目录
OUT_ROOT    = "./thingi10k_valid"  # 输出目录
GRID_SIZE   = 32
SAMPLE_K    = 1000
OVERSAMPLE  = 4000                  # 先面积采样这么多，再 FPS -> 1000
TRAIN_RATIO = 0.8
SEED        = 42
# -------------------------------------------

# optional: use thingi10k if available
_HAS_THINGI = False
try:
    import thingi10k
    _HAS_THINGI = True
except Exception:
    _HAS_THINGI = False


# -------------- Geometry utils --------------
def normalize_to_unit_cube(points: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    pmin = points.min(axis=0); pmax = points.max(axis=0)
    center = (pmin + pmax) / 2.0
    extent = (pmax - pmin).max()
    if extent < eps: extent = 1.0
    return (points - center) / extent  # ≈ [-0.5, 0.5]

def triangle_areas(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    tri = v[f]
    a = tri[:, 1] - tri[:, 0]
    b = tri[:, 2] - tri[:, 0]
    return np.linalg.norm(np.cross(a, b), axis=1) * 0.5

def farthest_point_sampling_np(points: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    N = points.shape[0]
    if k >= N or N == 0: return np.arange(N, dtype=np.int64)
    start = int(rng.integers(0, N))
    sel = np.empty(k, dtype=np.int64); sel[0] = start
    diff = points - points[start]; dmin = np.einsum('ij,ij->i', diff, diff)
    for i in range(1, k):
        nxt = int(np.argmax(dmin)); sel[i] = nxt
        diff = points - points[nxt]
        dnew = np.einsum('ij,ij->i', diff, diff)
        dmin = np.minimum(dmin, dnew)
    return sel

def sample_points_on_mesh(v: np.ndarray, f: np.ndarray, n_samples: int,
                          oversample: int, rng: np.random.Generator) -> np.ndarray:
    if f.size == 0:
        if len(v) == 0: return np.zeros((n_samples, 3), dtype=np.float32)
        idx = rng.integers(0, len(v), size=n_samples)
        return v[idx].astype(np.float32)
    areas = triangle_areas(v, f); total = areas.sum()
    if total <= 1e-12:
        idx = rng.integers(0, len(v), size=n_samples)
        return v[idx].astype(np.float32)
    m = max(n_samples, oversample)
    probs = areas / total
    face_idx = rng.choice(len(f), size=m, replace=True, p=probs)
    tri = v[f[face_idx]]  # (m,3,3)
    u = rng.random(m); v2 = rng.random(m)
    r1 = np.sqrt(u); r2 = v2
    A, B, C = tri[:, 0, :], tri[:, 1, :], tri[:, 2, :]
    samples = (1 - r1)[:, None]*A + (r1*(1 - r2))[:, None]*B + (r1*r2)[:, None]*C
    if m == n_samples: return samples.astype(np.float32)
    idx_fps = farthest_point_sampling_np(samples, k=n_samples, rng=rng)
    return samples[idx_fps].astype(np.float32)

def pointcloud_to_voxel(pc: np.ndarray, grid_size: int) -> np.ndarray:
    gs = int(grid_size)
    vol = np.zeros((gs, gs, gs), dtype=np.uint8)
    coords = ((pc + 0.5) * gs).astype(int)
    coords = np.clip(coords, 0, gs - 1)
    vol[coords[:, 0], coords[:, 1], coords[:, 2]] = 1
    return vol

def compute_closest_points(pc: np.ndarray, grid_size: int) -> np.ndarray:
    gs = int(grid_size)
    half = 1.0 / gs / 2.0
    coords = np.linspace(-0.5 + half, 0.5 - half, gs)
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
    centers = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)  # (gs^3,3)
    tree = cKDTree(pc)
    _, idx = tree.query(centers, k=1)
    return pc[idx].reshape(gs, gs, gs, 3)

def save_pt(out_path: str,
            Volume: np.ndarray,
            surfaceSamples: np.ndarray,
            vertices: np.ndarray,
            faces: np.ndarray,
            axisangle: np.ndarray,
            closestPoints: np.ndarray):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = {
        "Volume": torch.from_numpy(Volume).float(),
        "surfaceSamples": torch.from_numpy(surfaceSamples).float(),
        "vertices": torch.from_numpy(vertices).float(),
        "faces": (torch.from_numpy(faces.astype(np.int64)).long()
                  if faces.size else torch.zeros((0, 3), dtype=torch.long)),
        "axisangle": torch.from_numpy(axisangle.astype(np.float32)).float(),
        "closestPoints": torch.from_numpy(closestPoints).float(),
    }
    torch.save(payload, out_path)
# -------------------------------------------


def load_mesh_any(path: str) -> tuple[np.ndarray, np.ndarray]:
    if _HAS_THINGI:
        try:
            v, f = thingi10k.load_file(path)
            return np.asarray(v, dtype=np.float64), np.asarray(f, dtype=np.int64).reshape(-1, 3)
        except Exception:
            pass
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.lib.npyio.NpzFile):
        if "vertices" in data and "faces" in data:
            v, f = data["vertices"], data["faces"]
        elif "v" in data and "f" in data:
            v, f = data["v"], data["f"]
        else:
            raise KeyError(f"Unknown npz keys: {list(data.keys())}")
    else:
        obj = data.item()
        if "vertices" in obj and "faces" in obj:
            v, f = obj["vertices"], obj["faces"]
        elif "v" in obj and "f" in obj:
            v, f = obj["v"], obj["f"]
        else:
            raise KeyError(f"Unknown npy keys: {obj.keys()}")
    return np.asarray(v, dtype=np.float64), np.asarray(f, dtype=np.int64).reshape(-1, 3)


def main():
    rng = np.random.default_rng(SEED)
    random.seed(SEED)

    # Gather files
    if _HAS_THINGI:
        thingi10k.init(cache_dir=CACHE_DIR)
        entries = list(thingi10k.dataset())
        files = [(e.get("file_id", os.path.splitext(os.path.basename(e["file_path"]))[0]),
                  e["file_path"]) for e in entries]
    else:
        files = []
        for p in ("**/*.npz", "**/*.npy"):
            files += [(os.path.splitext(os.path.basename(path))[0], path)
                      for path in glob(os.path.join(CACHE_DIR, p), recursive=True)]

    if not files:
        print(f"[WARN] No mesh files found under: {CACHE_DIR}")
        return

    print(f"[INFO] Found {len(files)} files.")

    # Split 8:2 per-file
    idxs = list(range(len(files)))
    random.shuffle(idxs)
    split = int(len(files) * TRAIN_RATIO)
    idx_train = set(idxs[:split])

    out_train = os.path.join(OUT_ROOT, "train")
    out_test  = os.path.join(OUT_ROOT, "test")
    os.makedirs(out_train, exist_ok=True)
    os.makedirs(out_test,  exist_ok=True)

    n_ok, n_fail = 0, 0
    for i, (fid, path) in enumerate(files):
        try:
            v, f = load_mesh_any(path)
            if v.size == 0 or f.size == 0:
                raise ValueError("empty vertices/faces")

            v_norm = normalize_to_unit_cube(v)
            pts = sample_points_on_mesh(v_norm, f, n_samples=SAMPLE_K, oversample=OVERSAMPLE, rng=rng)
            vol = pointcloud_to_voxel(pts, GRID_SIZE)
            cp  = compute_closest_points(pts, GRID_SIZE)
            axisangle = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

            subdir = out_train if i in idx_train else out_test
            out_path = os.path.join(subdir, f"thingi10k_{fid}.pt")
            save_pt(out_path,
                    Volume=vol,
                    surfaceSamples=pts,
                    vertices=pts,
                    faces=np.zeros((0, 3), dtype=np.int64),
                    axisangle=axisangle,
                    closestPoints=cp)
            n_ok += 1

            if (i + 1) % 100 == 0:
                print(f"[{i+1}/{len(files)}] saved -> {out_path}")

        except Exception as e:
            n_fail += 1
            print(f"[ERR] {path}: {e}")

    print(f"[DONE] ok={n_ok}, fail={n_fail}, out_root={OUT_ROOT}")
    print("  train =", len(os.listdir(out_train)), "files")
    print("  test  =", len(os.listdir(out_test)), "files")


if __name__ == "__main__":
    main()
