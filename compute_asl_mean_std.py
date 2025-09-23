import os
import numpy as np
from os.path import join as pjoin
from tqdm import tqdm

def iter_side_files(data_root: str, side: str):
    """Yield full paths to <...>/<side>_3d.npy under data_root."""
    target = f"{side}_3d.npy"
    for root, _, files in os.walk(data_root):
        if target in files:
            yield pjoin(root, target)

def compute_stats_from_npy(data_root: str, out_dir: str, side: str):
    files = list(iter_side_files(data_root, side))
    if not files:
        raise FileNotFoundError(f"No '{side}_3d.npy' files under {data_root}")

    print(f"🔍 {side}: found {len(files)} files under {data_root}")
    total_frames = 0
    feat_dim = None
    sum_vec = None
    sumsq_vec = None

    for fp in tqdm(files, desc=f"Accumulating {side}"):
        arr = np.load(fp)  # expected [T, D]
        if arr.ndim == 3:
            arr = arr.reshape(arr.shape[0], -1)
        elif arr.ndim != 2:
            print(f"⚠️ Skipping {fp}, unexpected shape {arr.shape}")
            continue

        if arr.size == 0 or not np.any(arr):
            continue

        T, D = arr.shape
        if feat_dim is None:
            feat_dim = D
            sum_vec = np.zeros(D, dtype=np.float64)
            sumsq_vec = np.zeros(D, dtype=np.float64)
        elif D != feat_dim:
            print(f"⚠️ Skipping {fp}, dim mismatch: got {D}, expected {feat_dim}")
            continue

        sum_vec += arr.sum(axis=0, dtype=np.float64)
        sumsq_vec += np.square(arr, dtype=np.float64).sum(axis=0)
        total_frames += T

    if total_frames == 0:
        raise RuntimeError(f"No valid frames accumulated for side='{side}'.")

    mean = sum_vec / total_frames
    var = (sumsq_vec / total_frames) - np.square(mean)
    std = np.sqrt(np.maximum(var, 1e-12))

    os.makedirs(out_dir, exist_ok=True)
    mean_path = pjoin(out_dir, f"mean_{side}.npy")
    std_path  = pjoin(out_dir, f"std_{side}.npy")
    np.save(mean_path, mean.astype(np.float32))
    np.save(std_path,  std.astype(np.float32))

    print(f"✅ Saved {side} mean → {mean_path}  (dim={feat_dim}, frames={total_frames})")
    print(f"✅ Saved {side} std  → {std_path}")

if __name__ == "__main__":
    # 🔧 Hardcoded paths
    DATA_ROOT = r"ASL_Data/dmvb"         # folder with converted DMVB npy
    OUT_DIR   = r"ASL_Data/norm_stats"   # where to save stats

    # Compute for left + right
    compute_stats_from_npy(DATA_ROOT, OUT_DIR, side="left")
    compute_stats_from_npy(DATA_ROOT, OUT_DIR, side="right")
