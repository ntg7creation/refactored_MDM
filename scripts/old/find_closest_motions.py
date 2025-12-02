import os
import json
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist
from fastdtw import fastdtw  # install with: pip install fastdtw

import sys
# Ensure access to local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# === CONFIG ===
GENERATED_FOLDER = r"D:\repos\refactored_MDM\save\test_custom500_concat_11_4\infer_trainset_blind"   # where the 20 generated jsonl files are
ORIGINAL_TRAIN_JSONL = r"D:\repos\refactored_MDM\GigaHands_Data\train_custom_500.jsonl"
CONVERTED_ROOT = r"D:\repos\refactored_MDM\GigaHands_Data\coverted_motions\hand_poses_xyz"

OUTPUT_FILE = os.path.join(GENERATED_FOLDER, "closest_matches.json")

# ------------------------------------------------------
# 📦 Load helper: motion from generated jsonl (Nx3xT)
# ------------------------------------------------------
def load_generated_motion(path):
    with open(path, "r") as f:
        frames = [json.loads(line) for line in f]
    arr = np.array(frames)  # [T, 42, 4]
    arr = arr[..., :3]  # keep xyz
    arr = arr.transpose(1, 2, 0)  # → [42, 3, T]
    return arr.reshape(-1, arr.shape[-1])  # flatten spatial dims to [126, T]


# ------------------------------------------------------
# 📦 Load helper: motion from original dataset (Nx3xT)
# ------------------------------------------------------
def load_original_motion(scene, seq):
    npy_path = os.path.join(CONVERTED_ROOT, scene, "keypoints_3d", seq, "xyz_both.npy")
    if not os.path.exists(npy_path):
        return None
    arr = np.load(npy_path)

    # Handle multiple possible formats
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        # [T, 42, 3 or 4]
        arr = arr[..., :3].transpose(1, 2, 0).reshape(-1, arr.shape[0])  # [126, T]
    elif arr.ndim == 2:
        # [T, 126] already flattened
        arr = arr.T  # → [126, T]
    else:
        raise ValueError(f"Unexpected array shape {arr.shape} in {npy_path}")

    return arr


# ------------------------------------------------------
# 🔍 Compute MSE distance (same length)
# ------------------------------------------------------
def mse_distance(a, b):
    min_len = min(a.shape[1], b.shape[1])
    return np.mean((a[:, :min_len] - b[:, :min_len]) ** 2)


# ------------------------------------------------------
# 🔍 Compute DTW distance (time-aligned)
# ------------------------------------------------------
def dtw_distance(a, b):
    min_len = min(a.shape[0], b.shape[0])
    # flatten temporal structure for DTW along time axis
    aT = a[:, :min_len].T
    bT = b[:, :min_len].T
    dist, _ = fastdtw(aT, bT)
    return dist / min_len


# ------------------------------------------------------
# 📜 Load original dataset index
# ------------------------------------------------------
print("📖 Loading original train jsonl index...")
with open(ORIGINAL_TRAIN_JSONL, "r") as f:
    original_entries = [json.loads(line) for line in f]

original_index = []
for ann in original_entries:
    scene = ann["scene"]
    seq = ann["sequence"]
    motion = load_original_motion(scene, seq)
    if motion is not None:
        original_index.append({"scene": scene, "seq": seq, "motion": motion})
print(f"✅ Loaded {len(original_index)} original motions")


# ------------------------------------------------------
# 🚀 Compare each generated motion to originals
# ------------------------------------------------------
results = {}

gen_files = [f for f in os.listdir(GENERATED_FOLDER) if f.endswith(".jsonl")]
print(f"🔍 Found {len(gen_files)} generated motions")

for gfile in tqdm(gen_files, desc="Comparing motions"):
    gpath = os.path.join(GENERATED_FOLDER, gfile)
    gen_motion = load_generated_motion(gpath)

    scores = []
    for orig in original_index:
        d = mse_distance(gen_motion, orig["motion"])
        # d = dtw_distance(gen_motion, orig["motion"])
        scores.append((orig["scene"], orig["seq"], d))

    scores.sort(key=lambda x: x[2])
    top7 = scores[:7]

    results[gfile] = [{"scene": s, "seq": q, "mse": d} for s, q, d in top7]


# ------------------------------------------------------
# 💾 Save results
# ------------------------------------------------------
with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2)

print(f"✅ Done! Results saved to {OUTPUT_FILE}")
