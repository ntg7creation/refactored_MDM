

import os
import numpy as np
from tqdm import tqdm

def collect_all_motions(root_dir):
    """
    Recursively collect all xyz_both.npy files under root_dir.
    """
    file_paths = []
    for scene, _, files in os.walk(root_dir):
        for f in files:
            if f.endswith("xyz_both.npy"):
                file_paths.append(os.path.join(scene, f))
    return file_paths

def main():
    # Path to converted motions
    root_dir = r"D:\repos\refactored_MDM\GigaHands_Data\coverted_motions\hand_poses_xyz"
    out_dir = os.path.join(root_dir, "..", "norm_stats")
    os.makedirs(out_dir, exist_ok=True)

    file_paths = collect_all_motions(root_dir)
    assert len(file_paths) > 0, f"No xyz_both.npy found under {root_dir}"

    print(f"Found {len(file_paths)} motion files")

    # First pass: accumulate sums
    total_frames = 0
    sum_feats = None
    sum_sq_feats = None

    for path in tqdm(file_paths, desc="Accumulating stats"):
        motion = np.load(path)  # shape [T, 132]
        if motion.ndim != 2:
            raise ValueError(f"Unexpected shape {motion.shape} in {path}")

        if sum_feats is None:
            sum_feats = np.zeros(motion.shape[1], dtype=np.float64)
            sum_sq_feats = np.zeros(motion.shape[1], dtype=np.float64)

        sum_feats += motion.sum(axis=0)
        sum_sq_feats += (motion ** 2).sum(axis=0)
        total_frames += motion.shape[0]

    # Compute mean and std
    mean = sum_feats / total_frames
    var = (sum_sq_feats / total_frames) - (mean ** 2)
    std = np.sqrt(np.maximum(var, 1e-8))

    # Save
    mean_path = os.path.join(out_dir, "mean_both.npy")
    std_path = os.path.join(out_dir, "std_both.npy")
    np.save(mean_path, mean.astype(np.float32))
    np.save(std_path, std.astype(np.float32))

    print(f"✅ Saved mean to {mean_path}")
    print(f"✅ Saved std to {std_path}")
    print(f"Mean shape: {mean.shape}, Std shape: {std.shape}")

if __name__ == "__main__":
    main()
