import os
import numpy as np
from tqdm import tqdm


def collect_all_motions(root_dir):
    """
    Recursively collect all xyz_both_vel.npy files under root_dir.
    """
    file_paths = []
    for scene, _, files in os.walk(root_dir):
        for f in files:
            if f.endswith("xyz_both_vel.npy"):
                file_paths.append(os.path.join(scene, f))
    return file_paths


def main():
    # ----------------------------------------
    # PATHS (parallel to your original script)
    # ----------------------------------------
    repo_root = r"D:\repos\refactored_MDM"
    root_dir = os.path.join(repo_root, "GigaHands_Data", "converted_velocity", "hand_poses_xyz")
    out_dir = os.path.join(repo_root, "GigaHands_Data", "converted_velocity", "norm_stats")

    os.makedirs(out_dir, exist_ok=True)

    # ----------------------------------------
    # Collect files
    # ----------------------------------------
    file_paths = collect_all_motions(root_dir)
    assert len(file_paths) > 0, f"No xyz_both_vel.npy found under {root_dir}"

    print(f"Found {len(file_paths)} velocity-augmented motion files")

    total_frames = 0
    sum_feats = None
    sum_sq_feats = None

    # ----------------------------------------
    # Accumulate per-frame statistics
    # ----------------------------------------
    for path in tqdm(file_paths, desc="Accumulating mean/std over velocity motions"):
        motion = np.load(path)   # shape [T, 252]

        # If motion is saved as [42,6,T], reshape back to [T,252]
        if motion.ndim == 3 and motion.shape[0] == 42:
            T = motion.shape[2]
            motion = motion.transpose(2, 0, 1).reshape(T, -1)  # [T,252]

        if motion.ndim != 2 or motion.shape[1] != 252:
            raise ValueError(f"Unexpected shape {motion.shape} in {path}")

        if sum_feats is None:
            sum_feats = np.zeros(motion.shape[1], dtype=np.float64)
            sum_sq_feats = np.zeros(motion.shape[1], dtype=np.float64)

        sum_feats += motion.sum(axis=0)
        sum_sq_feats += (motion ** 2).sum(axis=0)
        total_frames += motion.shape[0]

    # ----------------------------------------
    # Compute mean/std
    # ----------------------------------------
    mean = sum_feats / total_frames
    var = (sum_sq_feats / total_frames) - (mean ** 2)
    std = np.sqrt(np.maximum(var, 1e-8))

    # ----------------------------------------
    # Save results
    # ----------------------------------------
    mean_path = os.path.join(out_dir, "mean_both_vel.npy")
    std_path = os.path.join(out_dir, "std_both_vel.npy")

    np.save(mean_path, mean.astype(np.float32))
    np.save(std_path, std.astype(np.float32))

    print(f"\n✅ Saved mean to: {mean_path}")
    print(f"✅ Saved std  to: {std_path}")
    print(f"Mean shape: {mean.shape}")
    print(f"Std  shape: {std.shape}")


if __name__ == "__main__":
    main()
