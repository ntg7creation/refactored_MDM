import os
import json
import numpy as np
from os.path import join as pjoin
from tqdm import tqdm

def load_jsonl_motion(file_path):
    """Load JSONL motion as [T, J, 3] array."""
    frames = []
    with open(file_path, "r") as f:
        for line in f:
            joints = json.loads(line.strip())
            coords = [[x, y, z] for (x, y, z, c) in joints]  # ignore confidence
            frames.append(coords)

    arr = np.array(frames, dtype=np.float32)  # [T, J, 3]
    return arr

def to_dmvb(motion_xyz):
    """
    Convert raw positions [T, J, 3] → DMVB [T, J*6].
    Each frame = [RIC, Vel].
    - RIC = positions with root joint centered
    - Vel = frame-to-frame difference of RIC (vel[0] = 0)
    """
    T, J, _ = motion_xyz.shape

    # Root-In-Center (RIC): subtract root joint (index 0)
    root = motion_xyz[:, 0:1, :]  # [T, 1, 3]
    ric = motion_xyz - root       # [T, J, 3]

    # Velocities (frame-to-frame difference in RIC)
    vel = np.zeros_like(ric)
    vel[1:] = ric[1:] - ric[:-1]  # first frame stays 0

    # Concatenate [RIC | Vel] per frame
    dmvb = np.concatenate([ric, vel], axis=-1)  # [T, J, 6]

    # Flatten to [T, J*6]
    return dmvb.reshape(T, -1)

def convert_all(data_root, out_root, side="left"):
    assert side in ["left", "right"]
    side_file = f"{side}_3d.jsonl"

    all_paths = []
    for root, _, files in os.walk(data_root):
        if side_file in files:
            motion_path = pjoin(root, side_file)
            relative = os.path.relpath(motion_path, data_root)
            # Replace .jsonl with _3d.npy
            out_path = pjoin(out_root, relative.replace("_3d.jsonl", "_3d.npy"))
            all_paths.append((motion_path, out_path))

    count = 0
    for motion_path, out_path in tqdm(all_paths, desc=f"Converting {side}_3d.jsonl to DMVB _3d.npy"):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        motion = load_jsonl_motion(motion_path)
        dmvb = to_dmvb(motion)
        np.save(out_path, dmvb)
        count += 1

    print(f"✅ Converted {count} {side}-hand files to DMVB _3d.npy in: {out_root}")


if __name__ == "__main__":
    # Hardcoded paths
    jsonl_root = "ASL_Data/jsonl"
    npy_root = "ASL_Data/dmvb"

    # Convert both left and right
    convert_all(jsonl_root, npy_root, side="left")
    convert_all(jsonl_root, npy_root, side="right")
