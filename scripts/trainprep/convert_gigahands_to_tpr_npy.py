import os
import json
import numpy as np
from tqdm import tqdm

# ============================================================
# Paths
# ============================================================

REPO_ROOT = r"D:\repos\refactored_MDM"
RAW_MOTION_ROOT = r"D:\repos\GigaHands\dataset\GigaHands\hand_poses"

OUT_ROOT = os.path.join(REPO_ROOT, "GigaHands_Data", "converted_tpr", "hand_poses_xyz")

# ============================================================
# Skeleton definition (GigaHands 21 joints)
# ============================================================

FINGERS = [
    [0, 1, 2, 3, 4],       # thumb
    [0, 5, 6, 7, 8],       # index
    [0, 9, 10, 11, 12],    # middle
    [0, 13, 14, 15, 16],   # ring
    [0, 17, 18, 19, 20],   # pinky
]

# ============================================================
# IO utils
# ============================================================

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def strip_w(frame):
    # each joint is [x,y,z,w?] -> keep xyz
    return [j[:3] for j in frame]

# ============================================================
# WORLD-SPACE theta/phi/r extraction (matches your JS viewer)
# ============================================================
# JS convention (world space):
#   x = r * cos(theta) * sin(phi)
#   y = r * sin(theta)
#   z = r * cos(theta) * cos(phi)
#
# Inverse:
#   r = ||v||
#   theta = asin(y/r)
#   phi = atan2(x, z)
#
# Output triple format:
#   [thetaDeg, phiDeg, r]
# ============================================================

def v_to_theta_phi_r_world(v, eps=1e-8):
    v = np.asarray(v, dtype=np.float32)
    r = float(np.linalg.norm(v))
    if r < eps:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    x, y, z = float(v[0]), float(v[1]), float(v[2])

    theta = np.degrees(np.arcsin(np.clip(y / r, -1.0, 1.0)))
    phi = np.degrees(np.arctan2(x, z))  # IMPORTANT: atan2(x, z) matches the JS mapping

    return np.array([theta, phi, r], dtype=np.float32)

def compute_hand_triples_world(xyz):
    """
    xyz: [T,21,3]
    returns: [T,5,4,3] where triple = [thetaDeg, phiDeg, r] in WORLD SPACE
    segments per finger:
      seg0 = j1-j0
      seg1 = j2-j1
      seg2 = j3-j2
      seg3 = j4-j3
    """
    xyz = np.asarray(xyz, dtype=np.float32)
    T = xyz.shape[0]
    out = np.zeros((T, 5, 4, 3), dtype=np.float32)

    for f_idx, finger in enumerate(FINGERS):
        j0, j1, j2, j3, j4 = finger
        for t in range(T):
            P = xyz[t]
            out[t, f_idx, 0] = v_to_theta_phi_r_world(P[j1] - P[j0])
            out[t, f_idx, 1] = v_to_theta_phi_r_world(P[j2] - P[j1])
            out[t, f_idx, 2] = v_to_theta_phi_r_world(P[j3] - P[j2])
            out[t, f_idx, 3] = v_to_theta_phi_r_world(P[j4] - P[j3])

    return out

def pack_both_hands_tpr(left_xyz, right_xyz):
    """
    left_xyz/right_xyz: [T,21,3]
    returns:
      tpr_both: [T, 120] float32
      where 120 = 2 hands * 5 fingers * 4 segs * 3 (theta,phi,r)
    """
    T = min(len(left_xyz), len(right_xyz))
    left_xyz = left_xyz[:T]
    right_xyz = right_xyz[:T]

    left_tpr = compute_hand_triples_world(left_xyz)   # [T,5,4,3]
    right_tpr = compute_hand_triples_world(right_xyz) # [T,5,4,3]

    both = np.stack([left_tpr, right_tpr], axis=1)    # [T,2,5,4,3]
    tpr_both = both.reshape(T, -1).astype(np.float32) # [T,120]
    return tpr_both

# ============================================================
# Directory walk + conversion
# ============================================================

def find_sequences(raw_root):
    """
    Yield tuples: (scene_name, sequence_name, motion_dir)
    motion_dir contains: keypoints_3d/<sequence>/left.jsonl,right.jsonl
    """
    # RAW_MOTION_ROOT/<scene>/keypoints_3d/<sequence>/*.jsonl
    if not os.path.isdir(raw_root):
        raise FileNotFoundError(f"RAW_MOTION_ROOT not found: {raw_root}")

    for scene in sorted(os.listdir(raw_root)):
        scene_dir = os.path.join(raw_root, scene)
        if not os.path.isdir(scene_dir):
            continue

        kp3d_dir = os.path.join(scene_dir, "keypoints_3d")
        if not os.path.isdir(kp3d_dir):
            continue

        for seq in sorted(os.listdir(kp3d_dir)):
            motion_dir = os.path.join(kp3d_dir, seq)
            if not os.path.isdir(motion_dir):
                continue

            left_path = os.path.join(motion_dir, "left.jsonl")
            right_path = os.path.join(motion_dir, "right.jsonl")
            if os.path.isfile(left_path) and os.path.isfile(right_path):
                yield scene, seq, motion_dir

def convert_all(save_overwrite=False):
    seqs = list(find_sequences(RAW_MOTION_ROOT))
    if len(seqs) == 0:
        raise RuntimeError(f"No sequences found under: {RAW_MOTION_ROOT}")

    print(f"Found {len(seqs)} sequences")

    for scene, seq, motion_dir in tqdm(seqs, desc="Converting to tpr_both.npy"):
        left_path = os.path.join(motion_dir, "left.jsonl")
        right_path = os.path.join(motion_dir, "right.jsonl")

        # Output path mirrors:
        # GigaHands_Data/converted_tpr/hand_poses_xyz/<scene>/keypoints_3d/<seq>/tpr_both.npy
        out_dir = os.path.join(OUT_ROOT, scene, "keypoints_3d", seq)
        out_path = os.path.join(out_dir, "tpr_both.npy")

        if (not save_overwrite) and os.path.isfile(out_path):
            continue

        os.makedirs(out_dir, exist_ok=True)

        left = load_jsonl(left_path)
        right = load_jsonl(right_path)

        if len(left) == 0 or len(right) == 0:
            continue

        left_xyz = np.array([strip_w(f) for f in left], dtype=np.float32)   # [T,21,3]
        right_xyz = np.array([strip_w(f) for f in right], dtype=np.float32)

        tpr_both = pack_both_hands_tpr(left_xyz, right_xyz)                 # [T,120]
        np.save(out_path, tpr_both)

def main():
    convert_all(save_overwrite=False)
    print("\n✅ Done. Output root:")
    print(OUT_ROOT)

if __name__ == "__main__":
    main()
