

import os
import json
import numpy as np
from tqdm import tqdm

# Constants for MDM format
NUM_JOINTS = 22
RIC_DIM = (NUM_JOINTS - 1) * 3
ROT6D_DIM = (NUM_JOINTS - 1) * 6
VEL_DIM = NUM_JOINTS * 3
TOTAL_DIM = 1 + 2 + 1 + RIC_DIM + ROT6D_DIM + VEL_DIM + 4  # = 263
FPS = 20  # Default HumanML3D frame rate

def load_jsonl(filepath):
    with open(filepath, "r") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]

def strip_w(pts):
    return np.array([[x, y, z] for x, y, z, _ in pts], dtype=np.float32)

def compute_velocity(positions):
    return positions[1:] - positions[:-1]  # [T-1, J, 3]

def compute_root_features(positions):
    root_pos = positions[:, 0]  # root is at index 0
    delta = root_pos[1:] - root_pos[:-1]  # [T-1, 3]
    rot_vel = np.arctan2(delta[:, 0], delta[:, 2]).reshape(-1, 1)  # Y-rotation only
    lin_vel = delta[:, [0, 2]]  # XZ
    root_y = root_pos[:-1, 1:2]  # Y
    root_feats = np.concatenate([rot_vel, lin_vel, root_y], axis=1)
    # print("Root features shape:", root_feats.shape)
    return root_feats  # [T-1, 4]


def convert_sequence(frames_xyz):
    """
    Convert raw 3D joint sequence into DMVB format:
    - No root duplication
    - No extra root_feats, rot6d, contact flags
    - Output shape: [T-1, 132] where 132 = 66 (xyz) + 66 (velocity)
    """
    positions = np.array(frames_xyz, dtype=np.float32)  # [T, 22, 3]
    velocities = positions[1:] - positions[:-1]         # [T-1, 22, 3]
    positions = positions[:-1]                          # Align with velocity

    flat_xyz = positions.reshape(positions.shape[0], -1)  # [T-1, 66]
    flat_vel = velocities.reshape(velocities.shape[0], -1)  # [T-1, 66]

    full_dmvb = np.concatenate([flat_xyz, flat_vel], axis=1)  # [T-1, 132]

    return full_dmvb


def main():
    base_path = os.path.join(os.path.dirname(__file__), "..", "GigaHands")
    annotation_path = os.path.join(base_path, "annotations_v2.jsonl")
    annotations = load_jsonl(annotation_path)

    for ann in tqdm(annotations):
        seq_name = ann["sequence"]
        scene = ann["scene"]

        motion_dir = os.path.join(base_path, "hand_poses", scene, "keypoints_3d", seq_name)
        output_dir = os.path.join(os.path.dirname(__file__), "..", "converted_motions", "hand_poses_dmvb", scene, "keypoints_3d", seq_name)
        os.makedirs(output_dir, exist_ok=True)

        out_left_path = os.path.join(output_dir, "dmvb_left.npy")
        out_right_path = os.path.join(output_dir, "dmvb_right.npy")

        # 🛑 Skip if both files already exist
        if os.path.exists(out_left_path) and os.path.exists(out_right_path):
            continue

        try:
            left = load_jsonl(os.path.join(motion_dir, "left.jsonl"))
            right = load_jsonl(os.path.join(motion_dir, "right.jsonl"))

            left = [strip_w(f) for f in left]
            right = [strip_w(f) for f in right]

            motion_left = convert_sequence(left)
            motion_right = convert_sequence(right)

            if not os.path.exists(out_left_path):
                np.save(out_left_path, motion_left)
            if not os.path.exists(out_right_path):
                np.save(out_right_path, motion_right)

        except Exception as e:
            print(f"❌ Error processing {scene}/{seq_name}: {e}")

if __name__ == "__main__":
    main()

