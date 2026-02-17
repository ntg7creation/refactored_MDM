

import os
import json
import numpy as np
from tqdm import tqdm

def load_jsonl(filepath):
    with open(filepath, "r") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]

def strip_w(pts):
    """Remove confidence (w) column -> keep [x, y, z]."""
    return np.array([[x, y, z] for x, y, z, _ in pts], dtype=np.float32)

def convert_sequence_xyz(frames_xyz):
    """
    Convert raw 3D joint sequence into flat XYZ positions only.
    Output shape: [T, 63] where 63 = 21 joints × 3 coords.
    """
    positions = np.array(frames_xyz, dtype=np.float32)  # [T, 21, 3]
    flat_xyz = positions.reshape(positions.shape[0], -1)  # [T, 63]
    return flat_xyz

def main():
    # Paths
    repo_root = r"D:\repos\refactored_MDM"
    annotations_path = os.path.join(repo_root, "GigaHands_Data", "annotations_v2.jsonl")
    raw_motion_root = r"D:\repos\GigaHands\dataset\GigaHands\hand_poses"
    output_root = os.path.join(repo_root, "GigaHands_Data", "coverted_motions", "hand_poses_xyz")

    annotations = load_jsonl(annotations_path)

    for ann in tqdm(annotations, desc="Converting GigaHands (XYZ only, 2 Hands, chosen frames)"):
        seq_name = ann["sequence"]
        scene = ann["scene"]

        # Input directories
        motion_dir = os.path.join(raw_motion_root, scene, "keypoints_3d", seq_name)

        # Output directories
        output_dir = os.path.join(output_root, scene, "keypoints_3d", seq_name)
        os.makedirs(output_dir, exist_ok=True)

        out_both_path = os.path.join(output_dir, "xyz_both.npy")

        # Skip if already exists
        if os.path.exists(out_both_path):
            continue

        try:
            # Load full frame sequences
            left_frames = load_jsonl(os.path.join(motion_dir, "left.jsonl"))
            right_frames = load_jsonl(os.path.join(motion_dir, "right.jsonl"))

            # Load chosen frame indices
            with open(os.path.join(motion_dir, "chosen_frames_left.json"), "r") as f:
                chosen_left = json.load(f)
            with open(os.path.join(motion_dir, "chosen_frames_right.json"), "r") as f:
                chosen_right = json.load(f)

            # Use intersection to keep alignment between left and right
            common_indices = sorted(set(chosen_left).intersection(set(chosen_right)))

            if len(common_indices) == 0:
                print(f"⚠️ No common chosen frames for {scene}/{seq_name}, skipping.")
                continue

            # Filter frames
            left_filtered = [left_frames[i] for i in common_indices if i < len(left_frames)]
            right_filtered = [right_frames[i] for i in common_indices if i < len(right_frames)]

            # Convert to flat xyz
            motion_left = convert_sequence_xyz([strip_w(f) for f in left_filtered])   # [T, 63]
            motion_right = convert_sequence_xyz([strip_w(f) for f in right_filtered]) # [T, 63]

            # Ensure same length (safety check)
            min_len = min(len(motion_left), len(motion_right))
            motion_left = motion_left[:min_len]
            motion_right = motion_right[:min_len]

            # Combine: [XYZ_left, XYZ_right] -> [T, 126]
            motion_combined = np.concatenate([motion_left, motion_right], axis=1)

            # Save combined file
            np.save(out_both_path, motion_combined)
            # Optional debug print
            # print(f"✅ Saved {out_both_path}, shape={motion_combined.shape}")

        except Exception as e:
            print(f"❌ Error processing {scene}/{seq_name}: {e}")

if __name__ == "__main__":
    main()
