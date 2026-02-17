import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm


def load_jsonl(path):
    """Load list of JSON lines."""
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def strip_w(frame):
    """Remove 4th value 'w' from each joint: [x,y,z,w] -> [x,y,z]."""
    return [joint[:3] for joint in frame]


def convert_sequence_xyz(seq_frames):
    """
    seq_frames: list of frames, each frame is list of 21 joints each [x,y,z]
    returns shape [T, 63]
    """
    T = len(seq_frames)
    arr = np.zeros((T, 21 * 3), dtype=np.float32)
    for t, frame in enumerate(seq_frames):
        flat = []
        for j in frame:
            flat.extend(j)      # XYZ
        arr[t] = np.array(flat, dtype=np.float32)
    return arr


def compute_velocity(xyz):
    """
    xyz: [T, D]
    vel: [T, D] with vel[0] = 0
    """
    vel = np.zeros_like(xyz)
    vel[1:] = xyz[1:] - xyz[:-1]
    return vel


def main():
    # ----------------------------------
    # SAME PATHS AS YOUR ORIGINAL SCRIPT
    # ----------------------------------
    repo_root = r"D:\repos\refactored_MDM"
    annotations_path = os.path.join(repo_root, "GigaHands_Data", "annotations_v2.jsonl")
    raw_motion_root = r"D:\repos\GigaHands\dataset\GigaHands\hand_poses"

    # NEW OUTPUT DIRECTORY
    output_root = os.path.join(repo_root, "GigaHands_Data", "converted_velocity", "hand_poses_xyz")

    # Load annotation list
    annotations = load_jsonl(annotations_path)

    for ann in tqdm(annotations, desc="Converting GigaHands → XYZ + Velocity (2 Hands)"):
        seq_name = ann["sequence"]
        scene = ann["scene"]

        # Input dirs
        motion_dir = os.path.join(raw_motion_root, scene, "keypoints_3d", seq_name)

        # Output dirs
        output_dir = os.path.join(output_root, scene, "keypoints_3d", seq_name)
        os.makedirs(output_dir, exist_ok=True)

        out_both_vel_path = os.path.join(output_dir, "xyz_both_vel.npy")

        # Skip existing
        if os.path.exists(out_both_vel_path):
            continue

        try:
            # --------------------------------------
            # Load RAW frames (left.jsonl, right.jsonl)
            # --------------------------------------
            left_frames = load_jsonl(os.path.join(motion_dir, "left.jsonl"))
            right_frames = load_jsonl(os.path.join(motion_dir, "right.jsonl"))

            # --------------------------------------
            # Load chosen frames to keep alignment
            # --------------------------------------
            with open(os.path.join(motion_dir, "chosen_frames_left.json")) as f:
                chosen_left = json.load(f)
            with open(os.path.join(motion_dir, "chosen_frames_right.json")) as f:
                chosen_right = json.load(f)

            # Intersection
            common_indices = sorted(set(chosen_left).intersection(set(chosen_right)))
            if len(common_indices) == 0:
                print(f"⚠️ No aligned chosen frames for {scene}/{seq_name}, skipping.")
                continue

            # --------------------------------------
            # Filter frames
            # --------------------------------------
            left_filtered = [left_frames[i] for i in common_indices if i < len(left_frames)]
            right_filtered = [right_frames[i] for i in common_indices if i < len(right_frames)]

            # --------------------------------------
            # Strip w + convert to XYZ
            # --------------------------------------
            left_xyz  = convert_sequence_xyz([strip_w(f) for f in left_filtered])   # [T, 63]
            right_xyz = convert_sequence_xyz([strip_w(f) for f in right_filtered])  # [T, 63]

            # Align lengths
            T = min(len(left_xyz), len(right_xyz))
            left_xyz = left_xyz[:T]
            right_xyz = right_xyz[:T]

            # --------------------------------------
            # Combine into [T, 126]
            # --------------------------------------
            xyz_both = np.concatenate([left_xyz, right_xyz], axis=1)

            # --------------------------------------
            # Compute velocity → [T,126]
            # --------------------------------------
            vel_both = compute_velocity(xyz_both)

            # --------------------------------------
            # Final output [T, 252] = [xyz | vel]
            # --------------------------------------
            xyz_vel = np.concatenate([xyz_both, vel_both], axis=1)  # [T,252]

            # (optional) reshape for mdm [42,6,T]
            # joints = 42 (21 per hand)
            # xyz_vel = xyz_vel.reshape(T, 42, 6).transpose(1, 2, 0)

            # --------------------------------------
            # Save velocity-augmented motion
            # --------------------------------------
            np.save(out_both_vel_path, xyz_vel)

        except Exception as e:
            print(f"❌ Error processing {scene}/{seq_name}: {e}")


if __name__ == "__main__":
    main()
