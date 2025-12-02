import json
import numpy as np

def load_jsonl(path):
    with open(path, "r") as f:
        lines = f.readlines()
    return [json.loads(line.strip()) for line in lines]

def preprocess_frame(frame, num_joints=6):
    return np.array([joint[:3] for joint in frame[:num_joints]])  # shape: [6, 3]

def flatten_motion(frames, num_joints=6):
    return np.array([preprocess_frame(f).reshape(-1) for f in frames])  # [T, 18]

def compute_mse(gt_array, pred_array):
    assert gt_array.shape == pred_array.shape, f"Shape mismatch: {gt_array.shape} vs {pred_array.shape}"
    return np.mean((gt_array - pred_array) ** 2)

def main():
    gt_path = r"D:\repos\Hand_Motion_Viewer\public\hand_poses\p001-folder\keypoints_3d\000\left.jsonl"
    pred_path = r"D:\repos\refactored_MDM\save\test_run\infer_test\reconstructed.jsonl"
    mean_path = r"D:\repos\mdm_custom_training\converted_motions\hand_poses\norm_stats\mean_left.npy"
    std_path = r"D:\repos\mdm_custom_training\converted_motions\hand_poses\norm_stats\std_left.npy"

    print(f"🔍 Loading motion files...")
    gt_frames = load_jsonl(gt_path)
    pred_frames = load_jsonl(pred_path)
    print(f"✅ Loaded {len(gt_frames)} GT frames, {len(pred_frames)} predicted frames")

    print("📊 Loading mean and std from dataset...")
    mean = np.load(mean_path)[:18]
    std = np.load(std_path)[:18]

    gt_array = flatten_motion(gt_frames)         # shape: [T, 18]
    pred_array = flatten_motion(pred_frames)     # shape: [T, 18]

    # Denormalize prediction (trained model uses normalized data)
    pred_array = (pred_array * std) + mean

    mse = compute_mse(gt_array, pred_array)
    print(f"📉 MSE over first 6 joints: {mse:.8f}")

if __name__ == "__main__":
    main()
