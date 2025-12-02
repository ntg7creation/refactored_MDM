import json
import numpy as np

def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line.strip()) for line in f]

def preprocess_frame(frame, num_joints=6):
    # frame: [[x, y, z, 1.0], ...] → take first 6 joints, strip the trailing 1.0
    return np.array([joint[:3] for joint in frame[:num_joints]])  # [6, 3]

def flatten_motion(frames, num_joints=6):
    # list-of-frames → [T, 18]
    return np.array([preprocess_frame(f).reshape(-1) for f in frames], dtype=np.float64)

def coerce_T18(arr):
    """
    Coerce npy-loaded dmvb to shape [T, 18].
    Accepts [T, 18], [18, T], or [..., 18] last-dim.
    """
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 2:
        # [T, 18] or [18, T]
        if arr.shape[1] == 18:
            return arr
        if arr.shape[0] == 18:
            return arr.T
    if arr.ndim >= 2 and arr.shape[-1] >= 18:
        # take first 18 of last dim and flatten leading dims into T
        T = int(np.prod(arr.shape[:-1]))
        return arr.reshape(T, arr.shape[-1])[:, :18]
    raise ValueError(f"dmvb array has unsupported shape {arr.shape}; expected [T,18] or [18,T].")

def compute_mse(a, b):
    assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"
    return np.mean((a - b) ** 2)

def main():
    # --- Paths ---
    pred_path = r"D:\repos\refactored_MDM\save\test_run\infer_test\reconstructed.jsonl"
    dmvb_path = r"D:\repos\mdm_custom_training\converted_motions\hand_poses\p001-folder\keypoints_3d\000\dmvb_left.npy"
    mean_path = r"D:\repos\mdm_custom_training\converted_motions\hand_poses\norm_stats\mean_left.npy"
    std_path  = r"D:\repos\mdm_custom_training\converted_motions\hand_poses\norm_stats\std_left.npy"

    print("🔍 Loading prediction (jsonl) and training target (dmvb_left.npy)...")
    pred_frames = load_jsonl(pred_path)                      # list of frames (each frame: [[x,y,z,1.0], ...])
    pred_T18     = flatten_motion(pred_frames)               # [T_pred, 18]

    dmvb_raw     = np.load(dmvb_path)                        # unknown shape, coerce next
    gt_T18       = coerce_T18(dmvb_raw)                      # [T_gt, 18]

    print(f"✅ Pred shape: {pred_T18.shape}, GT(dmvb) shape: {gt_T18.shape}")

    # --- Load stats & normalize prediction into training space ---
    mean = np.load(mean_path)[:18].astype(np.float64)
    std  = np.load(std_path)[:18].astype(np.float64)
    std  = np.clip(std, 1e-8, None)                          # avoid divide-by-zero

    pred_norm = (pred_T18 - mean) / std                      # compare in normalized feature space

    # --- Align time dimension ---
    T = min(pred_norm.shape[0], gt_T18.shape[0])
    if pred_norm.shape[0] != gt_T18.shape[0]:
        print(f"ℹ️ Truncating to common length T={T} (pred={pred_norm.shape[0]}, gt={gt_T18.shape[0]})")
    pred_aligned = pred_norm[:T]
    gt_aligned   = gt_T18[:T, :18]                           # keep first 18 dims just in case

    # --- Compute loss ---
    mse = compute_mse(gt_aligned, pred_aligned)
    print(f"📉 MSE in normalized (training) space vs dmvb_left.npy: {mse:.8f}")

if __name__ == "__main__":
    main()
