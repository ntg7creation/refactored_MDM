import os
import json
import numpy as np
from tqdm import tqdm


# ============================================================
# Hand skeleton (same indexing you described)
# ============================================================
FINGERS = [
    [0, 1, 2, 3, 4],       # finger 1
    [0, 5, 6, 7, 8],       # finger 2
    [0, 9, 10, 11, 12],    # finger 3
    [0, 13, 14, 15, 16],   # finger 4
    [0, 17, 18, 19, 20],   # finger 5
]


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def strip_w(frame):
    """Convert joints [x,y,z,w] -> [x,y,z]."""
    return [j[:3] for j in frame]


def frames_to_xyz(frames_21x3):
    """list[T][21][3] -> np[T,21,3]."""
    T = len(frames_21x3)
    arr = np.zeros((T, 21, 3), dtype=np.float32)
    for t, frame in enumerate(frames_21x3):
        arr[t] = np.asarray(frame, dtype=np.float32)
    return arr


def safe_normalize(v, eps=1e-8):
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v, dtype=np.float32)
    return (v / n).astype(np.float32)


def build_local_frame(z_axis, fallback=np.array([0.0, 0.0, 1.0], dtype=np.float32), eps=1e-8):
    """
    Build an orthonormal basis (x,y,z) where z=z_axis (normalized).
    x is chosen via cross(fallback, z); if near-parallel, use a different fallback.
    """
    z = safe_normalize(z_axis, eps=eps)
    if np.linalg.norm(z) < eps:
        # Degenerate; return identity basis
        return (
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.0, 1.0], dtype=np.float32),
        )

    x = np.cross(fallback, z).astype(np.float32)
    if np.linalg.norm(x) < eps:
        # fallback was parallel; pick another axis
        fallback2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        x = np.cross(fallback2, z).astype(np.float32)

    x = safe_normalize(x, eps=eps)
    y = np.cross(z, x).astype(np.float32)
    y = safe_normalize(y, eps=eps)

    return x, y, z


def vnext_to_theta_phi(v_prev, v_next, eps=1e-8):
    """
    v_prev: parent->joint direction
    v_next: joint->child direction

    Returns (theta_open_deg, phi_deg):
      - theta_open_deg = 180 - theta_deg, so straight finger ~ 180.
      - phi_deg in [-180, 180], rotation about v_prev axis.
    """
    vp = safe_normalize(v_prev, eps=eps)
    vn = safe_normalize(v_next, eps=eps)

    if np.linalg.norm(vp) < eps or np.linalg.norm(vn) < eps:
        # Degenerate -> treat as open
        return np.array([180.0, 0.0], dtype=np.float32)

    x, y, z = build_local_frame(vp, eps=eps)

    # Express v_next in local frame
    lx = float(np.dot(vn, x))
    ly = float(np.dot(vn, y))
    lz = float(np.dot(vn, z))

    # theta (0..pi)
    lz = float(np.clip(lz, -1.0, 1.0))
    theta = np.arccos(lz)  # radians
    theta_deg = np.degrees(theta)

    # Convert to your "open hand = 180°" convention
    theta_open_deg = 180.0 - theta_deg

    # phi (-pi..pi)
    phi = np.arctan2(ly, lx)
    phi_deg = np.degrees(phi)

    # If nearly straight, phi is unstable; pin to 0 for cleanliness
    if theta_deg < 1e-3:
        phi_deg = 0.0

    return np.array([theta_open_deg, phi_deg], dtype=np.float32)


def compute_hand_theta_phi(xyz_hand):
    """
    xyz_hand: [T,21,3]
    returns: out [T,5,4,2]
      5 fingers
      4 joints per finger: (root, mip, pip, dip/tip)
      2 angles per joint: (theta_open_deg, phi_deg)
    """
    T = xyz_hand.shape[0]
    out = np.zeros((T, 5, 4, 2), dtype=np.float32)

    for f_idx, finger in enumerate(FINGERS):
        j0, j1, j2, j3, j4 = finger  # wrist, root, mip, pip, tip

        for t in range(T):
            P = xyz_hand[t]

            # root @ j1: parent=j0, child=j2
            v_prev = P[j1] - P[j0]   # parent -> joint
            v_next = P[j2] - P[j1]   # joint  -> child
            out[t, f_idx, 0] = vnext_to_theta_phi(v_prev, v_next)

            # mip @ j2: parent=j1, child=j3
            v_prev = P[j2] - P[j1]
            v_next = P[j3] - P[j2]
            out[t, f_idx, 1] = vnext_to_theta_phi(v_prev, v_next)

            # pip @ j3: parent=j2, child=j4
            v_prev = P[j3] - P[j2]
            v_next = P[j4] - P[j3]
            out[t, f_idx, 2] = vnext_to_theta_phi(v_prev, v_next)

            # dip/tip (no child): default open
            out[t, f_idx, 3] = np.array([180.0, 0.0], dtype=np.float32)

    return out


def main():
    # ----------------------------------
    # Defaults copied from your velocity script (edit if needed)
    # ----------------------------------
    repo_root = r"D:\repos\refactored_MDM"
    annotations_path = os.path.join(repo_root, "GigaHands_Data", "annotations_v2.jsonl")
    raw_motion_root = r"D:\repos\GigaHands\dataset\GigaHands\hand_poses"

    # Output roots (TWO folders: npy + json)
    output_root_base = os.path.join(repo_root, "GigaHands_Data", "converted_angles_phi_theta")
    output_root_npy = os.path.join(output_root_base, "npy")
    output_root_json = os.path.join(output_root_base, "json")

    annotations = load_jsonl(annotations_path)

    for ann in tqdm(annotations, desc="Converting GigaHands → (theta,phi) (both hands)"):
        seq_name = ann["sequence"]
        scene = ann["scene"]
        motion_dir = os.path.join(raw_motion_root, scene, "keypoints_3d", seq_name)

        # Output dirs
        out_dir_npy = os.path.join(output_root_npy, scene, "keypoints_3d", seq_name)
        out_dir_json = os.path.join(output_root_json, scene, "keypoints_3d", seq_name)
        os.makedirs(out_dir_npy, exist_ok=True)
        os.makedirs(out_dir_json, exist_ok=True)

        out_npy_path = os.path.join(out_dir_npy, "angles_both.npy")          # [T,80]
        out_jsonl_path = os.path.join(out_dir_json, "angles_both.jsonl")

        if os.path.exists(out_npy_path) and os.path.exists(out_jsonl_path):
            continue

        try:
            left_frames = load_jsonl(os.path.join(motion_dir, "left.jsonl"))
            right_frames = load_jsonl(os.path.join(motion_dir, "right.jsonl"))

            with open(os.path.join(motion_dir, "chosen_frames_left.json"), "r", encoding="utf-8") as f:
                chosen_left = json.load(f)
            with open(os.path.join(motion_dir, "chosen_frames_right.json"), "r", encoding="utf-8") as f:
                chosen_right = json.load(f)

            common_indices = sorted(set(chosen_left).intersection(set(chosen_right)))
            if len(common_indices) == 0:
                print(f"⚠️ No aligned chosen frames for {scene}/{seq_name}, skipping.")
                continue

            left_filtered = [left_frames[i] for i in common_indices if i < len(left_frames)]
            right_filtered = [right_frames[i] for i in common_indices if i < len(right_frames)]

            left_xyz = frames_to_xyz([strip_w(f) for f in left_filtered])
            right_xyz = frames_to_xyz([strip_w(f) for f in right_filtered])

            T = min(left_xyz.shape[0], right_xyz.shape[0])
            if T == 0:
                print(f"⚠️ Empty after alignment for {scene}/{seq_name}, skipping.")
                continue
            left_xyz = left_xyz[:T]
            right_xyz = right_xyz[:T]

            # Compute (theta,phi) per hand: [T,5,4,2]
            left_ang = compute_hand_theta_phi(left_xyz)
            right_ang = compute_hand_theta_phi(right_xyz)

            # Pack both hands: [T,2,5,4,2]
            both_nested = np.stack([left_ang, right_ang], axis=1)

            # Flatten to [T,80]
            both_flat = both_nested.reshape(T, 80)

            np.save(out_npy_path, both_flat)

            with open(out_jsonl_path, "w", encoding="utf-8") as f:
                for t in range(T):
                    rec = {
                        "frame": t,
                        "angles_nested": both_nested[t].tolist(),  # [2][5][4][2]
                        "angles_flat": both_flat[t].tolist(),      # [80]
                        "layout": {
                            "hands": ["left", "right"],
                            "fingers": 5,
                            "joints_per_finger": ["root", "mip", "pip", "dip"],
                            "angles_per_joint": ["theta_open_deg", "phi_deg"],
                        }
                    }
                    f.write(json.dumps(rec) + "\n")

        except Exception as e:
            print(f"❌ Error processing {scene}/{seq_name}: {e}")


if __name__ == "__main__":
    main()
