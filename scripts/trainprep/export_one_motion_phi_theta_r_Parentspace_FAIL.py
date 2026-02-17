import os
import json
import numpy as np
from tqdm import tqdm

# ============================================================
# Paths (match your converter defaults)
# ============================================================

REPO_ROOT = r"D:\repos\refactored_MDM"
RAW_MOTION_ROOT = r"D:\repos\GigaHands\dataset\GigaHands\hand_poses"
ANNOTATION_FILE = os.path.join(
    REPO_ROOT,
    "GigaHands_Data",
    "annotations_v2.jsonl",
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# Skeleton definition (same as converter)
# ============================================================

FINGERS = [
    [0, 1, 2, 3, 4],       # thumb
    [0, 5, 6, 7, 8],       # index
    [0, 9, 10, 11, 12],    # middle
    [0, 13, 14, 15, 16],   # ring
    [0, 17, 18, 19, 20],   # pinky
]

FINGER_ORDER = ["thumb", "index", "middle", "ring", "pinky"]
JOINT_ORDER = ["root", "mip", "pip", "dip"]

# ============================================================
# Utilities
# ============================================================

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def strip_w(frame):
    return [j[:3] for j in frame]

def safe_normalize(v, eps=1e-8):
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n

# ============================================================
# Angle extraction (same math as converter)
# ============================================================

def build_local_frame(z_axis, eps=1e-8):
    z = safe_normalize(z_axis, eps)
    if np.linalg.norm(z) < eps:
        return np.eye(3)

    up = np.array([0.0, 0.0, 1.0])
    x = np.cross(up, z)
    if np.linalg.norm(x) < eps:
        up = np.array([0.0, 1.0, 0.0])
        x = np.cross(up, z)

    x = safe_normalize(x)
    y = np.cross(z, x)
    return x, y, z

def vnext_to_theta_phi(v_prev, v_next):
    vp = safe_normalize(v_prev)
    vn = safe_normalize(v_next)

    if np.linalg.norm(vp) < 1e-6 or np.linalg.norm(vn) < 1e-6:
        return np.array([180.0, 0.0], dtype=np.float32)

    x, y, z = build_local_frame(vp)

    lx = np.dot(vn, x)
    ly = np.dot(vn, y)
    lz = np.clip(np.dot(vn, z), -1.0, 1.0)

    theta = np.degrees(np.arccos(lz))
    theta_open = 180.0 - theta
    phi = np.degrees(np.arctan2(ly, lx))

    if theta < 1e-3:
        phi = 0.0

    return np.array([theta_open, phi], dtype=np.float32)

def compute_hand_angles(xyz):
    """
    xyz: [T,21,3]
    returns [T,5,4,2] -> [theta, phi]
    """
    T = xyz.shape[0]
    out = np.zeros((T, 5, 4, 2), dtype=np.float32)

    for f_idx, finger in enumerate(FINGERS):
        j0, j1, j2, j3, j4 = finger

        for t in range(T):
            P = xyz[t]

            out[t, f_idx, 0] = vnext_to_theta_phi(P[j1]-P[j0], P[j2]-P[j1])
            out[t, f_idx, 1] = vnext_to_theta_phi(P[j2]-P[j1], P[j3]-P[j2])
            out[t, f_idx, 2] = vnext_to_theta_phi(P[j3]-P[j2], P[j4]-P[j3])
            out[t, f_idx, 3] = np.array([180.0, 0.0])

    return out

# ============================================================
# Main exporter
# ============================================================

def export_single_motion_from_annotation(
    annotation_index=0,
    out_jsonl="Synthetic_motions/real_motion_from_annotation.jsonl",
    R=1.0,
):
    annotations = load_jsonl(ANNOTATION_FILE)
    ann = annotations[annotation_index]

    scene = ann["scene"]
    sequence = ann["sequence"]
    start = ann["start_frame_id"]
    end = ann["end_frame_id"]

    print(f"[Annotation] Using index {annotation_index}")
    print(f"  scene={scene}, sequence={sequence}")
    print(f"  frames {start} → {end}")
    print(f"  text: {ann['description']}")

    motion_dir = os.path.join(
        RAW_MOTION_ROOT,
        scene,
        "keypoints_3d",
        sequence,
    )

    left = load_jsonl(os.path.join(motion_dir, "left.jsonl"))
    right = load_jsonl(os.path.join(motion_dir, "right.jsonl"))

    if end == -1:
        end = min(len(left), len(right))

    left_xyz = np.array([strip_w(f) for f in left[start:end]], dtype=np.float32)
    right_xyz = np.array([strip_w(f) for f in right[start:end]], dtype=np.float32)

    T = min(len(left_xyz), len(right_xyz))
    left_xyz = left_xyz[:T]
    right_xyz = right_xyz[:T]

    left_ang = compute_hand_angles(left_xyz)
    right_ang = compute_hand_angles(right_xyz)

    # angles_nested[T][2][5][4][3] -> [phi, theta, r]
    angles_nested = np.zeros((T, 2, 5, 4, 3), dtype=np.float32)
    angles_nested[..., 2] = R

    angles_nested[:, 0, :, :, 0] = left_ang[:, :, :, 1]
    angles_nested[:, 0, :, :, 1] = left_ang[:, :, :, 0]
    angles_nested[:, 1, :, :, 0] = right_ang[:, :, :, 1]
    angles_nested[:, 1, :, :, 1] = right_ang[:, :, :, 0]

    out_path = os.path.join(SCRIPT_DIR, out_jsonl)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for t in tqdm(range(T), desc="Writing frames"):
            f.write(json.dumps({
                "frame": t,
                "angles_nested": angles_nested[t].tolist(),
            }) + "\n")

    print(f"[Done] Wrote {T} frames to:")
    print(out_path)

# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    export_single_motion_from_annotation(
        annotation_index=56,   # 👈 change this to test another annotation
        R=1.0,
    )
