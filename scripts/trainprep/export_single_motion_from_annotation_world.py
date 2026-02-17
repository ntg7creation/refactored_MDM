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
# Utilities
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
# so it matches your viewer's triple=[thetaDeg, phiDeg, r]
# ============================================================

def v_to_theta_phi_r_world(v, eps=1e-8):
    v = np.asarray(v, dtype=np.float32)
    r = float(np.linalg.norm(v))
    if r < eps:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    x, y, z = float(v[0]), float(v[1]), float(v[2])

    theta = np.degrees(np.arcsin(np.clip(y / r, -1.0, 1.0)))
    phi = np.degrees(np.arctan2(x, z))  # IMPORTANT: atan2(x, z) matches the JS mapping above

    return np.array([theta, phi, r], dtype=np.float32)

def compute_hand_triples_world(xyz):
    """
    xyz: [T,21,3]
    returns: [T,5,4,3] where triple = [thetaDeg, phiDeg, r] in WORLD SPACE
    segments are vectors:
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

# ============================================================
# Main exporter
# ============================================================

def export_single_motion_from_annotation_world(
    annotation_index=0,
    out_jsonl="Synthetic_motions/real_motion_from_annotation_world.jsonl",
    force_R=None,  # set to a float (e.g. 1.0) to override all r values; otherwise keep true lengths
):
    annotations = load_jsonl(ANNOTATION_FILE)
    if annotation_index < 0 or annotation_index >= len(annotations):
        raise IndexError(f"annotation_index out of range: {annotation_index} (0..{len(annotations)-1})")

    ann = annotations[annotation_index]

    scene = ann["scene"]
    sequence = ann["sequence"]
    start = int(ann.get("start_frame_id", 0))
    end = int(ann.get("end_frame_id", -1))

    print(f"[Annotation] Using index {annotation_index}")
    print(f"  scene={scene}, sequence={sequence}")
    print(f"  frames {start} → {end}")
    print(f"  text: {ann.get('description', '')}")

    motion_dir = os.path.join(
        RAW_MOTION_ROOT,
        scene,
        "keypoints_3d",
        sequence,
    )

    if not os.path.isdir(motion_dir):
        raise FileNotFoundError(f"motion_dir not found: {motion_dir}")

    left_path = os.path.join(motion_dir, "left.jsonl")
    right_path = os.path.join(motion_dir, "right.jsonl")

    if not os.path.isfile(left_path):
        raise FileNotFoundError(f"missing: {left_path}")
    if not os.path.isfile(right_path):
        raise FileNotFoundError(f"missing: {right_path}")

    left = load_jsonl(left_path)
    right = load_jsonl(right_path)

    if end == -1:
        end = min(len(left), len(right))

    if start < 0:
        start = 0
    if end <= start:
        raise ValueError(f"Bad frame range: start={start}, end={end} (lenL={len(left)}, lenR={len(right)})")

    left_xyz = np.array([strip_w(f) for f in left[start:end]], dtype=np.float32)
    right_xyz = np.array([strip_w(f) for f in right[start:end]], dtype=np.float32)

    T = min(len(left_xyz), len(right_xyz))
    left_xyz = left_xyz[:T]
    right_xyz = right_xyz[:T]

    # [T,5,4,3] where triple=[thetaDeg,phiDeg,r] in world space
    left_triples = compute_hand_triples_world(left_xyz)
    right_triples = compute_hand_triples_world(right_xyz)

    # angles_nested[T][2][5][4][3]
    angles_nested = np.zeros((T, 2, 5, 4, 3), dtype=np.float32)
    angles_nested[:, 0] = left_triples
    angles_nested[:, 1] = right_triples

    if force_R is not None:
        angles_nested[:, :, :, :, 2] = float(force_R)

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
    export_single_motion_from_annotation_world(
        annotation_index=56,   # 👈 change this to test another annotation
        force_R=None,          # None keeps real segment lengths; set to 1.0 to force constant r
    )
