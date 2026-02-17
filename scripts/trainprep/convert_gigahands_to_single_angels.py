import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm


def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def strip_w(frame):
    # frame is list of 21 joints, each [x,y,z,w] (or sometimes [x,y,z])
    out = []
    for j in frame:
        if len(j) >= 3:
            out.append(j[:3])
        else:
            out.append([0.0, 0.0, 0.0])
    return out


# Same mapping you already use in the viewer (handConstants.js)
FINGER_JOINTS = {
    "thumb":  [0, 1, 2, 3, 4],
    "index":  [0, 5, 6, 7, 8],
    "middle": [0, 9, 10, 11, 12],
    "ring":   [0, 13, 14, 15, 16],
    "pinky":  [0, 17, 18, 19, 20],
}
FINGER_ORDER = ["thumb", "index", "middle", "ring", "pinky"]


def angle_with_floor_deg(v, floor_normal=np.array([0.0, 1.0, 0.0], dtype=np.float32)):
    """
    v: (3,) bone vector in world coords
    returns angle to the FLOOR PLANE in degrees:
      0 = parallel to floor, 90 = perpendicular to floor.
    """
    eps = 1e-8
    norm = np.linalg.norm(v)
    if norm < eps:
        return 0.0
    vh = v / norm

    # sin(angle_to_plane) = |dot(vhat, normalhat)|
    s = abs(float(np.dot(vh, floor_normal)))
    s = max(0.0, min(1.0, s))
    return float(np.degrees(np.arcsin(s)))


def frame_to_hand_angles_floor(points21):
    """
    points21: list[21][3]
    returns: nested [5 fingers][4 bones] of scalar angles (deg)
    """
    pts = np.asarray(points21, dtype=np.float32)  # (21,3)
    out = []
    for finger in FINGER_ORDER:
        ids = FINGER_JOINTS[finger]  # 5 joints
        finger_angles = []
        for seg in range(4):
            a = ids[seg]
            b = ids[seg + 1]
            v = pts[b] - pts[a]
            finger_angles.append(angle_with_floor_deg(v))
        out.append(finger_angles)
    return out


def main():
    # ----------------------------------
    # YOUR PATHS
    # ----------------------------------
    repo_root = r"D:\repos\refactored_MDM"
    annotations_path = os.path.join(repo_root, "GigaHands_Data", "annotations_v2.jsonl")
    raw_motion_root = r"D:\repos\GigaHands\dataset\GigaHands\hand_poses"

    # ----------------------------------
    # OUTPUT (inside repo_root)
    # ----------------------------------
    output_root = os.path.join(
        repo_root,
        "GigaHands_Data",
        "converted_angles_floor",
        "json"
    )

    annotations = load_jsonl(annotations_path)

    for ann in tqdm(annotations, desc="Converting GigaHands → per-bone angle-to-floor (2 Hands)"):
        seq_name = ann["sequence"]
        scene = ann["scene"]

        motion_dir = os.path.join(raw_motion_root, scene, "keypoints_3d", seq_name)
        out_dir = os.path.join(output_root, scene, "keypoints_3d", seq_name)
        os.makedirs(out_dir, exist_ok=True)

        out_jsonl = os.path.join(out_dir, "angles_floor_both.jsonl")
        if os.path.exists(out_jsonl):
            continue

        try:
            left_frames = load_jsonl(os.path.join(motion_dir, "left.jsonl"))
            right_frames = load_jsonl(os.path.join(motion_dir, "right.jsonl"))

            with open(os.path.join(motion_dir, "chosen_frames_left.json"), "r") as f:
                chosen_left = json.load(f)
            with open(os.path.join(motion_dir, "chosen_frames_right.json"), "r") as f:
                chosen_right = json.load(f)

            common_indices = sorted(set(chosen_left).intersection(set(chosen_right)))
            if len(common_indices) == 0:
                print(f"⚠️ No aligned chosen frames for {scene}/{seq_name}, skipping.")
                continue

            # Filter to common chosen frames
            left_filtered = [left_frames[i] for i in common_indices if i < len(left_frames)]
            right_filtered = [right_frames[i] for i in common_indices if i < len(right_frames)]

            T = min(len(left_filtered), len(right_filtered))
            left_filtered = left_filtered[:T]
            right_filtered = right_filtered[:T]

            # Write JSONL
            with open(out_jsonl, "w", encoding="utf-8") as w:
                for t in range(T):
                    L = strip_w(left_filtered[t])
                    R = strip_w(right_filtered[t])

                    left_angles = frame_to_hand_angles_floor(L)    # [5][4]
                    right_angles = frame_to_hand_angles_floor(R)   # [5][4]

                    rec = {
                        "frame": t,
                        "angles_floor_nested": [left_angles, right_angles],  # [2][5][4]
                        "degrees": True,
                        "floor_normal": [0.0, 1.0, 0.0],  # documenting assumption
                    }
                    w.write(json.dumps(rec) + "\n")

        except Exception as e:
            print(f"❌ Error processing {scene}/{seq_name}: {e}")


if __name__ == "__main__":
    main()
