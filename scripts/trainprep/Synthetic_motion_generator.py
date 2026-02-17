import json
import numpy as np
from tqdm import tqdm
import os

# -------------------------------------------------
# Constants
# -------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

FINGER_ORDER = ["thumb", "index", "middle", "ring", "pinky"]
JOINT_ORDER = ["root", "mip", "pip", "dip"]

ANGLE_COMPONENTS = {
    "phi": 0,
    "theta": 1,
}

# -------------------------------------------------
# Generator
# -------------------------------------------------

def make_synthetic_angles_both_jsonl(
    out_jsonl_path: str,

    finger: str = "middle",
    num_frames: int = 360,

    sweep_start: float = -180.0,
    sweep_end: float = 180.0,

    component: str = "phi",   # "phi" or "theta"
    R: float = 1.0,

    # scale PIP so it bends less than MIP
    pip_scale: float = 0.65,

    # small constant offset so fingers are not identical
    const_finger: str = "index",
    const_joint: str = "mip",
    const_phi: float = 12.0,
    const_theta: float = 25.0,
):
    # -------------------------------------------------
    # Indices
    # -------------------------------------------------

    finger_i = FINGER_ORDER.index(finger)
    mip_i = JOINT_ORDER.index("mip")
    pip_i = JOINT_ORDER.index("pip")
    comp_i = ANGLE_COMPONENTS[component]

    const_finger_i = FINGER_ORDER.index(const_finger)
    const_joint_i = JOINT_ORDER.index(const_joint)

    # -------------------------------------------------
    # Base tensor: [hands=2][fingers=5][joints=4][phi,theta,r]
    # -------------------------------------------------

    base = np.zeros((2, 5, 4, 3), dtype=np.float32)
    base[..., 2] = R  # r is always 1

    # -------------------------------------------------
    # Sweep setup
    # -------------------------------------------------

    step = (sweep_end - sweep_start) / (num_frames - 1)

    abs_path = os.path.join(SCRIPT_DIR, out_jsonl_path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)

    print(f"[SyntheticMotion] Writing to: {abs_path}")
    print(f"[SyntheticMotion] Frames: {num_frames}")
    print(f"[SyntheticMotion] Sweeping {finger} MIP+PIP ({component})")
    print("-" * 50)

    # -------------------------------------------------
    # Frame loop
    # -------------------------------------------------

    with open(abs_path, "w", encoding="utf-8") as f:
        for t in tqdm(range(num_frames), desc="Generating frames", unit="frame"):
            angles = base.copy()

            # ---------------------------------------------
            # 1) constant bend on another finger
            # ---------------------------------------------
            angles[:, const_finger_i, const_joint_i, 0] = const_phi
            angles[:, const_finger_i, const_joint_i, 1] = const_theta

            # ---------------------------------------------
            # 2) current sweep angle
            # ---------------------------------------------
            a = sweep_start + step * t

            # ---------------------------------------------
            # 3) apply to BOTH hands, BOTH joints
            # ---------------------------------------------
            for hand in (0, 1):
                angles[hand, finger_i, mip_i, comp_i] = a
                angles[hand, finger_i, pip_i, comp_i] = a * pip_scale

            # ---------------------------------------------
            # write frame
            # ---------------------------------------------
            record = {
                "frame": t,
                "angles_nested": angles.tolist(),
            }
            f.write(json.dumps(record) + "\n")

    # print("✔ Done")
    print(f"[SyntheticMotion] ✔ Done writing to {abs_path}")


# -------------------------------------------------
# Run directly
# -------------------------------------------------

if __name__ == "__main__":
    make_synthetic_angles_both_jsonl(
        out_jsonl_path=os.path.join(
            "Synthetic_motions",
            "angles_both_synth_middle_mip_pip_phi.jsonl"
        ),
        finger="middle",
        num_frames=360,
        sweep_start=-180.0,
        sweep_end=180.0,
        component="phi",
        pip_scale=2,
        const_finger="index",
        const_joint="mip",
        const_phi=12.0,
        const_theta=25.0,
    )
