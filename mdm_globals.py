# ============================================
# mdm_globals.py — Minimal Global Definitions
# ============================================

from pathlib import Path

# ------------------------------------------------
# Automatically detect project (repo) root
# ------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[0]

# ------------------------------------------------
# Raw GigaHands location (external repo)
# ------------------------------------------------
RAW_GIGAHANDS = Path(r"D:\repos\GigaHands\dataset\GigaHands\hand_poses")

# ------------------------------------------------
# Annotation file
# ------------------------------------------------
ANNOTATIONS_PATH = REPO_ROOT / "GigaHands_Data" / "annotations_v2.jsonl"
SINGLE_ANNOTATION_INDEX = 56 # Which annotation index to force when load_mode == "single"
SINGLE_TEXT_SOURCE = "rewritten_annotation" # Which text field to use for that annotation
# ------------------------------------------------
# ACTIVE DATA PATH  (Decide between xyz-only or xyz+velocity or angles)
# ------------------------------------------------
DATA_ROOT = REPO_ROOT / "GigaHands_Data" / "converted_tpr"/ "npy" 
SAVE_DIR = "save/tpr_single_motion_test"

# ------------------------------------------------
# File Names
# ------------------------------------------------
NORM_STATS = REPO_ROOT / "GigaHands_Data" / "converted_tpr" / "norm_stats"
MEAN_PATH  = NORM_STATS / "mean_tpr_both.npy"
STD_PATH   = NORM_STATS / "std_tpr_both.npy"
FILE_NAME = "tpr_both.npy"  # The actual motion file name to load from each sample dir

# ============================================================
# Motion Representation Globals
# ============================================================

NUM_JOINTS = 42  # 21 left + 21 right

# Position + Velocity (xyz + vel)
DMVB_DIM_POS_VEL = NUM_JOINTS * 6    # 252

# Position only (xyz) – legacy
DMVB_DIM_POS     = NUM_JOINTS * 3    # 126

# Joint angles (5 fingers × 4 rotation-joints × 3 angles (x,y,z) × 2 hands)
DMVB_DIM_ANGLES  = 120

# Active representation
DMVB_DIM = DMVB_DIM_ANGLES


# ------------------------------------------------
# Utility: dump all globals
# ------------------------------------------------
def dump():
    print("\n===== MDM GLOBALS =====")
    for name, value in globals().items():
        if name.isupper():
            print(f"{name:20s} = {value}")
    print("=======================\n")


if __name__ == "__main__":
    dump()
