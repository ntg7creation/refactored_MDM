
from dataset_api.gigahands.gigadataset_loader import GigaHandsT2M
import numpy as np

dataset = GigaHandsT2M(
    root_dir=r"D:\repos\refactored_MDM\GigaHands_Data\coverted_motions\hand_poses_xyz",
    annotation_file=r"D:\repos\refactored_MDM\GigaHands_Data\annotations_v2.jsonl",
    mean_std_dir=r"D:\repos\refactored_MDM\GigaHands_Data\coverted_motions\norm_stats",
    motion_file_name='both',
    load_mode='identity',
    num_frames=120
)

motion = dataset[0][4].numpy()
print("✅ Motion shape:", motion.shape)
print("🧮 Mean of first frame:", motion[0].mean())
print("🧮 Std of first frame:", motion[0].std())

# scripts/sanity_check.py