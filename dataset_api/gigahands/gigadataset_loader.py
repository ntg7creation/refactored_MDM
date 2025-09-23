import os
from os.path import join as pjoin
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from data_loaders.humanml.utils.word_vectorizer import WordVectorizer


def build_dmvb(raw_motion: np.ndarray, layout_type: str = "full") -> np.ndarray:
    # print(raw_motion.shape)
    if True: #layout_type == "full":
        return raw_motion

    # elif False:#layout_type == "root+5":
    #     joint_indices = [0, 1, 5, 9, 13, 17]  # root + base of each finger
    #     d_per_joint = 3
    #     frame_dim = raw_motion.shape[1]
    #     T = raw_motion.shape[0]

    #     if frame_dim < 63:
    #         raise ValueError("Expected raw_motion to have at least 21 joints (63 dims)")

    #     joint_data = [raw_motion[:, j * d_per_joint : (j + 1) * d_per_joint] for j in joint_indices]
    #     # print(f"[build_dmvb] Selected joints indices: {joint_indices}, each with {d_per_joint} dims, total frame_dim={frame_dim}")
    #     return np.concatenate(joint_data, axis=1)

    # elif layout_type == "2d_only":
    #     raise NotImplementedError("DMVB layout '2d_only' is not implemented yet.")

    # elif layout_type == "velocity_only":
    #     raise NotImplementedError("DMVB layout 'velocity_only' is not implemented yet.")

    # elif layout_type == "flattened_xyz+vel":
    #     raise NotImplementedError("DMVB layout 'flattened_xyz+vel' is not implemented yet.")

    # else:
    #     raise NotImplementedError(f"DMVB layout '{layout_type}' is not implemented.")
    



class GigaHandsT2M(Dataset):
    """
    Core Dataset Loader for GigaHands T2M format.

    This class reads annotations from a JSONL file, finds the corresponding motion .npy files
    (either left or right hand), normalizes them using provided mean and std values, and returns
    them in the shape expected by the MDM model.

    Output:
        Each sample is a dictionary with:
            - 'inp': normalized motion tensor [263, 1, T]
            - 'text': corresponding textual annotation
            - 'lengths': number of frames
            - 'key': file path for reference

    This class is intended to be used internally by a wrapper that conforms to MDM's dataset expectations.
    """
    def __init__(self, root_dir, annotation_file, mean_std_dir, 
                 side='both', split='train', device='cpu',
                 num_frames=120, dmvb_size=126, dmvb_layout='full'):
        assert side in ['left', 'right', 'both']
        self.side = side
        self.root_dir = root_dir
        self.device = device
        self.num_frames = num_frames
        self.fixed_len = num_frames
        self.dmvb_size = dmvb_size
        self.max_text_len = 40
        self.dmvb_layout = dmvb_layout
        # self.w_vectorizer = WordVectorizer(encoder_type='bert')
       


        self.mean = np.load(pjoin(mean_std_dir, f'mean_{side}.npy'))
        self.std = np.load(pjoin(mean_std_dir, f'std_{side}.npy'))

        
        # Derive trimmed stats using layout
        dummy = np.expand_dims(self.mean, axis=0)  # shape [1, D]
        trimmed_mean = build_dmvb(dummy, layout_type=dmvb_layout)[0]

        dummy = np.expand_dims(self.std, axis=0)
        trimmed_std = build_dmvb(dummy, layout_type=dmvb_layout)[0]

        self.mean = trimmed_mean
        self.std = trimmed_std

        self.mean_gpu = torch.tensor(self.mean).to(device)[None, :, None, None]
        self.std_gpu = torch.tensor(self.std).to(device)[None, :, None, None]

        self.samples = []  # list of (motion_path, text)
        self._load_annotations(annotation_file, split)

    def _load_annotations(self, annotation_file, split):
        with open(annotation_file, 'r') as f:
            lines = f.readlines()

        for line in tqdm(lines, desc=f"Loading GigaHands [{split}]"):
            ann = json.loads(line)
            scene = ann['scene']
            seq = ann['sequence']
            text_list = ann['rewritten_annotation']
            motion_path = pjoin(self.root_dir, scene, 'keypoints_3d', seq, f'xyz_{self.side}.npy')

            if os.path.exists(motion_path):
                for text in text_list:
                    self.samples.append((motion_path, text))


        assert len(self.samples) > 0, "No valid samples found."

    def __len__(self):
        return len(self.samples)    



    def __getitem__(self, idx):

        # Load the current sample's annotation text
        # motion_path, text = self.samples[idx]

        # # By default: load the motion that matches this sample's path
        # motion = np.load(motion_path).astype(np.float32)

        # --- Identity training mode (optional) ---
        # If you want to force every sample to use the *same* motion
        # regardless of index (e.g., to test memorization),
        # Always use the chosen identity motion (ignores dataset index)
        fixed_scene = "p005-sandwich-salad-baking-monoply-boxing"
        fixed_seq = "018"
        motion_path = pjoin(
            self.root_dir, fixed_scene, "keypoints_3d", fixed_seq, "xyz_both.npy"
        )
        _, text = self.samples[idx]
        motion = np.load(motion_path).astype(np.float32)


        # motion = build_dmvb(motion, layout_type=self.dmvb_layout)
        # print (f"[DEBUG] Loaded motion from {motion_path} with shape {motion.shape}")
        # Normalize
        motion = (motion - self.mean) / (self.std + 1e-8)

        # Cut or pad motion to fixed length
        if self.fixed_len > 0:
            T = motion.shape[0]
            if T >= self.fixed_len:
                start = np.random.randint(0, T - self.fixed_len + 1) # start = 0  #  
                motion = motion[start:start + self.fixed_len]
            else:
                # Pad by repeating the last frame instead of zeros
                last_frame = motion[-1][None, :]                     # shape (1, D)
                pad = np.repeat(last_frame, self.fixed_len - T, axis=0)  # shape (fixed_len - T, D)
                motion = np.concatenate([motion, pad], axis=0)
            m_length = self.fixed_len
        else:
            m_length = motion.shape[0]

        # BERT-compatible tokenization
        tokens = text.split()
        tokens = ['sos/OTHER'] + tokens[:self.max_text_len] + ['eos/OTHER']
        sent_len = len(tokens)
        tokens += ['unk/OTHER'] * (self.max_text_len + 2 - len(tokens))

        word_embeddings = []
        pos_one_hots = []



        motion_tensor = torch.from_numpy(motion)


        return (
            None,
            None,
            text,
            sent_len,
            motion_tensor,   # [T, D]
            m_length,
            '_'.join(tokens)
        )

    
    def inv_transform(self, data):
        return data * (self.std + 1e-8) + self.mean










class GigaHandsML3D(Dataset):
    def __init__(self, mode, datapath=None, split="train", dmvb_size=126, **kwargs):
        self.mode = mode
        self.dataset_name = 'gigahands'
        self.dataname = 'gigahands'
        self.dmvb_size = dmvb_size
        self.dmvb_layout = kwargs.get('dmvb_layout', 'full')
        self.device = kwargs.get('device', 'cpu')

        # Paths (can be overridden via kwargs)
        self.root_dir = kwargs.get('root_dir', r"D:\repos\refactored_MDM\GigaHands_Data\coverted_motions\hand_poses_xyz")
        self.annotation_file = kwargs.get('annotation_file', r"D:\repos\refactored_MDM\GigaHands_Data\annotations_v2.jsonl")
        self.mean_std_dir = kwargs.get('mean_std_dir', r"D:\repos\refactored_MDM\GigaHands_Data\coverted_motions\norm_stats")

        self.fixed_len = kwargs.get('fixed_len', 0)
        self.use_cache = kwargs.get('use_cache', True)
        self.side = kwargs.get('side', 'both')

        # Load mean/std
        mean_path = pjoin(self.mean_std_dir, f'mean_{self.side}.npy')
        std_path = pjoin(self.mean_std_dir, f'std_{self.side}.npy')
        self.mean = np.load(mean_path)
        self.std = np.load(std_path)

        # Fake opt to mimic original behavior
        self.opt = type('', (), {})()
        self.opt.fixed_len = self.fixed_len
        self.opt.max_motion_length = self.fixed_len if self.fixed_len > 0 else 196
        self.opt.unit_length = 4
        self.opt.max_text_len = 64
        self.opt.disable_offset_aug = False

        # Load GigaHandsT2M
        self.t2m_dataset = GigaHandsT2M(
            root_dir=self.root_dir,
            annotation_file=self.annotation_file,
            mean_std_dir=self.mean_std_dir,
            split=split,
            side=self.side,
            num_frames=self.fixed_len if self.fixed_len > 0 else 196,
            device=self.device,
            dmvb_size=self.dmvb_size,
            dmvb_layout=self.dmvb_layout,
        )

        self.mean_gpu = torch.tensor(self.mean).to(self.device)[None, :, None, None]
        self.std_gpu = torch.tensor(self.std).to(self.device)[None, :, None, None]

        assert len(self.t2m_dataset) > 0, 'GigaHands dataset appears empty.'

    def __getitem__(self, idx):
        return self.t2m_dataset[idx]

    def __len__(self):
        return len(self.t2m_dataset)

    def __getitem__(self, idx):
        return self.t2m_dataset[idx]

    def __len__(self):
        return len(self.t2m_dataset)




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', required=True)
    parser.add_argument('--annotation_file', required=True)
    parser.add_argument('--mean_std_dir', required=True)
    parser.add_argument('--side', choices=['left', 'right','both'], default='both')
    parser.add_argument('--split', default='train')
    args = parser.parse_args()

    dataset = GigaHandsT2M(
        root_dir=args.root_dir,
        annotation_file=args.annotation_file,
        mean_std_dir=args.mean_std_dir,
        side=args.side,
        split=args.split
    )

    print(f"Loaded {len(dataset)} samples")
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        print(f"[{i}] {sample['key']}: {sample['text']} → motion shape {sample['inp'].shape}")
