import os
from os.path import join as pjoin
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from data_loaders.humanml.utils.word_vectorizer import WordVectorizer

def build_dmvb(raw_motion: np.ndarray, layout_type: str = "full") -> np.ndarray:
    return raw_motion  # passthrough for now

class ASLHandsT2M(Dataset):
    def __init__(self, root_dir, annotation_file, mean_std_dir,
                 split='train', device='cpu',
                 num_frames=120, dmvb_size=126, dmvb_layout='full'):
        self.side = "right"  # ✅ force right hand
        self.root_dir = root_dir
        self.device = device
        self.num_frames = num_frames
        self.fixed_len = num_frames
        self.dmvb_size = dmvb_size
        self.max_text_len = 40
        self.dmvb_layout = dmvb_layout

        # ✅ Load right-hand mean/std
        self.mean = np.load(pjoin(mean_std_dir, f'mean_{self.side}.npy'))
        self.std = np.load(pjoin(mean_std_dir, f'std_{self.side}.npy'))

        dummy = np.expand_dims(self.mean, axis=0)
        trimmed_mean = build_dmvb(dummy, layout_type=dmvb_layout)[0]
        dummy = np.expand_dims(self.std, axis=0)
        trimmed_std = build_dmvb(dummy, layout_type=dmvb_layout)[0]

        self.mean = trimmed_mean
        self.std = trimmed_std

        self.mean_gpu = torch.tensor(self.mean).to(device)[None, :, None, None]
        self.std_gpu = torch.tensor(self.std).to(device)[None, :, None, None]

        self.samples = []
        self._load_annotations(annotation_file, split)

    def _load_annotations(self, annotation_file, split):
        print(f"📄 Loading annotations from: {annotation_file}")
        with open(annotation_file, 'r') as f:
            lines = f.readlines()

        # Hardcode the two motions
        motion_id_top = "69370"   # "how"
        motion_id_bottom = "52348" # "sleepy"

        for i, line in enumerate(tqdm(lines, desc=f"Loading ASL [{split}]")):
            try:
                entry = json.loads(line)
                gloss = entry["gloss"]

                if i < 5037:
                    video_id = motion_id_top
                else:
                    video_id = motion_id_bottom

                motion_path = pjoin(self.root_dir, video_id, f"{self.side}_3d.npy")

                if os.path.exists(motion_path):
                    self.samples.append((motion_path, gloss))
                else:
                    print(f"⚠️ Missing motion file: {motion_path}")

            except Exception as e:
                print(f"❌ Error parsing line {i}: {e}")

        print(f"📦 Total valid samples: {len(self.samples)}")
        assert len(self.samples) > 0, "No valid samples found."


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        
        print("🚨 ERROR: Using ASLHandsT2M instead of GigaHandsT2M!")
        
        motion_path, text = self.samples[idx]
        motion = np.load(motion_path).astype(np.float32)

        motion = build_dmvb(motion, layout_type=self.dmvb_layout)
        motion = (motion - self.mean) / (self.std + 1e-8)

        T = motion.shape[0]
        if T >= self.fixed_len:
            start = np.random.randint(0, T - self.fixed_len + 1)
            motion = motion[start:start + self.fixed_len]
        else:
            pad = np.zeros((self.fixed_len - T, motion.shape[1]), dtype=np.float32)
            motion = np.concatenate([motion, pad], axis=0)
        m_length = self.fixed_len

        tokens = text.split()
        tokens = ['sos/OTHER'] + tokens[:self.max_text_len] + ['eos/OTHER']
        sent_len = len(tokens)
        tokens += ['unk/OTHER'] * (self.max_text_len + 2 - len(tokens))

        motion_tensor = torch.from_numpy(motion)

        return (
            None,
            None,
            text,
            sent_len,
            motion_tensor,
            m_length,
            '_'.join(tokens)
        )

    def inv_transform(self, data):
        return data * (self.std + 1e-8) + self.mean


class ASLHandsML3D(Dataset):
    def __init__(self, mode, datapath=None, split="train", dmvb_size=126, **kwargs):
        self.mode = mode
        self.dataset_name = 'asl'
        self.dataname = 'asl'
        self.dmvb_size = dmvb_size
        self.dmvb_layout = kwargs.get('dmvb_layout', 'full')
        self.device = kwargs.get('device', 'cpu')

        # self.root_dir = r"D:\repos\refactored_MDM\ASL_Data\dmvb"
        # self.annotation_file = './ASL_Data/annotations_minimal.jsonl'
        # self.mean_std_dir = r"D:\repos\refactored_MDM\ASL_Data\norm_stats"

        self.root_dir = kwargs.get('root_dir', r"D:\repos\refactored_MDM\ASL_Data\dmvb")
        self.annotation_file = kwargs.get('annotation_file', './ASL_Data/annotations_minimal.jsonl')
        self.mean_std_dir = kwargs.get('mean_std_dir', r"D:\repos\refactored_MDM\ASL_Data\norm_stats")


        self.fixed_len = kwargs.get('fixed_len', 0)
        self.use_cache = kwargs.get('use_cache', True)
        self.side = "right"  # ✅ force right hand

        # ✅ Load right-hand mean/std
        mean_path = pjoin(self.mean_std_dir, f'mean_{self.side}.npy')
        std_path = pjoin(self.mean_std_dir, f'std_{self.side}.npy')
        self.mean = np.load(mean_path)
        self.std = np.load(std_path)

        self.opt = type('', (), {})()
        self.opt.fixed_len = self.fixed_len
        self.opt.max_motion_length = self.fixed_len if self.fixed_len > 0 else 196
        self.opt.unit_length = 4
        self.opt.max_text_len = 64
        self.opt.disable_offset_aug = False

        self.t2m_dataset = ASLHandsT2M(
            root_dir=self.root_dir,
            annotation_file=self.annotation_file,
            mean_std_dir=self.mean_std_dir,
            split=split,
            num_frames=self.fixed_len if self.fixed_len > 0 else 196,
            device=self.device,
            dmvb_size=self.dmvb_size,
            dmvb_layout=self.dmvb_layout,
        )

        self.mean_gpu = torch.tensor(self.mean).to(self.device)[None, :, None, None]
        self.std_gpu = torch.tensor(self.std).to(self.device)[None, :, None, None]

        assert len(self.t2m_dataset) > 0, 'ASL dataset appears empty.'

    def __getitem__(self, idx):
        return self.t2m_dataset[idx]

    def __len__(self):
        return len(self.t2m_dataset)
