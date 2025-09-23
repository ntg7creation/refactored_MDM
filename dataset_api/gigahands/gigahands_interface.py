# datasets/gigahands_interface.py
import os
import yaml
from torch.utils.data import DataLoader
from dataset_api.dataset_interface import DatasetInterface, DatasetFunctions
from dataset_api.gigahands.gigadataset_loader import GigaHandsML3D


import platform

NUM_WORKERS = 0 if platform.system() == 'Windows' else 8  # or any number you find optimal on Linux

import torch
from torch.utils.data._utils.collate import default_collate as collate


def t2m_collate(batch, target_batch_size):
    repeat_factor = -(-target_batch_size // len(batch))  # Ceiling division
    repeated_batch = batch * repeat_factor 
    full_batch = repeated_batch[:target_batch_size]  # Truncate to the target batch size
    # batch.sort(key=lambda x: x[3], reverse=True)
    adapted_batch = [{
        'inp': torch.tensor(b[4].T).float().unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
        'text': b[2], #b[0]['caption']
        'tokens': b[6],
        'lengths': b[5],
        'key': b[7] if len(b) > 7 else None,
    } for b in full_batch]
    # Debug print shapes and a sample
    # for i, ab in enumerate(adapted_batch[:2]):  # only print first 2 to avoid flooding
        # print(f"[DEBUG] Batch {i}: inp shape={ab['inp'].shape}, "
        #     f"text={ab['text']}, tokens={ab['tokens']}, lengths={ab['lengths']}, key={ab['key']}")

    return collate(adapted_batch)


def t2m_prefix_collate(batch, pred_len):
    # batch.sort(key=lambda x: x[3], reverse=True)
    adapted_batch = [{
        'inp': torch.tensor(b[4].T).float().unsqueeze(1)[..., -pred_len:], # [seqlen, J] -> [J, 1, seqlen]
        'prefix': torch.tensor(b[4].T).float().unsqueeze(1)[..., :-pred_len],
        'text': b[2], #b[0]['caption']
        'tokens': b[6],
        'lengths': pred_len,  # b[5],
        'orig_lengths': b[5][0], #  For evaluation
        'key': b[7] if len(b) > 7 else None,
    } for b in batch]
    return collate(adapted_batch)

# def gigahands_prefix_collate(batch, pred_len):
#     adapted_batch = [{
#         'inp': torch.tensor(b[4].T).float().unsqueeze(1)[..., -pred_len:],
#         'prefix': torch.tensor(b[4].T).float().unsqueeze(1)[..., :-pred_len],
#         'text': b[2],
#         'tokens': b[6],
#         'lengths': pred_len,
#         'orig_lengths': b[5],  # already an int in GigaHands
#         'key': b[7] if len(b) > 7 else None,
#     } for b in batch]
#     return collate(adapted_batch)

# def gigahands_collate(batch, target_batch_size):
#     repeat_factor = -(-target_batch_size // len(batch))  # Ceiling division
#     repeated_batch = batch * repeat_factor
#     full_batch = repeated_batch[:target_batch_size]

#     adapted_batch = [{
#         'inp': torch.tensor(b[4].T).float().unsqueeze(1),
#         'text': b[2],
#         'tokens': b[6],
#         'lengths': b[5],
#         'key': b[7] if len(b) > 7 else None,
#     } for b in full_batch]
#     return collate(adapted_batch)



class GigaHandsInterface(DatasetInterface):
    def __init__(self):
        cur_dir = os.path.dirname(__file__)
        yaml_path = os.path.join(cur_dir, "gigahands.yaml")
        with open(yaml_path, "r") as f:
            self.config = yaml.safe_load(f)

    # ---------------- CONFIG ----------------
    def get_config(self) -> dict:
        return self.config

    # ---------------- CUSTOM COLLATE ----------------
    def collate_fn(self, batch):
        pred_len = self.config.get("pred_len", 0)
        return t2m_collate(batch, pred_len)

    # ---------------- DIRECT LOADER ----------------
    # def get_loader(self,  num_frames, split="train", batch_size=1,
    #             hml_mode="train", fixed_len=0, pred_len=0,
    #             device=None, autoregressive=False,):
    #     dataset = GigaHandsML3D(
    #         mode=hml_mode,
    #         split=split,
    #         device=device,
    #         num_frames=num_frames,
    #         device=device
    #         # fixed_len=fixed_len if fixed_len > 0 else (pred_len + self.config.get("context_len", 0)),
    #     )

    #     if pred_len > 0:
    #         collate = lambda x: t2m_prefix_collate(x, pred_len=pred_len)
    #     else:
    #         collate = lambda x: t2m_collate(x, batch_size)

    #     return DataLoader(
    #         dataset,
    #         batch_size=batch_size,
    #         shuffle=True,
    #         num_workers=NUM_WORKERS,
    #         drop_last=True,
    #         collate_fn=collate
    #     )

    def get_loader( device,  num_frames, split="train", batch_size=1,name = "gigahands",
                hml_mode="train",fixed_len = 0,  pred_len = 0 ):
        import data_loaders.get_data as get_data

        return get_data.get_dataset_loader(
            name="gigahands",
            batch_size=batch_size,
            num_frames=num_frames,
            split=split,
            hml_mode=hml_mode,
            fixed_len=fixed_len,  # ✅ match old call exactly
            pred_len=pred_len,
            device=device,
        )
    

    # ---------------- FUNCTION DISPATCH ----------------
    def get_function(self, name: str):
        fn = DatasetFunctions(name)
        if fn == DatasetFunctions.GET_LOADER:
            return self.get_loader
        elif fn == DatasetFunctions.EVALUATE:
            from eval import eval_gigahands
            return eval_gigahands.evaluate
        elif fn == DatasetFunctions.COLLATE:
            return self.collate_fn
        elif fn == DatasetFunctions.WRAPPER:
            return lambda device: None  # GigaHands doesn’t use evaluator wrapper
        elif fn == DatasetFunctions.CONFIG:
            return self.config
        raise NotImplementedError(f"GigaHandsInterface: function {name} not implemented")
