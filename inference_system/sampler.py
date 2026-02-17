from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
import os
import numpy as np
import torch
import sys

# add repo root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Your repo imports (same style as your current scripts)
from utils.fixseed import fixseed
from utils.model_util import load_saved_model, create_gaussian_diffusion
from utils import dist_util
from utils.sampler_util import ClassifierFreeSampleModel
from dataset_api.dataset_registry import DatasetInterfaceRegistry
from model.mdm import MDM

@dataclass
class LoadedInference:
    model: torch.nn.Module
    diffusion: Any
    mean: np.ndarray   # shape [1, J, C, 1]
    std: np.ndarray    # shape [1, J, C, 1]
    njoints: int
    nfeats: int

def _infer_joints_feats_from_mean(mean_1d: np.ndarray) -> Tuple[int,int]:
    D = int(mean_1d.shape[0])
    # for xyz-only DMVB: D = J*3
    if D % 3 != 0:
        raise ValueError(f"Cannot infer njoints/nfeats from mean dim D={D}. Pass explicit layout handling if needed.")
    return D // 3, 3

def load_model_and_data(args) -> LoadedInference:
    fixseed(0)
    dist_util.setup_dist(args.device)

    dataset_interface = DatasetInterfaceRegistry.get(args.dataset)
    data = dataset_interface.get_function("get_loader")(args)

    # infer J,C from dataset mean (keeps you compatible with different DMVB layouts)
    mean_1d = data.dataset.mean
    std_1d = data.dataset.std
    J, C = _infer_joints_feats_from_mean(mean_1d)

    model = MDM(
        modeltype='',
        njoints=J,
        nfeats=C,
        num_actions=1,
        translation=True,
        pose_rep='rot6d',
        glob=True,
        glob_rot=True,
        latent_dim=args.latent_dim,
        ff_size=1024,
        num_layers=args.layers,
        num_heads=4,
        dropout=0.1,
        activation="gelu",
        data_rep='hml_vec',
        cond_mode='text',
        cond_mask_prob=args.cond_mask_prob,
        arch=args.arch,
        emb_trans_dec=args.emb_trans_dec,
        clip_version='ViT-B/32',
        dataset=args.dataset,
        text_encoder_type='bert',
        pos_embed_max_len=args.pos_embed_max_len,
        mask_frames=args.mask_frames,
        pred_len=args.pred_len,
        context_len=args.context_len,
        emb_policy='add',
        multi_target_cond=args.multi_target_cond,
        multi_encoder_type=args.multi_encoder_type,
        target_enc_layers=args.target_enc_layers,
    )
    diffusion = create_gaussian_diffusion(args)

    load_saved_model(model, args.model_path, use_avg=args.use_ema)

    model.to(dist_util.dev())
    model.eval()

    # mean/std reshape to [1,J,C,1] like your current scripts
    mean = mean_1d.reshape(1, J, C, 1)
    std = std_1d.reshape(1, J, C, 1)

    return LoadedInference(model=model, diffusion=diffusion, mean=mean, std=std, njoints=J, nfeats=C)

@torch.no_grad()
def sample_text(
    loaded,
    args,
    text_prompt: str,
    guidance: float,
    batch_size: int,
    seed: int,
):
    fixseed(seed)

    model = loaded.model
    if guidance != 1.0:
        model = ClassifierFreeSampleModel(model)

    model_kwargs = {
        "y": {
            "text": [text_prompt] * batch_size,
            "lengths": torch.tensor(
                [args.absolote_frame_connt] * batch_size,
                device=dist_util.dev()
            ),
            "scale": torch.tensor(
                [guidance] * batch_size,
                device=dist_util.dev()
            ),
        }
    }

    model_kwargs["y"]["text_embed"] = (
        model.model.encode_text(model_kwargs["y"]["text"])
        if isinstance(model, ClassifierFreeSampleModel)
        else model.encode_text(model_kwargs["y"]["text"])
    )

    if args.context_len == 0:
        model_kwargs["y"]["prefix"] = torch.zeros(
            (batch_size, loaded.njoints, loaded.nfeats, 0),
            device=dist_util.dev()
        )
        model_kwargs["y"]["mask"] = torch.zeros(
            (batch_size, 1, 1, 0),
            device=dist_util.dev()
        )

    shape = (batch_size, loaded.njoints, loaded.nfeats, args.absolote_frame_connt)

    sample = loaded.diffusion.p_sample_loop(
        model,
        shape,
        clip_denoised=False,
        model_kwargs=model_kwargs,
        progress=True
    )

    motion = sample.detach().cpu().numpy()
    motion = motion * loaded.std + loaded.mean

    return motion  # shape [B, J, C, T]

