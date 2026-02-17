from __future__ import annotations
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import List, Optional, Dict, Any
import os
import json

def _maybe_parse_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@dataclass
class InferenceConfig:
    # --- core ---
    dataset: str = "gigahands"
    model_path: str = ""
    output_dir: str = ""
    device: int = 0
    use_ema: bool = True

    # --- sampling ---
    n_frames: int = 253
    guidance: List[float] = field(default_factory=lambda: [2.5])
    samples_per_prompt: int = 20
    base_seed: int = 0
    prompt_stride: int = 1000
    prompt_offset: int = 0

    # --- model knobs (keep aligned with how you trained) ---
    latent_dim: int = 512
    layers: int = 8
    cond_mask_prob: float = 0.01
    arch: str = "trans_dec"
    emb_trans_dec: bool = False
    pos_embed_max_len: int = 5000
    mask_frames: bool = True
    pred_len: int = 4
    context_len: int = 0
    multi_target_cond: bool = False
    multi_encoder_type: str = "single"
    target_enc_layers: int = 1

    # diffusion
    diffusion_steps: int = 1000
    noise_schedule: str = "linear"
    sigma_small: bool = True

    # dataset knobs (avoid hard-coded defaults from loader)
    root_dir: Optional[str] = None
    annotation_file: Optional[str] = None
    mean_std_dir: Optional[str] = None
    side: str = "both"
    fixed_len: int = 0
    dmvb_size: int = 126
    dmvb_layout: str = "full"
    load_mode: str = "custome"

    # prompt selection
    mode: str = "train"  # train|rewrite|unseen|custom
    train_jsonl: Optional[str] = None
    annotations_jsonl: Optional[str] = None
    input_file_list_js: Optional[str] = None  # for rewrite mode
    custom_prompts_path: Optional[str] = None # .txt or .jsonl

    max_prompts: Optional[int] = None
    sort: str = "stable"  # stable|alpha|len|random
    random_seed_for_sort: int = 123

    @staticmethod
    def from_json(path: str) -> "InferenceConfig":
        d = _maybe_parse_json(path)
        return InferenceConfig(**d)

    def to_args(self) -> SimpleNamespace:
        """
        Build args object compatible with:
        - DatasetInterfaceRegistry.get(...).get_loader(args)
        - create_gaussian_diffusion(args)
        - MDM(**get_model_args(args))
        """
        args = SimpleNamespace(
            dataset=self.dataset,
            model_path=self.model_path,
            output_dir=self.output_dir,
            device=self.device,
            use_ema=self.use_ema,
            guidance_param=1.0,   # we override per-sample in runtime
            num_samples=1,

            absolote_frame_connt=self.n_frames,
            pred_len=self.pred_len,
            context_len=self.context_len,

            latent_dim=self.latent_dim,
            layers=self.layers,
            cond_mask_prob=self.cond_mask_prob,
            arch=self.arch,
            emb_trans_dec=self.emb_trans_dec,
            pos_embed_max_len=self.pos_embed_max_len,
            mask_frames=self.mask_frames,

            multi_target_cond=self.multi_target_cond,
            multi_encoder_type=self.multi_encoder_type,
            target_enc_layers=self.target_enc_layers,

            diffusion_steps=self.diffusion_steps,
            noise_schedule=self.noise_schedule,
            sigma_small=self.sigma_small,

            lambda_rcxyz=0.0,
            lambda_vel=0.0,
            lambda_fc=0.0,
            lambda_target_loc=0.0,

            # dataset overrides (your loader reads these via kwargs / args)
            root_dir=self.root_dir,
            annotation_file=self.annotation_file,
            mean_std_dir=self.mean_std_dir,
            side=self.side,
            fixed_len=self.fixed_len,
            dmvb_size=self.dmvb_size,
            dmvb_layout=self.dmvb_layout,
            load_mode=self.load_mode,
        )
        return args
