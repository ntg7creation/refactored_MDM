import os
import re
import json
import numpy as np
import torch
from tqdm import tqdm
from types import SimpleNamespace
import sys

# Ensure access to local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.fixseed import fixseed
from utils.model_util import load_saved_model, create_gaussian_diffusion
from utils import dist_util
from utils.sampler_util import ClassifierFreeSampleModel
from dataset_api.dataset_registry import DatasetInterfaceRegistry
from model.mdm import MDM


# ==========================================================
# 🔧 CONFIGURATION
# ==========================================================
ARGS = SimpleNamespace(
    dataset="gigahands",
    model_path="save/test_custom100_concat_10_29/model000045000.pt",
    train_jsonl="GigaHands_Data/train_custom_100.jsonl",
    output_dir="save/test_custom100_concat_10_29/infer_trainset_blind",
    device=0,
    use_ema=True,
    guidance_param=2.5,
    num_samples=1,
    absolote_frame_connt=253,
    pred_len=4,
    context_len=0,
    latent_dim=512,
    layers=8,
    cond_mask_prob=0.01,
    arch="trans_dec",
    emb_trans_dec=False,
    pos_embed_max_len=5000,
    mask_frames=True,
    multi_target_cond=False,
    multi_encoder_type="single",
    target_enc_layers=1,
    diffusion_steps=1000,
    noise_schedule="linear",
    sigma_small=True,
    lambda_rcxyz=0.0,
    lambda_vel=0.0,
    lambda_fc=0.0,
    lambda_target_loc=0.0,
)


# ==========================================================
# 🧠 Model setup helpers
# ==========================================================
def get_model_args(args, data=None):
    return {
        'modeltype': '',
        'njoints': 21 * 2,
        'nfeats': 3,
        'num_actions': 1,
        'translation': True,
        'pose_rep': 'rot6d',
        'glob': True,
        'glob_rot': True,
        'latent_dim': args.latent_dim,
        'ff_size': 1024,
        'num_layers': args.layers,
        'num_heads': 4,
        'dropout': 0.1,
        'activation': "gelu",
        'data_rep': 'hml_vec',
        'cond_mode': 'text',
        'cond_mask_prob': args.cond_mask_prob,
        'arch': args.arch,
        'emb_trans_dec': args.emb_trans_dec,
        'clip_version': 'ViT-B/32',
        'dataset': args.dataset,
        'text_encoder_type': 'bert',
        'pos_embed_max_len': args.pos_embed_max_len,
        'mask_frames': args.mask_frames,
        'pred_len': args.pred_len,
        'context_len': args.context_len,
        'emb_policy': 'add',
        'multi_target_cond': args.multi_target_cond,
        'multi_encoder_type': args.multi_encoder_type,
        'target_enc_layers': args.target_enc_layers,
    }


def create_model_and_diffusion(args, data=None):
    model = MDM(**get_model_args(args, data))
    diffusion = create_gaussian_diffusion(args)
    return model, diffusion


# ==========================================================
# 🧹 Safe filename utilities
# ==========================================================
def sanitize_filename(s: str, max_length: int = 100) -> str:
    """Removes commas, bad chars, and trailing dots from text."""
    s = s.replace(",", "")
    s = re.sub(r'[^a-zA-Z0-9_\- ]+', '_', s)
    s = s.strip(" .")  # remove trailing space/dot
    s = "_".join(s.split())  # collapse spaces
    if len(s) > max_length:
        s = s[:max_length].rstrip("_")
    return s


def make_safe_name(index: int, scene: str, seq: str, key: str, text: str) -> str:
    scene = sanitize_filename(scene)
    seq = sanitize_filename(seq)
    text = sanitize_filename(text)
    return f"{index:02d}_{scene}_{seq}_{key}_{text}"


# ==========================================================
# 💾 JSONL reconstruction utility
# ==========================================================
def reconstruct_jsonl(motion, output_path):
    n_joints, n_coords, n_frames = motion.shape[1], motion.shape[2], motion.shape[3]
    with open(output_path, "w") as fout:
        for f_idx in range(n_frames):
            keypoints = [
                [float(motion[0, j, 0, f_idx]), float(motion[0, j, 1, f_idx]), float(motion[0, j, 2, f_idx]), 1.0]
                for j in range(n_joints)
            ]
            fout.write(json.dumps(keypoints) + "\n")
    print(f"✅ Saved {n_frames} frames to {output_path}")


# ==========================================================
# 🚀 MAIN
# ==========================================================
def main(args=ARGS):
    fixseed(0)
    dist_util.setup_dist(args.device)

    dataset_interface = DatasetInterfaceRegistry.get(args.dataset)
    get_loader_fn = dataset_interface.get_function("get_loader")
    data = get_loader_fn(args)

    model, diffusion = create_model_and_diffusion(args, data)
    load_saved_model(model, args.model_path, use_avg=args.use_ema)
    if args.guidance_param != 1.0:
        model = ClassifierFreeSampleModel(model)

    model.to(dist_util.dev())
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)
    mean, std = data.dataset.mean, data.dataset.std
    mean = mean.reshape(1, 42, 3, 1)
    std = std.reshape(1, 42, 3, 1)

    # --- Load training JSONL
    with open(args.train_jsonl, "r", encoding="utf-8") as f:
        lines = f.readlines()
    print(f"📄 Loaded {len(lines)} annotations from {args.train_jsonl}")

    file_names = []

    # --- Generate motion for each line (limit for testing: remove [:20] to run full)
    for i, line in enumerate(tqdm(lines[:20], desc="Generating")):
        ann = json.loads(line)
        scene = ann.get("scene", f"scene_{i:02d}")
        seq = ann.get("sequence", f"{i:03d}")

        for text_key in ["text","text_1", "text_2"]:
            if text_key not in ann or not ann[text_key].strip():
                continue

            text_prompt = ann[text_key].strip()

            # --- build model kwargs
            model_kwargs = {
                'y': {
                    'text': [text_prompt],
                    'lengths': torch.tensor([args.pred_len], device=dist_util.dev()),
                    'uncond': False,
                    'scale': torch.tensor([args.guidance_param], device=dist_util.dev())
                }
            }
            model_kwargs['y']['text_embed'] = model.encode_text(model_kwargs['y']['text'])
            model_kwargs['y']['uncond'] = False
            if args.context_len == 0:
                model_kwargs['y']['prefix'] = torch.zeros((1, 42, 3, 0), device=dist_util.dev())
                model_kwargs['y']['mask'] = torch.zeros((1, 1, 1, 0), device=dist_util.dev())

            # --- diffusion sample
            shape = (1, 42, 3, args.absolote_frame_connt)
            sample = diffusion.p_sample_loop(model, shape, clip_denoised=False,
                                             model_kwargs=model_kwargs, progress=False)
            motion = sample.detach().cpu().numpy()
            motion = motion * std + mean

            # --- create safe filename
            safe_name = make_safe_name(i, scene, seq, text_key, text_prompt)
            json_path = os.path.join(args.output_dir, f"{safe_name}.json")
            jsonl_path = os.path.join(args.output_dir, f"{safe_name}.jsonl")

            # --- save both
            output_data = {
                "motion": motion.tolist(),
                "text": text_prompt,
                "lengths": [args.absolote_frame_connt]
            }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f)
            reconstruct_jsonl(motion, jsonl_path)

            file_names.append(os.path.basename(jsonl_path))

    # --- write file list for viewer
    js_file = os.path.join(args.output_dir, "file_list.js")
    txt_file = os.path.join(args.output_dir, "filenames.txt")

    with open(js_file, "w", encoding="utf-8") as f:
        f.write("const FILE_LIST = [\n")
        for n in file_names:
            f.write(f'    "{n}",\n')
        f.write("];\nexport default FILE_LIST;\n")

    with open(txt_file, "w", encoding="utf-8") as f:
        f.write("\n".join(file_names))

    print(f"✅ Created {len(file_names)} motions.")
    print(f"📄 JS list saved to: {js_file}")
    print(f"📄 Text list saved to: {txt_file}")
    print(f"📁 Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
