import json
import numpy as np
import torch
import os
import sys

# Ensure access to local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from types import SimpleNamespace
from utils.fixseed import fixseed
from utils.model_util import load_saved_model
from utils import dist_util
from utils.sampler_util import ClassifierFreeSampleModel
from dataset_api.dataset_registry import DatasetInterfaceRegistry
from model.mdm import MDM
from utils.model_util import create_gaussian_diffusion


# =====================
# CONFIGURATION
# =====================
ARGS = SimpleNamespace(
    dataset="gigahands",
    model_path="save/test_dual_concat_dual_10_20/model000040000.pt",
    output_dir="save/test_dual_concat_dual_10_20/infer_test",
    text_prompt="SLAM THE CAN",   # <- your input text, will also be used as filename
    num_repetitions=1,
    absolote_frame_connt=253,
    motion_length=5,
    pred_len=4,
    context_len=0,
    device=0,
    use_ema=True,
    guidance_param=2.5,
    autoregressive=False,
    seed=10,
    batch_size=1,
    num_samples=1,
    text_encoder_type="bert",
    unconstrained=False,
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


# =====================
# HELPERS
# =====================
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
        'action_emb': 'tensor',
        'arch': args.arch,
        'emb_trans_dec': args.emb_trans_dec,
        'clip_version': 'ViT-B/32',
        'dataset': args.dataset,
        'text_encoder_type': args.text_encoder_type,
        'pos_embed_max_len': args.pos_embed_max_len,
        'mask_frames': args.mask_frames,
        'pred_len': args.pred_len,
        'context_len': args.context_len,
        'emb_policy': 'add',
        'all_goal_joint_names': [],
        'multi_target_cond': args.multi_target_cond,
        'multi_encoder_type': args.multi_encoder_type,
        'target_enc_layers': args.target_enc_layers,
    }


def create_model_and_diffusion(args, data=None):
    model = MDM(**get_model_args(args, data))
    diffusion = create_gaussian_diffusion(args)
    return model, diffusion


def reconstruct_jsonl(data_dict, output_path):
    motions = data_dict.get("motion", [])
    if len(motions) == 0:
        print("❌ No motion data to reconstruct.")
        return
    motion = motions[0]
    n_joints = len(motion)
    n_coords = len(motion[0])
    n_frames = len(motion[0][0])

    with open(output_path, "w") as fout:
        for frame_idx in range(n_frames):
            keypoints = []
            for joint in range(n_joints):
                x = motion[joint][0][frame_idx]
                y = motion[joint][1][frame_idx]
                z = motion[joint][2][frame_idx]
                keypoints.append([x, y, z, 1.0])  # Add confidence
            fout.write(json.dumps(keypoints) + "\n")
    print(f"✅ Reconstructed {n_frames} frames → {output_path}")


# =====================
# MAIN
# =====================
def main(args=ARGS):
    fixseed(args.seed)
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

    fps = 20
    n_frames = int(args.absolote_frame_connt)
    num_motionData, features_per_data = 42, 3
    motion_shape = (args.num_samples, num_motionData, features_per_data, n_frames)

    model_kwargs = {
        'y': {
            'text': [args.text_prompt] * args.num_samples,
            'lengths': torch.tensor([args.pred_len] * args.num_samples, device=dist_util.dev()),
            'uncond': False,
            'scale': torch.tensor([args.guidance_param], device=dist_util.dev())
        }
    }

    model_kwargs['y']['text_embed'] = model.encode_text(model_kwargs['y']['text'])
    model_kwargs['y']['uncond'] = False

    if args.context_len == 0:
        model_kwargs['y']['prefix'] = torch.zeros((args.num_samples, num_motionData, features_per_data, 0), device=dist_util.dev())
        model_kwargs['y']['mask'] = torch.zeros((args.num_samples, 1, 1, 0), device=dist_util.dev())

    print("🚀 Generating motion...")
    sample = diffusion.p_sample_loop(model, motion_shape, clip_denoised=False, model_kwargs=model_kwargs, progress=True)
    all_motions = sample.detach().cpu().numpy()

    # Denormalize
    mean, std = data.dataset.mean, data.dataset.std
    mean = mean.reshape(1, num_motionData, features_per_data, 1)
    std = std.reshape(1, num_motionData, features_per_data, 1)
    all_motions = all_motions * std + mean

    # Save using the input text
    safe_name = args.text_prompt.replace(" ", "_").replace("/", "_")
    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, f"{safe_name}.json")
    jsonl_path = os.path.join(args.output_dir, f"{safe_name}.jsonl")

    output_data = {"motion": all_motions.tolist(), "text": [args.text_prompt], "lengths": [n_frames]}
    with open(json_path, "w") as f:
        json.dump(output_data, f)
    reconstruct_jsonl(output_data, jsonl_path)
    print(f"✅ Done. Saved JSON + JSONL under: {args.output_dir}")


if __name__ == "__main__":
    main()
