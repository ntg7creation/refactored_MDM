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

def get_model_args(args, data=None):
    # Custom for our simplified GigaHands layout
    return {
        'modeltype': '',
        'njoints': 21 *2,
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
    if not hasattr(args, 'text_encoder_type') or args.text_encoder_type is None:
        raise ValueError("[Error] 'text_encoder_type' not found in args!")
    model = MDM(**get_model_args(args, data))
    diffusion = create_gaussian_diffusion(args)
    return model, diffusion
# === CONFIG ===
ARGS = SimpleNamespace(
    dataset="gigahands",
    model_path="save/test_dual_concat_dual_10_20/model000040000.pt",
    output_dir="save/test_dual_concat_dual_10_20/infer_test",
    # text_prompt="hiting the wall very hard like a boxer",
    text_prompt="p042-massage",
    num_repetitions=1,
    absolote_frame_connt = 253,
    motion_length=5,
    pred_len=4,
    context_len=0,
    device=0,
    use_ema=True,
        # ================================
    # 🎛️ guidance_param (hyperparameter)
    # ----------------
    # - Controls the strength of classifier-free guidance
    # - 1.0 → no guidance (pure diffusion)
    # - >1.0 → stronger text conditioning (but risk of collapse if too high)
    # - Typical range: 1.5 – 3.0 for HumanML3D
    # - Needs tuning for GigaHands unknown yet (try 2.0–3.0)
    # ================================
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

def denormalize(motion, mean, std):
    return motion * std + mean


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

    # Motion shape (for GigaHands: 6 joints * 6 features = 36)
    fps = 20
    n_frames = int(args.absolote_frame_connt)#args.motion_length * fps)
    num_motionData = 42
    features_per_data = 3
    motion_shape = (args.num_samples, num_motionData, features_per_data, n_frames)

   
    model_kwargs = {
        'y': {
            'text': ["ignored"] * args.num_samples,
            'lengths': torch.tensor([args.pred_len] * args.num_samples, device=dist_util.dev()),
            'uncond': False,
            'scale': torch.tensor([args.guidance_param], device=dist_util.dev())  # ✅ Required!
        }
    }

    # print("[DEBUG] model_kwargs keys:", model_kwargs['y'].keys())
    model_kwargs['y']['text_embed'] = model.encode_text(model_kwargs['y']['text'])
    model_kwargs['y']['uncond'] = False  # disables conditioning entirely if true

    if args.context_len == 0:
        model_kwargs['y']['prefix'] = torch.zeros(
            (args.num_samples, num_motionData, features_per_data, 0), device=dist_util.dev()
        )

        # 💡 Add dummy mask tensor of shape (B, 1, 1, 0)
        model_kwargs['y']['mask'] = torch.zeros(
            (args.num_samples, 1, 1, 0), device=dist_util.dev()
        )

    all_motions = []

    for _ in range(args.num_repetitions):
        sample = diffusion.p_sample_loop(
            model,
            motion_shape,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=True
        )
        all_motions.append(sample.detach().cpu().numpy())




    # 🟩 Add this line:
    print("🟩 Generated motion shape:", sample.shape)
    
    # Save output
    os.makedirs(args.output_dir, exist_ok=True)
    all_motions = np.concatenate(all_motions, axis=0)

    # 🔄 Denormalize before saving
    mean = data.dataset.mean  # shape [126]
    std = data.dataset.std    # shape [126]

    if mean.shape[0] != num_motionData * features_per_data:
        raise ValueError(f"Expected mean/std shape {num_motionData*features_per_data}, got {mean.shape[0]}")

    mean = mean.reshape(1, num_motionData, features_per_data, 1)
    std = std.reshape(1, num_motionData, features_per_data, 1)

    all_motions = all_motions * std + mean


    np.save(os.path.join(args.output_dir, "results.npy"), all_motions)

    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump({
            "motion": all_motions.tolist(),
            "text": [args.text_prompt] * args.num_samples,
            "lengths": [n_frames] * args.num_samples
        }, f)

    print("✅ Done. Saved to", args.output_dir)

if __name__ == "__main__":
    main()
