import os
import json
import sys

# Ensure access to local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import numpy as np
from types import SimpleNamespace

from utils.fixseed import fixseed
from utils.model_util import load_saved_model
from utils import dist_util
from utils.sampler_util import ClassifierFreeSampleModel
from sample.generate_lite import create_model_and_diffusion
from dataset_api.dataset_registry import DatasetInterfaceRegistry

def load_jsonl(path, num_joints=6):
    with open(path, "r") as f:
        return [json.loads(line.strip()) for line in f]

def preprocess_frame(frame, num_joints=6):
    return np.array([joint[:3] for joint in frame[:num_joints]])

def flatten_motion(frames, num_joints=6):
    return np.array([preprocess_frame(f).reshape(-1) for f in frames])

def reconstruct_frames(flat_array, num_joints=6):
    return [[[float(v) for v in joint] + [1.0] for joint in frame.reshape(num_joints, 3)] for frame in flat_array]

def main():
    args = SimpleNamespace(
        dataset="gigahands",
        model_path="save/test_run/model000025000.pt",
        text_encoder_type="bert",
        device=0,
        use_ema=True,
        guidance_param=0.0,
        latent_dim=512,
        layers=8,
        cond_mask_prob=0.1,
        arch="trans_dec",
        emb_trans_dec=False,
        pos_embed_max_len=5000,
        mask_frames=True,
        multi_target_cond=False,
        multi_encoder_type="single",
        target_enc_layers=1,
        pred_len=253,
        context_len=0,
        seed=42,
    )

    gt_path = r"D:\repos\Hand_Motion_Viewer\public\hand_poses\p001-folder\keypoints_3d\000\left.jsonl"
    mean_path = r"D:\repos\mdm_custom_training\converted_motions\hand_poses\norm_stats\mean_left.npy"
    std_path = r"D:\repos\mdm_custom_training\converted_motions\hand_poses\norm_stats\std_left.npy"

    print("📦 Loading motion and stats...")
    gt_frames = load_jsonl(gt_path)
    gt_array = flatten_motion(gt_frames)  # [T, 18]
    mean = np.load(mean_path)[:18]
    std = np.load(std_path)[:18]
    x_start = (gt_array - mean) / std
    x_start = torch.tensor(x_start.T[None], dtype=torch.float32).cuda()  # [1, 18, T]

    print("🧠 Loading model...")
    fixseed(args.seed)
    dist_util.setup_dist(args.device)

    dataset_interface = DatasetInterfaceRegistry.get(args.dataset)
    data = dataset_interface.get_function("get_loader")(args)
    model, diffusion = create_model_and_diffusion(args, data)
    load_saved_model(model, args.model_path, use_avg=args.use_ema)

    if args.guidance_param != 1.0:
        model = ClassifierFreeSampleModel(model)

    model.to(dist_util.dev())
    model.eval()

    print("🔁 Running q_sample...")
    t = torch.randint(0, diffusion.num_timesteps, (1,), device=dist_util.dev()).long()
    noise = torch.randn_like(x_start)
    x_t = diffusion.q_sample(x_start=x_start, t=t, noise=noise)

    print("🎯 Denoising...")
    with torch.no_grad():
        pred = model(x_t, t)

    mse = torch.nn.functional.mse_loss(pred, x_start)
    print(f"📉 Denoising MSE at t={t.item()}: {mse.item():.8f}")

    pred_np = pred[0].cpu().numpy().T
    pred_denorm = (pred_np * std) + mean
    frames = reconstruct_frames(pred_denorm)

    output_path = os.path.join("sample", "denoised_from_qsample.jsonl")
    with open(output_path, "w") as f:
        for frame in frames:
            json.dump({"keypoints_3d": frame}, f)
            f.write("\n")

    print(f"✅ Denoised motion saved to {output_path}")

if __name__ == "__main__":
    main()
