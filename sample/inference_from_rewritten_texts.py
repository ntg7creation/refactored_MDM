import os
import re
import json
import numpy as np
import torch
from types import SimpleNamespace
from tqdm import tqdm
import sys

# add repo root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.fixseed import fixseed
from utils.model_util import load_saved_model, create_gaussian_diffusion
from utils import dist_util
from utils.sampler_util import ClassifierFreeSampleModel
from dataset_api.dataset_registry import DatasetInterfaceRegistry
from model.mdm import MDM


# ==========================================================
# CONFIG
# ==========================================================
ARGS = SimpleNamespace(
    dataset="gigahands",
    model_path="save/test_custom500_concat_11_4/model000045000.pt",
    file_list_js="save/test_custom500_concat_11_4/infer_trainset/file_list.js",
    original_annotation="GigaHands_Data/annotations_v2.jsonl",
    output_dir="save/test_custom500_concat_11_4/infer_rewritten",
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
# UTILS
# ==========================================================
def sanitize_filename(s: str, max_length: int = 80) -> str:
    s = re.sub(r'[^a-zA-Z0-9_\-]+', '_', s).strip('_')
    return s[:max_length]


def reconstruct_jsonl(motion, output_path):
    n_joints, n_coords, n_frames = motion.shape[1], motion.shape[2], motion.shape[3]
    with open(output_path, "w") as fout:
        for f_idx in range(n_frames):
            keypoints = [
                [float(motion[0, j, 0, f_idx]),
                 float(motion[0, j, 1, f_idx]),
                 float(motion[0, j, 2, f_idx]),
                 1.0]
                for j in range(n_joints)
            ]
            fout.write(json.dumps(keypoints) + "\n")


def load_file_list(js_path):
    """Parse file_list.js to extract the list of scene/sequence pairs."""
    with open(js_path, "r", encoding="utf-8") as f:
        content = f.read()
    json_text = re.search(r"const FILE_LIST\s*=\s*(\[.*\]);", content, re.DOTALL)
    if not json_text:
        raise ValueError(f"Cannot parse {js_path}")
    return json.loads(json_text.group(1))


def index_annotations(annotation_path):
    """Create lookup table (scene, seq) -> annotation JSON."""
    lookup = {}
    with open(annotation_path, "r", encoding="utf-8") as f:
        for line in f:
            ann = json.loads(line)
            scene = ann["scene"]
            seq = ann["sequence"]
            lookup[(scene, seq)] = ann
    print(f"🔍 Indexed {len(lookup)} annotations from {annotation_path}")
    return lookup


# ==========================================================
# MAIN
# ==========================================================
def main(args=ARGS):
    fixseed(0)
    dist_util.setup_dist(args.device)

    dataset_interface = DatasetInterfaceRegistry.get(args.dataset)
    get_loader_fn = dataset_interface.get_function("get_loader")
    data = get_loader_fn(args)

    # Build model + diffusion
    model = MDM(
        modeltype='',
        njoints=42,
        nfeats=3,
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
    if args.guidance_param != 1.0:
        model = ClassifierFreeSampleModel(model)
    model.to(dist_util.dev())
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)
    mean, std = data.dataset.mean, data.dataset.std
    mean = mean.reshape(1, 42, 3, 1)
    std = std.reshape(1, 42, 3, 1)

    # ----------------------------------------------------------
    # Load previous inference list + original annotations
    # ----------------------------------------------------------
    file_list = load_file_list(args.file_list_js)
    ann_lookup = index_annotations(args.original_annotation)

    print(f"📁 Loaded {len(file_list)} motions from {args.file_list_js}")

    new_file_objects = []

    for entry in tqdm(file_list, desc="Generating rewritten motions"):
        scene = entry["scene"]
        seq = entry["sequence"]

        ann = ann_lookup.get((scene, seq))
        if ann is None:
            print(f"⚠️ Missing annotation for {scene}/{seq}")
            continue

        # collect alternative texts (exclude clarify_annotation)
        all_texts = []
        for field in ["description", "rewritten_annotation"]:
            val = ann.get(field)
            if isinstance(val, list):
                all_texts.extend(val)
            elif isinstance(val, str):
                all_texts.append(val)
        all_texts = [t for t in all_texts if t.strip() and t.strip() != ann.get("clarify_annotation", "").strip()]

        # generate for each text
        for idx, text_prompt in enumerate(all_texts[:6]):  # limit to 3 rewrites per motion
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

            shape = (1, 42, 3, args.absolote_frame_connt)
            sample = diffusion.p_sample_loop(model, shape, clip_denoised=False,
                                             model_kwargs=model_kwargs, progress=False)
            motion = sample.detach().cpu().numpy()
            motion = motion * std + mean

            safe_name = sanitize_filename(f"{scene}_{seq}_alt{idx}_{text_prompt}")
            jsonl_path = os.path.join(args.output_dir, f"{safe_name}.jsonl")
            reconstruct_jsonl(motion, jsonl_path)

            new_file_objects.append({
                "filename": os.path.basename(jsonl_path),
                "scene": scene,
                "sequence": seq,
                "text": text_prompt
            })

    # ----------------------------------------------------------
    # Write output lists
    # ----------------------------------------------------------
    js_file = os.path.join(args.output_dir, "file_list.js")
    txt_file = os.path.join(args.output_dir, "filenames.txt")

    with open(js_file, "w", encoding="utf-8") as f:
        f.write("const FILE_LIST = ")
        json.dump(new_file_objects, f, indent=4)
        f.write(";\nexport default FILE_LIST;\n")

    with open(txt_file, "w", encoding="utf-8") as f:
        f.write("\n".join([obj["filename"] for obj in new_file_objects]))

    print(f"✅ Created {len(new_file_objects)} new rewritten motions.")
    print(f"📄 JS list saved to: {js_file}")
    print(f"📄 Text list saved to: {txt_file}")
    print(f"📁 Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
