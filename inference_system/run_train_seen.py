# inference_system/run_train_seen.py

from config import InferenceConfig
from run_infer import build_prompts
from sampler import load_model_and_data, sample_text
from export import (
    sanitize_filename,
    save_motion_jsonl,
    save_motion_npy,
    write_file_list_js,
    write_manifest_jsonl,
)
from metrics import (
    compute_pairwise_distances,
    write_dist_csv,
    summarize_intra_prompt,
    compute_inter_prompt_distances,
    summarize_energy,
)



import os
from datetime import datetime
from typing import Dict, Any, List
import shutil

def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def main():
    # =========================
    # 1️⃣ DEFINE THE EXPERIMENT
    # =========================
    cfg = InferenceConfig(
        dataset="gigahands",
      
        model_path=r"save\test_custom_velocity_target_500\model000075000.pt",
        output_dir=r"save\test_custom_velocity_target_500\infer",
        mode="train",

        train_jsonl="GigaHands_Data/train_custom_500.jsonl",

        samples_per_prompt=3,
        guidance=[2.5],
        n_frames=253,

        max_prompts=10,          # 🔴 start small
        sort="stable",
        base_seed=1234,

        # dataset-specific
        root_dir="GigaHands_Data",
        annotation_file="GigaHands_Data/annotations_v2.jsonl",
        mean_std_dir="GigaHands_Data/mean_std",
        dmvb_size=126,
        dmvb_layout="full",
    )

    # =========================
    # 2️⃣ PREPARE OUTPUT FOLDER
    # =========================
    run_dir = os.path.join(
        cfg.output_dir,
        f"run_train_seen_{timestamp()}"
    )
    os.makedirs(run_dir, exist_ok=True)
    cfg.output_dir = run_dir

    print(f"[INFO] Output dir: {run_dir}")

    # =========================
    # 3️⃣ BUILD ARGS + PROMPTS
    # =========================
    args = cfg.to_args()
    prompts = build_prompts(cfg)

    print(f"[INFO] Loaded {len(prompts)} prompts")

    # =========================
    # 4️⃣ LOAD MODEL ONCE
    # =========================
    loaded = load_model_and_data(args)

    # =========================
    # 5️⃣ RUN SAMPLING
    # =========================
    file_list: List[Dict[str, Any]] = []
    manifest: List[Dict[str, Any]] = []
    sample_cache: Dict[str, Any] = {}

    samples_dir = os.path.join(run_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    for p_idx, p in enumerate(prompts):
        prompt_dir = os.path.join(
            samples_dir,
            f"{p_idx:04d}_{sanitize_filename(p.scene)}_{sanitize_filename(p.sequence)}"
        )
        os.makedirs(prompt_dir, exist_ok=True)

        for g in cfg.guidance:
            seed = cfg.base_seed + p_idx * 10000

            # 🔹 ONE diffusion call → B motions
            motions = sample_text(
                loaded,
                args,
                text_prompt=p.text,
                guidance=g,
                batch_size=cfg.samples_per_prompt,
                seed=seed,
            )  # motions shape: [B, J, C, T]

            for s_idx in range(cfg.samples_per_prompt):
                motion = motions[s_idx:s_idx + 1]  # keep [1, J, C, T]

                base = sanitize_filename(
                    f"{p_idx}_{p.scene}_{p.sequence}_g{g}_s{s_idx}_{p.text}",
                    max_length=140,
                )

                jsonl_path = os.path.join(prompt_dir, base + ".jsonl")
                save_motion_jsonl(motion, jsonl_path)

                sample_key = f"{p.uid}::g{g}::s{s_idx}::seed{seed}"
                sample_cache[sample_key] = motion

                file_list.append({
                    "filename": os.path.relpath(jsonl_path, run_dir).replace("\\", "/"),
                    "scene": p.scene,
                    "sequence": p.sequence,
                    "text": p.text,
                    "source": p.source,
                    "guidance": g,
                    "sample_idx": s_idx,
                    "seed": seed,
                })

                manifest.append({
                    "prompt_uid": p.uid,
                    "scene": p.scene,
                    "sequence": p.sequence,
                    "text": p.text,
                    "source": p.source,
                    "guidance": g,
                    "sample_idx": s_idx,
                    "seed": seed,
                    "sample_key": sample_key,
                    "jsonl": os.path.relpath(jsonl_path, run_dir).replace("\\", "/"),
                })


    # =========================
    # 6️⃣ SAVE METADATA
    # =========================
    write_file_list_js(file_list, os.path.join(run_dir, "file_list.js"))
    write_manifest_jsonl(manifest, os.path.join(run_dir, "manifest.jsonl"))

    # =========================
    # 7️⃣ METRICS
    # =========================
    metrics_dir = os.path.join(run_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    pairs = compute_pairwise_distances(sample_cache)
    write_dist_csv(pairs, os.path.join(metrics_dir, "pairwise_distances.csv"))
    summarize_intra_prompt(
        manifest,
        sample_cache,
        os.path.join(metrics_dir, "intra_prompt_diversity.csv"),
    )


    compute_inter_prompt_distances(
        manifest,
        sample_cache,
        os.path.join(metrics_dir, "inter_prompt_distances.csv"),
    )


    summarize_energy(
        manifest,
        sample_cache,
        os.path.join(metrics_dir, "motion_energy.csv"),
    )



    print(f"[DONE] Generated {len(manifest)} samples")


    # =========================
    # 8️ COPY ANALYSIS NOTEBOOK
    # =========================
    ANALYSIS_NOTEBOOK_TEMPLATE = "analysis.ipynb"  # repo-level template

    dst_nb = os.path.join(run_dir, "analysis.ipynb")

    if os.path.exists(ANALYSIS_NOTEBOOK_TEMPLATE):
        shutil.copy(ANALYSIS_NOTEBOOK_TEMPLATE, dst_nb)
        print(f"[INFO] Copied analysis notebook → {dst_nb}")
    else:
        print("[WARN] analysis.ipynb template not found, skipping copy")



if __name__ == "__main__":
    main()
