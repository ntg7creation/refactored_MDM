from __future__ import annotations
import os
import argparse
from typing import List, Dict, Any
from datetime import datetime

from config import InferenceConfig
from prompt_sources import (
    load_train_prompts, load_rewrite_prompts, load_unseen_prompts, load_custom_prompts,
    sort_prompts, PromptItem
)
from sampler import load_model_and_data, sample_text
from export import (
    sanitize_filename, save_motion_jsonl, save_motion_npy,
    write_file_list_js, write_manifest_jsonl
)
from metrics import compute_pairwise_distances, write_dist_csv, summarize_intra_prompt, compute_inter_prompt_distances, summarize_energy

def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def build_prompts(cfg: InferenceConfig) -> List[PromptItem]:
    if cfg.mode == "train":
        assert cfg.train_jsonl, "train_jsonl required for mode=train"
        items = load_train_prompts(cfg.train_jsonl)
    elif cfg.mode == "rewrite":
        assert cfg.input_file_list_js and cfg.annotations_jsonl, "file_list_js + annotations_jsonl required for mode=rewrite"
        items = load_rewrite_prompts(cfg.input_file_list_js, cfg.annotations_jsonl)
    elif cfg.mode == "unseen":
        assert cfg.train_jsonl and cfg.annotations_jsonl, "train_jsonl + annotations_jsonl required for mode=unseen"
        items = load_unseen_prompts(cfg.train_jsonl, cfg.annotations_jsonl)
    elif cfg.mode == "custom":
        assert cfg.custom_prompts_path, "custom_prompts_path required for mode=custom"
        items = load_custom_prompts(cfg.custom_prompts_path)
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")

    items = sort_prompts(items, cfg.sort, cfg.random_seed_for_sort)

    # =========================
    # APPLY STRIDE ONLY WHERE IT MAKES SENSE
    # =========================
    if cfg.max_prompts is not None:

        if cfg.mode in ["unseen"]:
            stride = 1000  # jump size
            selected = []

            idx = 0
            while idx < len(items) and len(selected) < cfg.max_prompts:
                selected.append(items[idx])
                idx += stride

            items = selected

        else:
            # rewrite / custom → keep original behavior
            items = items[:cfg.max_prompts]

    return items

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None, help="path to JSON config (optional)")
    ap.add_argument("--model-path", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--mode", type=str, default=None, choices=["train","rewrite","unseen","custom"])
    ap.add_argument("--samples-per-prompt", type=int, default=None)
    ap.add_argument("--guidance", type=float, nargs="*", default=None)
    ap.add_argument("--n-frames", type=int, default=None)
    ap.add_argument("--max-prompts", type=int, default=None)
    ap.add_argument("--sort", type=str, default=None, choices=["stable","alpha","len","random"])
    ap.add_argument("--base-seed", type=int, default=None)
    ap.add_argument("--train-jsonl", type=str, default=None)
    ap.add_argument("--annotations-jsonl", type=str, default=None)
    ap.add_argument("--file-list-js", type=str, default=None)
    ap.add_argument("--custom-prompts", type=str, default=None)

    args_cli = ap.parse_args()

    cfg = InferenceConfig() if args_cli.config is None else InferenceConfig.from_json(args_cli.config)

    # CLI overrides (so you can tweak without editing JSON)
    if args_cli.model_path: cfg.model_path = args_cli.model_path
    if args_cli.out: cfg.output_dir = args_cli.out
    if args_cli.mode: cfg.mode = args_cli.mode
    if args_cli.samples_per_prompt is not None: cfg.samples_per_prompt = args_cli.samples_per_prompt
    if args_cli.guidance is not None and len(args_cli.guidance) > 0: cfg.guidance = args_cli.guidance
    if args_cli.n_frames is not None: cfg.n_frames = args_cli.n_frames
    if args_cli.max_prompts is not None: cfg.max_prompts = args_cli.max_prompts
    if args_cli.sort: cfg.sort = args_cli.sort
    if args_cli.base_seed is not None: cfg.base_seed = args_cli.base_seed
    if args_cli.train_jsonl: cfg.train_jsonl = args_cli.train_jsonl
    if args_cli.annotations_jsonl: cfg.annotations_jsonl = args_cli.annotations_jsonl
    if args_cli.file_list_js: cfg.input_file_list_js = args_cli.file_list_js
    if args_cli.custom_prompts: cfg.custom_prompts_path = args_cli.custom_prompts

    assert cfg.model_path, "model_path must be set"
    assert cfg.output_dir, "output_dir must be set"

    run_dir = os.path.join(cfg.output_dir, f"run_{cfg.mode}_{_timestamp()}")
    os.makedirs(run_dir, exist_ok=True)

    cfg.output_dir = run_dir  # lock outputs into the run folder
    args = cfg.to_args()

    prompts = build_prompts(cfg)
    print(f"[run] mode={cfg.mode} prompts={len(prompts)} samples_per_prompt={cfg.samples_per_prompt} guidance={cfg.guidance}")

    loaded = load_model_and_data(args)

    file_list_objects: List[Dict[str, Any]] = []
    manifest_rows: List[Dict[str, Any]] = []
    sample_cache: Dict[str, Any] = {}  # sample_key -> motion (for metrics)

    samples_dir = os.path.join(run_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    for p_i, p in enumerate(prompts):
        prompt_dir = os.path.join(samples_dir, f"{p_i:04d}_{sanitize_filename(p.scene)}_{sanitize_filename(p.sequence)}")
        os.makedirs(prompt_dir, exist_ok=True)

        for g in cfg.guidance:
            seed = cfg.base_seed + (p_i * 100000) + (int(g * 1000) * 100)

            motions = sample_text(
                loaded,
                args,
                text_prompt=p.text,
                guidance=g,
                batch_size=cfg.samples_per_prompt,
                seed=seed,
            )  # [B, J, C, T]

            for s_i in range(cfg.samples_per_prompt):
                motion = motions[s_i:s_i + 1]

                base = sanitize_filename(
                    f"{p_i:04d}_{p.scene}_{p.sequence}_{p.source}_g{g}_s{s_i}_seed{seed}_{p.text}",
                    140,
                )

                jsonl_path = os.path.join(prompt_dir, base + ".jsonl")
                save_motion_jsonl(motion, jsonl_path)

                sample_key = f"{p.uid}::g{g}::s{s_i}::seed{seed}"
                sample_cache[sample_key] = motion


                file_list_objects.append({
                    "filename": os.path.relpath(jsonl_path, run_dir).replace("\\", "/"),
                    "scene": p.scene,
                    "sequence": p.sequence,
                    "text": p.text,
                    "source": p.source,
                    "guidance": g,
                    "sample_idx": s_i,
                    "seed": seed,
                })

                manifest_rows.append({
                    "prompt_uid": p.uid,
                    "scene": p.scene,
                    "sequence": p.sequence,
                    "text": p.text,
                    "source": p.source,
                    "guidance": g,
                    "sample_idx": s_i,
                    "seed": seed,
                    "sample_key": sample_key,
                    "jsonl": os.path.relpath(jsonl_path, run_dir).replace("\\", "/"),
                    "npy": os.path.relpath(npy_path, run_dir).replace("\\", "/"),
                })

    # outputs for viewer + reproducibility
    write_file_list_js(file_list_objects, os.path.join(run_dir, "file_list.js"))
    write_manifest_jsonl(manifest_rows, os.path.join(run_dir, "manifest.jsonl"))

    # metrics
    metrics_dir = os.path.join(run_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    pairs = compute_pairwise_distances(sample_cache)
    write_dist_csv(pairs, os.path.join(metrics_dir, "pairwise_distances.csv"))
    summarize_intra_prompt(manifest_rows, sample_cache, os.path.join(metrics_dir, "intra_prompt_diversity.csv"))
    compute_inter_prompt_distances(manifest_rows, sample_cache, os.path.join(metrics_dir, "inter_prompt_distances.csv"),)
    summarize_energy(manifest_rows,sample_cache,os.path.join(metrics_dir, "motion_energy.csv"))

    print(f"[done] wrote {len(manifest_rows)} samples into {run_dir}")

if __name__ == "__main__":
    main()
