from __future__ import annotations
from typing import Dict, List, Tuple
import os
import csv
import numpy as np

def motion_vec(motion: np.ndarray) -> np.ndarray:
    # motion [1,J,C,T] -> flatten
    return motion.reshape(-1).astype(np.float32)

def l2(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    return float(np.sqrt(np.mean(d * d)))

def compute_pairwise_distances(samples: Dict[str, np.ndarray]) -> List[Tuple[str, str, float]]:
    """
    samples: key -> motion array
    returns (key_i, key_j, dist)
    """
    keys = list(samples.keys())
    vecs = {k: motion_vec(samples[k]) for k in keys}
    out = []
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            ki, kj = keys[i], keys[j]
            out.append((ki, kj, l2(vecs[ki], vecs[kj])))
    return out

def write_dist_csv(rows: List[Tuple[str,str,float]], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["key_i", "key_j", "l2_rmse"])
        w.writerows(rows)

def summarize_intra_prompt(manifest: List[dict], samples: Dict[str, np.ndarray], out_path: str) -> None:
    """
    manifest rows must include: prompt_uid, sample_key
    """
    by_prompt: Dict[str, List[str]] = {}
    for row in manifest:
        by_prompt.setdefault(row["prompt_uid"], []).append(row["sample_key"])

    rows = []
    for prompt_uid, keys in by_prompt.items():
        if len(keys) < 2:
            continue
        sub = {k: samples[k] for k in keys if k in samples}
        pairs = compute_pairwise_distances(sub)
        vals = [p[2] for p in pairs]
        rows.append((prompt_uid, len(keys), float(np.mean(vals)), float(np.std(vals)), float(np.min(vals)), float(np.max(vals))))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["prompt_uid", "n_samples", "mean_l2", "std_l2", "min_l2", "max_l2"])
        w.writerows(rows)


def compute_inter_prompt_distances(
    manifest: List[dict],
    samples: Dict[str, np.ndarray],
    out_path: str,
    representative_sample_idx: int = 0,
):
    """
    Computes distances between different prompts using one representative
    motion per prompt.
    """

    # pick one motion per prompt
    rep: Dict[str, np.ndarray] = {}

    for row in manifest:
        if row["sample_idx"] != representative_sample_idx:
            continue
        k = row["sample_key"]
        if k in samples:
            rep[row["prompt_uid"]] = samples[k]

    prompt_ids = list(rep.keys())
    vecs = {pid: motion_vec(rep[pid]) for pid in prompt_ids}

    rows = []
    for i in range(len(prompt_ids)):
        for j in range(i + 1, len(prompt_ids)):
            pi, pj = prompt_ids[i], prompt_ids[j]
            dist = l2(vecs[pi], vecs[pj])
            rows.append((pi, pj, dist))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["prompt_i", "prompt_j", "l2_rmse"])
        w.writerows(rows)


def compute_motion_energy(
    motion: np.ndarray,
    n_joints: int = 42,
) -> float:
    """
    motion: [1, J, C, T]
    Uses first n_joints, recomputes velocity, returns total kinetic energy
    """
    # extract xyz
    pos = motion[0, :n_joints, :3, :]  # [J, 3, T]

    # finite difference velocity
    vel = pos[:, :, 1:] - pos[:, :, :-1]  # [J, 3, T-1]

    # kinetic energy = 1/2 * sum(v^2)
    energy = 0.5 * np.sum(vel ** 2)

    return float(energy)


def summarize_energy(
    manifest: List[dict],
    samples: Dict[str, np.ndarray],
    out_path: str,
):
    rows = []

    for row in manifest:
        k = row["sample_key"]
        if k not in samples:
            continue

        energy = compute_motion_energy(samples[k])

        rows.append((
            row["prompt_uid"],
            row["sample_idx"],
            energy,
        ))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["prompt_uid", "sample_idx", "energy"])
        w.writerows(rows)

