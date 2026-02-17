from __future__ import annotations
from typing import Dict, Any, List
import os, re, json
import numpy as np

def sanitize_filename(s: str, max_length: int = 120) -> str:
    s = s.replace(",", "")
    s = re.sub(r"[^a-zA-Z0-9_\- ]+", "_", s)
    s = s.strip(" .")
    s = "_".join(s.split())
    if len(s) > max_length:
        s = s[:max_length].rstrip("_")
    return s

def save_motion_jsonl(motion: np.ndarray, out_path: str) -> None:
    # motion: [1, J, C, T]
    J, C, T = motion.shape[1], motion.shape[2], motion.shape[3]
    with open(out_path, "w", encoding="utf-8") as f:
        for t in range(T):
            keypoints = [
                [float(motion[0, j, 0, t]), float(motion[0, j, 1, t]), float(motion[0, j, 2, t]), 1.0]
                for j in range(J)
            ]
            f.write(json.dumps(keypoints) + "\n")

def save_motion_npy(motion: np.ndarray, out_path: str) -> None:
    np.save(out_path, motion)

def write_file_list_js(file_objects: List[Dict[str, Any]], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("const FILE_LIST = ")
        json.dump(file_objects, f, indent=4)
        f.write(";\nexport default FILE_LIST;\n")

def write_manifest_jsonl(manifest_rows: List[Dict[str, Any]], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row) + "\n")
