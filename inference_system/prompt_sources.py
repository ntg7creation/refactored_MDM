from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Iterable, Set
import json
import re
import random
import os

@dataclass
class PromptItem:
    uid: str
    scene: str
    sequence: str
    text: str
    source: str              # train|rewrite|unseen|custom
    meta: Dict = None

def _read_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def load_train_prompts(train_jsonl: str, keys=("text","text_1","text_2")) -> List[PromptItem]:
    out: List[PromptItem] = []
    for i, ann in enumerate(_read_jsonl(train_jsonl)):
        scene = ann.get("scene", f"scene_{i:02d}")
        seq = ann.get("sequence", f"{i:03d}")
        for k in keys:
            t = (ann.get(k) or "").strip()
            if not t:
                continue
            uid = f"train::{scene}::{seq}::{k}"
            out.append(PromptItem(uid=uid, scene=scene, sequence=seq, text=t, source="train", meta={"key": k}))
    return out

def _parse_file_list_js(js_path: str) -> List[Dict]:
    with open(js_path, "r", encoding="utf-8") as f:
        content = f.read()
    m = re.search(r"const FILE_LIST\s*=\s*(\[.*\]);", content, re.DOTALL)
    if not m:
        raise ValueError(f"Cannot parse {js_path}")
    return json.loads(m.group(1))

def load_rewrite_prompts(file_list_js: str, annotations_jsonl: str, max_alts_per_motion: int = 6) -> List[PromptItem]:
    ann_lookup: Dict[Tuple[str,str], Dict] = {}
    for ann in _read_jsonl(annotations_jsonl):
        ann_lookup[(ann["scene"], ann["sequence"])] = ann

    file_list = _parse_file_list_js(file_list_js)
    out: List[PromptItem] = []
    for entry in file_list:
        scene = entry["scene"]
        seq = entry["sequence"]
        ann = ann_lookup.get((scene, seq))
        if ann is None:
            continue

        clarify = (ann.get("clarify_annotation") or "").strip()
        texts: List[str] = []

        for field in ("description", "rewritten_annotation"):
            val = ann.get(field)
            if isinstance(val, list):
                texts.extend([v for v in val if isinstance(v, str)])
            elif isinstance(val, str):
                texts.append(val)

        # remove empty + clarify duplicates
        texts = [t.strip() for t in texts if t and t.strip() and t.strip() != clarify]

        for j, t in enumerate(texts[:max_alts_per_motion]):
            uid = f"rewrite::{scene}::{seq}::alt{j}"
            out.append(PromptItem(uid=uid, scene=scene, sequence=seq, text=t, source="rewrite", meta={"alt_idx": j}))
    return out

def load_unseen_prompts(train_jsonl: str, annotations_jsonl: str, max_per_motion: int = 3) -> List[PromptItem]:
    """
    "Unseen" here means: any text in annotations_jsonl that does NOT appear in the train_jsonl texts.
    """
    seen: Set[str] = set()
    for p in load_train_prompts(train_jsonl):
        seen.add(p.text.strip())

    out: List[PromptItem] = []
    per_motion_count: Dict[Tuple[str,str], int] = {}

    for ann in _read_jsonl(annotations_jsonl):
        scene = ann["scene"]
        seq = ann["sequence"]
        clarify = (ann.get("clarify_annotation") or "").strip()

        texts: List[str] = []
        for field in ("description", "rewritten_annotation"):
            val = ann.get(field)
            if isinstance(val, list):
                texts.extend([v for v in val if isinstance(v, str)])
            elif isinstance(val, str):
                texts.append(val)

        texts = [t.strip() for t in texts if t and t.strip() and t.strip() != clarify]
        # keep only unseen
        texts = [t for t in texts if t not in seen]

        key = (scene, seq)
        for t in texts:
            c = per_motion_count.get(key, 0)
            if c >= max_per_motion:
                break
            uid = f"unseen::{scene}::{seq}::{c}"
            out.append(PromptItem(uid=uid, scene=scene, sequence=seq, text=t, source="unseen", meta={"rank": c}))
            per_motion_count[key] = c + 1

    return out

def load_custom_prompts(path: str) -> List[PromptItem]:
    """
    Accepts:
      - .txt : one prompt per line
      - .jsonl : expects {"text": "..."} per line (scene/sequence optional)
    """
    ext = os.path.splitext(path)[1].lower()
    out: List[PromptItem] = []

    if ext == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                t = line.strip()
                if not t:
                    continue
                out.append(PromptItem(uid=f"custom::{i}", scene="custom", sequence=f"{i:04d}", text=t, source="custom", meta={}))
        return out

    if ext == ".jsonl":
        for i, ann in enumerate(_read_jsonl(path)):
            t = (ann.get("text") or "").strip()
            if not t:
                continue
            scene = ann.get("scene", "custom")
            seq = ann.get("sequence", f"{i:04d}")
            out.append(PromptItem(uid=f"custom::{scene}::{seq}::{i}", scene=scene, sequence=seq, text=t, source="custom", meta={k:v for k,v in ann.items() if k not in ("text","scene","sequence")}))
        return out

    raise ValueError(f"Unsupported custom prompts file type: {ext}")

def sort_prompts(items: List[PromptItem], mode: str, seed: int = 123) -> List[PromptItem]:
    if mode == "stable":
        return items
    if mode == "alpha":
        return sorted(items, key=lambda x: x.text.lower())
    if mode == "len":
        return sorted(items, key=lambda x: len(x.text))
    if mode == "random":
        rnd = random.Random(seed)
        items2 = items[:]
        rnd.shuffle(items2)
        return items2
    raise ValueError(f"Unknown sort mode: {mode}")
