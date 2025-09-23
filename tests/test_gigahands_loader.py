# tests/test_gigahands_loader.py
import json
import numpy as np
import torch
from pathlib import Path
import sys

# Make repo root importable (so we can import your loader module)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset_api.gigahands.Gigadataset_loader import GigaHandsT2M


def _build_fake_dataset(tmp: Path, side="left", D=263, T0=80, T1=120):
    """
    Creates a minimal on-disk dataset:

    <root>/
      norm_stats/
        mean_left.npy
        std_left.npy
      sceneA/keypoints_3d/seq000/dmvb_left.npy    (T0 x D, all ones)
      sceneB/keypoints_3d/seq111/dmvb_left.npy    (T1 x D, all sevens)
    annotations.jsonl  (two lines)
    """
    root = tmp / "hand_poses"
    stats = root / "norm_stats"
    a0 = root / "sceneA" / "keypoints_3d" / "seq000"
    a1 = root / "sceneB" / "keypoints_3d" / "seq111"
    stats.mkdir(parents=True, exist_ok=True)
    a0.mkdir(parents=True, exist_ok=True)
    a1.mkdir(parents=True, exist_ok=True)

    # mean/std expected by GigaHandsT2M: mean_left.npy / std_left.npy
    np.save(stats / f"mean_{side}.npy", np.zeros((D,), dtype=np.float32))
    np.save(stats / f"std_{side}.npy", np.ones((D,), dtype=np.float32))

    # motions
    m0 = np.ones((T0, D), dtype=np.float32)      # first motion (canonical)
    m1 = np.full((T1, D), 7.0, dtype=np.float32) # different content
    np.save(a0 / f"dmvb_{side}.npy", m0)
    np.save(a1 / f"dmvb_{side}.npy", m1)

    # annotations .jsonl
    ann = tmp / "annotations.jsonl"
    with open(ann, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "scene": "sceneA",
            "sequence": "seq000",
            "rewritten_annotation": ["first sample caption"],
        }) + "\n")
        f.write(json.dumps({
            "scene": "sceneB",
            "sequence": "seq111",
            "rewritten_annotation": ["second sample caption"],
        }) + "\n")

    return root, ann, stats

def test_structure_and_types(tmp_path: Path):
    """
    Verifies the tuple structure and shapes:
      (word_embeddings, pos_one_hots, text, sent_len, motion, m_length, token_key)
    """
    D = 263
    T0 = 80   # choose fixed_len == T0 to avoid random cropping variance
    T1 = 100
    root, ann, stats = _build_fake_dataset(tmp_path, D=D, T0=T0, T1=T1)

    ds = GigaHandsT2M(
        root_dir=str(root),
        annotation_file=str(ann),
        mean_std_dir=str(stats),
        side="left",
        split="train",
        device="cpu",
        num_frames=T0,  # fixed_len
    )

    assert len(ds) == 2

    we, pos, text, slen, motion, mlen, tokkey = ds[0]

    # embeddings / POS shapes match your current max_text_len (=40) + sos/eos (=2) -> 42
    assert isinstance(we, torch.Tensor) and we.shape == (ds.max_text_len + 2, 300)
    assert isinstance(pos, torch.Tensor) and pos.shape == (ds.max_text_len + 2, 49)

    # text + length
    assert isinstance(text, str) and len(text) > 0
    assert isinstance(slen, int) and 1 <= slen <= (ds.max_text_len + 2)

    # motion (this is your DMVB) -> [T, D]
    assert isinstance(motion, torch.Tensor)
    assert motion.ndim == 2 and motion.shape == (T0, D)
    assert isinstance(mlen, int) and mlen == T0

    # tokenized key is a string
    assert isinstance(tokkey, str) and len(tokkey) > 0


def test_always_uses_first_motion(tmp_path: Path):
    """
    With your two-line change in __getitem__:
      _, text = self.samples[idx]
      motion_path, _ = self.samples[0]
    both idx=0 and idx=1 must return the SAME motion tensor.
    """
    D = 263
    T0 = 80
    T1 = 120
    root, ann, stats = _build_fake_dataset(tmp_path, D=D, T0=T0, T1=T1)

    ds = GigaHandsT2M(
        root_dir=str(root),
        annotation_file=str(ann),
        mean_std_dir=str(stats),
        side="left",
        split="train",
        device="cpu",
        num_frames=T0,  # fixed_len == T0 ⇒ cropping always starts at 0
    )

    m0 = ds[0][4]  # motion
    m1 = ds[1][4]
    assert torch.allclose(m0, m1), "Expected identical motions (always loading first motion)."


def test_inv_transform_roundtrip(tmp_path: Path):
    """
    Quick sanity: inv_transform(normalize(x)) ≈ x on a single frame.
    """
    D = 263
    T0 = 16
    T1 = 32
    root, ann, stats = _build_fake_dataset(tmp_path, D=D, T0=T0, T1=T1)

    ds = GigaHandsT2M(
        root_dir=str(root),
        annotation_file=str(ann),
        mean_std_dir=str(stats),
        side="left",
        split="train",
        device="cpu",
        num_frames=T0,
    )

    # Load raw first motion directly to compare
    raw = np.load(root / "sceneA" / "keypoints_3d" / "seq000" / "dmvb_left.npy").astype(np.float32)
    # What __getitem__ returns is normalized; undo it and compare a frame
    motion = ds[0][4].numpy()  # [T, D], normalized
    restored = ds.inv_transform(motion)
    assert np.allclose(restored[0], raw[0], atol=1e-6)


def test_dynamic_dmvb_size(tmp_path: Path):
    """
    Confirms that motion vectors returned by GigaHandsT2M match the given DMVB layout shape,
    such as for "root+5" (18D) or "full" (263D).
    """
    D = 263
    T0 = 64
    root, ann, stats = _build_fake_dataset(tmp_path, D=D, T0=T0, T1=100)

    for layout, expected_D in [("full", 263), ("root+5", 18)]:
        dataset = GigaHandsT2M(
            root_dir=str(root),
            annotation_file=str(ann),
            mean_std_dir=str(stats),
            side='left',
            split='train',
            device='cpu',
            num_frames=T0,
            dmvb_layout=layout
        )

        assert len(dataset) == 2
        motion = dataset[0][4]  # [T, D]

        assert motion.shape[0] == T0
        assert motion.shape[1] == expected_D, \
            f"Layout {layout} expected D={expected_D}, got {motion.shape[1]}"
    

def test_dmvb_layout_root5(tmp_path: Path):
    """
    Checks if 'root+5' layout correctly produces 18D motion and extracts correct values.
    """
    D = 263
    T0 = 80
    T1 = 100
    root, ann, stats = _build_fake_dataset(tmp_path, D=D, T0=T0, T1=T1)

    # Patch motion to increasing integers to verify indexing
    raw = np.arange(T0 * D).reshape(T0, D).astype(np.float32)
    np.save(root / "sceneA" / "keypoints_3d" / "seq000" / "dmvb_left.npy", raw)
    np.save(root / "sceneB" / "keypoints_3d" / "seq111" / "dmvb_left.npy", raw + 1)

    ds = GigaHandsT2M(
        root_dir=str(root),
        annotation_file=str(ann),
        mean_std_dir=str(stats),
        side="left",
        split="train",
        device="cpu",
        num_frames=T0,
        dmvb_layout="root+5"
    )

    motion = ds[0][4].numpy()  # [T, 18]

    # Expected joint indices [0, 1, 5, 9, 13, 17]
    expected = np.concatenate([
        raw[:, j * 3 : (j + 1) * 3] for j in [0, 1, 5, 9, 13, 17]
    ], axis=1)

    assert motion.shape == expected.shape
    np.testing.assert_array_equal(motion, expected)

