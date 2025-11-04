import os
import sys
import json
import random

# ✅ Ensure access to project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ==========================================================
# CONFIG
# ==========================================================
ANNOTATION_PATH = os.path.join("GigaHands_Data", "annotations_v2.jsonl")
OUTPUT_NAME = "train_custom"       # change this to train_diez / train_small / etc
OUTPUT_PATH = os.path.join("GigaHands_Data", f"{OUTPUT_NAME}_500.jsonl")

NUM_SAMPLES = 500  # how many random samples to include


# ==========================================================
# MAIN LOGIC
# ==========================================================
def main():
    if not os.path.exists(ANNOTATION_PATH):
        raise FileNotFoundError(f"Annotation file not found at {ANNOTATION_PATH}")

    print(f"📥 Reading annotations from: {ANNOTATION_PATH}")

    samples = []
    with open(ANNOTATION_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                ann = json.loads(line.strip())
                scene = ann.get("scene")
                seq = ann.get("sequence")
                text = ann.get("clarify_annotation", "")
                if scene and seq and text:
                    samples.append({
                        "scene": scene,
                        "sequence": seq,
                        "text": text.strip()
                    })
            except json.JSONDecodeError:
                continue

    print(f"📊 Loaded {len(samples)} valid samples")

    if len(samples) == 0:
        raise ValueError("No valid entries with 'scene', 'sequence', and 'clarify_annotation' found.")

    # Pick random subset
    selected = random.sample(samples, min(NUM_SAMPLES, len(samples)))

    # Write output JSONL
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
        for s in selected:
            fout.write(json.dumps(s) + "\n")

    print(f"✅ Saved {len(selected)} samples to {OUTPUT_PATH}")
    print(f"Example entry:\n  {selected[0]}")


if __name__ == "__main__":
    main()
