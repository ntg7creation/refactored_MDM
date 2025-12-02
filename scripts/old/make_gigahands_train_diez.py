import json
import os

# --- Configuration ---
ANNOTATION_FILE = r"D:\repos\refactored_MDM\GigaHands_Data\annotations_v2.jsonl"
OUTPUT_FILE = r"D:\repos\refactored_MDM\GigaHands_Data\gigahands_train_diez.jsonl"

FIXED_SAMPLES = [
    ("p005-sandwich-salad-baking-monoply-boxing", "018"),
    ("p004-mindmap", "006"),
    ("p003-packing", "034"),
    ("p008-boxing", "007"),
    ("p009-tool", "002"),
    ("p010-fastfood", "004"),
    ("p021-monopoly", "007"),
    ("p031-sewing", "005"),
    ("p042-baking", "001"),
    ("p046-cleaning-tool", "003"),
]


def main():
    if not os.path.exists(ANNOTATION_FILE):
        raise FileNotFoundError(f"Missing annotation file: {ANNOTATION_FILE}")

    print(f"📘 Loading annotations from {ANNOTATION_FILE}")
    with open(ANNOTATION_FILE, "r", encoding="utf-8") as f:
        annotations = [json.loads(line) for line in f]

    fixed_keys = set(FIXED_SAMPLES)
    selected = []

    for ann in annotations:
        key = (ann.get("scene"), ann.get("sequence"))
        if key in fixed_keys:
            rewritten = ann.get("rewritten_annotation", [])
            if len(rewritten) >= 2:
                entry = {
                    "scene": ann["scene"],
                    "sequence": ann["sequence"],
                    "text_1": rewritten[0],
                    "text_2": rewritten[1],
                }
                selected.append(entry)
                print(f"✅ Added {key}: {rewritten[0][:40]} / {rewritten[1][:40]}")
            else:
                print(f"⚠️ Skipped {key} — not enough rewritten annotations ({len(rewritten)})")

    if not selected:
        print("❌ No matching samples found!")
        return

    with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
        for entry in selected:
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\n💾 Saved {len(selected)} entries to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
