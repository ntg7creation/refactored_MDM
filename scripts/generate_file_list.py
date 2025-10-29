import json
import re
import os
import sys

# Ensure access to local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

INPUT_FILE = r"GigaHands_Data\gigahands_train_diez.jsonl"
OUTPUT_FILE = "file_list.js"

def safe_name(s: str) -> str:
    """Sanitize to a filesystem-safe identifier."""
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', s.strip())

def make_full_filename(ann: dict, key: str, index: int) -> str:
    """Build the full motion file name, e.g.
       12_p042-baking_001_text_1_Place_the_flour_bag_beside_the_bowl.jsonl"""
    scene = ann.get("scene", "unknownscene")
    seq = ann.get("sequence", "000")
    text = safe_name(ann.get(key, ""))
    return f"{index:02d}_{scene}_{seq}_{key}_{text}.jsonl"

def main():
    file_list = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            ann = json.loads(line)
            for key in ["text_1", "text_2"]:
                if key in ann and ann[key].strip():
                    file_list.append(make_full_filename(ann, key, i))

    # Write JS file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("const FILE_LIST = [\n")
        for name in file_list:
            f.write(f'    "{name}",\n')
        f.write("];\nexport default FILE_LIST;\n")

    print(f"✅ Done! Wrote {len(file_list)} entries to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
