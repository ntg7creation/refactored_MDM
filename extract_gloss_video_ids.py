import json
from pathlib import Path

# Input and output paths
input_path = Path("ASL_Data/dataset_filtered.json")
output_path = Path("ASL_Data/annotations_minimal.jsonl")

with input_path.open("r", encoding="utf-8") as f:
    full_data = json.load(f)

with output_path.open("w", encoding="utf-8") as f_out:
    count = 0
    for entry in full_data:
        gloss = entry.get("gloss")
        for inst in entry.get("instances", []):
            video_id = inst.get("video_id")
            if gloss and video_id:
                obj = {"gloss": gloss, "video_id": video_id}
                f_out.write(json.dumps(obj) + "\n")
                count += 1

print(f"✅ Wrote {count} minimal entries to: {output_path}")
