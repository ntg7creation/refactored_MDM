import os
import json
import numpy as np
import matplotlib.pyplot as plt

import sys
# Ensure access to local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# === CONFIG ===
RESULTS_FILE = r"D:\repos\refactored_MDM\save\test_custom500_concat_11_4\infer_trainset_blind\closest_matches.json"
OUTPUT_DIR = os.path.dirname(RESULTS_FILE)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------
# 📦 Load results
# ------------------------------------------------------
with open(RESULTS_FILE, "r") as f:
    results = json.load(f)

print(f"✅ Loaded {len(results)} generated motions from {RESULTS_FILE}")

# Sort for consistency
results = dict(sorted(results.items()))

# ------------------------------------------------------
# 🎨 Plot each generated motion's closest matches
# ------------------------------------------------------
for gfile, matches in results.items():
    scenes = [f"{m['scene']}/{m['seq']}" for m in matches]
    mse_values = [m["mse"] for m in matches]

    fig, ax = plt.subplots(figsize=(8, 4))
    y_pos = np.arange(len(scenes))

    ax.barh(y_pos, mse_values, color="skyblue", edgecolor="black")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(scenes, fontsize=8)
    ax.invert_yaxis()  # highest at top
    ax.set_xlabel("MSE Distance")
    ax.set_title(f"Closest motions for {gfile}")

    for i, v in enumerate(mse_values):
        ax.text(v, i, f"{v:.4f}", va="center", ha="left", fontsize=7)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(gfile)[0]}_closest.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

print(f"✅ Saved all bar charts to {OUTPUT_DIR}")
