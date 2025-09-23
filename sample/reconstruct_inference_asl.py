import json
import os

def reconstruct_jsonl(input_path: str, output_path: str):
    print(f"📥 Reading: {input_path}")
    if not os.path.exists(input_path):
        print(f"❌ Input file does not exist: {input_path}")
        return

    with open(input_path, "r") as f:
        data = json.load(f)

    motions = data.get("motion", [])
    if len(motions) != 1:
        print(f"❌ Expected 1 motion sample, found {len(motions)}.")
        return
    
    print("✅ Loaded motion sample")

    motion = motions[0]  # shape: [6, 3, 4]
    n_joints = len(motion)
    n_coords = len(motion[0])
    n_frames = len(motion[0][0])

    print(f"ℹ️ Shape: joints={n_joints}, coords={n_coords}, frames={n_frames}")
    print(f"📤 Writing to: {output_path}")

    with open(output_path, "w") as fout:
        for frame_idx in range(n_frames):
            keypoints = []
            for joint in range(n_joints):
                x = motion[joint][0][frame_idx]
                y = motion[joint][1][frame_idx]
                z = motion[joint][2][frame_idx]
                keypoints.append([x, y, z, 1.0])  # Add confidence score

            fout.write(json.dumps(keypoints) + "\n")


    print(f"✅ Reconstructed {n_frames} frames → saved to: {output_path}")


def main():
    print("🚀 Starting reconstruct_inference.py")

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    infer_dir = os.path.join(root, "save", "ASL_01", "infer_test")

    input_path = os.path.join(infer_dir, "results.json")
    output_path = os.path.join(infer_dir, "reconstructed_ASL.jsonl")

    print(f"🔍 Root directory: {root}")
    print(f"📂 Inference directory: {infer_dir}")
    
    reconstruct_jsonl(input_path, output_path)


if __name__ == "__main__":
    main()
