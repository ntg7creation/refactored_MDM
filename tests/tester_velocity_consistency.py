import os
import random
import numpy as np
from pathlib import Path


def find_all_velocity_files(root):
    files = []
    for scene, _, fns in os.walk(root):
        for f in fns:
            if f.endswith("xyz_both_vel.npy"):
                files.append(os.path.join(scene, f))
    return files


def test_one_file(path, tol=1e-5):
    """
    Load motion and check: xyz[t] ≈ xyz[t-1] + vel[t]
    motion shape: [T, 252]  (# 126 xyz + 126 vel)
    """
    data = np.load(path)  # [T, 252]
    if data.ndim != 2 or data.shape[1] != 252:
        return None, None, f"Invalid shape {data.shape}"

    T = data.shape[0]

    xyz = data[:, :126]
    vel = data[:, 126:]

    # expected[t] = xyz[t-1] + vel[t]
    prev = xyz[:-1]
    moved = prev + vel[1:]

    # actual positions at t >= 1
    actual = xyz[1:]

    # compute abs error
    err = np.abs(actual - moved)
    max_err = err.max()
    mean_err = err.mean()

    ok = max_err < tol
    return max_err, mean_err, ok


def main():
    repo_root = Path(r"D:\repos\refactored_MDM")
    vel_root = repo_root / "GigaHands_Data" / "converted_velocity" / "hand_poses_xyz"

    print("Searching for velocity motions in:", vel_root)
    files = find_all_velocity_files(vel_root)
    print(f"Found {len(files)} files total")

    if len(files) == 0:
        print("❌ Error: No files found.")
        return

    # Select 100 random files (or all if <100)
    sample_size = min(100, len(files))
    test_files = random.sample(files, sample_size)

    print(f"\n🧪 Testing {sample_size} random motions...\n")

    passes = 0
    fails = 0
    detailed = []

    for path in test_files:
        max_err, mean_err, ok = test_one_file(path)

        if bool(ok):
            passes += 1
        else:
            fails += 1

        detailed.append((path, max_err, mean_err, ok))

    print("==========================================")
    print(f"✔ PASSED: {passes}")
    print(f"✘ FAILED: {fails}")
    print("==========================================")

    if fails > 0:
        print("\n---- FAIL DETAILS ----")
        for path, max_err, mean_err, ok in detailed:
            if ok is not True:
                print(f"\n{path}")
                print(f"  Max error : {max_err}")
                print(f"  Mean error: {mean_err}")


    global_max = max([d[1] for d in detailed])
    global_mean = np.mean([d[2] for d in detailed])

    print("\n==========================================")
    print(f"Global max error : {global_max}")
    print(f"Global mean error: {global_mean}")
    print("==========================================\n")


if __name__ == "__main__":
    main()
