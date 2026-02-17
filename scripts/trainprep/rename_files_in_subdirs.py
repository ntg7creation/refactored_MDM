import os
import argparse

def rename_files_recursive(root_dir, old_name, new_name, dry_run=False):
    """
    Recursively rename files named `old_name` to `new_name`
    under `root_dir`.
    """

    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Directory not found: {root_dir}")

    total = 0

    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname == old_name:
                old_path = os.path.join(dirpath, fname)
                new_path = os.path.join(dirpath, new_name)

                if os.path.exists(new_path):
                    print(f"[SKIP] Target exists: {new_path}")
                    continue

                print(f"[RENAME] {old_path} -> {new_path}")

                if not dry_run:
                    os.rename(old_path, new_path)

                total += 1

    print(f"\nDone. Renamed {total} files.")
    if dry_run:
        print("Dry-run mode: no files were actually renamed.")


def main():
    parser = argparse.ArgumentParser(
        description="Rename files recursively inside subdirectories."
    )
    parser.add_argument("root_dir", help="Root directory to scan")
    parser.add_argument("old_name", help="Exact old filename (e.g. left_xyz.npy)")
    parser.add_argument("new_name", help="New filename (e.g. left_tpr.npy)")
    parser.add_argument("--dry-run", action="store_true", help="Preview only")

    args = parser.parse_args()

    rename_files_recursive(
        root_dir=args.root_dir,
        old_name=args.old_name,
        new_name=args.new_name,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
