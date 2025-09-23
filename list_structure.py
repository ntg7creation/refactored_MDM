import os
import builtins

# Folders to ignore (by name, anywhere in the tree)
IGNORE_FOLDERS = {
    ".git", "node_modules", "dist", "build", "__pycache__", "data", "datasets", "save", ".pytest_cache","ASL_Data"
}

# Folders to ignore by relative path (from root)
IGNORE_PATHS = {
    os.path.normpath("GigaHands_Data/coverted_motions"),
    # add more like: os.path.normpath("some/other/path")
}




OUTPUT_FILE = "file_structure.txt"

def walk_directory(root, prefix=""):
    try:
        entries = sorted(os.listdir(root))
    except PermissionError:
        return

    for i, entry in enumerate(entries):
        path = os.path.join(root, entry)
        rel_path = os.path.normpath(os.path.relpath(path, "."))  # relative path from root
        is_last = i == len(entries) - 1
        connector = "└── " if is_last else "├── "

        if os.path.isdir(path):
            if entry in IGNORE_FOLDERS or rel_path in IGNORE_PATHS:
                continue
            print(prefix + connector + entry + "/")
            walk_directory(path, prefix + ("    " if is_last else "│   "))
        else:
            print(prefix + connector + entry)

if __name__ == "__main__":
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        # Save original print
        original_print = builtins.print

        def file_print(*args, **kwargs):
            original_print(*args, **kwargs, file=f)

        # Replace global print with file_print
        globals()["print"] = file_print

        print(".")
        walk_directory(".")

    original_print(f"File structure written to {OUTPUT_FILE}")
