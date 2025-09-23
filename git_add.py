import os

root = "."

for dirpath, dirnames, filenames in os.walk(root):
    # Ignore .git and hidden folders
    if ".git" in dirpath or dirpath.startswith("./."):
        continue
    
    # If folder has no files/subfiles (except hidden), add .gitkeep
    if not any(fname for fname in filenames if not fname.startswith(".")):
        gitkeep_path = os.path.join(dirpath, ".gitkeep")
        if not os.path.exists(gitkeep_path):
            with open(gitkeep_path, "w") as f:
                f.write("")  # create empty file
            print(f"Created: {gitkeep_path}")

