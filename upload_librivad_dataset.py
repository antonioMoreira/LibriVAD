import os
import shutil
import tempfile
from huggingface_hub import HfApi, login

login()

print("Login successful!")

# Define the base directory of the repository
base_dir = os.getcwd()

# Define the folders to exclude (absolute paths)
exclude_paths = [
    os.path.join(base_dir, "Results"),
    os.path.join(base_dir, "Results_large"),
    os.path.join(base_dir, "Files"),#, "Labels"),
    #os.path.join(base_dir, "Files", "Datasets"),
    os.path.join(base_dir, ".venv"),
    os.path.join(base_dir, "Scripts", "__pycache__"),
]

# Create a temporary directory
with tempfile.TemporaryDirectory() as tmpdir:
    print(f"Created temporary directory: {tmpdir}")

    # Walk through the base directory and copy non-excluded files to the temporary directory
    for root, dirs, files in os.walk(base_dir):
        # Determine if the current root or any of its parents should be excluded
        should_exclude_root = False
        for exclude_path in exclude_paths:
            if root.startswith(exclude_path):
                should_exclude_root = True
                break
        if should_exclude_root:
            continue

        # Create corresponding directory in the temporary structure
        relative_root = os.path.relpath(root, base_dir)
        temp_root = os.path.join(tmpdir, relative_root)
        os.makedirs(temp_root, exist_ok=True)

        for file in files:
            src_file_path = os.path.join(root, file)
            dest_file_path = os.path.join(temp_root, file)

            # Check if the file itself is in an excluded path
            should_exclude_file = False
            for exclude_path in exclude_paths:
                if src_file_path.startswith(exclude_path):
                    should_exclude_file = True
                    break
            if should_exclude_file:
                continue

            shutil.copy2(src_file_path, dest_file_path)
            print(f"Copied: {src_file_path} to {dest_file_path}")

    # Upload the temporary directory to Hugging Face Hub
    repo_id = "LibriVAD/LibriVAD" # Organization/DatasetName
    api = HfApi()

    print(f"Uploading folder '{tmpdir}' to Hugging Face Hub as dataset '{repo_id}'...")
    api.upload_folder(
        folder_path=tmpdir,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Upload LibriVAD repository excluding specified folders"
    )
    print("Upload complete!")

print("Temporary directory cleaned up.")
