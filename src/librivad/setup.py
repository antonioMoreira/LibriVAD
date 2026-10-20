# Author: Ioannis Stylianou
# Date: September 2025
# Setup script for downloading and preparing the LibriSpeech dataset,
# the LibriSpeechConcat dataset and their corresponding labels.

import argparse
import os
import shutil
import stat
import subprocess
import sys
import tarfile
import zipfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

# --- Hugging Face URLs for additional data ---
FORCED_ALIGNMENTS_URL: str = os.getenv(
    "FORCED_ALIGNMENTS_URL",
    "https://huggingface.co/datasets/LibriVAD/LibriVAD/resolve/main/Files/Forced_alignments.zip",
)
NOISES_URL: str = os.getenv(
    "NOISES_URL",
    "https://huggingface.co/datasets/LibriVAD/LibriVAD/resolve/main/Files/Noises.zip",
)


# --- Configuration ---
# Dictionary mapping the region argument to its corresponding base URL.
URL_MIRRORS: dict[str, str] = {
    "US": os.getenv("URL_MIRROR_US", "https://us.openslr.org/resources/12"),
    "EU": os.getenv("URL_MIRROR_EU", "https://openslr.elda.org/resources/12"),
    "CN": os.getenv("URL_MIRROR_CN", "https://openslr.magicdatatech.com/resources/12"),
}

# List of files to download and extract.
FILES_TO_DOWNLOAD: list[str] = [
    item.strip()
    for item in os.getenv(
        "FILES_TO_DOWNLOAD", "train-clean-100.tar.gz,dev-clean.tar.gz,test-clean.tar.gz"
    ).split(",")
    if item.strip()
]


# --- Helper Function for Downloading ---
def download_file(url: str, destination: str | Path) -> None:
    """Downloads a file from a URL to a destination with a progress bar."""
    print(f"Downloading {os.path.basename(destination)}...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
            total_size: int = int(r.headers.get("content-length", 0))

            with (
                open(destination, "wb") as f,
                tqdm(
                    total=total_size,
                    unit="iB",
                    unit_scale=True,
                    desc=os.path.basename(destination),
                ) as pbar,
            ):
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        # Clean up partially downloaded file
        if os.path.exists(destination):
            os.remove(destination)
        sys.exit(1)


def download_and_extract_zip(url: str, destination_dir: str | Path) -> None:
    """Downloads a zip file if not already present and extracts its contents
    to the specified directory if empty.
    """
    zip_filename: str = os.path.basename(url)
    temp_zip_path: Path = (
        Path(destination_dir).parent / zip_filename
    )  # Download to parent dir temporarily

    # Check if destination directory is already populated
    if Path(destination_dir).exists() and any(Path(destination_dir).iterdir()):
        print(
            f"Skipping download and extraction for {zip_filename}: "
            f"{destination_dir} is already populated."
        )
        return

    # If destination is empty or doesn't exist, proceed.
    Path(destination_dir).mkdir(parents=True, exist_ok=True)

    # Check if the zip file already exists locally
    if not temp_zip_path.exists():
        print(f"Downloading {zip_filename}...")
        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                total_size: int = int(r.headers.get("content-length", 0))
                with (
                    open(temp_zip_path, "wb") as f,
                    tqdm(total=total_size, unit="iB", unit_scale=True, desc=zip_filename) as pbar,
                ):
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {url}: {e}")
            sys.exit(1)

    print(f"Extracting {zip_filename} to {destination_dir}...")
    try:
        with zipfile.ZipFile(temp_zip_path, "r") as zip_ref:
            zip_ref.extractall(destination_dir)
        print(f"Successfully extracted {zip_filename} to {destination_dir}.")
    except zipfile.BadZipFile:
        print(
            f"Error: Downloaded file {zip_filename} is not a valid zip file. "
            f"Please delete {temp_zip_path} and try again."
        )
        sys.exit(1)
    finally:
        # Clean up the downloaded zip file after successful extraction
        if temp_zip_path.exists():
            os.remove(temp_zip_path)


def run_command(command: Sequence[str]) -> None:
    """Runs a command and exits if it fails."""
    print(f"\n> Running script: {Path(command[-1]).stem}.py")
    try:
        # Using sys.executable ensures we use the python from the correct venv
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {' '.join(command)}")
        print(f"Return code: {e.returncode}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: Command '{command[0]}' not found. Is Python in your PATH?")
        sys.exit(1)


def onerror(
    func: Callable[..., Any],
    path: str,
    exc_info: BaseException,
) -> None:
    """
    Source: https://stackoverflow.com/questions/2656322/shutil-rmtree-fails-on-windows-with-access-is-denied
    Error handler for ``shutil.rmtree``.

    If the error is due to an access error (read only file)
    it attempts to add write permission and then retries.

    If the error is for another reason it re-raises the error.

    Usage : ``shutil.rmtree(path, onerror=onerror)``
    """
    # Is the error an access error?
    if not os.access(path, os.W_OK):
        os.chmod(path, stat.S_IWUSR)
        func(path)
    else:
        raise


def main() -> None:
    # --- 1. Argument Parsing ---
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Download and set up LibriSpeech dataset."
    )
    parser.add_argument(
        "region",
        choices=URL_MIRRORS.keys(),
        help="The download region (mirror) to use.",
    )
    args: argparse.Namespace = parser.parse_args()

    # --- 2. Set up Paths and URLs ---
    base_url: str = URL_MIRRORS[args.region]

    # Use pathlib for cross-platform path handling
    base_dir: Path = Path(os.getenv("PROJECT_ROOT", str(Path(__file__).parent.resolve())))
    flac_dir: Path = base_dir / os.getenv("FLAC_DIR", "Files/Datasets/Flac")
    scripts_dir: Path = base_dir / os.getenv("SCRIPTS_DIR", "src/librivad/creation_scripts")

    # --- Download and extract additional data (Forced Alignments and Noises) ---
    forced_alignments_dir: Path = base_dir / os.getenv(
        "FORCED_ALIGNMENTS_DIR", "Files/Forced_alignments"
    )
    noises_dir: Path = base_dir / os.getenv("NOISES_DIR", "Files/Noises")

    download_and_extract_zip(FORCED_ALIGNMENTS_URL, forced_alignments_dir)
    download_and_extract_zip(NOISES_URL, noises_dir)

    # Create the target directory if it doesn't exist
    flac_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using base URL: {base_url}")
    print(f"Downloading files to: {flac_dir}")

    # --- 3. Download, Extract, and Clean Up Archives ---
    for filename in FILES_TO_DOWNLOAD:
        file_url: str = f"{base_url}/{filename}"
        archive_path: Path = flac_dir / filename

        # Download the file
        download_file(file_url, archive_path)

        # Extract the archive
        print(f"Extracting {filename}...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=flac_dir)

        # Remove the archive file after extraction
        print(f"Removing {filename}...")
        os.remove(archive_path)

    # --- 4. Run the Python Preprocessing Scripts ---
    run_command([sys.executable, str(scripts_dir / "create_LibriSpeech_wav.py")])

    # Clean up the Flac directory which is no longer needed
    print(f"Removing temporary directory: {flac_dir}")
    shutil.rmtree(flac_dir, onexc=onerror)

    run_command([sys.executable, str(scripts_dir / "create_LibriSpeechConcat.py")])
    run_command([sys.executable, str(scripts_dir / "create_labels.py")])

    print("\nSetup completed successfully!")


if __name__ == "__main__":
    main()
