# Validation script for the LibriVAD data generation pipeline.

# This script performs a visual check to confirm that the LibriSpeechConcat
# dataset and its corresponding VAD labels have been generated correctly.


# Usage:
#    python validate.py

import os
import random
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wav
from dotenv import load_dotenv
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from numpy.typing import NDArray

load_dotenv()


def reconstruct_original_paths(concat_wav_path: Path) -> tuple[Path, Path] | tuple[None, None]:
    """
    Reconstructs the paths of the two original source audio files from a
    LibriSpeechConcat filename.

    This function is the inverse of the naming logic used in the
    create_LibriSpeechConcat.py and create_labels.py scripts.

    Args:
        concat_wav_path (Path): Path to the LibriSpeechConcat .wav file.

    Returns:
        tuple[Path, Path] | tuple[None, None]: A tuple containing the paths to the
        two original files, or (None, None) if reconstruction fails.
    """
    try:
        # The filename contains parts of the original filenames separated by '_+_'.
        # e.g., "PART1_+_PART2.wav"
        concat_parts: list[str] = concat_wav_path.stem.split("_+_")
        part1_stem: str = concat_parts[0]
        part2_info: str = concat_parts[1]

        # The first source file is always in the same directory as the concat file,
        # but in the 'LibriSpeech' dataset folder instead of 'LibriSpeechConcat'.
        # We find this by replacing the dataset name in the path parts.
        parts: list[str] = list(concat_wav_path.parts)
        idx: int = parts.index("LibriSpeechConcat")
        parts[idx] = "LibriSpeech"

        # Reconstruct the base path to the original 'LibriSpeech' split directory.
        original_base_path: Path = Path(*parts).parent
        file1: Path = original_base_path / f"{part1_stem}.wav"

        # The logic for the second file's path depends on its filename structure.
        part2_sub_parts: list[str] = part2_info.split("-")
        current_speaker: str = concat_wav_path.parts[-3]
        file2: Path

        if len(part2_sub_parts) == 1:
            # Case 1: Same speaker, same chapter. e.g., ..._+_0004.wav
            current_chapter: str = concat_wav_path.parts[-2]
            file2_stem: str = f"{current_speaker}-{current_chapter}-{part2_info}"
            file2 = original_base_path / f"{file2_stem}.wav"
        elif len(part2_sub_parts) == 2:
            # Case 2: Same speaker, DIFFERENT chapter. e.g., ..._+_64039-0004.wav
            new_chapter: str = part2_sub_parts[0]
            speaker_dir: Path = original_base_path.parent
            file2_stem = f"{current_speaker}-{part2_info}"
            file2 = speaker_dir / new_chapter / f"{file2_stem}.wav"
        else:
            # Case 3: Different speaker entirely. e.g., ..._+_23-2938-0001.wav
            new_speaker: str = part2_sub_parts[0]
            new_chapter = part2_sub_parts[1]
            split_dir: Path = original_base_path.parent.parent
            file2 = split_dir / new_speaker / new_chapter / f"{part2_info}.wav"

        return file1, file2

    except (IndexError, ValueError) as e:
        print(f"Error reconstructing paths from {concat_wav_path.name}: {e}")
        return None, None


def plot_validation(
    sr: int,
    concat_wav: NDArray[Any],
    concat_label: NDArray[Any],
    wav1: NDArray[Any],
    label1: NDArray[Any],
    wav2: NDArray[Any],
    label2: NDArray[Any],
    filename: str,
) -> None:
    """
    Creates a detailed plot for visually validating the data generation.
    """
    # Create the figure and a grid for complex subplot layout.
    fig: Figure = plt.figure(figsize=(15, 8))
    gs: GridSpec = fig.add_gridspec(2, 2)
    fig.suptitle(f"Validation for: {filename}", fontsize=16)

    # --- Top Plot: Concatenated Signal ---
    ax1: Axes = fig.add_subplot(gs[0, :])  # Top plot spans both columns
    time_concat: NDArray[np.floating[Any]] = np.arange(len(concat_wav)) / sr
    ax1.plot(
        time_concat, concat_wav, label="Concatenated Waveform", color="royalblue", linewidth=0.7
    )
    # Shade the background where VAD label is 1 (speech).
    ax1.fill_between(
        time_concat,
        ax1.get_ylim()[0],
        ax1.get_ylim()[1],
        where=concat_label == 1,
        color="green",
        alpha=0.2,
        label="VAD Label (Speech)",
    )
    ax1.set_title("Full Concatenated Signal and VAD Label")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle="--", alpha=0.6)

    # --- Bottom-Left Plot: First Original Signal ---
    ax2: Axes = fig.add_subplot(gs[1, 0])
    time1: NDArray[np.floating[Any]] = np.arange(len(wav1)) / sr
    ax2.plot(time1, wav1, label="Source Waveform 1", color="darkorange", linewidth=0.7)
    ax2.fill_between(
        time1,
        ax2.get_ylim()[0],
        ax2.get_ylim()[1],
        where=label1 == 1,
        color="green",
        alpha=0.2,
        label="VAD Label (Speech)",
    )
    ax2.set_title("Original Signal 1")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Amplitude")
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle="--", alpha=0.6)

    # --- Bottom-Right Plot: Second Original Signal ---
    ax3: Axes = fig.add_subplot(gs[1, 1])
    time2: NDArray[np.floating[Any]] = np.arange(len(wav2)) / sr
    ax3.plot(time2, wav2, label="Source Waveform 2", color="darkred", linewidth=0.7)
    ax3.fill_between(
        time2,
        ax3.get_ylim()[0],
        ax3.get_ylim()[1],
        where=label2 == 1,
        color="green",
        alpha=0.2,
        label="VAD Label (Speech)",
    )
    ax3.set_title("Original Signal 2")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Amplitude")
    ax3.legend(loc="upper right")
    ax3.grid(True, linestyle="--", alpha=0.6)

    # Ensure y-axis limits are consistent for easy visual comparison.
    max_amp: float = (
        float(max(np.abs(concat_wav).max(), np.abs(wav1).max(), np.abs(wav2).max())) * 1.1
    )
    ax1.set_ylim(-max_amp, max_amp)
    ax2.set_ylim(-max_amp, max_amp)
    ax3.set_ylim(-max_amp, max_amp)

    plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))  # Adjust layout to make room for suptitle
    plt.show()


def main() -> None:
    """Main function to drive the validation process."""
    # Define root paths for the project, datasets, and labels.
    project_root: Path = Path(os.getenv("PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))
    datasets_root: Path = project_root / os.getenv("DATASETS_DIR", "files/Datasets")
    labels_root_dir: Path = project_root / os.getenv("LABELS_DIR", "files/Labels")
    concat_wav_dir: Path = project_root / os.getenv(
        "LIBRISPEECH_CONCAT_DIR", "files/Datasets/LibriSpeechConcat"
    )

    # 1. Find a random LibriSpeechConcat file.
    print(f"Searching for .wav files in: {concat_wav_dir}")
    all_concat_files: list[Path] = list(concat_wav_dir.rglob("*.wav"))

    if not all_concat_files:
        print("\nFATAL ERROR: No .wav files found in the LibriSpeechConcat directory.")
        print("Please ensure the data generation scripts have been run successfully.")
        sys.exit(1)

    random_concat_file: Path = random.choice(all_concat_files)
    print(f"\nSelected random file for validation:\n  -> {random_concat_file}")

    # 2. Reconstruct paths to original files.
    original_file1: Path | None
    original_file2: Path | None
    original_file1, original_file2 = reconstruct_original_paths(random_concat_file)
    if not original_file1 or not original_file2:
        print("Could not reconstruct original file paths. Exiting.")
        sys.exit(1)

    print("Reconstructed original source files:")
    print(f"  -> {original_file1}")
    print(f"  -> {original_file2}")

    try:
        # 3. Load all necessary waveforms and labels.
        print("\nLoading audio waveforms and VAD labels...")

        # Load concatenated data
        sr: int
        concat_wav: NDArray[Any]
        sr, concat_wav = wav.read(random_concat_file)
        concat_label_path: Path = labels_root_dir / random_concat_file.relative_to(datasets_root)
        concat_label: NDArray[Any] = np.load(concat_label_path.with_suffix(".npy"))

        # Load data for original file 1
        wav1: NDArray[Any]
        _, wav1 = wav.read(original_file1)
        label1_path: Path = labels_root_dir / original_file1.relative_to(datasets_root)
        label1: NDArray[Any] = np.load(label1_path.with_suffix(".npy"))

        # Load data for original file 2
        wav2: NDArray[Any]
        _, wav2 = wav.read(original_file2)
        label2_path: Path = labels_root_dir / original_file2.relative_to(datasets_root)
        label2: NDArray[Any] = np.load(label2_path.with_suffix(".npy"))

        def sync_len(label: NDArray[Any], waveform: NDArray[Any]) -> NDArray[Any]:
            if len(label) < len(waveform):
                # Pad the label with 0s (silence) at the end.
                padding: NDArray[Any] = np.zeros(len(waveform) - len(label), dtype=label.dtype)
                return np.concatenate((label, padding))
            elif len(label) > len(waveform):
                # Truncate the label.
                return label[: len(waveform)]
            return label

        concat_label = sync_len(concat_label, concat_wav)
        label1 = sync_len(label1, wav1)
        label2 = sync_len(label2, wav2)
        # -----------------------------------------------------------------

        # 4. Plot the results for visual inspection.
        print("Data loaded successfully. Generating plot...")
        plot_validation(
            sr, concat_wav, concat_label, wav1, label1, wav2, label2, random_concat_file.name
        )

    except FileNotFoundError as e:
        print("\nFATAL ERROR: A required file was not found.")
        print(f"Could not find: {e.filename}")
        print(
            "This likely means the label generation failed for this specific file "
            "or the path is incorrect."
        )
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
