# Script used to create the LibriSpeechConcat dataset by pairing and merging
# audio files from the LibriSpeech dataset.

import glob
import os
import sys
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io.wavfile as wav
from dotenv import load_dotenv
from numpy.typing import NDArray
from tqdm import tqdm

from librivad import librispeech_functions as lf

load_dotenv()

# Add the parent 'Scripts' directory to the system path to import local modules.
scripts_path: str = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(scripts_path)


def generate_silence_signal(
    list_of_paths: Sequence[str | Path], alignments_dir: str | Path
) -> NDArray[Any]:
    """
    Generates a single, long silence signal by extracting and concatenating
    silent segments from multiple LibriSpeech audio files in parallel.

    Args:
        list_of_paths (list): Paths to audio files for silence extraction.
        alignments_dir (str): Directory containing forced alignment files.

    Returns:
        np.array: The resulting concatenated silence signal.
    """
    all_silence_segments: list[NDArray[Any]] = []

    # Use a thread pool to extract silence from files concurrently.
    with ThreadPoolExecutor() as executor:
        # Create a dictionary to map each future back to its file path for error reporting.
        future_to_path: dict[Future[NDArray[Any]], str | Path] = {
            executor.submit(lf.extract_silence, path, alignments_dir): path
            for path in list_of_paths
        }

        # Use tqdm for a progress bar that updates as tasks complete.
        for future in tqdm(
            as_completed(future_to_path), total=len(list_of_paths), desc="Extracting silence"
        ):
            try:
                silence_segment: NDArray[Any] = future.result()
                if silence_segment is not None and len(silence_segment) > 0:
                    all_silence_segments.append(silence_segment)
            except Exception as exc:
                path: str | Path = future_to_path[future]
                print(f"Error extracting silence from {path}: {exc}")

    # Combine all extracted segments into one continuous signal.
    if not all_silence_segments:
        print("Warning: No silence could be extracted. Returning empty signal.")
        return np.array([], dtype=np.int16)

    return np.concatenate(all_silence_segments)


def process_pair(task_info: tuple[str | Path, str | Path, int], full_silence: NDArray[Any]) -> None:
    """
    Processes a pair of audio files, concatenates them with a slice of silence,
    and saves the resulting audio file. This function is designed to be run
    in a parallel thread and is fully self-contained.

    Args:
        task_info (tuple): Contains (path1_str, path2_str, silence_start_index).
        full_silence (np.array): The complete silence signal used for slicing.
    """
    path1_str: str | Path
    path2_str: str | Path
    silence_start: int
    path1_str, path2_str, silence_start = task_info

    # Use pathlib for robust, cross-platform path handling.
    path1: Path = Path(path1_str)
    path2: Path = Path(path2_str)

    sr: int
    wav1: NDArray[Any]
    wav2: NDArray[Any]
    sr, wav1 = wav.read(path1)
    sr, wav2 = wav.read(path2)

    # Calculate the required length of silence (25% of the combined audio length).
    n_silence: int = len(full_silence)
    inbetween_silence_len: int = int((len(wav1) + len(wav2)) / 4)

    if inbetween_silence_len > n_silence:
        raise ValueError("Generated silence signal is too short for the required length.")

    # Slice the silence signal, wrapping around to the beginning if necessary.
    silence_end: int = silence_start + inbetween_silence_len
    inbetween_silence: NDArray[Any]
    if silence_end > n_silence:
        first_chunk: NDArray[Any] = full_silence[silence_start:]
        remaining_len: int = silence_end - n_silence
        second_chunk: NDArray[Any] = full_silence[:remaining_len]
        inbetween_silence = np.concatenate((first_chunk, second_chunk))
    else:
        inbetween_silence = full_silence[silence_start:silence_end]

    # Create the new audio signal by concatenating wav1 -> silence -> wav2.
    sample_concat: NDArray[Any] = np.concatenate((wav1, 0.001 * inbetween_silence, wav2))
    output: NDArray[np.int16] = lf.scale_signal(sample_concat)

    # --- Robustly construct the output filename and path ---
    # The new filename is a combination of the two source filenames.
    file_name: str
    if path1.parts[-3] != path2.parts[-3]:  # Different speakers
        file_name = f"{path1.stem}_+_{path2.name}"
    elif path1.parts[-2] != path2.parts[-2]:  # Different chapters, same speaker
        file_name = f"{path1.stem}_+_{'-'.join(path2.stem.split('-')[1:])}.wav"
    else:  # Same chapter and speaker
        file_name = f"{path1.stem}_+_{path2.stem.split('-')[-1]}.wav"

    # Create the new path by modifying the path of the second file.
    parts2: list[str] = list(path2.parts)
    try:
        # Find and replace 'LibriSpeech' with 'LibriSpeechConcat'.
        idx: int = parts2.index("LibriSpeech")
        parts2[idx] = "LibriSpeechConcat"
    except ValueError:
        print("Warning: 'LibriSpeech' not found in path. Output path may be incorrect.")

    parts2[-1] = file_name
    final_path: Path = Path(*parts2)  # Reassemble the path from its parts.

    # Create the directory and save the new .wav file.
    final_path.parent.mkdir(parents=True, exist_ok=True)
    wav.write(final_path, sr, output)


def create_LibriSpeechConcat(list_of_paths: list[str], silence_path: str | Path) -> None:
    """
    Creates the LibriSpeechConcat dataset for a given list of audio files.
    It pre-calculates all necessary information for each pair before starting
    the parallel processing, ensuring each task is independent.

    Args:
      - list_of_paths (list): A list of .wav file paths to be merged.
      - silence_path (Path): Path to the generated silence .wav file.
    """
    sr: int
    full_silence: NDArray[Any]
    sr, full_silence = wav.read(silence_path)
    n_silence: int = len(full_silence)

    # Ensure the list of files is even by removing the last file if necessary.
    if len(list_of_paths) % 2 != 0:
        list_of_paths.pop()

    # Group the file paths into pairs.
    pairs: list[tuple[str, str]] = [
        (list_of_paths[i], list_of_paths[i + 1]) for i in range(0, len(list_of_paths), 2)
    ]

    # Pre-calculate the silence start index for each pair sequentially.
    # This makes each task independent and safe for parallel processing.
    tasks: list[tuple[str, str, int]] = []
    current_silence_start: int = 0
    for path1, path2 in pairs:
        # Read audio files to determine their length for the calculation.
        wav1: NDArray[Any]
        wav2: NDArray[Any]
        sr, wav1 = wav.read(path1)
        sr, wav2 = wav.read(path2)
        inbetween_silence_len: int = int((len(wav1) + len(wav2)) / 4)

        # Add the task info (paths and calculated start index) to the list.
        tasks.append((path1, path2, current_silence_start))

        # Update the next start index, wrapping around the silence buffer.
        current_silence_start = (current_silence_start + inbetween_silence_len) % n_silence

    # Process all the prepared tasks in parallel.
    with ThreadPoolExecutor() as executor:
        future_to_task: dict[Future[None], tuple[str, str, int]] = {
            executor.submit(process_pair, task, full_silence): task for task in tasks
        }

        for future in tqdm(
            as_completed(future_to_task), total=len(tasks), desc="Concatenating files"
        ):
            try:
                future.result()
            except Exception as exc:
                task: tuple[str, str, int] = future_to_task[future]
                path1_name: str = Path(task[0]).name
                path2_name: str = Path(task[1]).name
                print(f"Error processing pair ({path1_name}, {path2_name}): {exc}")


if __name__ == "__main__":
    project_root: Path = Path(os.environ["PROJECT_ROOT"])

    # Build all necessary paths from the project root directory.
    wav_dir: Path = project_root / os.environ["LIBRISPEECH_WAV_DIR"]
    silence_path: Path = wav_dir / "train100_silence.wav"
    alignments_dir: Path = project_root / os.environ["LIBRISPEECH_ALIGNMENTS_DIR"]
    unaligned_path: Path = alignments_dir / "unaligned.txt"

    print("\nSearching LibriSpeech files...")
    # Find all .wav files in the specified subdirectories.
    paths_train: list[str] = sorted(
        glob.glob(str(wav_dir / "train-clean-100" / "**" / "*.wav"), recursive=True)
    )
    paths_val: list[str] = sorted(
        glob.glob(str(wav_dir / "dev-clean" / "**" / "*.wav"), recursive=True)
    )
    paths_test: list[str] = sorted(
        glob.glob(str(wav_dir / "test-clean" / "**" / "*.wav"), recursive=True)
    )

    # Remove paths that do not have corresponding alignment files.
    paths_train = lf.remove_unaligned_paths(paths_train, unaligned_path)

    print("\nGenerating silence signal...")
    silence_signal: NDArray[Any] = generate_silence_signal(
        paths_train, alignments_dir / "train-clean-100"
    )
    wav.write(silence_path, 16000, silence_signal)

    # Process each data split sequentially.
    print("Preparing LibriSpeechConcat train-clean-100. This might take a few minutes...")
    create_LibriSpeechConcat(paths_train, silence_path)

    print("Preparing LibriSpeechConcat dev-clean...")
    create_LibriSpeechConcat(paths_val, silence_path)

    print("Preparing LibriSpeechConcat test-clean...")
    create_LibriSpeechConcat(paths_test, silence_path)

    print("\nLibriSpeechConcat dataset creation completed.")
