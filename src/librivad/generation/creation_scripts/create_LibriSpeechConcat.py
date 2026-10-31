# Script used to create the LibriSpeechConcat dataset by pairing and merging
# audio files from the LibriSpeech dataset.
#
# The pairing, gap length, gain and normalization now live in the `middle`
# scenario of configs/scenarios/librispeech_middle.yaml, rendered by the
# scenario builder (librivad.generation.build_scenarios). Its dev-clean and
# test-clean output is bit-identical to the files this script used to write
# (tests/test_generation_parity.py). This script only builds the silence signal
# the scenario cuts its gaps from, then runs the builder.

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

from librivad.generation import build_scenarios
from librivad.generation import librispeech_functions as lf

load_dotenv()


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


if __name__ == "__main__":
    project_root: Path = Path(os.environ["PROJECT_ROOT"])

    # Build all necessary paths from the project root directory.
    wav_dir: Path = project_root / os.environ["LIBRISPEECH_WAV_DIR"]
    silence_path: Path = wav_dir / "train100_silence.wav"
    alignments_dir: Path = project_root / os.environ["LIBRISPEECH_ALIGNMENTS_DIR"]
    unaligned_path: Path = alignments_dir / "unaligned.txt"
    config_path: Path = project_root / "configs" / "scenarios" / "librispeech_middle.yaml"

    if not silence_path.is_file():
        print("\nGenerating silence signal...")
        paths_train: list[str] = sorted(
            glob.glob(str(wav_dir / "train-clean-100" / "**" / "*.wav"), recursive=True)
        )
        # Remove paths that do not have corresponding alignment files.
        paths_train = lf.remove_unaligned_paths(paths_train, unaligned_path)
        silence_signal: NDArray[Any] = generate_silence_signal(
            paths_train, alignments_dir / "train-clean-100"
        )
        wav.write(silence_path, 16000, silence_signal)

    print("Preparing LibriSpeechConcat. This might take a few minutes...")
    sys.exit(build_scenarios.main(["--config", str(config_path)]))
