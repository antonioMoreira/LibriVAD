# Authors: Ioannis Stylianou
# Date: September 2025

# Description: The main script for generating the LibriVAD dataset.

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

scripts_path: str = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(scripts_path)

PROJECT_ROOT: Path = Path(os.environ["PROJECT_ROOT"])
DEFAULT_DATASETS_DIR: Path = PROJECT_ROOT / os.environ["DATASETS_DIR"]
LIBRISPEECH_ALIGNMENTS_DIR: Path = PROJECT_ROOT / os.environ["LIBRISPEECH_ALIGNMENTS_DIR"]
NOISES_DIR: Path = PROJECT_ROOT / os.environ["NOISES_DIR"]
DEFAULT_RESULTS_DIR: Path = PROJECT_ROOT / os.environ["RESULTS_DIR"]


def load_files(dataset: str, dataset_dir: str | Path, split: str) -> list[str]:
    """
    Function for loading the file paths of the corresponding LibriSpeech split

    Inputs:
    - dataset     : Indicates which dataset to use between LibriSpeech and
                    LibriSpeechConcat (str)
    - dataset_dir : The directory of the dataset (str)
    - split       : Indicates which split to generate between "train", "val"
                    and "test" (str)
    """

    # Locating the paths of the utterances for each split
    search_path: str
    file_paths: list[str]
    if split == "train":
        search_path = os.path.join(Path(dataset_dir), dataset, "train-clean-100", "*", "*", "*.wav")
        file_paths = glob.glob(search_path)

    elif split == "val":
        search_path = os.path.join(Path(dataset_dir), dataset, "dev-clean", "*", "*", "*.wav")
        file_paths = glob.glob(search_path)

    elif split == "test":
        search_path = os.path.join(Path(dataset_dir), dataset, "test-clean", "*", "*", "*.wav")
        file_paths = glob.glob(search_path)
    else:
        raise ValueError(
            'Split parameter should be a choice between "train", \
  "val" and "test"'
        )

    # Removing (the few) utterances that are not paired with forced alignments
    if dataset == "LibriSpeech":
        librispeech_unaligned: Path = LIBRISPEECH_ALIGNMENTS_DIR / "unaligned.txt"

        file_paths = lf.remove_unaligned_paths(file_paths, librispeech_unaligned)

    file_paths.sort()

    return file_paths


def extract_info(file: str) -> tuple[NDArray[Any], np.floating[Any], int, str]:
    """
    Function for computing the speech Root Mean Square, the waveform,
    and the length of a signal. These are required for accelerating the
    dataset generation process
    """

    _sr: int
    signal: NDArray[Any]
    _sr, signal = wav.read(file)

    # The RMS of the signal is computed only for the speech part of the
    # utterance
    omitted_silence_signal: NDArray[Any] = lf.remove_silence(signal, file)
    rms: np.floating[Any] = lf.RMS(omitted_silence_signal)

    return signal, rms, len(signal), file


def load_noise(noise: str, split: str) -> tuple[NDArray[Any], int]:
    """
    Function for loading the noisy signals. Returns the waveform and the length
    of the noise
    """

    noise_path: Path = NOISES_DIR / noise / f"{noise}_{split}.wav"

    _sr: int
    noise_sgn: NDArray[Any]
    _sr, noise_sgn = wav.read(noise_path)

    return noise_sgn, len(noise_sgn)


def create_LibriVAD(
    dataset: str,
    dataset_dir: str | Path,
    SNRs: Sequence[int],
    Noises: Sequence[str],
    split: str,
    size: str,
    batch_size: int = 500,
    save_dir: str | Path | None = None,
) -> None:
    """
    Creates the LibriVAD dataset by mixing clean audio with various noises at
    specified SNRs. Samples individual files for different
    dataset sizes ('small', 'medium', 'large').

    Args:
      - dataset (str): 'LibriSpeech' or 'LibriSpeechConcat'.
      - dataset_dir (str): The root directory of the clean datasets.
      - SNRs (list): A list of Signal-to-Noise Ratios to generate.
      - Noises (list): A list of noises to mix in.
      - split (str): The data split to generate ('train', 'val', or 'test').
      - size (str): The target dataset size. 'small' (saves every 100th file),
                    'medium' (every 10th), or 'large' (every file).
      - batch_size (int): The number of files to process in memory at once.
      - save_dir (str, optional): The root directory to save the generated files.
                                  Defaults to "LibriVAD/Results" if None.
    """
    print(f"\nPreparing {split} files for {dataset} dataset (size: {size})...")

    # Setting up the output directory
    output_root: Path = DEFAULT_RESULTS_DIR if save_dir is None else Path(save_dir)

    # Determine the sampling step based on size
    step: int
    if size == "large":
        step = 1
    elif size == "medium":
        step = 10
    elif size == "small":
        step = 100
    else:
        raise ValueError('Size must be one of "small", "medium", or "large".')

    # Loading file paths
    file_paths_all: list[str] = load_files(dataset, dataset_dir, split)
    file_paths: list[str] = file_paths_all[::step]
    print(f"Total files to process after applying size filter: {len(file_paths)}")
    n_total_files: int = len(file_paths)
    sr: int = 16000

    # The main loop iterates through the data in batches for memory efficiency.
    for i_batch in range(0, n_total_files, batch_size):
        batch_paths: list[str] = file_paths[i_batch : i_batch + batch_size]
        n_batch: int = len(batch_paths)

        print(
            f"\nProcessing batch {i_batch // batch_size + 1}/"
            f"{(n_total_files - 1) // batch_size + 1}..."
        )

        # Pre-fetch info for the current batch in parallel.
        Signals: list[NDArray[Any] | None] = [None] * n_batch
        RMSs: list[np.floating[Any] | None] = [None] * n_batch
        Lens: list[int | None] = [None] * n_batch
        Files: list[str | None] = [None] * n_batch
        file_index_map: dict[str, int] = {path: i for i, path in enumerate(batch_paths)}

        with ThreadPoolExecutor() as executor:
            futures: dict[Future[tuple[NDArray[Any], np.floating[Any], int, str]], str] = {
                executor.submit(extract_info, path): path for path in batch_paths
            }

            for future in tqdm(as_completed(futures), total=n_batch, desc="Loading batch data"):
                try:
                    signal: NDArray[Any]
                    rms: np.floating[Any]
                    sgn_len: int
                    file: str
                    signal, rms, sgn_len, file = future.result()
                    index: int = file_index_map[file]
                    Signals[index], RMSs[index], Lens[index], Files[index] = (
                        signal,
                        rms,
                        sgn_len,
                        file,
                    )
                except Exception as exc:
                    print(f"Error loading info for file: {futures[future]}, Error: {exc}")

        # Iterate through each noise and SNR combination.
        for noise in Noises:
            noise_sgn: NDArray[Any]
            noise_dur: int
            noise_sgn, noise_dur = load_noise(noise, split)
            noise_sgn_temp: NDArray[Any] = np.tile(noise_sgn, 2)  # For seamless recycling

            for snr in SNRs:
                print(f"\nMixing with {noise} at {snr} SNR...")
                noise_idx: int = 0

                for i in tqdm(range(n_batch), desc="Mixing audio"):
                    signal_length: int | None = Lens[i]
                    rms_val: np.floating[Any] | None = RMSs[i]
                    sig_val: NDArray[Any] | None = Signals[i]
                    file_val: str | None = Files[i]
                    if (
                        signal_length is None
                        or rms_val is None
                        or sig_val is None
                        or file_val is None
                    ):
                        continue

                    noise_subset: NDArray[Any]
                    if noise_idx + signal_length > noise_dur:
                        noise_subset = noise_sgn_temp[noise_idx : noise_idx + signal_length]
                        noise_idx = (noise_idx + signal_length) % noise_dur
                    else:
                        noise_subset = noise_sgn[noise_idx : noise_idx + signal_length]
                        noise_idx += signal_length

                    scale: np.floating[Any] = rms_val / (lf.RMS(noise_subset) * (10 ** (snr / 20)))
                    corrupted_signal: NDArray[Any] = sig_val + scale * noise_subset
                    output: NDArray[np.int16] = lf.scale_signal(corrupted_signal)

                    file_path: Path = Path(file_val)

                    # Extract the split folder name (train, val, test)
                    split_folder: str = file_path.parts[-4]

                    save_dir_path: Path = output_root / dataset / split_folder / noise / str(snr)
                    save_dir_path.mkdir(parents=True, exist_ok=True)

                    save_path: Path = save_dir_path / file_path.name
                    wav.write(save_path, sr, output)

    print("\nData generation completed!")


if __name__ == "__main__":
    import argparse

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Generate the LibriVAD dataset."
    )
    parser.add_argument(
        "size", choices=["small", "medium", "large"], help="Size of the dataset to generate."
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=str(DEFAULT_DATASETS_DIR),
        help="Root directory of datasets.",
    )
    parser.add_argument(
        "--snrs", nargs="+", type=int, default=[-5, 0, 5, 10, 15, 20], help="List of SNRs to use."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2900,
        help="Number of audio files to load into memory at once.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Directory to save the generated dataset. If None, saves to LibriVAD/Results.",
    )

    # To maintain the structure of the dataset these should be fixed;
    # one can experiment with them however
    Datasets: list[str] = ["LibriSpeech", "LibriSpeechConcat"]
    Noises: list[str] = [
        "Babble_noise",
        "SSN_noise",
        "Domestic_noise",
        "Nature_noise",
        "Office_noise",
        "Public_noise",
        "Street_noise",
        "Transport_noise",
        "City_noise",
    ]
    Splits: list[str] = ["train", "val", "test"]

    args: argparse.Namespace = parser.parse_args()

    for dataset in Datasets:
        print(f"\n--- Generating data based on the {dataset} dataset ---")
        for split in Splits:
            create_LibriVAD(
                dataset,
                args.dataset_dir,
                args.snrs,
                Noises,
                split,
                args.size,
                args.batch_size,
                args.save_dir,
            )
