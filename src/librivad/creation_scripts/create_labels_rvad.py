# Description: Generates the binary VAD labels (.npy, one value per sample at 16 kHz,
# 1 = speech and 0 = non-speech) of a prepared dataset using rVADfast, for datasets that do
# not come with forced alignments. The labels are an estimate from an unsupervised VAD, not
# ground truth like the forced alignment labels of create_labels.py.
#
# Usage:
#   uv run src/librivad/creation_scripts/create_labels_rvad.py --dataset Skylar

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io.wavfile as wav
from dotenv import load_dotenv
from numpy.typing import NDArray
from rVADfast import rVADfast
from tqdm import tqdm

load_dotenv()

scripts_path: str = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(scripts_path)

vad: Any = None


def expand_frame_labels(
    frame_labels: NDArray[Any], n_samples: int, frame_len: int = 160
) -> NDArray[np.int16]:
    """
    Expands labels given every 10 ms (160 samples at 16 kHz) to one label per sample. The
    result has exactly `n_samples` values, since later steps index the signal with it.
    """
    per_sample: NDArray[np.int16] = np.repeat(frame_labels.astype(np.int16), frame_len)

    if len(per_sample) >= n_samples:
        return per_sample[:n_samples]

    return np.pad(per_sample, (0, n_samples - len(per_sample)))


def label_file(wav_path: str, label_path: str) -> bool:
    """
    Creates the label of one wav file. Returns False when no speech was found: no label is
    written, so create_LibriVAD.py skips the file instead of mixing an empty speech RMS.
    """
    global vad
    if vad is None:
        vad = rVADfast()

    sr: int
    signal: NDArray[np.int16]
    sr, signal = wav.read(wav_path)

    frame_labels: NDArray[Any]
    frame_labels, _ = vad(signal.astype(np.float32) / 32768.0, sr)

    label: NDArray[np.int16] = expand_frame_labels(frame_labels, len(signal), sr // 100)
    if not label.any():
        return False

    Path(label_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(label_path, label)

    return True


def label_path_for(wav_path: Path, datasets_dir: Path, labels_dir: Path) -> Path:
    """Mirrors the dataset tree under the labels directory, as create_labels.py does."""
    return (labels_dir / wav_path.relative_to(datasets_dir)).with_suffix(".npy")


if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Generate rVADfast VAD labels for a prepared dataset."
    )
    parser.add_argument("--dataset", required=True, help="Dataset folder name, e.g. Skylar.")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Parallel processes.")
    args: argparse.Namespace = parser.parse_args()

    project_root: Path = Path(os.environ["PROJECT_ROOT"])
    datasets_dir: Path = project_root / os.environ["DATASETS_DIR"]
    labels_dir: Path = project_root / os.environ["LABELS_DIR"]

    wav_paths: list[Path] = sorted((datasets_dir / args.dataset).rglob("*.wav"))
    if not wav_paths:
        sys.exit(f"No .wav files found in {datasets_dir / args.dataset}.")

    # Labels that already exist are kept, so the process can be resumed.
    pending: list[Path] = [
        p for p in wav_paths if not label_path_for(p, datasets_dir, labels_dir).is_file()
    ]
    print(f"{len(wav_paths)} files, {len(pending)} without labels.")

    no_speech: list[Path] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = executor.map(
            label_file,
            [str(p) for p in pending],
            [str(label_path_for(p, datasets_dir, labels_dir)) for p in pending],
            chunksize=16,
        )
        for path, labeled in tqdm(zip(pending, results, strict=True), total=len(pending)):
            if not labeled:
                no_speech.append(path)

    if no_speech:
        print(f"{len(no_speech)} files had no speech detected and got no label.")
        print(f"First one: {no_speech[0]}")

    print("\nLabel generation completed.")
