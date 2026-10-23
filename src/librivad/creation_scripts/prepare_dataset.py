# Description: Downloads a dataset described by a DatasetAdapter, converts it to 16 kHz mono
# wav and writes it in the layout expected by the LibriVAD generation scripts:
#   <DATASETS_DIR>/<name>/<split dir>/<speaker>/<chapter>/<id>.wav
#
# Usage:
#   uv run src/librivad/creation_scripts/prepare_dataset.py skylar

import argparse
import io
import os
import sys
import time
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import scipy.io.wavfile as wav
import soundfile as sf
from dotenv import load_dotenv
from numpy.typing import NDArray
from tqdm import tqdm

from librivad.dataset_adapter import ADAPTERS, DatasetAdapter, Sample

load_dotenv()

scripts_path: str = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(scripts_path)

RETRIES: int = 6
MAX_WAIT_SECONDS: float = 120.0


def retry_wait(exc: Exception, attempt: int) -> float:
    """
    Seconds to wait before the next attempt. Honours the Retry-After header of the server
    (sent with HTTP 429, Too Many Requests) and otherwise backs off exponentially.
    """
    response: Any = getattr(exc, "response", None)
    retry_after: str | None = response.headers.get("Retry-After") if response is not None else None
    if retry_after is not None and retry_after.isdigit():
        return min(float(retry_after), MAX_WAIT_SECONDS)

    return min(2.0 ** (attempt + 1), MAX_WAIT_SECONDS)


def convert_audio(raw: bytes, target_sr: int = 16000) -> NDArray[np.int16]:
    """
    Decodes an audio file, mixes it down to mono, resamples it to `target_sr` (only
    if needed) and returns it as 16-bit PCM samples.
    """
    signal: NDArray[Any]
    sr: int
    signal, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    mono: NDArray[np.float32] = signal.mean(axis=1)

    if sr != target_sr:
        mono = librosa.resample(mono, orig_sr=sr, target_sr=target_sr)

    return np.asarray(np.clip(mono, -1.0, 1.0) * 32767, dtype=np.int16)


def prepare_sample(
    adapter: DatasetAdapter, sample: Sample, split: str, datasets_dir: str | Path
) -> bool:
    """
    Fetches, converts and saves a single sample. Returns False if it already existed.
    """
    out_path: Path = adapter.audio_path(datasets_dir, split, sample)
    if out_path.is_file():
        return False

    raw: bytes | None = None
    for attempt in range(RETRIES):
        try:
            raw = adapter.fetch(sample)
            break
        except Exception as exc:
            if attempt == RETRIES - 1:
                raise
            time.sleep(retry_wait(exc, attempt))

    assert raw is not None
    audio: NDArray[np.int16] = convert_audio(raw, adapter.sample_rate)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Written through a temporary name so an interrupted run never leaves a partial wav.
    tmp_path: Path = out_path.with_suffix(".part")
    with open(tmp_path, "wb") as f:
        wav.write(f, adapter.sample_rate, audio)
    tmp_path.replace(out_path)

    return True


def prepare_dataset(
    adapter: DatasetAdapter,
    datasets_dir: str | Path,
    workers: int = 4,
    limit: int | None = None,
) -> None:
    """
    Prepares every sample of the adapter's dataset. Already prepared files are skipped, so
    the process can be interrupted and resumed.
    """
    print(f"Listing the samples of {adapter.name}...")
    samples: Sequence[Sample] = adapter.list_samples()
    if limit is not None:
        samples = samples[:limit]

    splits: dict[str, str] = {s.id: adapter.assign_split(s.group) for s in samples}
    counts: dict[str, int] = {split: 0 for split in adapter.split_dirs}
    for split in splits.values():
        counts[split] += 1
    print(f"{len(samples)} samples. Split sizes: {counts}")

    new_files: int = 0
    # Downloading is I/O bound, so threads are enough.
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures: dict[Future[Any], Sample] = {
            executor.submit(prepare_sample, adapter, s, splits[s.id], datasets_dir): s
            for s in samples
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Preparing files"):
            try:
                new_files += int(future.result())
            except Exception as exc:
                print(f"Error preparing {futures[future].id}: {exc}")

    print(f"Done. {new_files} new files written, {len(samples) - new_files} skipped or failed.")


if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Prepare a dataset for the LibriVAD generation scripts."
    )
    parser.add_argument("adapter", choices=sorted(ADAPTERS), help="Dataset to prepare.")
    parser.add_argument("--workers", type=int, default=4, help="Parallel downloads.")
    parser.add_argument("--limit", type=int, default=None, help="Only prepare the first N samples.")
    args: argparse.Namespace = parser.parse_args()

    project_root: Path = Path(os.environ["PROJECT_ROOT"])
    datasets_dir: Path = project_root / os.environ["DATASETS_DIR"]

    prepare_dataset(ADAPTERS[args.adapter](), datasets_dir, args.workers, args.limit)
