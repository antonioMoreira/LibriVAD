"""
Plots a histogram of the duration of every .wav file found recursively under
a root folder (e.g. Results/LibriSpeechConcat). Durations are read with PyAV
from the container header, so the audio is not decoded.

Usage:
    uv run scripts/duration_histogram.py Results/LibriSpeechConcat
    uv run scripts/duration_histogram.py Results/LibriSpeechConcat --bins 80 \
        --output durations.png --csv durations.csv
"""

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import av
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm


def wav_duration(path: Path) -> float | None:
    """
    Returns the duration of an audio file in seconds, or None if it cannot be read.
    """
    try:
        # WAV duration is in the header, so skip FFmpeg's stream probing.
        with av.open(str(path), options={"probesize": "32", "analyzeduration": "0"}) as container:
            stream = container.streams.audio[0]
            if stream.duration is not None and stream.time_base is not None:
                return float(stream.duration * stream.time_base)
            if container.duration is not None:
                return container.duration / av.time_base
    except (av.FFmpegError, IndexError) as exc:
        print(f"Could not read {path}: {exc}")
    return None


def collect_durations(root: Path, workers: int) -> tuple[list[Path], NDArray[np.float64]]:
    """
    Finds every .wav file under root and reads its duration in parallel.
    """
    paths: list[Path] = sorted(p for p in root.rglob("*") if p.suffix.lower() == ".wav")
    if not paths:
        raise SystemExit(f"No .wav files found under {root}")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        durations: list[float | None] = list(
            tqdm(executor.map(wav_duration, paths), total=len(paths), desc="Reading durations")
        )

    valid: list[tuple[Path, float]] = [
        (p, d) for p, d in zip(paths, durations, strict=True) if d is not None
    ]
    return [p for p, _ in valid], np.array([d for _, d in valid], dtype=np.float64)


def plot_histogram(durations: NDArray[np.float64], root: Path, bins: int, output: Path) -> None:
    """
    Saves the duration histogram with the mean and median marked.
    """
    mean: float = float(np.mean(durations))
    median: float = float(np.median(durations))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(durations, bins=bins, color="#2563eb", edgecolor="white", linewidth=0.5)
    ax.axvline(mean, color="#dc2626", linestyle="--", label=f"mean = {mean:.2f} s")
    ax.axvline(median, color="#059669", linestyle=":", label=f"median = {median:.2f} s")
    ax.set_xlabel("Duration (s)")
    ax.set_ylabel("Number of files")
    ax.set_title(f"{root} ({len(durations)} files)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Histogram of .wav durations under a folder.")
    parser.add_argument("root", type=Path, help="Folder searched recursively for .wav files.")
    parser.add_argument("--bins", type=int, default=50, help="Number of histogram bins.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path. Defaults to <root name>_durations.png in the current folder.",
    )
    parser.add_argument("--csv", type=Path, default=None, help="Optionally save path,duration.")
    parser.add_argument("--workers", type=int, default=16, help="Threads used to read files.")
    args = parser.parse_args()

    root: Path = args.root.resolve()
    if not root.is_dir():
        raise SystemExit(f"{root} is not a folder")
    output: Path = args.output or Path(f"{root.name}_durations.png")

    paths, durations = collect_durations(root, args.workers)

    print(f"\nFiles:  {len(durations)}")
    print(f"Total:  {durations.sum() / 3600:.2f} h")
    print(f"Min:    {durations.min():.2f} s")
    print(f"Mean:   {durations.mean():.2f} s")
    print(f"Median: {np.median(durations):.2f} s")
    print(f"Max:    {durations.max():.2f} s")

    plot_histogram(durations, root, args.bins, output)
    print(f"\nHistogram saved to {output}")

    if args.csv is not None:
        with open(args.csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "duration_s"])
            writer.writerows((p, f"{d:.4f}") for p, d in zip(paths, durations, strict=True))
        print(f"Durations saved to {args.csv}")


if __name__ == "__main__":
    main()
