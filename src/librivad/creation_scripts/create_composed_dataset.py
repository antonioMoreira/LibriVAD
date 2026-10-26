# Description: Creates clips of up to 30 seconds made of speech segments, pauses and up to two
# noise layers, each one with its own level, from any dataset that has VAD labels (LibriSpeech,
# Skylar, ...). Unlike create_LibriVAD.py, which mixes every utterance with every noise on a
# fixed grid, each clip here draws its duration, segments, pauses and noises from the ranges
# given by the user, so the labels are created together with the audio.
#
# Every parameter has a default (see ComposeConfig) and can be set, from lowest to highest
# priority, by a TOML or YAML file (--config, see configs/composer_example.yaml) and by the
# command line.
#
# Usage:
#   uv run src/librivad/creation_scripts/create_composed_dataset.py --dataset Skylar \
#       --noise Street_noise:-35 --noise Babble_noise:-40 --duration 10 30
#
# Output (under --save_dir, default <RESULTS_DIR>_composed):
#   <Dataset>/<split>/<clip id>.wav        the audio
#   Labels/<Dataset>/<split>/<clip id>.npy the VAD label, one value per sample
#   <Dataset>/manifest.csv                 the recipe of every clip

import argparse
import csv
import json
import os
import sys
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io.wavfile as wav
import yaml
from dotenv import load_dotenv
from numpy.typing import NDArray
from tqdm import tqdm

from librivad.composer import (
    SAMPLE_RATE,
    Clip,
    ComposeConfig,
    NoiseLayer,
    Segment,
    compose_clip,
)
from librivad.dataset_adapter import SPLIT_DIRS

load_dotenv()

scripts_path: str = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(scripts_path)

SPLITS: tuple[str, ...] = tuple(SPLIT_DIRS)

MANIFEST_COLUMNS: list[str] = [
    "clip_id",
    "split",
    "duration",
    "mode",
    "speech_db",
    "gain_db",
    "clipped",
    "noises",
    "segments",
    "pauses",
    "pauses_samples",
    "seed",
]


def parse_range(values: list[float] | None) -> tuple[float, float] | None:
    """One value means a fixed value, two values a (min, max) range."""
    if values is None:
        return None

    return (values[0], values[-1])


def parse_noise_layer(text: str) -> NoiseLayer:
    name, separator, db = text.rpartition(":")
    if not separator or not name:
        raise argparse.ArgumentTypeError(f"'{text}' is not in the NAME:DB format.")
    try:
        return NoiseLayer(name, float(db))
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{db}' is not a number (in '{text}').") from None


def build_parser() -> argparse.ArgumentParser:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Create clips with speech, pauses and noise layers of chosen levels.",
        epilog="Parameters left out use the defaults of ComposeConfig (see docs/composer.md).",
    )
    parser.add_argument("--dataset", required=True, help="Dataset folder name, e.g. Skylar.")
    parser.add_argument("--config", type=Path, help="TOML or YAML file with parameters.")
    parser.add_argument("--dataset_dir", type=str, default=None, help="Root of the datasets.")
    parser.add_argument("--labels_dir", type=str, default=None, help="Root of the labels.")
    parser.add_argument("--noises_dir", type=str, default=None, help="Root of the noises.")
    parser.add_argument("--save_dir", type=str, default=None, help="Where to save the clips.")

    parser.add_argument("--mode", choices=["speech", "noise_only"])
    parser.add_argument("--noise-only-ratio", type=float, help="Fraction of clips without speech.")
    parser.add_argument("--duration", nargs="+", type=float, help="Seconds: MAX, or MIN MAX.")
    parser.add_argument("--speech-db", nargs="+", type=float, help="dBFS: LEVEL, or MIN MAX.")
    parser.add_argument(
        "--noise",
        action="append",
        type=parse_noise_layer,
        metavar="NAME:DB",
        help="A noise layer and its level in dBFS. Repeat it (at most twice) for two noises.",
    )
    parser.add_argument("--noise-pool", nargs="+", help="Noises to sample from.")
    parser.add_argument("--noise-count", nargs="+", type=int, help="Sampled noises: N, or MIN MAX.")
    parser.add_argument("--noise-db", nargs="+", type=float, help="Sampled noise dBFS: MIN MAX.")
    parser.add_argument("--pause-range", nargs="+", type=float, help="Pause seconds: MIN MAX.")
    parser.add_argument("--pause-prob", type=float, help="Chance of a pause before each segment.")
    parser.add_argument("--num-clips", nargs=3, type=int, metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--normalize", choices=["limit", "none"])

    return parser


def load_config_file(path: Path) -> dict[str, Any]:
    """Reads the parameters of a .toml, .yaml or .yml file, according to its extension."""
    suffix: str = path.suffix.lower()
    if suffix == ".toml":
        with open(path, "rb") as f:
            return tomllib.load(f)

    if suffix in (".yaml", ".yml"):
        with open(path) as f:
            values: Any = yaml.safe_load(f)
        if values is None:  # an empty file, or one with every line commented out
            return {}
        if not isinstance(values, dict):
            raise ValueError(f"{path}: expected 'key: value' pairs at the top level.")
        return values

    raise ValueError(f"{path}: the config file must be .toml, .yaml or .yml.")


def resolve_config(args: argparse.Namespace) -> ComposeConfig:
    """Defaults, then the config file, then the command line."""
    values: dict[str, Any] = {}
    if args.config is not None:
        values = load_config_file(args.config)
    config: ComposeConfig = ComposeConfig.from_dict(values)

    changes: dict[str, Any] = {
        "mode": args.mode,
        "noise_only_ratio": args.noise_only_ratio,
        "duration": parse_range(args.duration),
        "speech_db": parse_range(args.speech_db),
        "noises": None if args.noise is None else tuple(args.noise),
        "noise_pool": None if args.noise_pool is None else tuple(args.noise_pool),
        "noise_count": parse_range(args.noise_count),
        "noise_db": parse_range(args.noise_db),
        "pause_range": parse_range(args.pause_range),
        "pause_prob": args.pause_prob,
        "num_clips": None if args.num_clips is None else tuple(args.num_clips),
        "seed": args.seed,
        "normalize": args.normalize,
    }
    config = config.replace(**{k: v for k, v in changes.items() if v is not None})
    config.validate()

    return config


def noise_path(noises_dir: Path, name: str, split: str) -> Path:
    return noises_dir / name / f"{name}_{split}.wav"


def available_noises(noises_dir: Path, split: str) -> list[str]:
    """Noise folders that have a file for the split."""
    if not noises_dir.is_dir():
        return []

    return [
        d.name
        for d in sorted(noises_dir.iterdir())
        if d.is_dir() and noise_path(noises_dir, d.name, split).is_file()
    ]


def label_path_for(wav_path: Path, datasets_dir: Path, labels_dir: Path) -> Path:
    return (labels_dir / wav_path.relative_to(datasets_dir)).with_suffix(".npy")


def list_speech_files(datasets_dir: Path, labels_dir: Path, dataset: str, split: str) -> list[Path]:
    """Wavs of the split that have a label (the others cannot be level-matched)."""
    split_dir: Path = datasets_dir / dataset / SPLIT_DIRS[split]

    return [
        p
        for p in sorted(split_dir.rglob("*.wav"))
        if label_path_for(p, datasets_dir, labels_dir).is_file()
    ]


def load_segment(path: Path, datasets_dir: Path, labels_dir: Path) -> Segment:
    sr: int
    audio: NDArray[Any]
    sr, audio = wav.read(path)
    if sr != SAMPLE_RATE:
        raise ValueError(f"{path} is {sr} Hz; expected {SAMPLE_RATE} Hz.")

    label: NDArray[Any] = np.load(label_path_for(path, datasets_dir, labels_dir))

    return Segment(source=str(path.relative_to(datasets_dir)), audio=audio, label=label)


def create_split(
    config: ComposeConfig,
    dataset: str,
    split: str,
    split_index: int,
    paths: dict[str, Path],
    save_dir: Path,
) -> list[dict[str, Any]]:
    """Creates the clips of one split and returns their manifest rows."""
    datasets_dir: Path = paths["datasets"]
    labels_dir: Path = paths["labels"]
    noises_dir: Path = paths["noises"]
    n_clips: int = config.num_clips[split_index]

    pool: list[str] = available_noises(noises_dir, split)
    if config.noise_pool is not None:
        missing: list[str] = [n for n in config.noise_pool if n not in pool]
        if missing:
            raise SystemExit(f"Noises without a '{split}' file in {noises_dir}: {missing}")
        pool = list(config.noise_pool)
    for layer in config.noises:
        if not noise_path(noises_dir, layer.name, split).is_file():
            raise SystemExit(f"Noise file not found: {noise_path(noises_dir, layer.name, split)}")

    speech_files: list[Path] = list_speech_files(datasets_dir, labels_dir, dataset, split)
    needs_speech: bool = config.mode == "speech"
    if needs_speech and not speech_files:
        raise SystemExit(
            f"No labeled .wav files for {dataset}/{SPLIT_DIRS[split]}. "
            "Prepare the dataset and its labels first."
        )

    noise_cache: dict[str, NDArray[Any]] = {}

    def get_noise(name: str) -> NDArray[Any]:
        if name not in noise_cache:
            # Memory-mapped: the noise files are hours long.
            _sr, noise_cache[name] = wav.read(noise_path(noises_dir, name, split), mmap=True)
        return noise_cache[name]

    def next_segment(rng: np.random.Generator) -> Segment:
        path: Path = speech_files[int(rng.integers(0, len(speech_files)))]
        return load_segment(path, datasets_dir, labels_dir)

    audio_dir: Path = save_dir / dataset / split
    label_dir: Path = save_dir / "Labels" / dataset / split
    audio_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for i in tqdm(range(n_clips), desc=f"{dataset} {split}"):
        # One generator per clip: a clip does not depend on the ones created before it.
        rng: np.random.Generator = np.random.default_rng([config.seed, split_index, i])
        clip: Clip = compose_clip(rng, config, next_segment, get_noise, pool)

        clip_id: str = f"{split}_{i:06d}"
        wav.write(audio_dir / f"{clip_id}.wav", SAMPLE_RATE, clip.audio)
        np.save(label_dir / f"{clip_id}.npy", clip.label)

        recipe: dict[str, Any] = clip.recipe
        rows.append(
            {
                "clip_id": clip_id,
                "split": split,
                "duration": recipe["duration"],
                "mode": recipe["mode"],
                "speech_db": recipe["speech_db"],
                "gain_db": recipe["gain_db"],
                "clipped": recipe["clipped"],
                "noises": json.dumps(recipe["noises"]),
                "segments": json.dumps(recipe["segments"]),
                "pauses": json.dumps(recipe["pauses"]),
                "pauses_samples": json.dumps(recipe["pauses_samples"]),
                "seed": config.seed,
            }
        )

    return rows


if __name__ == "__main__":
    args: argparse.Namespace = build_parser().parse_args()
    config: ComposeConfig = resolve_config(args)

    project_root: Path = Path(os.environ["PROJECT_ROOT"])
    paths: dict[str, Path] = {
        "datasets": Path(args.dataset_dir or project_root / os.environ["DATASETS_DIR"]),
        "labels": Path(args.labels_dir or project_root / os.environ["LABELS_DIR"]),
        "noises": Path(args.noises_dir or project_root / os.environ["NOISES_DIR"]),
    }
    save_dir: Path = Path(
        args.save_dir or project_root / f"{os.environ['RESULTS_DIR'].rstrip('/')}_composed"
    )

    print(f"Configuration: {config}")

    all_rows: list[dict[str, Any]] = []
    for split_index, split in enumerate(SPLITS):
        all_rows.extend(create_split(config, args.dataset, split, split_index, paths, save_dir))

    manifest_path: Path = save_dir / args.dataset / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer: csv.DictWriter[str] = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nCreated {len(all_rows)} clips. Manifest: {manifest_path}")
