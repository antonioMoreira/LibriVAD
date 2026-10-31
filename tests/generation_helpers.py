"""Builders for a small corpus that follows skylar-dataset's `clean` contract, and noises."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

from librivad.generation.tables import clean_schema, write_rows

SR: int = 16000
BUCKETS: dict[str, float] = {"05-10": 6.0, "10-15": 12.0, "25-30": 27.0}
"""Bucket -> duration of the fixture file placed in it."""


def speech_like(duration: float, seed: int) -> tuple[NDArray[np.int16], list[dict[str, Any]]]:
    """Tone bursts ("words") separated by silent pauses, and the word list that marks them."""
    rng: np.random.Generator = np.random.default_rng(seed)
    n: int = round(duration * SR)
    audio: NDArray[np.float64] = np.zeros(n)
    words: list[dict[str, Any]] = []
    position: float = 0.3
    index: int = 0
    while position + 0.5 < duration - 0.3:
        length: float = 0.25 + 0.05 * (index % 4)
        start: int = round(position * SR)
        end: int = round((position + length) * SR)
        t: NDArray[np.float64] = np.arange(end - start) / SR
        audio[start:end] = 9000 * np.sin(2 * np.pi * (300 + 40 * (index % 5)) * t)
        audio[start:end] += rng.normal(0, 300, end - start)
        words.append(
            {
                "word": f"w{index}",
                "start": round(position, 3),
                "end": round(position + length, 3),
                "probability": 0.9,
            }
        )
        position += length + 0.2
        index += 1
    return np.clip(np.round(audio), -32768, 32767).astype(np.int16), words


def write_clean_corpus(
    root: Path, items_per_cell: int = 1, languages: tuple[str, ...] = ("en", "es")
) -> Path:
    """A `SkylarDataset` folder with `clean/<split>/<lang>/<bucket>/<id>.wav` and metadata."""
    clean: Path = root / "clean"
    (root / ".provenance").mkdir(parents=True, exist_ok=True)
    (root / ".provenance" / "schema.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "metadata_file": "metadata.parquet",
                "fields": clean_schema().names,
            }
        )
    )
    seed: int = 0
    for split in ("train", "val", "test"):
        for language in languages:
            folder: Path = clean / split / language
            rows: list[dict[str, Any]] = []
            for bucket, duration in BUCKETS.items():
                for copy in range(items_per_cell):
                    seed += 1
                    item_id: str = f"{seed:016x}"
                    audio, words = speech_like(duration + 0.01 * copy, seed)
                    (folder / bucket).mkdir(parents=True, exist_ok=True)
                    sf.write(str(folder / bucket / f"{item_id}.wav"), audio, SR, subtype="PCM_16")
                    text: str = " ".join(w["word"] for w in words)
                    rows.append(
                        {
                            "file_name": f"{bucket}/{item_id}.wav",
                            "id": item_id,
                            "sample_id": f"{language}_{bucket}_{item_id}",
                            "language": language,
                            "locale": "es_ES" if language == "es" else None,
                            "duration_bucket": bucket,
                            "duration": round(len(audio) / SR, 4),
                            "text": text,
                            "word_count": len(words),
                            "has_question": False,
                            # null in every row of es/val, like a FLEURS-only folder
                            "speaker": None
                            if (language, split) == ("es", "val")
                            else f"spk{seed % 3}",
                            "dnsmos": 3.1 if language == "en" else None,
                            "source_id": f"src{seed}",
                            "source": None,
                            "dataset": "fixture",
                            "component": None,
                            "split_group": f"spk{seed % 3}",
                            "words": words,
                        }
                    )
            folder.mkdir(parents=True, exist_ok=True)
            write_rows(folder / "metadata.parquet", rows, clean_schema())
    return clean


def write_noises(
    root: Path, kinds: tuple[str, ...] = ("babble", "office"), seconds: float = 90.0
) -> dict[str, str]:
    """One recording per kind and split; returns kind -> path template with `{split}`."""
    templates: dict[str, str] = {}
    rng: np.random.Generator = np.random.default_rng(7)
    for kind in kinds:
        for split in ("train", "val", "test"):
            path: Path = root / kind / f"{kind}_{split}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            noise: NDArray[np.float64] = rng.normal(0, 2000, round(seconds * SR))
            sf.write(
                str(path), np.clip(noise, -32768, 32767).astype(np.int16), SR, subtype="PCM_16"
            )
        templates[kind] = str(root / kind / f"{kind}_{{split}}.wav")
    return templates


def skylar_raw_config(
    clean: Path, noises: dict[str, str], output: Path, snrs: tuple[int, ...] = (0, 5, 35)
) -> dict[str, Any]:
    """The plan 2 §5.1 config, cut down to the fixture's noises and a few levels."""
    return {
        "version": 2,
        "seed": 1234,
        "dataset": {
            "name": "skylar_clean",
            "adapter": "audiofolder",
            "root": str(clean),
            "schema_version": 1,
            "sample_rate": SR,
            "splits": {"train": "train", "val": "val", "test": "test"},
            "id_field": "id",
            "passthrough_fields": [
                "language",
                "duration_bucket",
                "text",
                "speaker",
                "dnsmos",
                "dataset",
                "component",
                "locale",
                "source_id",
                "source",
            ],
            "labels": {"source": "words", "field": "words", "bridge_gaps_ms": 0},
            "continuity": None,
        },
        "noise_sources": {
            name: {"type": "file_per_split", "path": path} for name, path in noises.items()
        },
        "output": {
            "root": str(output),
            "path": "{split}/{scenario}/{noise}/snr_{snr_db:02d}/{language}"
            "/{duration_bucket}/{id}.wav",
            "metadata_dir": "{split}/{scenario}/{noise}/snr_{snr_db:02d}/{language}",
            "normalization": "none",
            "clip_guard": "per_group",
            "drop_fields": ["seek", "tokens", "temperature", "compression_ratio"],
            "limits": {"max_files_total": 100000, "max_files_per_folder": 10000},
        },
        "defaults": {
            "speech_block": {"composition": "single"},
            "noise_length": {"mode": "fill_to_target", "target_duration_s": 30, "min_noise_s": 2.0},
            "noise_block": {
                "slicing": "random_offset",
                "key": ["id", "scenario", "noise"],
                "fade_ms": 10,
            },
            "loudness": {"measure": "rms", "speech_only": True, "min_speech_s": 0.5},
            "constraints": {"min_snr_db": 0},
        },
        "scenarios": [
            {"name": "noise_first", "layout": ["noise", "speech"]},
            {"name": "noise_last", "layout": ["speech", "noise"]},
        ],
        "conditions": {"noise": list(noises), "snr_db": list(snrs)},
        "population": {
            "train": {"crossed": [], "assigned": ["scenario", "noise", "snr_db"]},
            "val": {"crossed": ["snr_db"], "assigned": ["scenario", "noise"]},
            "test": {"crossed": ["snr_db", "scenario"], "assigned": ["noise"]},
            "strata": ["language", "duration_bucket"],
        },
    }
