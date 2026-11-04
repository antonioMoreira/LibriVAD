"""Plan 5: `noise_block.span: file` with a `[speech]` layout does the arithmetic of the
paper's `create_LibriVAD.py` (noise over the whole utterance, peak-normalized).

Only the per-file arithmetic is compared: the script restarts its noise cursor every batch
and walks files in `glob` order, so its excerpts can't be planned the same way. Each output
is recomputed with the script's own functions from the excerpt the builder chose.

Slow (reads every dev-clean file to plan); run with `uv run pytest -m slow`. Skipped
without `files/`.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import scipy.io.wavfile as wav

from librivad.generation import librispeech_functions as lf
from librivad.generation.build_scenarios import build, make_noises
from librivad.generation.config import config_from_dict
from librivad.generation.corpus import make_adapter
from librivad.generation.planner import Planner

REPO: Path = Path(__file__).resolve().parents[1]
LIBRISPEECH: Path = REPO / "files/datasets/LibriSpeech/dev-clean"
LABELS: Path = REPO / "files/Labels/LibriSpeech"
NOISE: Path = REPO / "files/Noises/Office_noise/Office_noise_val.wav"


def paper_config(output: Path) -> dict[str, Any]:
    return {
        "version": 2,
        "dataset": {
            "name": "librispeech",
            "adapter": "path_pattern",
            "root": "files/datasets/LibriSpeech",
            "splits": {"val": "dev-clean"},
            "path_pattern": "{split}/{speaker}/{chapter}/{speaker}-{chapter}-{utt}.wav",
            "labels": {
                "source": "npy",
                "dir": "files/Labels/LibriSpeech",
                "path": "{split_folder}/{speaker}/{chapter}/{stem}.npy",
            },
        },
        "noise_sources": {
            "office": {
                "type": "file_per_split",
                "path": "files/Noises/Office_noise/Office_noise_{split}.wav",
            }
        },
        "output": {
            "root": str(output),
            "path": "{split_folder}/office/{snr_db}/{first[stem]}.wav",
            "normalization": "peak",
            "peak_dbfs": 0.0,
        },
        "scenarios": [
            {
                "name": "paper",
                "layout": ["speech"],
                "speech_block": {"composition": "single", "edge_fade_ms": 0},
                "noise_block": {"source": "office", "span": "file", "slicing": "sequential_wrap"},
                "loudness": {"measure": "rms", "speech_only": True},
                "conditions": {"snr_db": [-5, 20]},
            }
        ],
    }


@pytest.mark.slow
@pytest.mark.skipif(not (LIBRISPEECH.is_dir() and NOISE.is_file()), reason="files/ not set up")
def test_span_file_matches_create_librivad(tmp_path: Path) -> None:
    config = config_from_dict(paper_config(tmp_path), base=REPO)
    adapter = make_adapter(config.dataset)
    noises = make_noises(config, adapter)
    result = Planner(config, adapter, noises).plan(["val"], limit=8)
    items = adapter.index(["val"])
    assert build(config, result, items, noises, workers=4, overwrite=False)["failed"] == 0

    _, noise = wav.read(NOISE)
    tiled: np.ndarray = np.tile(noise, 2)
    compared: int = 0
    worst: int = 0
    for spec in (s for s in result.specs if s.selected):
        item = items[spec.speech_items()[0].id]
        _, signal = wav.read(item.path)
        label_path: Path = LABELS / "dev-clean" / item.fields["speaker"] / item.fields["chapter"]
        label_path = label_path / f"{item.path.stem}.npy"
        speech_rms = lf.RMS(lf.remove_silence(signal, item.path, label_path))
        subset: np.ndarray = tiled[spec.noise_offset : spec.noise_offset + len(signal)]
        assert spec.snr_db is not None
        scale = speech_rms / (lf.RMS(subset) * (10 ** (spec.snr_db / 20)))
        paper: np.ndarray = lf.scale_signal(signal + scale * subset)
        _, ours = wav.read(tmp_path / spec.out_path)
        worst = max(worst, int(np.max(np.abs(paper.astype(int) - ours.astype(int)))))
        compared += 1
    print(f"\ncompared {compared} files, max difference {worst} LSB")
    assert compared == 16
    assert worst <= 1
