"""Phase 6: the `middle` scenario reproduces the paper's LibriSpeechConcat (dev and test).

Slow (renders 2,661 files); run with `uv run pytest -m slow`. Skipped without `files/`.
"""

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf
import yaml

from librivad.generation.build_scenarios import build, make_noises
from librivad.generation.config import config_from_dict
from librivad.generation.corpus import make_adapter
from librivad.generation.labels import load_labels
from librivad.generation.planner import FileSpec, Planner

REPO: Path = Path(__file__).resolve().parents[1]
PAPER_AUDIO: Path = REPO / "files/datasets/LibriSpeechConcat"
PAPER_LABELS: Path = REPO / "files/Labels/LibriSpeechConcat"


def compare(spec: FileSpec, ours_root: Path, labels: np.ndarray) -> dict[str, Any]:
    """Audio difference and label agreement of one output against the paper's files."""
    paper: np.ndarray = sf.read(str(PAPER_AUDIO / spec.out_path), dtype="int16")[0]
    ours: np.ndarray = sf.read(str(ours_root / spec.out_path), dtype="int16")[0]
    result: dict[str, Any] = {"same_length": len(paper) == len(ours)}
    if result["same_length"]:
        result["max_diff"] = int(np.max(np.abs(paper.astype(int) - ours.astype(int))))
    paper_label: np.ndarray = np.load(PAPER_LABELS / Path(spec.out_path).with_suffix(".npy"))
    result["label_equal"] = len(paper_label) == len(labels) and bool(
        np.array_equal(paper_label, labels)
    )
    result["label_length_diff"] = len(paper_label) - len(labels)
    return result


@pytest.mark.slow
@pytest.mark.skipif(not PAPER_AUDIO.is_dir(), reason="LibriSpeechConcat not generated")
def test_middle_reproduces_the_paper(tmp_path: Path) -> None:
    raw: dict[str, Any] = yaml.safe_load(
        (REPO / "configs/scenarios/librispeech_middle.yaml").read_text()
    )
    raw["output"]["root"] = str(tmp_path)
    config = config_from_dict(raw, base=REPO)
    adapter = make_adapter(config.dataset)
    noises = make_noises(config, adapter)
    result = Planner(config, adapter, noises).plan(["val", "test"])
    items = adapter.index(["val", "test"])
    built: dict[str, Any] = build(config, result, items, noises, workers=8, overwrite=False)
    assert built["failed"] == 0

    paper_files: set[str] = {
        str(p.relative_to(PAPER_AUDIO))
        for split in ("dev-clean", "test-clean")
        for p in (PAPER_AUDIO / split).rglob("*.wav")
    }
    assert {s.out_path for s in result.specs} == paper_files

    stats: Counter[str] = Counter()
    worst: int = 0
    for spec in result.specs:
        labels: np.ndarray = np.concatenate(
            [
                np.zeros(b.length, dtype=np.int16)
                if b.kind == "noise"
                else load_labels(items[b.items[0].id], config.dataset).astype(np.int16)
                for b in spec.blocks
            ]
        )
        outcome: dict[str, Any] = compare(spec, tmp_path, labels)
        stats["files"] += 1
        stats["same_length"] += outcome["same_length"]
        stats["label_equal"] += outcome["label_equal"]
        worst = max(worst, outcome.get("max_diff", 0))
    print(f"\nparity: {dict(stats)}, max audio difference {worst} LSB")
    assert stats["same_length"] == stats["files"]
    assert worst <= 1
