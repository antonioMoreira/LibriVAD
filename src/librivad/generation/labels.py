"""
Sample-level speech (1) / non-speech (0) labels for source files.

Sources: `textgrid` (forced alignments, an empty word is silence), `npy` (a precomputed 0/1
array per file), `words` (the word list of an `audiofolder` row) and `none` (the whole
file counts as speech). Every label has exactly the length of its audio.
"""

import math
from pathlib import Path
from typing import Any

import numpy as np
import textgrids
from numpy.typing import NDArray

from librivad.generation.config import DatasetConfig
from librivad.generation.corpus import ContractError, SourceItem


def fit_length(label: NDArray[Any], n_samples: int) -> NDArray[np.int8]:
    """Truncate `label`, or pad it with non-speech, to exactly `n_samples`."""
    fitted: NDArray[np.int8] = np.zeros(n_samples, dtype=np.int8)
    n: int = min(n_samples, len(label))
    fitted[:n] = label[:n]
    return fitted


def textgrid_label(path: Path, tier: str, silence_token: str, sample_rate: int) -> NDArray[np.int8]:
    """Label from a TextGrid, with the same arithmetic as `create_labels.py`.

    End times are kept as float32 and multiplied by the sample rate before truncation, as
    `librispeech_functions.find_timeframes` and `_generate_label_from_timeframes` do, so the
    boundaries fall on the same samples.
    """
    grid: textgrids.TextGrid = textgrids.TextGrid(path)
    intervals: Any = grid[tier]
    ends: NDArray[np.float32] = np.array([i.xmax for i in intervals], dtype=np.float32)
    words: list[Any] = [i.text for i in intervals]
    if ends.size == 0:
        return np.zeros(0, dtype=np.int8)

    label: NDArray[np.int8] = np.ones(int(ends[-1] * sample_rate), dtype=np.int8)
    silent: list[int] = [j for j, w in enumerate(words) if w == silence_token or w is None]
    for j in silent:
        start: int = 0 if j == 0 else int(ends[j - 1] * sample_rate)
        label[start : int(ends[j] * sample_rate)] = 0
    return label


def words_label(
    words: list[dict[str, Any]], n_samples: int, sample_rate: int, bridge_gaps_ms: float = 0.0
) -> NDArray[np.int8]:
    """Label from a word list: samples `[floor(start·sr), ceil(end·sr))` of each word are speech.

    Gaps between consecutive words shorter than `bridge_gaps_ms` are labelled speech too.
    """
    label: NDArray[np.int8] = np.zeros(n_samples, dtype=np.int8)
    bounds: list[tuple[int, int]] = []
    for word in words:
        start: int = max(0, math.floor(float(word["start"]) * sample_rate))
        end: int = min(n_samples, math.ceil(float(word["end"]) * sample_rate))
        if end > start:
            label[start:end] = 1
            bounds.append((start, end))
    bridge: int = round(bridge_gaps_ms / 1000 * sample_rate)
    if bridge > 0:
        bounds.sort()
        for (_, previous_end), (next_start, _) in zip(bounds, bounds[1:], strict=False):
            if 0 < next_start - previous_end < bridge:
                label[previous_end:next_start] = 1
    return label


def load_labels(item: SourceItem, config: DatasetConfig) -> NDArray[np.int8]:
    """The 0/1 label of `item`, with exactly `item.n_samples` samples."""
    labels = config.labels
    sr: int = config.sample_rate
    if labels.source == "none":
        return np.ones(item.n_samples, dtype=np.int8)
    if labels.source == "words":
        if item.row is None:
            raise ContractError(f"{item.path}: labels.source words needs a metadata row")
        words: Any = item.row.get(labels.field)
        if not isinstance(words, list):
            raise ContractError(f"{item.path}: `{labels.field}` is not a list")
        return words_label(words, item.n_samples, sr, labels.bridge_gaps_ms)

    assert labels.dir is not None and labels.path is not None  # checked by check_config
    path: Path = labels.dir / labels.path.format(**item.fields)
    if not path.is_file():
        raise ContractError(f"{item.path}: label file {path} not found")
    if labels.source == "textgrid":
        raw: NDArray[Any] = textgrid_label(path, labels.tier, labels.silence_token, sr)
    else:
        raw = np.load(path)
    return fit_length(raw, item.n_samples)
