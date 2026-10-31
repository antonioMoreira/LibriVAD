"""
Noise recordings and the excerpts cut from them.

Two source types:
  - `file_per_split`: one recording per split (`{split}` in the path), e.g. LibriVAD's
    `files/Noises/<Kind>_noise/<Kind>_noise_{split}.wav`.
  - `extracted_from_dataset`: the label-0 regions of one corpus split, concatenated and
    cached to a WAV (LibriSpeech's `train100_silence.wav`).

All audio is returned in int16 units (PCM_16 values as float64), so a fixed gain means the
same thing as in the legacy scripts.
"""

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

from librivad.generation.config import NoiseSourceConfig
from librivad.generation.corpus import ContractError, CorpusAdapter
from librivad.generation.labels import load_labels

INT16_SCALE: float = 32768.0


def read_int16_units(path: Path, start: int = 0, frames: int = -1) -> NDArray[np.float64]:
    """Read `frames` samples from `start` as float64 in int16 units (PCM_16 values unchanged)."""
    info: Any = sf.info(str(path))
    if info.subtype == "PCM_16":
        data: NDArray[Any] = sf.read(str(path), start=start, frames=frames, dtype="int16")[0]
        return data.astype(np.float64)
    data = sf.read(str(path), start=start, frames=frames, dtype="float64")[0]
    return np.asarray(data, dtype=np.float64) * INT16_SCALE


def keyed_offset(seed: int, key: list[str], available: int) -> int:
    """Offset in `[0, available]` chosen by a generator seeded with a hash of `seed` and `key`.

    The same key always gives the same offset, whatever order files are planned in.
    """
    digest: bytes = hashlib.sha256(":".join([str(seed), *key]).encode()).digest()
    rng: np.random.Generator = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    return int(rng.integers(0, available + 1))


def raised_cosine(n: int) -> NDArray[np.float64]:
    """Fade-in curve of `n` samples, from just above 0 to just below 1."""
    if n <= 0:
        return np.ones(0)
    return 0.5 - 0.5 * np.cos(np.pi * (np.arange(n) + 0.5) / n)


class NoiseSource:
    """One named noise source. Lengths and files are resolved per split."""

    def __init__(
        self, name: str, config: NoiseSourceConfig, sample_rate: int, adapter: CorpusAdapter
    ) -> None:
        self.name = name
        self.config = config
        self.sample_rate = sample_rate
        self.adapter = adapter
        self._lengths: dict[Path, int] = {}

    def file(self, split: str) -> Path:
        if self.config.type == "file_per_split":
            return Path((self.config.path or "").format(split=split))
        assert self.config.cache is not None
        if not self.config.cache.is_file():
            self._build_cache()
        return self.config.cache

    def length(self, split: str) -> int:
        path: Path = self.file(split)
        if path not in self._lengths:
            if not path.is_file():
                raise ContractError(f"noise {self.name}: {path} not found")
            info: Any = sf.info(str(path))
            if info.samplerate != self.sample_rate or info.channels != 1:
                raise ContractError(
                    f"noise {self.name}: {path} is {info.samplerate} Hz with {info.channels}"
                    f" channels, expected {self.sample_rate} Hz mono"
                )
            self._lengths[path] = int(info.frames)
        return self._lengths[path]

    def read(self, split: str, offset: int, n: int) -> NDArray[np.float64]:
        """`n` samples from `offset`, wrapping to the start of the file when it runs out."""
        path: Path = self.file(split)
        total: int = self.length(split)
        if total == 0:
            raise ContractError(f"noise {self.name}: {path} is empty")
        chunks: list[NDArray[np.float64]] = []
        position: int = offset % total
        remaining: int = n
        while remaining > 0:
            take: int = min(remaining, total - position)
            chunks.append(read_int16_units(path, position, take))
            remaining -= take
            position = 0
        return np.concatenate(chunks) if chunks else np.zeros(0)

    def _build_cache(self) -> None:
        """Concatenate the label-0 samples of every file in `config.split` into `cache`."""
        assert self.config.cache is not None and self.config.split is not None
        pieces: list[NDArray[np.int16]] = []
        for item in self.adapter.items(self.config.split):
            label: NDArray[np.int8] = load_labels(item, self.adapter.config)
            audio: NDArray[np.float64] = read_int16_units(item.path)
            pieces.append(audio[label == 0].astype(np.int16))
        self.config.cache.parent.mkdir(parents=True, exist_ok=True)
        data: NDArray[np.int16] = np.concatenate(pieces) if pieces else np.zeros(0, np.int16)
        sf.write(str(self.config.cache), data, self.sample_rate, subtype="PCM_16")
