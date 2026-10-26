# Description: Adapters that bring datasets other than LibriSpeech into the
# layout the LibriVAD generation scripts expect.
#
# `DatasetAdapter` holds the default behaviour, which mirrors what the repo already does for
# LibriSpeech: 16 kHz wavs organised as
#     <datasets_dir>/<name>/<split dir>/<speaker>/<chapter>/<file>.wav
# with the split folders named after the LibriSpeech ones. A new dataset only overrides what
# differs (see `SkylarAdapter`) and is registered in `ADAPTERS`.

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import requests

# Split folder names used by create_LibriVAD.py when it looks for the clean audio.
SPLIT_DIRS: dict[str, str] = {
    "train": "train-clean-100",
    "val": "dev-clean",
    "test": "test-clean",
}


@dataclass(frozen=True)
class Sample:
    """
    A single utterance of a dataset.

    Fields:
    - id       : Unique file stem. The generated mixtures are saved without the
                 speaker/chapter folders, so it has to be unique within the dataset.
    - source   : Where the audio is fetched from (a URL or a path).
    - group    : Speaker-like key. All samples of a group end up in the same split.
    - subgroup : Chapter-like folder name.
    """

    id: str
    source: str
    group: str
    subgroup: str


def _safe_dir_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


class DatasetAdapter:
    """
    Default behaviour, equivalent to the layout that already exists for LibriSpeech.
    Subclasses override only the differences.
    """

    name: str = "LibriSpeech"
    split_dirs: dict[str, str] = SPLIT_DIRS
    sample_rate: int = 16000
    # Fractions of the groups assigned to train, val and test.
    split_fractions: tuple[float, float, float] = (0.8, 0.1, 0.1)
    split_salt: str = "librivad"

    def list_samples(self) -> list[Sample]:
        """Lists every sample of the dataset."""
        raise NotImplementedError(
            f"{self.name}: LibriSpeech is prepared by setup.py; "
            "other datasets must implement list_samples()."
        )

    def fetch(self, sample: Sample) -> bytes:
        """Returns the raw bytes of the audio file of a sample."""
        raise NotImplementedError(f"{self.name}: fetch() is not implemented.")

    def assign_split(self, group: str) -> str:
        """
        Deterministically maps a group to 'train', 'val' or 'test'. Hash based, so the
        result does not depend on the order or on the rest of the dataset, and every
        sample of a group lands in the same split.
        """
        digest: bytes = hashlib.sha256(f"{self.split_salt}:{group}".encode()).digest()
        value: float = int.from_bytes(digest[:8], "big") / 2**64

        train_end: float = self.split_fractions[0]
        val_end: float = train_end + self.split_fractions[1]
        if value < train_end:
            return "train"
        if value < val_end:
            return "val"
        return "test"

    def relative_path(self, split: str, sample: Sample, suffix: str) -> Path:
        """<name>/<split dir>/<speaker>/<chapter>/<id><suffix>"""
        return (
            Path(self.name)
            / self.split_dirs[split]
            / _safe_dir_name(sample.group)
            / _safe_dir_name(sample.subgroup)
            / f"{sample.id}{suffix}"
        )

    def audio_path(self, datasets_dir: str | Path, split: str, sample: Sample) -> Path:
        return Path(datasets_dir) / self.relative_path(split, sample, ".wav")

    def label_path(self, labels_dir: str | Path, split: str, sample: Sample) -> Path:
        return Path(labels_dir) / self.relative_path(split, sample, ".npy")


class SkylarAdapter(DatasetAdapter):
    """
    skylar-ai-hf/skylar-dataset: an English and Spanish ASR corpus (~30 h) made of samples
    from Emilia, FLEURS and VoxPopuli. It only has a `train` split and no speech/silence
    annotation; clips are 24 kHz (English) or 16 kHz (Spanish).
    """

    name: str = "Skylar"
    repo_url: str = "https://huggingface.co/datasets/skylar-ai-hf/skylar-dataset/resolve/main"
    languages: tuple[str, ...] = ("en", "es")
    timeout: int = 60

    def _headers(self) -> dict[str, str]:
        token: str | None = os.environ.get("HF_TOKEN")
        return {"Authorization": f"Bearer {token}"} if token else {}

    def parse_metadata(self, language: str, lines: list[str]) -> list[Sample]:
        """Turns the lines of a `<language>/metadata.jsonl` into samples."""
        samples: list[Sample] = []
        for line in lines:
            if not line.strip():
                continue
            row: dict[str, object] = json.loads(line)

            file_name: str = str(row["file_name"])
            bucket: str = str(row["bucket"])
            stem: str = Path(file_name).stem
            # Some Spanish stems repeat across buckets, so the bucket is part of the id.
            sample_id: str = f"{language}_{bucket}_{stem}"

            # FLEURS has no speaker; its source_id (the sentence) is the next best key.
            who: str = str(row["speaker"] or row["source_id"])
            group: str = f"{language}:{row['dataset']}:{who}"

            samples.append(
                Sample(
                    id=sample_id,
                    source=f"{self.repo_url}/{language}/{file_name}",
                    group=group,
                    subgroup=bucket,
                )
            )

        return samples

    def list_samples(self) -> list[Sample]:
        samples: list[Sample] = []
        for language in self.languages:
            response: requests.Response = requests.get(
                f"{self.repo_url}/{language}/metadata.jsonl",
                headers=self._headers(),
                timeout=self.timeout,
            )
            response.raise_for_status()
            samples.extend(self.parse_metadata(language, response.text.splitlines()))

        return samples

    def fetch(self, sample: Sample) -> bytes:
        response: requests.Response = requests.get(
            sample.source, headers=self._headers(), timeout=self.timeout
        )
        response.raise_for_status()

        return response.content


ADAPTERS: dict[str, type[DatasetAdapter]] = {
    "skylar": SkylarAdapter,
}
