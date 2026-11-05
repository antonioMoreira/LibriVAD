"""
Writes rendered outputs and keeps the output folder consistent across runs.

  - Each wav (and `.npy` label) is written to `<name>.tmp` and renamed, so a crash never
    leaves a half-written file under its final name.
  - Metadata rows are `metadata.parquet` files with one explicit schema (see `tables.py`).
    Parquet cannot be appended to, so the rows of each folder are kept in memory and the
    file is rewritten atomically when the folder's last planned output is written, every
    `checkpoint_rows` rows, and at the end. Rows are sorted by `file_name`, so the content
    does not depend on the number of workers.
  - Before a run, leftovers of an interrupted run are cleaned: `.tmp` files are deleted,
    planned wavs without a row (written after the last checkpoint) are moved to
    `.orphans/<timestamp>/` and rendered again, and rows whose wav is gone are dropped.
    Outputs with both wav and row are skipped (resume).
"""

import datetime
import os
import shutil
import threading
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import soundfile as sf

from librivad.generation.config import Config
from librivad.generation.planner import FileSpec
from librivad.generation.render import Rendered
from librivad.generation.tables import read_rows, write_rows

METADATA_NAME: str = "metadata.parquet"
CHECKPOINT_ROWS: int = 500


@dataclass
class Cleanup:
    removed_tmp: int = 0
    orphans: list[str] = field(default_factory=list)
    dropped_rows: int = 0


class OutputWriter:
    def __init__(
        self, config: Config, schema: pa.Schema | None, checkpoint_rows: int = CHECKPOINT_ROWS
    ) -> None:
        self.config = config
        self.schema = schema
        self.checkpoint_rows = checkpoint_rows
        self.root: Path = config.output.root
        self.labels_root: Path | None = config.output.labels_root
        self.sample_rate: int = config.dataset.sample_rate
        self._lock = threading.Lock()
        self._rows: dict[Path, dict[str, dict[str, Any]]] = {}
        self._dirty: set[Path] = set()
        self._remaining: Counter[Path] = Counter()
        self._since_checkpoint: int = 0

    def wav_path(self, spec: FileSpec) -> Path:
        return self.root / spec.out_path

    def label_path(self, spec: FileSpec) -> Path | None:
        if spec.label_path is None or self.labels_root is None:
            return None
        return self.labels_root / spec.label_path

    def metadata_path(self, spec: FileSpec) -> Path | None:
        return None if spec.metadata_dir is None else self.root / spec.metadata_dir / METADATA_NAME

    def rows(self, path: Path) -> dict[str, dict[str, Any]]:
        """Rows of one metadata file by id, read from disk the first time."""
        if path not in self._rows:
            self._rows[path] = {row["id"]: row for row in read_rows(path)}
        return self._rows[path]

    def is_done(self, spec: FileSpec) -> bool:
        if not self.wav_path(spec).is_file():
            return False
        label: Path | None = self.label_path(spec)
        if label is not None and not label.is_file():
            return False
        metadata: Path | None = self.metadata_path(spec)
        return metadata is None or spec.row_id in self.rows(metadata)

    def prepare(
        self, specs: list[FileSpec], overwrite: bool = False
    ) -> tuple[list[FileSpec], Cleanup]:
        """Clean leftovers and return the specs still to render."""
        cleanup = Cleanup()
        for tmp in [*self.root.rglob("*.tmp"), *(self.labels_root or self.root).rglob("*.tmp")]:
            if ".orphans" not in tmp.parts and tmp.is_file():
                tmp.unlink()
                cleanup.removed_tmp += 1

        planned_ids: set[str] = {s.row_id for s in specs}
        for path in sorted({p for s in specs if (p := self.metadata_path(s)) is not None}):
            rows: dict[str, dict[str, Any]] = self.rows(path)
            for row_id, row in list(rows.items()):
                gone: bool = not (path.parent / row["file_name"]).is_file()
                if gone or (overwrite and row_id in planned_ids):
                    del rows[row_id]
                    cleanup.dropped_rows += 1
                    self._dirty.add(path)

        stamp: str = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
        pending: list[FileSpec] = []
        for spec in specs:
            if not overwrite and self.is_done(spec):
                continue
            wav: Path = self.wav_path(spec)
            metadata: Path | None = self.metadata_path(spec)
            if not overwrite and wav.is_file() and metadata is not None:
                target: Path = self.root / ".orphans" / stamp / spec.out_path
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(wav, target)
                cleanup.orphans.append(spec.out_path)
            if metadata is not None:
                self._remaining[metadata] += 1
            pending.append(spec)
        for path in sorted(self._dirty):
            if self._remaining[path] == 0:
                self._flush(path)
        return pending, cleanup

    def unplanned(self, specs: list[FileSpec]) -> tuple[list[Path], list[Path]]:
        """Wavs and metadata files under `root` that no spec plans (hidden folders aside)."""
        planned: set[str] = {s.out_path for s in specs}
        folders: set[str] = {s.metadata_dir for s in specs if s.metadata_dir is not None}
        wavs: list[Path] = []
        metadata: list[Path] = []
        if not self.root.is_dir():
            return wavs, metadata
        for path in sorted(self.root.rglob("*")):
            relative = path.relative_to(self.root)
            if any(part.startswith(".") for part in relative.parts) or not path.is_file():
                continue
            if path.suffix == ".wav" and relative.as_posix() not in planned:
                wavs.append(path)
            elif path.name == METADATA_NAME and relative.parent.as_posix() not in folders:
                metadata.append(path)
        return wavs, metadata

    def prune(self, specs: list[FileSpec]) -> int:
        """Delete what `unplanned` finds, with each wav's label; rows of a deleted wav that
        sit in a planned metadata file are dropped by `prepare`. Returns the wavs deleted."""
        wavs, metadata = self.unplanned(specs)
        for wav in wavs:
            wav.unlink()
            if self.labels_root is not None:
                label: Path = self.labels_root / wav.relative_to(self.root).with_suffix(".npy")
                label.unlink(missing_ok=True)
        for path in metadata:
            path.unlink()
        return len(wavs)

    def write(self, rendered: Rendered, row: dict[str, Any] | None) -> None:
        spec: FileSpec = rendered.spec
        wav: Path = self.wav_path(spec)
        wav.parent.mkdir(parents=True, exist_ok=True)
        tmp: Path = wav.with_name(wav.name + ".tmp")
        sf.write(str(tmp), rendered.audio, self.sample_rate, subtype="PCM_16", format="WAV")
        os.replace(tmp, wav)

        label: Path | None = self.label_path(spec)
        if label is not None:
            label.parent.mkdir(parents=True, exist_ok=True)
            label_tmp: Path = label.with_name(label.name + ".tmp")
            with open(label_tmp, "wb") as handle:
                np.save(handle, rendered.labels.astype(np.int16))
            os.replace(label_tmp, label)

        metadata: Path | None = self.metadata_path(spec)
        if metadata is None or row is None:
            return
        with self._lock:
            self.rows(metadata)[row["id"]] = row
            self._dirty.add(metadata)
            self._remaining[metadata] -= 1
            self._since_checkpoint += 1
            if self._remaining[metadata] <= 0:
                self._flush(metadata)
            if self._since_checkpoint >= self.checkpoint_rows:
                for path in sorted(self._dirty):
                    self._flush(path)
                self._since_checkpoint = 0

    def _flush(self, path: Path) -> None:
        """Rewrite one metadata file with its rows sorted by `file_name`."""
        assert self.schema is not None
        rows: list[dict[str, Any]] = sorted(self.rows(path).values(), key=lambda r: r["file_name"])
        if rows:
            write_rows(path, rows, self.schema)
        elif path.is_file():
            path.unlink()
        self._dirty.discard(path)

    def finalize(self) -> None:
        """Write every metadata file that still has unwritten rows."""
        with self._lock:
            for path in sorted(self._dirty):
                self._flush(path)
