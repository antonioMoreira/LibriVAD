"""
Lists the source files of a corpus, with the fields that templates and grouping use.

Two adapters cover the corpora the builder reads:
  - `path_pattern`: fields parsed from each file's path (LibriSpeech).
  - `audiofolder`: an AudioFolder corpus described by one `metadata.parquet` per folder
    (the skylar-dataset `clean` subset, contract in skylar's plan 1 §2 and §7).

No corpus is special-cased in code: everything corpus-specific comes from the config.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from string import Formatter
from typing import Any

import pyarrow as pa
import soundfile as sf

from librivad.generation.config import DatasetConfig
from librivad.generation.tables import CLEAN_TYPES, read_table_rows

AUDIO_EXTENSIONS: tuple[str, ...] = (".wav", ".flac")


class ContractError(ValueError):
    """Raised when the input corpus breaks its contract. The CLI exits with code 4."""


@dataclass(frozen=True)
class SourceItem:
    """One source audio file and everything the planner and renderer need about it."""

    id: str
    path: Path
    split: str
    split_folder: str
    n_samples: int
    fields: dict[str, Any] = field(default_factory=dict)
    """Template fields: parsed path parts or metadata row values, plus split and stem."""
    row: dict[str, Any] | None = None
    """The full metadata row (`audiofolder` only)."""


class CorpusAdapter:
    """Base class. `items(split)` returns the split's source files sorted by relative path."""

    def __init__(self, config: DatasetConfig) -> None:
        self.config = config
        self._cache: dict[str, list[SourceItem]] = {}
        self.field_types: dict[str, pa.DataType] = {}
        """Arrow type of each metadata column (`audiofolder` only)."""

    def items(self, split: str) -> list[SourceItem]:
        if split not in self._cache:
            if split not in self.config.splits:
                raise ContractError(f"split {split!r} is not in dataset.splits")
            self._cache[split] = self._load(split)
        return self._cache[split]

    def index(self, splits: list[str]) -> dict[str, SourceItem]:
        """Items of the given splits by id."""
        return {item.id: item for split in splits for item in self.items(split)}

    def _load(self, split: str) -> list[SourceItem]:
        raise NotImplementedError

    def _check_audio(self, path: Path) -> int:
        """Frame count of `path`, after checking its sample rate and channel count."""
        try:
            info: Any = sf.info(str(path))
        except (RuntimeError, sf.LibsndfileError) as exc:
            raise ContractError(f"{path}: cannot read audio header: {exc}") from exc
        if info.samplerate != self.config.sample_rate:
            raise ContractError(
                f"{path}: {info.samplerate} Hz, expected {self.config.sample_rate} Hz"
                " (the builder never resamples)"
            )
        if info.channels != 1:
            raise ContractError(f"{path}: {info.channels} channels, expected mono")
        return int(info.frames)


def pattern_regex(pattern: str, split_folder: str) -> re.Pattern[str]:
    """Regex for a path pattern: each `{name}` is one path component, repeats must match."""
    regex: str = ""
    seen: set[str] = set()
    for literal, name, _, _ in Formatter().parse(pattern):
        regex += re.escape(literal)
        if name is None:
            continue
        if name == "split":
            regex += re.escape(split_folder)
        elif name in seen:
            regex += f"(?P={name})"
        else:
            regex += f"(?P<{name}>[^/]+?)"
            seen.add(name)
    return re.compile(regex)


class PathPatternAdapter(CorpusAdapter):
    """Fields come from the path relative to `root`, e.g. `{split}/{speaker}/{chapter}/...`."""

    def _excluded(self) -> set[str]:
        if self.config.exclude_list is None:
            return set()
        lines: list[str] = self.config.exclude_list.read_text(encoding="utf-8").splitlines()
        return {line.split(" ")[0] for line in lines if line.strip() and not line.startswith("#")}

    def _load(self, split: str) -> list[SourceItem]:
        pattern: str = self.config.path_pattern or ""
        folder: str = self.config.splits[split]
        regex: re.Pattern[str] = pattern_regex(pattern, folder)
        suffix: str = Path(pattern).suffix
        search_root: Path = self.config.root / folder if "{split}" in pattern else self.config.root
        excluded: set[str] = self._excluded()

        items: list[SourceItem] = []
        for path in sorted(search_root.rglob(f"*{suffix}")):
            relative: str = path.relative_to(self.config.root).as_posix()
            match: re.Match[str] | None = regex.fullmatch(relative)
            if match is None or path.stem in excluded:
                continue
            fields: dict[str, Any] = {
                **match.groupdict(),
                "split": split,
                "split_folder": folder,
                "stem": path.stem,
                "id": path.stem,
            }
            items.append(
                SourceItem(
                    id=path.stem,
                    path=path,
                    split=split,
                    split_folder=folder,
                    n_samples=self._check_audio(path),
                    fields=fields,
                )
            )
        if len({item.id for item in items}) != len(items):
            raise ContractError(f"{split}: file stems are not unique, ids would collide")
        return items


class AudioFolderAdapter(CorpusAdapter):
    """Rows of `metadata.parquet` files; `file_name` is relative to each metadata file.

    Every metadata file must have the same column types; their union is kept in
    `field_types` so the outputs can reuse them.
    """

    def __init__(self, config: DatasetConfig) -> None:
        super().__init__(config)
        self._seen_ids: dict[str, str] = {}
        self._schema_origin: str = ""
        self.metadata_file: str = config.metadata_file or "metadata.parquet"
        self._check_schema()

    def _check_schema(self) -> None:
        schema_path: Path = self.config.root.parent / ".provenance" / "schema.json"
        if not schema_path.is_file():
            if self.config.schema_version is not None:
                raise ContractError(f"{schema_path} not found (dataset.schema_version is set)")
            return
        try:
            schema: Any = json.loads(schema_path.read_text(encoding="utf-8"))
            found: Any = schema.get("schema_version")
            listed: Any = schema.get("metadata_file")
        except (json.JSONDecodeError, AttributeError) as exc:
            raise ContractError(f"{schema_path}: invalid JSON: {exc}") from exc
        if self.config.schema_version is not None and found != self.config.schema_version:
            raise ContractError(
                f"{schema_path}: schema_version {found}, config expects"
                f" {self.config.schema_version}"
            )
        if self.config.metadata_file is None and listed:
            self.metadata_file = str(listed)

    def _required_fields(self) -> list[str]:
        required: list[str] = ["file_name", self.config.id_field, *self.config.passthrough_fields]
        if self.config.labels.source == "words":
            required.append(self.config.labels.field)
        return required

    def _read(self, metadata_path: Path) -> list[dict[str, Any]]:
        try:
            rows, schema = read_table_rows(metadata_path)
        except (OSError, ValueError, pa.ArrowException) as exc:
            raise ContractError(f"{metadata_path}: cannot read: {exc}") from exc
        if schema is None:
            for name in rows[0] if rows else {}:
                self.field_types.setdefault(name, CLEAN_TYPES.get(name, pa.string()))
            return rows
        if missing := [n for n in self._required_fields() if n not in schema.names]:
            raise ContractError(f"{metadata_path}: missing columns {missing}")
        types: dict[str, pa.DataType] = {f.name: f.type for f in schema}
        for name, kind in types.items():
            known: pa.DataType | None = self.field_types.get(name)
            if known is not None and known != kind:
                raise ContractError(
                    f"{metadata_path}: column {name} is {kind}, but {known} in"
                    f" {self._schema_origin}; every metadata file needs the same types"
                )
            if known is None:
                self.field_types[name] = kind
                self._schema_origin = self._schema_origin or str(metadata_path)
        return rows

    def _load(self, split: str) -> list[SourceItem]:
        folder: str = self.config.splits[split]
        glob: str = self.config.metadata_glob.format(split=folder, metadata_file=self.metadata_file)
        required: list[str] = self._required_fields()
        sr: int = self.config.sample_rate

        items: list[SourceItem] = []
        for metadata_path in sorted(self.config.root.glob(glob)):
            for number, row in enumerate(self._read(metadata_path), start=1):
                where: str = f"{metadata_path} row {number}"
                if missing := [name for name in required if name not in row]:
                    raise ContractError(f"{where}: missing fields {missing}")
                if self.config.labels.source == "words" and row[self.config.labels.field] is None:
                    raise ContractError(f"{where}: `{self.config.labels.field}` is null")

                item_id: str = str(row[self.config.id_field])
                if item_id in self._seen_ids:
                    raise ContractError(
                        f"{where}: id {item_id} already used in {self._seen_ids[item_id]}"
                    )
                self._seen_ids[item_id] = where

                path: Path = metadata_path.parent / row["file_name"]
                if not path.is_file():
                    raise ContractError(f"{where}: {path} does not exist")
                n_samples: int = self._check_audio(path)
                if row.get("duration") is not None and (
                    abs(float(row["duration"]) * sr - n_samples) > 2
                ):
                    raise ContractError(
                        f"{where}: duration {row['duration']} does not match {n_samples} frames"
                    )
                fields: dict[str, Any] = {
                    **row,
                    "split": split,
                    "split_folder": folder,
                    "stem": path.stem,
                    "id": item_id,
                }
                items.append(
                    SourceItem(
                        id=item_id,
                        path=path,
                        split=split,
                        split_folder=folder,
                        n_samples=n_samples,
                        fields=fields,
                        row=row,
                    )
                )
        if self.config.order == "path":
            items.sort(key=lambda item: item.path.relative_to(self.config.root).as_posix())
        return items


def make_adapter(config: DatasetConfig) -> CorpusAdapter:
    if config.adapter == "path_pattern":
        return PathPatternAdapter(config)
    return AudioFolderAdapter(config)
