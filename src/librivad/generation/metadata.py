"""
Builds the metadata row of each output: the condition, where speech and noise sit, the
gains applied, fields carried over from the source row, and the source's word timestamps
shifted to their new position. `output_schema` gives the one Arrow schema every output
`metadata.parquet` is written with.
"""

from pathlib import Path, PurePosixPath
from typing import Any

import pyarrow as pa

from librivad.generation.config import Config
from librivad.generation.corpus import SourceItem
from librivad.generation.planner import FileSpec
from librivad.generation.render import Rendered
from librivad.generation.tables import CLEAN_TYPES, SEGMENTS_TYPE, WORDS_TYPE

TIMESTAMP_FIELDS: tuple[str, ...] = ("words", "segments")

OUTPUT_TYPES: dict[str, pa.DataType] = {
    "file_name": pa.string(),
    "id": pa.string(),
    "clean_id": pa.string(),
    "scenario": pa.string(),
    "noise": pa.string(),
    "duration": pa.float64(),
    "source_duration": pa.float64(),
    "speech_start": pa.float64(),
    "speech_end": pa.float64(),
    "noise_start": pa.float64(),
    "noise_end": pa.float64(),
    "noise_ratio": pa.float64(),
    "noise_file": pa.string(),
    "noise_offset_s": pa.float64(),
    "noise_gain_db": pa.float64(),
    "clip_gain_db": pa.float64(),
    "measured_snr_db": pa.float64(),
    "segments": SEGMENTS_TYPE,
}
"""Types of the columns the builder adds; passthrough columns keep their input types."""

MEASURES: tuple[str, ...] = (
    "duration",
    "source_duration",
    "speech_start",
    "speech_end",
    "noise_start",
    "noise_end",
    "noise_ratio",
    "noise_file",
    "noise_offset_s",
    "noise_gain_db",
    "clip_gain_db",
    "measured_snr_db",
)


def output_schema(config: Config, input_types: dict[str, pa.DataType]) -> pa.Schema:
    """The schema of every output metadata file, in `build_row`'s column order.

    Passthrough and timestamp columns take their type from the input corpus
    (`input_types`), else from the clean contract; `snr_db` is int64 when every configured
    level is an integer.
    """
    passthrough: list[str] = config.dataset.passthrough_fields
    factors: list[str] = config.factors()[1:]
    names: list[str] = ["file_name", "id", "clean_id", "scenario", *factors]
    names += [n for n in ("language", "duration_bucket") if n in passthrough]
    names += [*MEASURES, *passthrough]
    names += [f for f in TIMESTAMP_FIELDS if f in input_types]

    levels: list[object] = [v for s in config.scenarios for v in s.conditions.get("snr_db", [])]
    integer_levels: bool = all(isinstance(v, int) and not isinstance(v, bool) for v in levels)

    fields: list[pa.Field] = []
    for name in dict.fromkeys(names):
        if name == "snr_db":
            kind: pa.DataType = pa.int64() if integer_levels else pa.float64()
        elif name == "segments":
            kind = SEGMENTS_TYPE
        elif name in OUTPUT_TYPES and name not in passthrough:
            kind = OUTPUT_TYPES[name]
        elif name == "words":
            kind = input_types.get(name, WORDS_TYPE)
        else:
            kind = input_types.get(name) or CLEAN_TYPES.get(name, pa.string())
        fields.append(pa.field(name, kind, nullable=True))
    return pa.schema(fields)


def shift_timestamps(value: Any, offset_s: float, drop_fields: list[str]) -> Any:
    """Copy of `value` with every `start`/`end` moved by `offset_s` and `drop_fields` removed.

    Works at any nesting level, so `segments[].words[]` is shifted as well. Times are
    rounded to 3 decimals.
    """
    if isinstance(value, list):
        return [shift_timestamps(v, offset_s, drop_fields) for v in value]
    if isinstance(value, dict):
        shifted: dict[str, Any] = {}
        for key, item in value.items():
            if key in drop_fields:
                continue
            if key in ("start", "end") and isinstance(item, int | float):
                shifted[key] = round(item + offset_s, 3)
            else:
                shifted[key] = shift_timestamps(item, offset_s, drop_fields)
        return shifted
    return value


def build_row(rendered: Rendered, items: dict[str, SourceItem], config: Config) -> dict[str, Any]:
    """The metadata row of one output, keys in the order of skylar's plan 3 §2."""
    spec: FileSpec = rendered.spec
    sr: int = config.dataset.sample_rate
    refs = spec.speech_items()
    first: SourceItem = items[refs[0].id]
    source_row: dict[str, Any] = first.row or {}
    speech_blocks = [b for b in spec.blocks if b.kind == "speech"]
    noise = spec.noise_block()
    assert spec.metadata_dir is not None and noise.noise_file is not None

    row: dict[str, Any] = {
        "file_name": str(PurePosixPath(spec.out_path).relative_to(spec.metadata_dir)),
        "id": spec.row_id,
        "clean_id": first.id,
        "scenario": spec.scenario,
        **spec.conditions,
    }
    for name in ("language", "duration_bucket"):
        if name in config.dataset.passthrough_fields:
            row[name] = source_row.get(name)
    row |= {
        "duration": round(spec.total_samples / sr, 4),
        "source_duration": round(spec.speech_samples / sr, 4),
        "speech_start": round(speech_blocks[0].start / sr, 4),
        "speech_end": round((speech_blocks[-1].start + speech_blocks[-1].length) / sr, 4),
        "noise_start": round(noise.start / sr, 4),
        "noise_end": round((noise.start + noise.length) / sr, 4),
        "noise_ratio": round(spec.noise_samples / spec.total_samples, 4),
        "noise_file": Path(noise.noise_file).name,
        "noise_offset_s": round((noise.noise_offset or 0) / sr, 4),
        "noise_gain_db": round(rendered.noise_gain_db, 3),
        "clip_gain_db": round(rendered.clip_gain_db, 3),
        "measured_snr_db": (
            round(rendered.measured_snr_db, 3) if rendered.measured_snr_db is not None else None
        ),
    }
    for name in config.dataset.passthrough_fields:
        row.setdefault(name, source_row.get(name))

    for field in TIMESTAMP_FIELDS:
        if not any(field in (items[ref.id].row or {}) for ref in refs):
            continue
        shifted: list[Any] = []
        for ref in refs:
            value: Any = (items[ref.id].row or {}).get(field) or []
            shifted += shift_timestamps(value, ref.offset / sr, config.output.drop_fields)
        row[field] = shifted
    return row
