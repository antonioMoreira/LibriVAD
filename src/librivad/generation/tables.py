"""
Metadata tables: reading the input corpus's `metadata.parquet` and writing the outputs'.

AudioFolder infers the Arrow types of each metadata file separately, so a column that is
null in every row of one file (e.g. `dnsmos` for Spanish) would get type `null` there and
`float64` elsewhere, and `datasets.load_dataset` refuses to combine them. Parquet stores the
type even when every value is null, so every file is written with one explicit schema.
"""

import json
import os
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

WORDS_TYPE: pa.DataType = pa.list_(
    pa.struct(
        [
            ("word", pa.string()),
            ("start", pa.float64()),
            ("end", pa.float64()),
            ("probability", pa.float64()),
        ]
    )
)
SEGMENTS_TYPE: pa.DataType = pa.list_(
    pa.struct([("start", pa.float64()), ("end", pa.float64()), ("text", pa.string())])
)

CLEAN_TYPES: dict[str, pa.DataType] = {
    "file_name": pa.string(),
    "id": pa.string(),
    "sample_id": pa.string(),
    "language": pa.string(),
    "locale": pa.string(),
    "duration_bucket": pa.string(),
    "duration": pa.float64(),
    "text": pa.string(),
    "word_count": pa.int64(),
    "has_question": pa.bool_(),
    "speaker": pa.string(),
    "dnsmos": pa.float64(),
    "source_id": pa.string(),
    "source": pa.string(),
    "dataset": pa.string(),
    "component": pa.string(),
    "split_group": pa.string(),
    "words": WORDS_TYPE,
    "words_source": pa.string(),
    "words_reviewed": pa.bool_(),
}
"""Column types of skylar-dataset's clean `metadata.parquet` (schema_version 2)."""


def clean_schema() -> pa.Schema:
    return pa.schema([pa.field(name, kind, nullable=True) for name, kind in CLEAN_TYPES.items()])


def read_table_rows(path: Path) -> tuple[list[dict[str, Any]], pa.Schema | None]:
    """Rows of a metadata file, and its Arrow schema (None for JSON lines)."""
    if path.suffix == ".parquet":
        table: pa.Table = pq.read_table(path)
        return table.to_pylist(), table.schema
    lines: list[str] = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()], None


def read_rows(path: Path) -> list[dict[str, Any]]:
    """Rows of a metadata file, or an empty list when it does not exist."""
    return read_table_rows(path)[0] if path.is_file() else []


def _conform(value: Any, kind: pa.DataType) -> Any:
    """Keep only the struct fields the type declares, so extra keys never break a write."""
    if value is None:
        return None
    if pa.types.is_list(kind) and isinstance(value, list):
        return [_conform(v, kind.value_type) for v in value]
    if pa.types.is_struct(kind) and isinstance(value, dict):
        return {
            kind.field(i).name: _conform(value.get(kind.field(i).name), kind.field(i).type)
            for i in range(kind.num_fields)
        }
    return value


def write_rows(path: Path, rows: list[dict[str, Any]], schema: pa.Schema) -> None:
    """Write `rows` as Parquet with `schema`, atomically (`<name>.tmp` then rename)."""
    conformed: list[dict[str, Any]] = [
        {field.name: _conform(row.get(field.name), field.type) for field in schema} for row in rows
    ]
    table: pa.Table = pa.Table.from_pylist(conformed, schema=schema)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp: Path = path.with_name(path.name + ".tmp")
    pq.write_table(table, tmp)
    os.replace(tmp, path)
