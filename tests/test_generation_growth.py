"""A corpus that grows: with `dataset.order: metadata`, existing outputs keep their assignment;
outputs the plan no longer has are pruned."""

import copy
import hashlib
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
from generation_helpers import SR, speech_like

from librivad.generation.build_scenarios import run
from librivad.generation.config import config_from_dict
from librivad.generation.tables import clean_schema, write_rows


def snapshot(output: Path) -> dict[str, tuple[str, int]]:
    """Every output wav: its content hash and modification time."""
    return {
        p.relative_to(output).as_posix(): (
            hashlib.sha256(p.read_bytes()).hexdigest(),
            p.stat().st_mtime_ns,
        )
        for p in output.rglob("*.wav")
    }


def append_rows(clean: Path, count: int, seed: int = 500) -> None:
    """Add `count` files at the end of every train metadata file, as skylar's collect does."""
    for metadata in sorted((clean / "train").glob("*/metadata.parquet")):
        rows: list[dict[str, Any]] = pq.read_table(metadata).to_pylist()
        template: dict[str, Any] = rows[0]
        for _ in range(count):
            seed += 1
            item_id: str = f"{seed:016x}"
            audio, words = speech_like(12.0, seed)
            sf.write(str(metadata.parent / "10-15" / f"{item_id}.wav"), audio, SR, subtype="PCM_16")
            text: str = " ".join(w["word"] for w in words)
            rows.append(
                {
                    **template,
                    "file_name": f"10-15/{item_id}.wav",
                    "id": item_id,
                    "sample_id": f"{template['language']}_10-15_{item_id}",
                    "duration_bucket": "10-15",
                    "duration": round(len(audio) / SR, 4),
                    "text": text,
                    "word_count": len(words),
                    "words": words,
                }
            )
        write_rows(metadata, rows, clean_schema())


def growing_raw(setup: dict[str, Any]) -> dict[str, Any]:
    raw: dict[str, Any] = copy.deepcopy(setup["raw"])
    raw["dataset"]["order"] = "metadata"
    return raw


def test_growing_the_corpus_keeps_existing_outputs(skylar_setup: dict[str, Any]) -> None:
    raw = growing_raw(skylar_setup)
    first: dict[str, Any] = run(config_from_dict(raw), workers=2, progress=False)
    assert first["build"]["failed"] == 0
    before = snapshot(skylar_setup["output"])

    append_rows(skylar_setup["clean"], 3)
    second: dict[str, Any] = run(
        config_from_dict(raw), workers=2, progress=False, prune_unplanned=True
    )
    after = snapshot(skylar_setup["output"])
    assert second["build"]["unplanned"] == second["build"]["pruned"] == 0
    assert {k: after[k] for k in before} == before  # same bytes, never rewritten
    assert second["build"]["written"] == len(after) - len(before) > 0


def test_outputs_the_plan_dropped_are_pruned(skylar_setup: dict[str, Any]) -> None:
    raw = growing_raw(skylar_setup)
    run(config_from_dict(raw), workers=2, progress=False)
    metadata: Path = sorted((skylar_setup["clean"] / "train").glob("*/metadata.parquet"))[0]
    rows = pq.read_table(metadata).to_pylist()
    dropped: str = rows[0]["id"]
    write_rows(metadata, rows[1:], clean_schema())

    counted: dict[str, Any] = run(
        config_from_dict(raw), splits=["train"], progress=False, prune_unplanned=True
    )
    # A narrowed plan can't tell stale outputs from those outside it: nothing counted or deleted.
    assert counted["build"]["unplanned"] is None and counted["build"]["pruned"] == 0

    report: dict[str, Any] = run(
        config_from_dict(raw), workers=2, progress=False, prune_unplanned=True
    )
    noises, levels = len(raw["conditions"]["noise"]), len(raw["conditions"]["snr_db"])
    assert report["build"]["unplanned"] == report["build"]["pruned"] > 0
    assert report["build"]["pruned"] <= noises * levels * 2  # at most every output of one file
    leftover = [p for p in skylar_setup["output"].rglob("*.wav") if dropped in p.name]
    rows_left = [
        r
        for m in skylar_setup["output"].rglob("metadata.parquet")
        for r in pq.read_table(m).to_pylist()
    ]
    assert not leftover and all(r["clean_id"] != dropped for r in rows_left)
    assert (
        run(config_from_dict(raw), progress=False, prune_unplanned=True)["build"]["unplanned"] == 0
    )


def test_a_metadata_file_with_an_outdated_schema_is_rewritten(
    skylar_setup: dict[str, Any],
) -> None:
    """A config that dropped a level factor (`loudness_lu`) leaves its column in files the
    next build does not touch; the build rewrites them so every file has the same columns."""
    raw = growing_raw(skylar_setup)
    run(config_from_dict(raw), workers=2, progress=False)
    files: list[Path] = sorted(skylar_setup["output"].rglob("metadata.parquet"))
    current = pq.read_schema(files[0])
    old = pq.read_table(files[0])
    pq.write_table(old.append_column("loudness_lu", pa.nulls(len(old), pa.int64())), files[0])
    before = snapshot(skylar_setup["output"])

    report: dict[str, Any] = run(config_from_dict(raw), workers=2, progress=False)
    assert report["build"]["written"] == 0
    assert report["build"]["cleanup"]["schema_rewrites"] == 1
    assert all(pq.read_schema(f).equals(current, check_metadata=False) for f in files)
    assert pq.read_table(files[0]).num_rows == old.num_rows
    assert snapshot(skylar_setup["output"]) == before  # no wav rewritten
    again = run(config_from_dict(raw), progress=False)
    assert again["build"]["cleanup"]["schema_rewrites"] == 0


def test_outputs_named_flac_are_flac_with_the_same_samples(skylar_setup: dict[str, Any]) -> None:
    """The output path's extension picks the format: FLAC holds the same 16-bit samples."""
    raw = growing_raw(skylar_setup)
    run(config_from_dict(raw), workers=2, progress=False)
    flac_raw = copy.deepcopy(raw)
    flac_raw["output"]["root"] = str(skylar_setup["tmp"] / "flac")
    flac_raw["output"]["path"] = raw["output"]["path"].removesuffix(".wav") + ".flac"
    report: dict[str, Any] = run(config_from_dict(flac_raw), workers=2, progress=False)
    assert report["build"]["failed"] == 0
    wavs = sorted(skylar_setup["output"].rglob("*.wav"))
    flacs = sorted((skylar_setup["tmp"] / "flac").rglob("*.flac"))
    assert len(wavs) == len(flacs) > 0
    for wav, flac in zip(wavs, flacs, strict=True):
        assert sf.info(str(flac)).format == "FLAC" and sf.info(str(flac)).subtype == "PCM_16"
        assert (sf.read(str(wav), dtype="int16")[0] == sf.read(str(flac), dtype="int16")[0]).all()
    names = {
        r["file_name"]
        for m in (skylar_setup["tmp"] / "flac").rglob("metadata.parquet")
        for r in pq.read_table(m).to_pylist()
    }
    assert names and all(name.endswith(".flac") for name in names)
