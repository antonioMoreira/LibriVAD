import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf
import yaml
from generation_helpers import SR

from librivad.generation import build_scenarios
from librivad.generation.build_scenarios import main
from librivad.generation.render import rms

TOTAL: int = 30 * SR


def write_config(skylar_setup: dict[str, Any], raw: dict[str, Any] | None = None) -> Path:
    path: Path = skylar_setup["tmp"] / "scenarios.yaml"
    path.write_text(yaml.safe_dump(raw or skylar_setup["raw"]), encoding="utf-8")
    return path


def rows_by_folder(output: Path) -> dict[Path, list[dict[str, Any]]]:
    return {
        path.parent: pq.read_table(path).to_pylist()
        for path in sorted(output.rglob("metadata.parquet"))
    }


def clean_rows(clean: Path) -> dict[str, tuple[Path, dict[str, Any]]]:
    rows: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in clean.rglob("metadata.parquet"):
        for row in pq.read_table(path).to_pylist():
            rows[row["id"]] = (path.parent / row["file_name"], row)
    return rows


@pytest.fixture
def built(skylar_setup: dict[str, Any]) -> dict[str, Any]:
    report: Path = skylar_setup["tmp"] / "report.json"
    config: Path = write_config(skylar_setup)
    assert main(["--config", str(config), "--report", str(report), "--workers", "4"]) == 0
    return {**skylar_setup, "report": json.loads(report.read_text()), "config": config}


def test_layout_and_metadata(built: dict[str, Any]) -> None:
    output: Path = built["output"]
    folders = rows_by_folder(output)
    for folder, rows in folders.items():
        split, scenario, noise, level, language = folder.relative_to(output).parts
        assert split in ("train", "val", "test") and level.startswith("snr_")
        for row in rows:
            assert (folder / row["file_name"]).is_file()
            assert row["scenario"] == scenario and row["noise"] == noise
            assert f"snr_{row['snr_db']:02d}" == level and row["language"] == language
            assert row["file_name"].split("/")[0] == row["duration_bucket"]
            assert isinstance(row["duration"], float) and row["duration"] == 30.0
            assert row["id"] == f"{row['clean_id']}-{scenario}-{noise}-{row['snr_db']:02d}"
    wavs: list[Path] = sorted(output.rglob("*.wav"))
    assert len(wavs) == sum(len(rows) for rows in folders.values())
    assert all(path.parent.parent in folders for path in wavs)


def test_population_counts(built: dict[str, Any]) -> None:
    per_split: Counter[str] = Counter()
    per_clean: dict[str, Counter[str]] = {}
    for folder, rows in rows_by_folder(built["output"]).items():
        split: str = folder.relative_to(built["output"]).parts[0]
        for row in rows:
            per_split[split] += 1
            per_clean.setdefault(split, Counter())[row["clean_id"]] += 1
    items_per_split: int = 2 * 3  # 2 languages x 3 buckets
    assert per_split == {
        "train": items_per_split,
        "val": items_per_split * 3,
        "test": items_per_split * 6,
    }
    assert set(per_clean["val"].values()) == {3} and set(per_clean["test"].values()) == {6}


def test_audio_speech_and_timestamps(built: dict[str, Any]) -> None:
    clean = clean_rows(built["clean"])
    checked_identical: int = 0
    for folder, rows in rows_by_folder(built["output"]).items():
        for row in rows:
            audio: np.ndarray = sf.read(str(folder / row["file_name"]), dtype="int16")[0]
            assert len(audio) == TOTAL
            source_path, source_row = clean[row["clean_id"]]
            source: np.ndarray = sf.read(str(source_path), dtype="int16")[0]
            start: int = round(row["speech_start"] * SR)
            assert row["speech_end"] - row["speech_start"] == pytest.approx(len(source) / SR)
            if row["clip_gain_db"] == 0.0:
                assert np.array_equal(audio[start : start + len(source)], source)
                checked_identical += 1
            for shifted, original in zip(row["words"], source_row["words"], strict=True):
                assert shifted["start"] == pytest.approx(
                    original["start"] + row["speech_start"], abs=5e-4
                )
                assert shifted["end"] == pytest.approx(
                    original["end"] + row["speech_start"], abs=5e-4
                )
            assert abs(row["measured_snr_db"] - row["snr_db"]) <= 0.1
            assert row["noise_ratio"] == pytest.approx(1 - len(source) / TOTAL, abs=1e-4)
    assert checked_identical > 0


def test_levels_share_the_excerpt(built: dict[str, Any]) -> None:
    blocks: dict[tuple[str, str, str], list[np.ndarray]] = {}
    for folder, rows in rows_by_folder(built["output"]).items():
        split = folder.relative_to(built["output"]).parts[0]
        if split == "train":
            continue
        for row in rows:
            audio = sf.read(str(folder / row["file_name"]), dtype="int16")[0].astype(float)
            start, end = round(row["noise_start"] * SR) + 160, round(row["noise_end"] * SR) - 160
            blocks.setdefault((row["clean_id"], row["scenario"], row["noise"]), []).append(
                audio[start:end]
            )
    for excerpts in blocks.values():
        assert len(excerpts) == 3
        reference = excerpts[0] / rms(excerpts[0])
        for other in excerpts[1:]:
            correlation = float(np.mean(reference * other / rms(other)))
            assert correlation >= 0.9999


def _digests(output: Path) -> dict[str, str]:
    return {
        str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.rglob("*"))
        if p.is_file()
    }


def test_resume_and_worker_count_do_not_change_outputs(
    built: dict[str, Any], tmp_path: Path
) -> None:
    before: dict[str, str] = _digests(built["output"])
    report: Path = tmp_path / "again.json"
    assert main(["--config", str(built["config"]), "--report", str(report)]) == 0
    again: dict[str, Any] = json.loads(report.read_text())["build"]
    assert again["written"] == 0 and again["already_done"] == built["report"]["plan"]["outputs"]
    assert _digests(built["output"]) == before

    raw: dict[str, Any] = copy.deepcopy(built["raw"])
    raw["output"]["root"] = str(tmp_path / "single_worker")
    config: Path = tmp_path / "single.yaml"
    config.write_text(yaml.safe_dump(raw))
    assert main(["--config", str(config), "--workers", "1"]) == 0
    assert _digests(tmp_path / "single_worker") == before


def test_interrupted_build_is_cleaned(built: dict[str, Any]) -> None:
    output: Path = built["output"]
    metadata: Path = next(output.rglob("metadata.parquet"))
    table: pa.Table = pq.read_table(metadata)
    lost: dict[str, Any] = table.slice(0, 1).to_pylist()[0]
    pq.write_table(table.slice(1), metadata)  # its wav now has no row
    (output / "stale.wav.tmp").write_bytes(b"partial")
    assert main(["--config", str(built["config"])]) == 0
    assert not (output / "stale.wav.tmp").exists()
    assert len(list((output / ".orphans").rglob("*.wav"))) == 1
    restored = [row["id"] for row in pq.read_table(metadata).to_pylist()]
    assert lost["id"] in restored


def test_dry_run_writes_the_plan(skylar_setup: dict[str, Any]) -> None:
    report: Path = skylar_setup["tmp"] / "dry" / "report.json"
    assert (
        main(["--config", str(write_config(skylar_setup)), "--dry-run", "--report", str(report)])
        == 0
    )
    summary: dict[str, Any] = json.loads(report.read_text())
    plan_lines: list[str] = (report.parent / "plan.jsonl").read_text().splitlines()
    assert len(plan_lines) == summary["plan"]["outputs"] == 6 + 18 + 36
    assert summary["librivad"]["commit"]
    assert not skylar_setup["output"].exists()


def test_check_finds_a_damaged_output(built: dict[str, Any]) -> None:
    assert main(["--config", str(built["config"]), "--check"]) == 0
    wav: Path = next(built["output"].rglob("*.wav"))
    sf.write(str(wav), np.zeros(TOTAL, dtype=np.int16), SR, subtype="PCM_16")
    assert main(["--config", str(built["config"]), "--check"]) == 1


def test_exit_codes(skylar_setup: dict[str, Any]) -> None:
    raw: dict[str, Any] = copy.deepcopy(skylar_setup["raw"])
    raw["conditions"]["snr_db"] = [-5]
    assert main(["--config", str(write_config(skylar_setup, raw))]) == build_scenarios.EXIT_CONFIG

    raw = copy.deepcopy(skylar_setup["raw"])
    raw["output"]["limits"]["max_files_total"] = 5
    assert main(["--config", str(write_config(skylar_setup, raw))]) == build_scenarios.EXIT_LIMITS
    assert not skylar_setup["output"].exists()

    wav: Path = next(skylar_setup["clean"].rglob("*.wav"))
    audio = sf.read(str(wav), dtype="int16")[0]
    sf.write(str(wav), audio, 24000, subtype="PCM_16")
    assert main(["--config", str(write_config(skylar_setup))]) == build_scenarios.EXIT_CONTRACT


def test_schema_version_is_checked(skylar_setup: dict[str, Any]) -> None:
    schema: Path = skylar_setup["clean"].parent / ".provenance" / "schema.json"
    schema.write_text(json.dumps({"schema_version": 1}))
    assert (
        main(["--config", str(write_config(skylar_setup)), "--dry-run"])
        == build_scenarios.EXIT_CONTRACT
    )


def test_every_output_file_has_one_schema(built: dict[str, Any]) -> None:
    """Columns that are null in a whole folder keep their type (AudioFolder needs that)."""
    schemas = {pq.read_schema(p) for p in built["output"].rglob("metadata.parquet")}
    assert len(schemas) == 1
    schema = schemas.pop()
    assert schema.field("snr_db").type == pa.int64()
    assert schema.field("dnsmos").type == pa.float64()
    assert schema.field("speaker").type == pa.string()
    assert (
        schema.field("words").type
        == pq.read_schema(next(built["clean"].rglob("metadata.parquet"))).field("words").type
    )
    es_rows = [
        r
        for p in built["output"].rglob("es/metadata.parquet")
        for r in pq.read_table(p).to_pylist()
    ]
    assert es_rows and all(r["dnsmos"] is None for r in es_rows)


def test_silent_sources_are_skipped_not_failed(skylar_setup: dict[str, Any]) -> None:
    """Regression: 7 real FLEURS rows were digital silence (peak 2, RMS 0.7 LSB) with aligned
    words. Their noise rounded to nothing (measured 237 dB at a 15 dB target)."""
    metadata: Path = skylar_setup["clean"] / "train" / "es" / "metadata.parquet"
    row: dict[str, Any] = pq.read_table(metadata).to_pylist()[0]
    rng = np.random.default_rng(3)
    silent = rng.integers(-2, 3, round(row["duration"] * SR)).astype(np.int16)
    sf.write(str(metadata.parent / row["file_name"]), silent, SR, subtype="PCM_16")

    report: Path = skylar_setup["tmp"] / "silent.json"
    assert main(["--config", str(write_config(skylar_setup)), "--report", str(report)]) == 0
    summary: dict[str, Any] = json.loads(report.read_text())
    assert summary["build"]["failed"] == 0
    assert summary["plan"]["skipped"] == {"train/too_quiet": 1}
    outputs = [r for rows in rows_by_folder(skylar_setup["output"]).values() for r in rows]
    assert row["id"] not in {r["clean_id"] for r in outputs}
