"""Plan 5: one noise excerpt under the whole file (`noise_block.span: file`), levelled by SNR
or by loudness (`loudness.measure: lufs`)."""

import copy
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import pyloudnorm
import pytest
import soundfile as sf
from generation_helpers import SR, speech_like

from librivad.generation.build_scenarios import ConfigError, run
from librivad.generation.config import config_from_dict, format_template, format_value
from librivad.generation.loudness import integrated_loudness
from librivad.generation.noise import raised_cosine

SNRS: list[int] = [-5, 10]
LUS: list[int] = [-5, 10]
FADE: int = round(0.010 * SR)


def span_raw(setup: dict[str, Any]) -> dict[str, Any]:
    """Four scenarios like skylar's: two layouts, each levelled by SNR and by loudness."""
    raw: dict[str, Any] = copy.deepcopy(setup["raw"])
    raw["defaults"]["noise_block"]["span"] = "file"
    raw["defaults"]["constraints"] = {}
    raw["conditions"] = {"noise": ["babble", "office"]}
    base: str = "{split}/{scenario}/{noise}/%s/{language}"
    scenarios: list[dict[str, Any]] = []
    for layout_name, layout in (
        ("noise_first", ["noise", "speech"]),
        ("noise_last", ["speech", "noise"]),
    ):
        for measure, factor, folder, levels in (
            ("rms", "snr_db", "snr_{snr_db:02d}", SNRS),
            ("lufs", "loudness_lu", "lu_{loudness_lu:02d}", LUS),
        ):
            directory: str = base % folder
            scenarios.append(
                {
                    "name": f"{layout_name}_{'snr' if measure == 'rms' else 'lufs'}",
                    "layout": layout,
                    "loudness": {"measure": measure},
                    "conditions": {factor: levels},
                    "metadata_dir": directory,
                    "output_path": directory + "/{duration_bucket}/{id}.wav",
                }
            )
    raw["scenarios"] = scenarios
    raw["population"] = {
        "train": {"crossed": [], "assigned": ["scenario", "noise", "snr_db", "loudness_lu"]},
        "val": {"crossed": ["snr_db", "loudness_lu"], "assigned": ["scenario", "noise"]},
        "test": {"crossed": ["snr_db", "loudness_lu", "scenario"], "assigned": ["noise"]},
        "strata": ["language", "duration_bucket"],
    }
    return raw


@pytest.fixture
def span_built(skylar_setup: dict[str, Any]) -> dict[str, Any]:
    config = config_from_dict(span_raw(skylar_setup))
    report: dict[str, Any] = run(config, workers=4, progress=False)
    assert report["build"]["failed"] == 0, report["build"]["failures"]
    rows: list[tuple[Path, dict[str, Any]]] = [
        (path.parent, row)
        for path in sorted(skylar_setup["output"].rglob("metadata.parquet"))
        for row in pq.read_table(path).to_pylist()
    ]
    clean: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in skylar_setup["clean"].rglob("metadata.parquet"):
        for row in pq.read_table(path).to_pylist():
            clean[row["id"]] = (path.parent / row["file_name"], row)
    return {**skylar_setup, "report": report, "rows": rows, "clean_rows": clean}


def expected_parts(
    folder: Path, row: dict[str, Any], clean: dict[str, Any], noises: dict[str, str], split: str
) -> tuple[np.ndarray, np.ndarray]:
    """The faded clean speech at its place and the scaled excerpt, rebuilt from the row."""
    total: int = round(row["duration"] * SR)
    source, _ = clean[row["clean_id"]]
    audio: np.ndarray = sf.read(str(source), dtype="int16")[0].astype(np.float64)
    audio[:FADE] *= raised_cosine(FADE)
    audio[len(audio) - FADE :] *= raised_cosine(FADE)[::-1]
    speech: np.ndarray = np.zeros(total)
    start: int = round(row["speech_start"] * SR)
    speech[start : start + len(audio)] = audio
    noise_path: str = noises[row["noise"]].format(split=split)
    offset: int = round(row["noise_offset_s"] * SR)
    excerpt: np.ndarray = sf.read(noise_path, dtype="int16", start=offset, frames=total)[0]
    gain: float = 10 ** (row["noise_gain_db"] / 20)
    clip: float = 10 ** (row["clip_gain_db"] / 20)
    return speech * clip, excerpt.astype(np.float64) * gain * clip


# --- rendering ------------------------------------------------------------------------


def test_the_noise_runs_under_the_whole_file(span_built: dict[str, Any]) -> None:
    output: Path = span_built["output"]
    for folder, row in span_built["rows"]:
        split: str = folder.relative_to(output).parts[0]
        written: np.ndarray = sf.read(str(folder / row["file_name"]), dtype="int16")[0]
        speech, noise = expected_parts(
            folder, row, span_built["clean_rows"], span_built["noises"], split
        )
        # The noise and clip gains are stored to 0.001 dB, each a relative error up to
        # 5.8e-5, plus the rounding to int16 on both sides.
        tolerance: np.ndarray = 1 + 1.2e-4 * (np.abs(speech) + np.abs(noise))
        assert np.all(np.abs(written - np.round(speech + noise)) <= tolerance)
        assert (row["noise_start"], row["noise_end"]) == (0.0, row["duration"])


def test_both_levels_are_measured_and_the_chosen_one_is_met(span_built: dict[str, Any]) -> None:
    output: Path = span_built["output"]
    meter = pyloudnorm.Meter(SR)
    for folder, row in span_built["rows"]:
        split: str = folder.relative_to(output).parts[0]
        speech, noise = expected_parts(
            folder, row, span_built["clean_rows"], span_built["noises"], split
        )
        start: int = round(row["speech_start"] * SR)
        end: int = round(row["speech_end"] * SR)
        independent: float = meter.integrated_loudness(
            speech[start:end] / 32768
        ) - meter.integrated_loudness(noise / 32768)
        assert row["measured_loudness_lu"] == pytest.approx(independent, abs=0.1)
        assert row["measured_snr_db"] is not None
        if row["scenario"].endswith("_snr"):
            assert row["loudness_lu"] is None
            assert row["measured_snr_db"] == pytest.approx(row["snr_db"], abs=0.1)
        else:
            assert row["snr_db"] is None
            assert row["measured_loudness_lu"] == pytest.approx(row["loudness_lu"], abs=0.1)


def test_every_level_of_a_file_shares_the_excerpt(span_built: dict[str, Any]) -> None:
    offsets: dict[tuple[str, str, str], set[float]] = defaultdict(set)
    for _, row in span_built["rows"]:
        offsets[(row["clean_id"], row["scenario"], row["noise"])].add(row["noise_offset_s"])
    assert offsets and all(len(found) == 1 for found in offsets.values())


def test_words_follow_the_speech(span_built: dict[str, Any]) -> None:
    for _, row in span_built["rows"]:
        _, clean = span_built["clean_rows"][row["clean_id"]]
        shift: float = row["speech_start"]
        assert [w["start"] for w in row["words"]] == pytest.approx(
            [round(w["start"] + shift, 3) for w in clean["words"]], abs=1e-3
        )


def test_population_with_two_level_factors(span_built: dict[str, Any]) -> None:
    output: Path = span_built["output"]
    per_clean: dict[str, Counter[str]] = defaultdict(Counter)
    for folder, row in span_built["rows"]:
        per_clean[folder.relative_to(output).parts[0]][row["clean_id"]] += 1
    assert set(per_clean["train"].values()) == {1}
    assert set(per_clean["val"].values()) == {len(SNRS)}  # one scenario, all its levels
    assert set(per_clean["test"].values()) == {2 * len(SNRS) + 2 * len(LUS)}
    levels = {(r["snr_db"], r["loudness_lu"]) for _, r in span_built["rows"]}
    assert {lu for _, lu in levels if lu is not None} == set(LUS)


def test_negative_levels_get_padded_folder_names(span_built: dict[str, Any]) -> None:
    folders = {
        folder.relative_to(span_built["output"]).parts[3] for folder, _ in span_built["rows"]
    }
    assert {"snr_-05", "snr_10", "lu_-05", "lu_10"} <= folders


def test_check_only_measures_the_written_files(span_built: dict[str, Any]) -> None:
    config = config_from_dict(span_raw(span_built))
    report: dict[str, Any] = run(config, check_only=True, progress=False)
    assert report["check"]["problems"] == 0, report["check"]["details"]


# --- units ----------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["noise", "speech", "tone"])
def test_loudness_matches_pyloudnorm(kind: str) -> None:
    rng = np.random.default_rng(3)
    if kind == "noise":
        signal: np.ndarray = rng.normal(0, 3000, 20 * SR)
    elif kind == "speech":
        signal = speech_like(12.0, 5)[0].astype(np.float64)
    else:
        signal = 8000 * np.sin(2 * np.pi * 1000 * np.arange(5 * SR) / SR)
    ours: float = integrated_loudness(signal, SR)
    assert ours == pytest.approx(pyloudnorm.Meter(SR).integrated_loudness(signal / 32768), abs=0.1)


def test_loudness_of_silence_and_short_signals_is_minus_infinity() -> None:
    assert integrated_loudness(np.zeros(5 * SR), SR) == float("-inf")
    assert integrated_loudness(np.ones(SR // 10), SR) == float("-inf")


def test_negative_integers_keep_two_digits() -> None:
    assert [format_value(v) for v in (-5, 0, 5, -12)] == ["-05", "00", "05", "-12"]
    assert format_template("snr_{snr_db:02d}", {"snr_db": -5}) == "snr_-05"
    assert format_template("snr_{snr_db:02d}", {"snr_db": 20}) == "snr_20"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"loudness": {"measure": "lufs"}}, "needs conditions.loudness_lu"),
        ({"conditions": {"snr_db": [0], "loudness_lu": [0]}}, "excludes loudness_lu"),
        ({"layout": ["speech"], "noise_block": {"span": "block"}}, "exactly one `noise`"),
    ],
)
def test_span_and_loudness_rules(
    skylar_setup: dict[str, Any], change: dict[str, Any], message: str
) -> None:
    raw: dict[str, Any] = span_raw(skylar_setup)
    raw["scenarios"][0] = {**raw["scenarios"][0], **change}
    with pytest.raises(ConfigError, match=message):
        config_from_dict(raw)


def test_a_layout_of_speech_only_is_allowed_with_span_file(skylar_setup: dict[str, Any]) -> None:
    raw: dict[str, Any] = span_raw(skylar_setup)
    raw["scenarios"] = [{**raw["scenarios"][0], "layout": ["speech"]}]
    raw["population"] = None
    config = config_from_dict(raw)
    report: dict[str, Any] = run(config, splits=["val"], workers=2, progress=False)
    assert report["build"]["failed"] == 0
    for path in skylar_setup["output"].rglob("metadata.parquet"):
        for row in pq.read_table(path).to_pylist():
            assert row["noise_ratio"] == 0 and row["speech_start"] == 0
            assert row["duration"] == row["source_duration"]
