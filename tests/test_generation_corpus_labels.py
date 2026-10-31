import copy
import glob
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf
from generation_helpers import SR

from librivad.generation import librispeech_functions as lf
from librivad.generation.build_scenarios import make_noises
from librivad.generation.config import DatasetConfig, config_from_dict
from librivad.generation.corpus import (
    ContractError,
    PathPatternAdapter,
    make_adapter,
    pattern_regex,
)
from librivad.generation.labels import load_labels, textgrid_label, words_label
from librivad.generation.planner import Planner
from librivad.generation.render import Rendered, Renderer, clip_gain
from librivad.generation.tables import clean_schema, write_rows

REPO: Path = Path(__file__).resolve().parents[1]
ALIGNMENTS: Path = REPO / "files/Forced_alignments/librispeech_alignments"
LIBRISPEECH: Path = REPO / "files/datasets/LibriSpeech"


def test_pattern_regex_requires_repeated_fields_to_match() -> None:
    regex = pattern_regex("{split}/{speaker}/{chapter}/{speaker}-{chapter}-{utt}.wav", "dev-clean")
    match = regex.fullmatch("dev-clean/84/121123/84-121123-0001.wav")
    assert match is not None and match.groupdict() == {
        "speaker": "84",
        "chapter": "121123",
        "utt": "0001",
    }
    assert regex.fullmatch("dev-clean/84/121123/85-121123-0001.wav") is None


def test_path_pattern_adapter_reads_fields(tmp_path: Path) -> None:
    for stem in ("84-121123-0001", "84-121123-0000", "174-50561-0000"):
        speaker, chapter, _ = stem.split("-")
        path: Path = tmp_path / "dev-clean" / speaker / chapter / f"{stem}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), np.zeros(SR, dtype=np.int16), SR, subtype="PCM_16")
    (tmp_path / "unaligned.txt").write_text("# dev-clean\n84-121123-0001 reason\n")
    config = DatasetConfig(
        name="toy",
        adapter="path_pattern",
        root=tmp_path,
        splits={"val": "dev-clean"},
        path_pattern="{split}/{speaker}/{chapter}/{speaker}-{chapter}-{utt}.wav",
        exclude_list=tmp_path / "unaligned.txt",
    )
    items = PathPatternAdapter(config).items("val")
    assert [i.id for i in items] == ["174-50561-0000", "84-121123-0000"]
    assert items[1].fields["chapter"] == "121123" and items[1].n_samples == SR


def test_words_label_boundaries_and_bridging() -> None:
    words: list[dict[str, Any]] = [
        {"word": "a", "start": 0.0001, "end": 0.0009375},  # samples 1.6 .. 15.0
        {"word": "b", "start": 0.002, "end": 0.0029},  # samples 32.0 .. 46.4
    ]
    label = words_label(words, 64, SR)
    assert np.flatnonzero(label).tolist() == [*range(1, 15), *range(32, 47)]
    bridged = words_label(words, 64, SR, bridge_gaps_ms=1.2)
    assert np.flatnonzero(bridged).tolist() == list(range(1, 47))
    assert words_label(words, 64, SR, bridge_gaps_ms=1.0).sum() == label.sum()


@pytest.mark.skipif(not ALIGNMENTS.is_dir(), reason="LibriSpeech alignments not downloaded")
def test_textgrid_labels_match_create_labels() -> None:
    from librivad.generation.creation_scripts.create_labels import _generate_label_from_timeframes

    paths: list[str] = sorted(glob.glob(str(LIBRISPEECH / "dev-clean/*/*/*.wav")))[:5]
    assert paths
    for path in paths:
        timeframes, words = lf.find_timeframes(path, ALIGNMENTS / "dev-clean")
        legacy = _generate_label_from_timeframes(timeframes, words)
        speaker, chapter = Path(path).parts[-3], Path(path).parts[-2]
        grid: Path = ALIGNMENTS / "dev-clean" / speaker / chapter / f"{Path(path).stem}.TextGrid"
        ours = textgrid_label(grid, "words", "", SR)
        assert np.array_equal(ours, legacy.astype(np.int8))


def _broken_corpus(skylar_setup: dict[str, Any], edit: Any) -> None:
    metadata: Path = next(skylar_setup["clean"].rglob("metadata.parquet"))
    rows: list[dict[str, Any]] = pq.read_table(metadata).to_pylist()
    edit(rows)
    kept = [f for f in clean_schema() if all(f.name in r for r in rows)]
    write_rows(metadata, rows, pa.schema(kept))


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda rows: [r.pop("language") for r in rows], "missing columns"),
        (lambda rows: rows[1].update(id=rows[0]["id"]), "already used"),
        (lambda rows: rows[0].update(words=None), "is null"),
        (lambda rows: rows[0].update(duration=rows[0]["duration"] + 0.5), "does not match"),
        (lambda rows: rows[0].update(file_name="05-10/missing.wav"), "does not exist"),
    ],
)
def test_adapter_rejects_broken_rows(skylar_setup: dict[str, Any], edit: Any, message: str) -> None:
    _broken_corpus(skylar_setup, edit)
    config = config_from_dict(skylar_setup["raw"])
    adapter = make_adapter(config.dataset)
    with pytest.raises(ContractError, match=message):
        for split in ("train", "val", "test"):
            adapter.items(split)


def test_words_labels_from_rows(skylar_setup: dict[str, Any]) -> None:
    config = config_from_dict(skylar_setup["raw"])
    item = make_adapter(config.dataset).items("train")[0]
    label = load_labels(item, config.dataset)
    assert len(label) == item.n_samples
    assert item.row is not None
    first = item.row["words"][0]
    assert label[int(first["start"] * SR) + 10] == 1 and label[int(first["start"] * SR) - 10] == 0


def test_clip_guard_is_shared_by_a_group(skylar_setup: dict[str, Any]) -> None:
    for wav in skylar_setup["clean"].rglob("*.wav"):
        audio = sf.read(str(wav), dtype="int16")[0].astype(float)
        sf.write(
            str(wav), np.clip(audio * 3.4, -32768, 32767).astype(np.int16), SR, subtype="PCM_16"
        )
    raw: dict[str, Any] = copy.deepcopy(skylar_setup["raw"])
    config = config_from_dict(raw)
    adapter = make_adapter(config.dataset)
    noises = make_noises(config, adapter)
    specs = Planner(config, adapter, noises).plan(["test"]).specs
    renderer = Renderer(config, adapter.index(["test"]), noises)
    group = [s for s in specs if s.group == specs[0].group]
    results = renderer.render_group(group)
    assert all(isinstance(r, Rendered) for r in results)
    rendered = [r for r in results if isinstance(r, Rendered)]
    gains = {r.clip_gain_db for r in rendered}
    assert len(gains) == 1 and gains.pop() < 0
    assert any(r.clip_triggered for r in rendered)
    for r in rendered:
        assert abs((r.measured_snr_db or 0) - (r.spec.snr_db or 0)) <= 0.1
    speech = [
        r.audio[r.spec.blocks[1].start :] for r in rendered if r.spec.scenario == "noise_first"
    ]
    assert all(np.array_equal(speech[0], s) for s in speech[1:])


def test_adapter_rejects_metadata_files_with_different_types(skylar_setup: dict[str, Any]) -> None:
    metadata: Path = next(skylar_setup["clean"].rglob("es/metadata.parquet"))
    table: pa.Table = pq.read_table(metadata)
    pq.write_table(
        table.drop_columns(["dnsmos"]).append_column("dnsmos", pa.nulls(len(table))), metadata
    )
    config = config_from_dict(skylar_setup["raw"])
    adapter = make_adapter(config.dataset)
    with pytest.raises(ContractError, match="needs the same types"):
        for split in ("train", "val", "test"):
            adapter.items(split)


def test_metadata_file_name_comes_from_the_schema_file(skylar_setup: dict[str, Any]) -> None:
    for path in skylar_setup["clean"].rglob("metadata.parquet"):
        path.rename(path.with_name("rows.parquet"))
    schema: Path = skylar_setup["clean"].parent / ".provenance" / "schema.json"
    schema.write_text(json.dumps({"schema_version": 1, "metadata_file": "rows.parquet"}))
    adapter = make_adapter(config_from_dict(skylar_setup["raw"]).dataset)
    assert len(adapter.items("train")) == 6


def test_clip_gain_acts_only_when_int16_would_clip() -> None:
    assert clip_gain(np.array([32767.4, -32768.4])) == 1.0
    assert clip_gain(np.array([32767.6, 0.0])) == pytest.approx(32767 / 32767.6)
    assert clip_gain(np.array([0.0, -32769.0])) == pytest.approx(32768 / 32769)
    assert clip_gain(np.array([40000.0, -50000.0])) == pytest.approx(32768 / 50000)


def test_full_scale_sources_stay_untouched_at_low_noise(skylar_setup: dict[str, Any]) -> None:
    """Real sources often peak at int16 full scale; quiet noise must not trigger the guard."""
    for wav in skylar_setup["clean"].rglob("*.wav"):
        audio = sf.read(str(wav), dtype="int16")[0].astype(float)
        audio = np.round(audio / np.max(np.abs(audio)) * 32768).clip(-32768, 32767)
        sf.write(str(wav), audio.astype(np.int16), SR, subtype="PCM_16")
    raw: dict[str, Any] = copy.deepcopy(skylar_setup["raw"])
    raw["conditions"]["snr_db"] = [35]
    config = config_from_dict(raw)
    adapter = make_adapter(config.dataset)
    noises = make_noises(config, adapter)
    specs = Planner(config, adapter, noises).plan(["test"]).specs
    items = adapter.index(["test"])
    renderer = Renderer(config, items, noises)
    for spec in specs:
        (result,) = renderer.render_group([spec])
        assert isinstance(result, Rendered)
        assert result.clip_gain_db == 0.0 and not result.clip_triggered
        speech = next(b for b in spec.blocks if b.kind == "speech")
        source = sf.read(str(items[spec.unit].path), dtype="int16")[0]
        assert np.array_equal(result.audio[speech.start : speech.start + speech.length], source)
