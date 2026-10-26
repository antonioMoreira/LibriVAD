import io
import json
from pathlib import Path

import numpy as np
import pytest
import scipy.io.wavfile as wav
from numpy.typing import NDArray

from librivad.creation_scripts import prepare_dataset as prepare_module
from librivad.creation_scripts.create_labels_rvad import expand_frame_labels
from librivad.creation_scripts.prepare_dataset import convert_audio, prepare_sample, retry_wait
from librivad.dataset_adapter import ADAPTERS, DatasetAdapter, Sample, SkylarAdapter


def _row(**overrides: object) -> str:
    row: dict[str, object] = {
        "file_name": "11-20/abc.wav",
        "text": "hello",
        "duration": 12.0,
        "bucket": "11-20",
        "speaker": "S1",
        "dnsmos": None,
        "source_id": "src1",
        "source": None,
        "dataset": "amphion/Emilia-Dataset",
        "component": "Emilia",
        "locale": None,
    }
    row.update(overrides)
    return json.dumps(row)


def _wav_bytes(sr: int, n_samples: int) -> bytes:
    t: NDArray[np.float64] = np.arange(n_samples) / sr
    signal: NDArray[np.int16] = (np.sin(2 * np.pi * 440 * t) * 10000).astype(np.int16)
    buffer: io.BytesIO = io.BytesIO()
    wav.write(buffer, sr, signal)
    return buffer.getvalue()


def test_default_layout_matches_librispeech() -> None:
    adapter: DatasetAdapter = DatasetAdapter()
    sample: Sample = Sample(id="19-198-0001", source="", group="19", subgroup="198")

    path: Path = adapter.audio_path("files/datasets", "val", sample)
    assert path == Path("files/datasets/LibriSpeech/dev-clean/19/198/19-198-0001.wav")

    label: Path = adapter.label_path("files/Labels", "train", sample)
    assert label == Path("files/Labels/LibriSpeech/train-clean-100/19/198/19-198-0001.npy")


def test_default_adapter_does_not_list_samples() -> None:
    with pytest.raises(NotImplementedError):
        DatasetAdapter().list_samples()


def test_assign_split_is_deterministic_and_proportional() -> None:
    adapter: DatasetAdapter = DatasetAdapter()
    groups: list[str] = [f"speaker-{i}" for i in range(2000)]

    first: list[str] = [adapter.assign_split(g) for g in groups]
    assert first == [adapter.assign_split(g) for g in groups]

    for split, expected in (("train", 0.8), ("val", 0.1), ("test", 0.1)):
        assert first.count(split) / len(groups) == pytest.approx(expected, abs=0.04)


def test_skylar_registered() -> None:
    assert ADAPTERS["skylar"] is SkylarAdapter


def test_skylar_parse_metadata() -> None:
    adapter: SkylarAdapter = SkylarAdapter()
    lines: list[str] = [
        _row(),
        # FLEURS style: no speaker, grouped by source_id.
        _row(
            file_name="5-10/xyz.wav",
            bucket="5-10",
            speaker=None,
            source_id="1168",
            dataset="google/fleurs",
        ),
        "",
    ]

    samples: list[Sample] = adapter.parse_metadata("es", lines)

    assert [s.id for s in samples] == ["es_11-20_abc", "es_5-10_xyz"]
    assert samples[0].source == f"{adapter.repo_url}/es/11-20/abc.wav"
    assert samples[0].group == "es:amphion/Emilia-Dataset:S1"
    assert samples[1].group == "es:google/fleurs:1168"
    assert samples[1].subgroup == "5-10"


def test_skylar_ids_are_unique_when_stems_repeat_across_buckets() -> None:
    lines: list[str] = [
        _row(file_name="5-10/same.wav", bucket="5-10"),
        _row(file_name="11-20/same.wav", bucket="11-20"),
    ]

    samples: list[Sample] = SkylarAdapter().parse_metadata("es", lines)

    assert len({s.id for s in samples}) == 2


def test_same_group_always_lands_in_the_same_split() -> None:
    adapter: SkylarAdapter = SkylarAdapter()
    lines: list[str] = [_row(file_name=f"11-20/f{i}.wav") for i in range(20)]

    samples: list[Sample] = adapter.parse_metadata("en", lines)

    assert len({adapter.assign_split(s.group) for s in samples}) == 1


def test_convert_audio_resamples_to_16k_int16() -> None:
    audio: NDArray[np.int16] = convert_audio(_wav_bytes(24000, 24000))

    assert audio.dtype == np.int16
    assert len(audio) == pytest.approx(16000, abs=2)
    assert np.abs(audio).max() > 5000


def test_convert_audio_keeps_16k_untouched() -> None:
    audio: NDArray[np.int16] = convert_audio(_wav_bytes(16000, 8000))

    assert len(audio) == 8000


def test_prepare_sample_writes_and_skips_existing(tmp_path: Path) -> None:
    class FakeAdapter(DatasetAdapter):
        name = "Fake"

        def fetch(self, sample: Sample) -> bytes:
            return _wav_bytes(24000, 2400)

    adapter: FakeAdapter = FakeAdapter()
    sample: Sample = Sample(id="x_1", source="", group="g", subgroup="b")

    assert prepare_sample(adapter, sample, "train", tmp_path) is True
    out: Path = tmp_path / "Fake" / "train-clean-100" / "g" / "b" / "x_1.wav"
    sr: int
    data: NDArray[np.int16]
    sr, data = wav.read(out)
    assert sr == 16000
    assert len(data) == pytest.approx(1600, abs=2)
    assert not out.with_suffix(".part").exists()

    assert prepare_sample(adapter, sample, "train", tmp_path) is False


class _FakeResponse:
    def __init__(self, headers: dict[str, str]) -> None:
        self.headers: dict[str, str] = headers


class _TooManyRequests(Exception):
    def __init__(self, retry_after: str | None = None) -> None:
        super().__init__("429 Too Many Requests")
        headers: dict[str, str] = {} if retry_after is None else {"Retry-After": retry_after}
        self.response: _FakeResponse = _FakeResponse(headers)


def test_retry_wait_honours_retry_after_and_backs_off() -> None:
    assert retry_wait(_TooManyRequests("7"), 0) == 7.0
    assert retry_wait(_TooManyRequests("100000"), 0) == prepare_module.MAX_WAIT_SECONDS
    assert retry_wait(_TooManyRequests(), 0) == 2.0
    assert retry_wait(_TooManyRequests(), 2) == 8.0
    assert retry_wait(ValueError("no response attribute"), 1) == 4.0


def test_prepare_sample_retries_after_rate_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    waits: list[float] = []
    monkeypatch.setattr(prepare_module.time, "sleep", waits.append)

    class FlakyAdapter(DatasetAdapter):
        name = "Flaky"
        calls = 0

        def fetch(self, sample: Sample) -> bytes:
            self.calls += 1
            if self.calls <= 2:
                raise _TooManyRequests("3")
            return _wav_bytes(16000, 1600)

    adapter: FlakyAdapter = FlakyAdapter()
    sample: Sample = Sample(id="f_1", source="", group="g", subgroup="b")

    assert prepare_sample(adapter, sample, "train", tmp_path) is True
    assert adapter.calls == 3
    assert waits == [3.0, 3.0]


def test_prepare_sample_gives_up_after_all_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(prepare_module.time, "sleep", lambda _: None)

    class DownAdapter(DatasetAdapter):
        name = "Down"
        calls = 0

        def fetch(self, sample: Sample) -> bytes:
            self.calls += 1
            raise _TooManyRequests()

    adapter: DownAdapter = DownAdapter()
    sample: Sample = Sample(id="d_1", source="", group="g", subgroup="b")

    with pytest.raises(_TooManyRequests):
        prepare_sample(adapter, sample, "train", tmp_path)
    assert adapter.calls == prepare_module.RETRIES


def test_expand_frame_labels_matches_signal_length() -> None:
    frames: NDArray[np.int64] = np.array([0, 1, 1, 0])

    exact: NDArray[np.int16] = expand_frame_labels(frames, 640)
    assert exact.tolist() == [0] * 160 + [1] * 320 + [0] * 160

    trimmed: NDArray[np.int16] = expand_frame_labels(frames, 500)
    assert len(trimmed) == 500

    padded: NDArray[np.int16] = expand_frame_labels(frames, 700)
    assert len(padded) == 700
    assert padded[640:].sum() == 0
