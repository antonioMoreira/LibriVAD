from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from numpy.typing import NDArray

from librivad.composer import (
    MAX_DURATION_SECONDS,
    SAMPLE_RATE,
    Clip,
    ComposeConfig,
    NoiseLayer,
    Segment,
    amplitude_to_db,
    compose_clip,
    fit_label,
    rms,
    take_wrapped,
)
from librivad.creation_scripts.create_composed_dataset import (
    build_parser,
    load_config_file,
    resolve_config,
)

SECOND: int = SAMPLE_RATE


def _tone(seconds: float, amplitude: float = 0.1) -> NDArray[np.float32]:
    t: NDArray[np.float64] = np.arange(int(seconds * SECOND)) / SECOND
    return (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def _segment(seconds: float = 4.0, silent_edge: float = 0.5) -> Segment:
    """A tone with `silent_edge` seconds of non-speech (label 0) at each end."""
    audio: NDArray[np.float32] = _tone(seconds)
    label: NDArray[np.int16] = np.ones(len(audio), dtype=np.int16)
    edge: int = int(silent_edge * SECOND)
    label[:edge] = 0
    label[-edge:] = 0
    return Segment(source="seg.wav", audio=audio, label=label)


def _noise_source(seconds: float = 20.0, seed: int = 0) -> NDArray[np.float32]:
    rng: np.random.Generator = np.random.default_rng(seed)
    return (rng.standard_normal(int(seconds * SECOND)) * 0.05).astype(np.float32)


NOISES: dict[str, NDArray[np.float32]] = {"A": _noise_source(seed=1), "B": _noise_source(seed=2)}


def _compose(config: ComposeConfig, seed: int = 0, pool: tuple[str, ...] = ("A", "B")) -> Clip:
    return compose_clip(
        np.random.default_rng(seed),
        config,
        lambda rng: _segment(),
        lambda name: NOISES[name],
        pool,
    )


def _db(signal: NDArray[Any]) -> float:
    return amplitude_to_db(rms(signal))


def test_default_config_is_valid() -> None:
    ComposeConfig().validate()
    assert ComposeConfig().duration[1] <= MAX_DURATION_SECONDS


@pytest.mark.parametrize(
    "changes",
    [
        {"duration": (10.0, 31.0)},
        {"duration": (20.0, 10.0)},
        {"mode": "other"},
        {"normalize": "other"},
        {"pause_prob": 1.5},
        {"noise_count": (0, 3)},
        {"noises": (NoiseLayer("A", -30), NoiseLayer("B", -30), NoiseLayer("C", -30))},
    ],
)
def test_invalid_config_is_rejected(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        ComposeConfig().replace(**changes).validate()


def test_from_dict_converts_and_rejects_unknown_keys() -> None:
    config: ComposeConfig = ComposeConfig.from_dict(
        {"duration": [5, 8], "noises": [{"name": "A", "db": -30}], "seed": 3}
    )
    assert config.duration == (5, 8)
    assert config.noises == (NoiseLayer("A", -30.0),)

    with pytest.raises(ValueError):
        ComposeConfig.from_dict({"not_a_parameter": 1})


EXAMPLE_CONFIG: Path = Path(__file__).parent.parent / "configs" / "composer_example.yaml"


def test_example_yaml_is_valid_and_matches_the_defaults() -> None:
    config: ComposeConfig = ComposeConfig.from_dict(load_config_file(EXAMPLE_CONFIG))

    assert config == ComposeConfig()


def test_example_yaml_alternatives_are_valid_when_uncommented() -> None:
    lines: list[str] = EXAMPLE_CONFIG.read_text().splitlines()
    start: int = next(i for i, line in enumerate(lines) if line.startswith("# --- Alternatives"))

    uncommented: list[str] = []
    for line in lines[start + 1 :]:
        # Only the lines of the alternatives: the examples, not the explanations.
        if line.startswith(("# noises:", "#   - {", "# mode:", "# noise_count:")):
            uncommented.append(line[2:])
    text: str = "\n".join(uncommented)
    assert "noises:" in text and "mode:" in text

    # The two alternatives are independent, so they are tried one at a time.
    fixed: str = "\n".join(line for line in uncommented if not line.startswith(("mode", "noise_c")))
    only_noise: str = "\n".join(
        line for line in uncommented if line.startswith(("mode", "noise_c"))
    )

    two_noises: ComposeConfig = ComposeConfig.from_dict(yaml.safe_load(fixed))
    assert [n.name for n in two_noises.noises] == ["Street_noise", "Babble_noise"]
    noise_only: ComposeConfig = ComposeConfig.from_dict(yaml.safe_load(only_noise))
    assert noise_only.mode == "noise_only"
    assert noise_only.noise_count == (1, 2)


def test_toml_and_yaml_files_give_the_same_config(tmp_path: Path) -> None:
    toml_file: Path = tmp_path / "config.toml"
    toml_file.write_text(
        'duration = [4, 6]\nseed = 5\nnoises = [{ name = "City_noise", db = -38 }]\n'
    )
    yaml_file: Path = tmp_path / "config.yaml"
    yaml_file.write_text("duration: [4, 6]\nseed: 5\nnoises:\n  - {name: City_noise, db: -38}\n")
    yml_file: Path = tmp_path / "config.yml"
    yml_file.write_text(yaml_file.read_text())

    from_toml: ComposeConfig = ComposeConfig.from_dict(load_config_file(toml_file))

    assert from_toml == ComposeConfig.from_dict(load_config_file(yaml_file))
    assert from_toml == ComposeConfig.from_dict(load_config_file(yml_file))
    assert from_toml.noises == (NoiseLayer("City_noise", -38.0),)


def test_yaml_empty_and_null_entries_keep_the_defaults(tmp_path: Path) -> None:
    empty: Path = tmp_path / "empty.yaml"
    empty.write_text("# everything commented out\n")
    nulls: Path = tmp_path / "nulls.yaml"
    nulls.write_text("noise_pool: null\nnoises:\nseed: 4\n")

    assert ComposeConfig.from_dict(load_config_file(empty)) == ComposeConfig()
    assert ComposeConfig.from_dict(load_config_file(nulls)) == ComposeConfig(seed=4)


def test_config_file_must_be_a_known_format_and_a_mapping(tmp_path: Path) -> None:
    other: Path = tmp_path / "config.json"
    other.write_text("{}")
    listing: Path = tmp_path / "list.yaml"
    listing.write_text("- 1\n- 2\n")

    with pytest.raises(ValueError, match="toml, .yaml or .yml"):
        load_config_file(other)
    with pytest.raises(ValueError, match="key: value"):
        load_config_file(listing)


def test_command_line_wins_over_the_yaml_file(tmp_path: Path) -> None:
    config_file: Path = tmp_path / "config.yaml"
    config_file.write_text("seed: 5\nduration: [4, 6]\npause_prob: 0.2\n")

    args = build_parser().parse_args(
        ["--dataset", "Skylar", "--config", str(config_file), "--seed", "9"]
    )
    config: ComposeConfig = resolve_config(args)

    assert config.seed == 9
    assert config.duration == (4, 6)
    assert config.pause_prob == 0.2


def test_take_wrapped_loops_around_the_end() -> None:
    source: NDArray[np.float32] = np.arange(10, dtype=np.float32)
    assert take_wrapped(source, 8, 5).tolist() == [8, 9, 0, 1, 2]
    assert len(take_wrapped(source, 3, 25)) == 25


def test_fit_label_trims_and_pads() -> None:
    assert fit_label(np.ones(5), 3).tolist() == [1, 1, 1]
    assert fit_label(np.ones(3), 5).tolist() == [1, 1, 1, 0, 0]


def test_each_layer_gets_its_own_level() -> None:
    config: ComposeConfig = ComposeConfig(
        duration=(12.0, 12.0),
        speech_db=(-26.0, -26.0),
        noises=(NoiseLayer("A", -40.0), NoiseLayer("B", -34.0)),
        pause_prob=0.0,
    )
    clip: Clip = _compose(config)

    # Rebuild the layers with the recipe offsets to measure each one on its own.
    n: int = len(clip.audio)
    info: dict[str, Any] = {n["name"]: n for n in clip.recipe["noises"]}
    for name, db in (("A", -40.0), ("B", -34.0)):
        layer: NDArray[np.float32] = take_wrapped(
            NOISES[name], int(round(info[name]["offset"] * SECOND)), n
        )
        layer = layer * np.float32(10 ** (db / 20) / rms(layer))
        assert _db(layer) == pytest.approx(db, abs=0.01)

    assert info["A"]["snr_db"] == pytest.approx(14.0)
    assert info["B"]["snr_db"] == pytest.approx(8.0)
    assert clip.recipe["speech_db"] == -26.0


def test_speech_level_is_measured_on_the_active_samples() -> None:
    config: ComposeConfig = ComposeConfig(
        duration=(4.0, 4.0), speech_db=(-20.0, -20.0), noise_count=(0, 0), pause_prob=0.0
    )
    clip: Clip = _compose(config)

    audio: NDArray[np.float32] = clip.audio.astype(np.float32) / 32767
    assert _db(audio[clip.label == 1]) == pytest.approx(-20.0, abs=0.05)


def test_speech_level_is_exact_when_the_last_segment_is_cut() -> None:
    def uneven_segment(rng: np.random.Generator) -> Segment:
        # Loud first half, quiet second half: cutting it changes the RMS of what is kept.
        audio: NDArray[np.float32] = np.concatenate([_tone(2.0, 0.3), _tone(2.0, 0.02)])
        return Segment("uneven.wav", audio, np.ones(len(audio), dtype=np.int16))

    config: ComposeConfig = ComposeConfig(
        duration=(3.0, 3.0), speech_db=(-22.0, -22.0), noise_count=(0, 0), pause_prob=0.0
    )
    clip: Clip = compose_clip(
        np.random.default_rng(0), config, uneven_segment, lambda name: NOISES[name], ()
    )

    assert clip.recipe["segments"][-1]["truncated"] is True
    audio: NDArray[np.float32] = clip.audio.astype(np.float32) / 32767
    assert _db(audio[clip.label == 1]) == pytest.approx(-22.0, abs=0.01)


def test_recipe_positions_are_exact_in_samples() -> None:
    config: ComposeConfig = ComposeConfig(
        mode="noise_only",
        duration=(6.0, 6.0),
        noises=(NoiseLayer("A", -35.0), NoiseLayer("B", -41.0)),
    )
    clip: Clip = _compose(config)

    # The noise layers can be rebuilt from the recipe alone.
    n: int = len(clip.audio)
    rebuilt: NDArray[np.float32] = np.zeros(n, dtype=np.float32)
    for info in clip.recipe["noises"]:
        layer: NDArray[np.float32] = take_wrapped(NOISES[info["name"]], info["offset_samples"], n)
        rebuilt += layer * np.float32(10 ** (info["db"] / 20) / rms(layer))
    assert np.max(np.abs(clip.audio.astype(np.float32) / 32767 - rebuilt)) < 1e-3

    speech: Clip = _compose(
        ComposeConfig(duration=(15.0, 15.0), pause_range=(1.0, 2.0), pause_prob=1.0), seed=3
    )
    covered: NDArray[np.bool_] = np.zeros(len(speech.audio), dtype=bool)
    for start, end in speech.recipe["pauses_samples"]:
        assert speech.label[start:end].sum() == 0
        covered[start:end] = True
    for segment in speech.recipe["segments"]:
        covered[segment["start_sample"] : segment["end_sample"]] = True
        assert segment["start_sample"] / SECOND == pytest.approx(segment["start"], abs=1e-3)

    # Segments and pauses tile the whole clip, with nothing left over or overlapping.
    assert covered.all()


def test_duration_never_exceeds_the_limit_and_matches_label() -> None:
    config: ComposeConfig = ComposeConfig(duration=(5.0, 30.0), pause_prob=0.5)
    for seed in range(15):
        clip: Clip = _compose(config, seed)
        assert len(clip.audio) == len(clip.label)
        assert len(clip.audio) <= MAX_DURATION_SECONDS * SECOND


def test_long_clip_concatenates_several_segments() -> None:
    config: ComposeConfig = ComposeConfig(duration=(20.0, 20.0), pause_prob=0.0)
    clip: Clip = _compose(config)

    segments: list[dict[str, Any]] = clip.recipe["segments"]
    assert len(segments) == 5
    assert segments[-1]["end"] == pytest.approx(20.0)
    assert segments[-1]["truncated"] is False


def test_pauses_have_only_noise_and_label_zero() -> None:
    config: ComposeConfig = ComposeConfig(
        duration=(15.0, 15.0),
        speech_db=(-20.0, -20.0),
        noises=(NoiseLayer("A", -50.0),),
        pause_range=(2.0, 2.0),
        pause_prob=1.0,
    )
    clip: Clip = _compose(config)

    assert clip.recipe["pauses"]
    start: float
    end: float
    start, end = clip.recipe["pauses"][0]
    inside: slice = slice(int(start * SECOND), int(end * SECOND))
    assert clip.label[inside].sum() == 0
    # Quiet noise only: far below the -20 dBFS speech.
    assert _db(clip.audio[inside].astype(np.float32) / 32767) < -45.0


def test_noise_only_clip_has_empty_label_and_requested_levels() -> None:
    config: ComposeConfig = ComposeConfig(
        mode="noise_only",
        duration=(6.0, 6.0),
        noises=(NoiseLayer("A", -35.0), NoiseLayer("B", -35.0)),
    )
    clip: Clip = _compose(config)

    assert clip.recipe["mode"] == "noise_only"
    assert clip.label.sum() == 0
    assert clip.recipe["segments"] == []
    assert clip.recipe["speech_db"] is None
    assert [n["snr_db"] for n in clip.recipe["noises"]] == [None, None]
    # Two independent noises of -35 dBFS add up to about -32 dBFS.
    assert _db(clip.audio.astype(np.float32) / 32767) == pytest.approx(-32.0, abs=1.0)


def test_noise_only_always_has_at_least_one_noise() -> None:
    config: ComposeConfig = ComposeConfig(mode="noise_only", noise_count=(0, 0), duration=(2, 2))
    clip: Clip = _compose(config)

    assert len(clip.recipe["noises"]) == 1


def test_noise_only_ratio_mixes_both_kinds() -> None:
    config: ComposeConfig = ComposeConfig(noise_only_ratio=0.5, duration=(2.0, 2.0))
    modes: set[str] = {_compose(config, seed).recipe["mode"] for seed in range(30)}

    assert modes == {"speech", "noise_only"}


def test_same_seed_reproduces_the_clip() -> None:
    config: ComposeConfig = ComposeConfig(noise_count=(1, 2), duration=(5.0, 20.0))

    first: Clip = _compose(config, seed=7)
    second: Clip = _compose(config, seed=7)
    other: Clip = _compose(config, seed=8)

    assert np.array_equal(first.audio, second.audio)
    assert first.recipe == second.recipe
    assert first.recipe != other.recipe


def test_limit_attenuates_without_changing_relative_levels() -> None:
    config: ComposeConfig = ComposeConfig(
        duration=(4.0, 4.0), speech_db=(-1.0, -1.0), noise_count=(0, 0), pause_prob=0.0
    )
    clip: Clip = _compose(config)

    assert clip.recipe["gain_db"] < 0
    assert clip.recipe["clipped"] is False
    assert np.abs(clip.audio).max() <= 32767

    hard: Clip = _compose(config.replace(normalize="none"))
    assert hard.recipe["gain_db"] == 0.0
    assert hard.recipe["clipped"] is True
