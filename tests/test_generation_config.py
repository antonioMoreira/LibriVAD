import copy
from pathlib import Path
from typing import Any

import pytest

from librivad.generation.config import (
    ConfigError,
    conditions,
    config_from_dict,
    deep_merge,
    load_config,
)

REPO: Path = Path(__file__).resolve().parents[1]


def test_example_configs_load() -> None:
    for name in (
        "docs/concat_scenarios.example.yaml",
        "configs/scenarios/librispeech_middle.yaml",
        "configs/scenarios/skylar_noisy.example.yaml",
    ):
        config = load_config(REPO / name, base=REPO)
        assert config.scenarios


def test_skylar_grid_has_216_conditions() -> None:
    """4 scenarios (2 layouts x SNR or loudness) x 9 noises x 6 levels."""
    config = load_config(REPO / "configs/scenarios/skylar_noisy.example.yaml", base=REPO)
    assert sum(len(list(conditions(s))) for s in config.scenarios) == 216
    assert config.factors() == ["scenario", "noise", "snr_db", "loudness_lu"]


def test_scenarios_are_deep_merged_over_defaults(skylar_setup: dict[str, Any]) -> None:
    raw: dict[str, Any] = copy.deepcopy(skylar_setup["raw"])
    raw["scenarios"][1]["noise_block"] = {"fade_ms": 5}
    config = config_from_dict(raw)
    first, last = config.scenarios
    assert first.noise_block.fade_ms == 10
    assert last.noise_block.fade_ms == 5
    assert last.noise_block.key == ["id", "scenario", "noise"]  # kept from defaults
    assert last.noise_length.target_duration_s == 30
    assert last.conditions["snr_db"] == [0, 5, 35]


def test_deep_merge_replaces_lists() -> None:
    assert deep_merge({"a": {"b": 1, "c": [1]}}, {"a": {"c": [2]}}) == {"a": {"b": 1, "c": [2]}}


def _broken(raw: dict[str, Any], path: str, value: Any) -> dict[str, Any]:
    broken: dict[str, Any] = copy.deepcopy(raw)
    node: Any = broken
    keys: list[str] = path.split(".")
    for key in keys[:-1]:
        node = node[int(key)] if isinstance(node, list) else node[key]
    if value is DELETE:
        del node[keys[-1]]
    else:
        node[keys[-1]] = value
    return broken


DELETE = object()

RULES: list[tuple[str, Any, str]] = [
    ("scenarios.0.layout", ["noise", "speech", "noise"], "exactly one `noise`"),
    ("scenarios.0.layout", ["noise"], "`speech` once or twice"),
    ("defaults.speech_block", {"composition": "contiguous"}, "needs dataset.continuity"),
    (
        "defaults.speech_block",
        {"composition": "concatenate", "group_by": ["nope"]},
        "unknown fields",
    ),
    ("defaults.noise_length", {"mode": "ratio", "noise_ratio": 1.0}, "must be in [0, 1)"),
    ("defaults.noise_length", {"mode": "ratio"}, "needs a noise_ratio"),
    (
        "defaults.noise_length",
        {"mode": "fill_to_target", "target_duration_s": 2, "min_noise_s": 2},
        "target_duration_s > min_noise_s",
    ),
    ("conditions.snr_db", [-5, 0], "below constraints.min_snr_db"),
    ("defaults.loudness", {"measure": "fixed_gain", "gain": 0.001}, "excludes snr_db"),
    ("conditions.snr_db", DELETE, "needs conditions.snr_db"),
    ("conditions.noise", ["babble", "jet"], "not in noise_sources"),
    ("conditions.loudness_lufs", [1], "unknown condition keys"),
    ("population.val", {"crossed": ["snr_db"], "assigned": ["noise"]}, "exactly once"),
    (
        "population.val",
        {"crossed": ["snr_db", "volume"], "assigned": ["scenario", "noise"]},
        "are not factors",
    ),
    ("population.strata", ["nope"], "strata uses unknown fields"),
    ("output.path", "{split}/{wrong}/{id}.wav", "unknown placeholders"),
    ("output.metadata_dir", "{split}/elsewhere", "parent folder of the output path"),
    ("dataset.labels", {"source": "textgrid"}, "needs labels.dir and labels.path"),
    ("dataset.splits", {"train": "train", "dev": "val"}, "unknown split names"),
    ("version", 1, "version"),
]


@pytest.mark.parametrize(("path", "value", "message"), RULES)
def test_each_rule_fails_with_a_clear_message(
    skylar_setup: dict[str, Any], path: str, value: Any, message: str
) -> None:
    with pytest.raises(ConfigError, match=None) as caught:
        config_from_dict(_broken(skylar_setup["raw"], path, value))
    assert message in str(caught.value)


def test_extracted_noise_needs_labels(skylar_setup: dict[str, Any]) -> None:
    raw: dict[str, Any] = copy.deepcopy(skylar_setup["raw"])
    raw["dataset"]["labels"] = {"source": "none"}
    raw["noise_sources"]["silence"] = {
        "type": "extracted_from_dataset",
        "split": "train",
        "cache": "x.wav",
    }
    with pytest.raises(ConfigError, match="needs labels"):
        config_from_dict(raw)


def test_relative_paths_resolve_against_base(skylar_setup: dict[str, Any], tmp_path: Path) -> None:
    raw: dict[str, Any] = copy.deepcopy(skylar_setup["raw"])
    raw["output"]["root"] = "relative/out"
    config = config_from_dict(raw, base=tmp_path)
    assert config.output.root == tmp_path / "relative/out"
