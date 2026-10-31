import copy
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from generation_helpers import SR, write_noises

from librivad.generation.build_scenarios import make_noises
from librivad.generation.config import Config, config_from_dict
from librivad.generation.corpus import CorpusAdapter, SourceItem
from librivad.generation.noise import keyed_offset, raised_cosine
from librivad.generation.planner import Planner, last_suffix, noise_length


class MemoryAdapter(CorpusAdapter):
    """Items built in memory: the planner never reads their audio, only rows and lengths."""

    def __init__(self, config: Any, items: dict[str, list[SourceItem]]) -> None:
        super().__init__(config)
        self._items = items

    def _load(self, split: str) -> list[SourceItem]:
        return self._items[split]


def memory_items(per_stratum: int, seconds: float = 12.0) -> dict[str, list[SourceItem]]:
    items: dict[str, list[SourceItem]] = {}
    for split in ("train", "val", "test"):
        items[split] = []
        for language in ("en", "es"):
            for bucket in ("05-10", "10-15"):
                for k in range(per_stratum):
                    item_id: str = f"{split}-{language}-{bucket}-{k:04d}"
                    row: dict[str, Any] = {
                        "id": item_id,
                        "language": language,
                        "duration_bucket": bucket,
                        "words": [{"word": "a", "start": 0.5, "end": 2.5, "probability": 1.0}],
                    }
                    items[split].append(
                        SourceItem(
                            id=item_id,
                            path=Path(f"/fake/{item_id}.wav"),
                            split=split,
                            split_folder=split,
                            n_samples=round(seconds * SR),
                            fields={**row, "split": split, "split_folder": split, "stem": item_id},
                            row=row,
                        )
                    )
    return items


def planner_for(
    skylar_setup: dict[str, Any],
    items: dict[str, list[SourceItem]],
    raw: dict[str, Any] | None = None,
) -> Planner:
    config: Config = config_from_dict(raw or skylar_setup["raw"])
    adapter = MemoryAdapter(config.dataset, items)
    planner = Planner(config, adapter, make_noises(config, adapter))
    # The items have no audio; give every one a speech-only RMS of 1000 (about -30 dBFS).
    planner.speech_power = lambda item, speech_only: (1e6 * 100, 100)  # type: ignore[method-assign]
    return planner


def test_noise_length_is_exact() -> None:
    assert noise_length(100, 0.2) == 25
    assert noise_length(2, 0.6) == 3  # float gives 2 * 1.4999999999999998 -> 2
    assert noise_length(10, 0.7) == 23
    assert noise_length(171, 0.2) == int(171 / 4)


def test_last_suffix_follows_the_paper_naming() -> None:
    assert last_suffix("84-121123-0001", "84-121123-0002") == "0002"
    assert last_suffix("84-121123-0010", "84-121550-0000") == "121550-0000"
    assert last_suffix("84-121550-0035", "174-50561-0000") == "174-50561-0000"


def test_counts_follow_the_population(skylar_setup: dict[str, Any]) -> None:
    planner: Planner = planner_for(skylar_setup, memory_items(per_stratum=6))
    result = planner.plan(["train", "val", "test"])
    by_split: Counter[str] = Counter(s.split for s in result.specs)
    n_items: int = 2 * 2 * 6
    assert by_split["train"] == n_items  # everything assigned
    assert by_split["val"] == n_items * 3  # snr crossed (3 levels)
    assert by_split["test"] == n_items * 3 * 2  # snr x scenario crossed


def test_assigned_values_are_balanced_per_stratum(skylar_setup: dict[str, Any]) -> None:
    planner: Planner = planner_for(skylar_setup, memory_items(per_stratum=37))
    specs = planner.plan(["train"]).specs
    per_stratum: dict[str, Counter[str]] = {}
    for spec in specs:
        stratum: str = spec.unit.rsplit("-", 1)[0]
        key: str = f"{spec.scenario}/{spec.conditions['noise']}/{spec.conditions['snr_db']}"
        per_stratum.setdefault(stratum, Counter())[key] += 1
    for counts in per_stratum.values():
        values: list[int] = [counts.get(k, 0) for k in _all_keys()]
        assert max(values) - min(values) <= 1


def _all_keys() -> list[str]:
    return [
        f"{s}/{n}/{v}"
        for s in ("noise_first", "noise_last")
        for n in ("babble", "office")
        for v in (0, 5, 35)
    ]


def test_plan_is_deterministic_and_split_independent(skylar_setup: dict[str, Any]) -> None:
    items = memory_items(per_stratum=4)
    full = planner_for(skylar_setup, items).plan(["train", "val", "test"]).specs
    again = planner_for(skylar_setup, items).plan(["train", "val", "test"]).specs
    assert [s.model_dump_json() for s in full] == [s.model_dump_json() for s in again]
    only_test = planner_for(skylar_setup, items).plan(["test"]).specs
    assert [s.model_dump_json() for s in only_test] == [
        s.model_dump_json() for s in full if s.split == "test"
    ]


def test_levels_share_the_excerpt_and_noise_kinds_do_not(skylar_setup: dict[str, Any]) -> None:
    specs = planner_for(skylar_setup, memory_items(per_stratum=2)).plan(["test"]).specs
    by_group: dict[str, set[int | None]] = {}
    for spec in specs:
        by_group.setdefault(spec.group, set()).add(spec.noise_block().noise_offset)
    assert all(len(offsets) == 1 for offsets in by_group.values())
    first = next(s for s in specs if s.scenario == "noise_first")
    last = next(s for s in specs if s.unit == first.unit and s.scenario == "noise_last")
    assert first.noise_block().noise_offset != last.noise_block().noise_offset


def test_fill_to_target_skips_at_the_boundary(skylar_setup: dict[str, Any]) -> None:
    exact = memory_items(per_stratum=1, seconds=28.0)
    over = memory_items(per_stratum=1, seconds=28.0 + 1 / SR)
    assert planner_for(skylar_setup, exact).plan(["test"]).specs
    planner = planner_for(skylar_setup, over)
    assert planner.plan(["test"]).specs == []
    planner = planner_for(skylar_setup, over)
    result = planner.plan(["test"])
    # 4 test files (2 languages x 2 buckets), each would have made 3 levels x 2 scenarios
    assert result.report["skipped"] == {"test/too_long": 24}
    assert result.report["skipped_sources"] == {"test/too_long": 4}


def test_noise_block_sits_where_the_layout_says(skylar_setup: dict[str, Any]) -> None:
    specs = planner_for(skylar_setup, memory_items(per_stratum=1)).plan(["test"]).specs
    for spec in specs:
        assert spec.total_samples == 30 * SR
        noise = spec.noise_block()
        assert noise.length == 18 * SR
        if spec.scenario == "noise_first":
            assert (noise.start, noise.fade_in, noise.fade_out) == (0, 0, 160)
        else:
            assert (noise.start, noise.fade_in, noise.fade_out) == (12 * SR, 160, 0)


def test_limits_are_reported(skylar_setup: dict[str, Any]) -> None:
    raw: dict[str, Any] = copy.deepcopy(skylar_setup["raw"])
    raw["output"]["limits"] = {"max_files_total": 10, "max_files_per_folder": 1}
    result = planner_for(skylar_setup, memory_items(per_stratum=3), raw).plan(["test"])
    assert result.over_limits()
    assert len(result.report["limits"]["exceeded"]) == 2


def test_keyed_offset_depends_only_on_the_key() -> None:
    assert keyed_offset(1, ["a", "b"], 10_000) == keyed_offset(1, ["a", "b"], 10_000)
    assert keyed_offset(1, ["a", "b"], 10_000) != keyed_offset(1, ["a", "c"], 10_000)
    assert keyed_offset(2, ["a", "b"], 10_000) != keyed_offset(1, ["a", "b"], 10_000)


def test_raised_cosine_reaches_the_edges() -> None:
    fade = raised_cosine(160)
    assert fade[0] < 1e-3 and fade[-1] > 1 - 1e-3
    assert np.all(np.diff(fade) > 0)


def test_noise_read_wraps_like_tile(tmp_path: Path, skylar_setup: dict[str, Any]) -> None:
    templates = write_noises(tmp_path / "short", kinds=("hum",), seconds=1.0)
    raw: dict[str, Any] = copy.deepcopy(skylar_setup["raw"])
    raw["noise_sources"]["hum"] = {"type": "file_per_split", "path": templates["hum"]}
    raw["conditions"]["noise"] = ["hum"]
    planner = planner_for(skylar_setup, memory_items(per_stratum=1), raw)
    source = planner.noises["hum"]
    whole = source.read("test", 0, SR)
    wrapped = source.read("test", SR - 100, int(2.5 * SR))
    expected = np.tile(whole, 4)[SR - 100 : SR - 100 + int(2.5 * SR)]
    assert np.array_equal(wrapped, expected)


@pytest.mark.parametrize("composition", ["concatenate", "contiguous"])
def test_block_compositions(skylar_setup: dict[str, Any], composition: str) -> None:
    raw: dict[str, Any] = copy.deepcopy(skylar_setup["raw"])
    raw["defaults"]["speech_block"] = {"composition": composition, "max_utterances": 2}
    if composition == "contiguous":
        raw["dataset"]["continuity"] = {"group_by": ["language"], "order_by": "id"}
    raw["defaults"]["noise_length"]["min_noise_s"] = 2.0
    items = memory_items(per_stratum=2, seconds=6.0)
    specs = planner_for(skylar_setup, items, raw).plan(["test"]).specs
    units: set[str] = {s.unit for s in specs}
    assert all(u.count("+") == 1 for u in units)  # two 6 s files per 30 s output
    if composition == "contiguous":
        assert all(u.split("+")[0].split("-")[1] == u.split("+")[1].split("-")[1] for u in units)


def test_quiet_sources_are_skipped_once_per_output(skylar_setup: dict[str, Any]) -> None:
    items = memory_items(per_stratum=2)
    planner = planner_for(skylar_setup, items)
    quiet: set[str] = {items["train"][0].id, items["val"][0].id}

    def power(item: SourceItem, speech_only: bool) -> tuple[float, int]:
        rms: float = 0.7 if item.id in quiet else 1000.0  # 0.7 = the silent FLEURS rows
        return (rms**2 * 100, 100)

    planner.speech_power = power  # type: ignore[method-assign]
    result = planner.plan(["train", "val"])
    assert quiet.isdisjoint(s.unit for s in result.specs)
    assert result.report["skipped"] == {"train/too_quiet": 1, "val/too_quiet": 3}
    assert result.report["skipped_sources"] == {"train/too_quiet": 1, "val/too_quiet": 1}


def test_quiet_threshold_covers_the_highest_level(skylar_setup: dict[str, Any]) -> None:
    planner = planner_for(skylar_setup, memory_items(per_stratum=1))
    scenario = planner.config.scenarios[0]
    # snr 35 needs the noise at >= ~1.89 LSB, so the speech at >= ~106 LSB (-49.8 dBFS),
    # stricter than the configured -60 dBFS (32.8 LSB).
    assert planner.min_speech_rms(scenario) == pytest.approx(1.8925 * 10 ** (35 / 20), rel=1e-3)
    scenario.conditions["snr_db"] = [0, 5]
    assert planner.min_speech_rms(scenario) == pytest.approx(32768 * 10 ** (-60 / 20))
