"""
Turns a config and a corpus into the full list of output files, before anything is rendered.

Each `FileSpec` holds everything needed to render one output: its sources and where they
go, the noise file and offset, the target level and the output paths. It serializes to JSON,
so `--dry-run` can write the whole plan (`plan.jsonl`) and report counts, disk use and
limits without touching audio.

The plan is a pure function of config, corpus and noise files: every random choice is
seeded by a hash of the values that identify it, never by its position in a list.
"""

import hashlib
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import pairwise
from pathlib import PurePosixPath
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel

from librivad.generation.config import (
    Condition,
    Config,
    ConfigError,
    ScenarioConfig,
    conditions,
    format_value,
)
from librivad.generation.corpus import CorpusAdapter, SourceItem
from librivad.generation.labels import load_labels
from librivad.generation.noise import NoiseSource, keyed_offset, read_int16_units

WAV_HEADER_BYTES: int = 44
NPY_HEADER_BYTES: int = 128
FULL_SCALE_RMS: float = 32768.0
LEVEL_TOLERANCE_DB: float = 0.1
"""Largest allowed gap between an output's measured and target level."""
NOISE_FLOOR_RMS: float = math.sqrt((1 / 12) / (10 ** (LEVEL_TOLERANCE_DB / 10) - 1))
"""Smallest noise RMS (int16 units, about 1.89) whose level int16 rounding (1/12 LSB² of
error power) keeps within `LEVEL_TOLERANCE_DB` of its target."""


class ItemRef(BaseModel):
    """A source file placed in an output, starting at sample `offset`."""

    id: str
    path: str
    n_samples: int
    offset: int


class BlockSpec(BaseModel):
    kind: Literal["speech", "noise"]
    start: int
    length: int
    items: list[ItemRef] = []
    noise: str | None = None
    noise_file: str | None = None
    noise_offset: int | None = None
    fade_in: int = 0
    """Samples faded in at the start of the noise block (speech before it)."""
    fade_out: int = 0
    """Samples faded out at the end of the noise block (speech after it)."""


class FileSpec(BaseModel):
    split: str
    split_folder: str
    scenario: str
    conditions: dict[str, Any]
    unit: str
    """Id of the speech unit: the source id, or ids joined by `+` for several sources."""
    group: str
    """Outputs that share sources and noise excerpt and differ only by level."""
    total_samples: int
    noise_samples: int
    speech_samples: int
    blocks: list[BlockSpec]
    out_path: str
    metadata_dir: str | None
    label_path: str | None
    row_id: str
    snr_db: float | None
    fixed_gain: float | None
    selected: bool = True
    """False when `--limit` leaves this output out (it still counts for its group's guard)."""

    def noise_block(self) -> BlockSpec:
        return next(b for b in self.blocks if b.kind == "noise")

    def speech_items(self) -> list[ItemRef]:
        return [i for b in self.blocks if b.kind == "speech" for i in b.items]


@dataclass
class Unit:
    """The sources filling the speech slots of one output (one list per `speech` slot)."""

    slots: list[list[SourceItem]]

    @property
    def id(self) -> str:
        return "+".join(item.id for slot in self.slots for item in slot)

    @property
    def first(self) -> SourceItem:
        return self.slots[0][0]

    @property
    def last(self) -> SourceItem:
        return self.slots[-1][-1]


@dataclass
class PlanResult:
    specs: list[FileSpec]
    report: dict[str, Any] = field(default_factory=dict)

    def over_limits(self) -> bool:
        return bool(self.report.get("limits", {}).get("exceeded"))


def noise_length(speech: int, ratio: float) -> int:
    """`floor(S·p / (1 − p))` in exact integer arithmetic (floats lose a sample for 0.6, 0.7)."""
    p: Fraction = Fraction(ratio).limit_denominator()
    return math.floor(speech * p / (1 - p))


def _seeded_rng(seed: int, *parts: str) -> np.random.Generator:
    digest: bytes = hashlib.sha256(":".join([str(seed), *parts]).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def last_suffix(first_stem: str, last_stem: str) -> str:
    """`last_stem` without the `-`-separated prefix it shares with `first_stem`.

    For LibriSpeech stems (`speaker-chapter-utt`) this is the paper's naming: the full stem
    for another speaker, `chapter-utt` for another chapter, `utt` otherwise.
    """
    a: list[str] = first_stem.split("-")
    b: list[str] = last_stem.split("-")
    common: int = 0
    while common < min(len(a), len(b)) - 1 and a[common] == b[common]:
        common += 1
    return "-".join(b[common:])


def _order_key(value: Any) -> tuple[int, Any]:
    text: str = str(value)
    return (0, int(text)) if text.isdigit() else (1, text)


class Planner:
    def __init__(
        self, config: Config, adapter: CorpusAdapter, noises: dict[str, NoiseSource]
    ) -> None:
        self.config = config
        self.adapter = adapter
        self.noises = noises
        self.sr: int = config.dataset.sample_rate
        self._speech_cache: dict[str, int] = {}
        self._power_cache: dict[tuple[str, bool], tuple[float, int]] = {}
        self.skipped_sources: Counter[str] = Counter()
        self._skipped_outputs: dict[str, Fraction] = defaultdict(Fraction)
        self.warnings: Counter[str] = Counter()

    # --- speech units -------------------------------------------------------------------

    def speech_label_samples(self, item: SourceItem) -> int:
        if item.id not in self._speech_cache:
            self._speech_cache[item.id] = int(load_labels(item, self.config.dataset).sum())
        return self._speech_cache[item.id]

    def speech_power(self, item: SourceItem, speech_only: bool) -> tuple[float, int]:
        """Sum of squares and count of the item's speech samples (all samples if not
        `speech_only`), in int16 units. Reads the audio once per item."""
        key: tuple[str, bool] = (item.id, speech_only)
        if key not in self._power_cache:
            audio: np.ndarray = read_int16_units(item.path)
            if speech_only:
                audio = audio[load_labels(item, self.config.dataset)[: len(audio)] == 1]
            self._power_cache[key] = (float(np.sum(audio**2)), len(audio))
        return self._power_cache[key]

    def min_speech_rms(self, scenario: ScenarioConfig) -> float:
        """Quietest speech-only RMS (int16 units) a source may have in this scenario.

        The larger of `loudness.min_speech_rms_dbfs` and the level at which the noise of
        the scenario's highest `snr_db` would still sit above `NOISE_FLOOR_RMS`.
        """
        configured: float = FULL_SCALE_RMS * 10 ** (scenario.loudness.min_speech_rms_dbfs / 20)
        levels: list[float] = [float(v) for v in scenario.conditions.get("snr_db", [])]
        representable: float = NOISE_FLOOR_RMS * 10 ** (max(levels) / 20) if levels else 0.0
        return max(configured, representable)

    def _block_length(self, items: list[SourceItem], scenario: ScenarioConfig) -> int:
        gap: int = round(scenario.speech_block.join_gap_s * self.sr)
        return sum(i.n_samples for i in items) + gap * (len(items) - 1)

    def _speech_target(self, scenario: ScenarioConfig, condition: Condition) -> int | None:
        length = scenario.noise_length
        if length.mode == "ratio" and length.target_duration_s is not None:
            ratio: float = condition.values.get("noise_ratio", length.noise_ratio)
            return round((1 - ratio) * length.target_duration_s * self.sr)
        return None

    def _max_speech(self, scenario: ScenarioConfig) -> int | None:
        length = scenario.noise_length
        if length.mode == "fill_to_target" and length.target_duration_s is not None:
            return round((length.target_duration_s - length.min_noise_s) * self.sr)
        return None

    def _groups(self, items: list[SourceItem], scenario: ScenarioConfig) -> list[list[SourceItem]]:
        block = scenario.speech_block
        if block.composition == "contiguous":
            continuity = self.config.dataset.continuity
            assert continuity is not None  # checked by check_config
            keys: list[str] = continuity.group_by
            order: str = continuity.order_by
        elif block.composition == "concatenate" and block.group_by:
            keys, order = block.group_by, ""
        else:
            return [items]
        groups: dict[tuple[Any, ...], list[SourceItem]] = defaultdict(list)
        for item in items:
            groups[tuple(item.fields[k] for k in keys)].append(item)
        result: list[list[SourceItem]] = list(groups.values())
        if order:
            result = [sorted(g, key=lambda i: _order_key(i.fields[order])) for g in result]
        return result

    def compose(
        self, items: list[SourceItem], scenario: ScenarioConfig, target: int | None, split: str
    ) -> list[Unit]:
        """Speech units for one scenario. Leftovers that cannot fill a unit are dropped."""
        block = scenario.speech_block
        n_slots: int = scenario.n_speech
        ordered: list[SourceItem] = list(items)
        if block.selection == "shuffled":
            rng: np.random.Generator = _seeded_rng(self.config.seed, "select", split, scenario.name)
            ordered = [ordered[i] for i in rng.permutation(len(ordered))]

        blocks: list[list[SourceItem]] = []
        if block.composition == "single":
            if target is not None and block.duration_tolerance is not None:
                tolerance: float = block.duration_tolerance * target
                ordered = [i for i in ordered if abs(i.n_samples - target) <= tolerance]
            blocks = [[item] for item in ordered]
        else:
            maximum: int | None = self._max_speech(scenario)
            for group in self._groups(ordered, scenario):
                current: list[SourceItem] = []
                for item in group:
                    if (
                        maximum is not None
                        and current
                        and (self._block_length([*current, item], scenario) > maximum)
                    ):
                        blocks.append(current)
                        current = []
                    current.append(item)
                    full: bool = len(current) >= block.max_utterances
                    reached: bool = (
                        target is not None and self._block_length(current, scenario) >= target
                    )
                    if full or reached:
                        if target is None or reached:
                            blocks.append(current)
                        current = []
                if current and target is None:
                    blocks.append(current)

        usable: int = len(blocks) - len(blocks) % n_slots
        return [Unit(slots=blocks[i : i + n_slots]) for i in range(0, usable, n_slots)]

    # --- planning -----------------------------------------------------------------------

    def plan(
        self, splits: list[str], scenario_names: list[str] | None = None, limit: int | None = None
    ) -> PlanResult:
        scenarios: list[ScenarioConfig] = [
            s for s in self.config.scenarios if scenario_names is None or s.name in scenario_names
        ]
        specs: list[FileSpec] = []
        for split in splits:
            specs += self._plan_split(split, scenarios)

        paths: Counter[str] = Counter(s.out_path for s in specs)
        if duplicated := [p for p, n in paths.items() if n > 1]:
            raise ConfigError(
                f"output.path gives the same file to several outputs, e.g. {duplicated[0]};"
                " add the missing condition placeholders"
            )
        if limit is not None:
            taken: Counter[str] = Counter()
            for spec in specs:
                condition = Condition(scenario=spec.scenario, values=spec.conditions)
                key: str = f"{spec.split}/{condition.key()}"
                spec.selected = taken[key] < limit
                taken[key] += 1
        result: PlanResult = PlanResult(specs=specs)
        result.report = self.report(specs)
        return result

    def _candidates(
        self, split: str, scenarios: list[ScenarioConfig]
    ) -> tuple[
        dict[str, Unit],
        dict[str, list[tuple[ScenarioConfig, Condition]]],
        dict[str, list[tuple[ScenarioConfig, str]]],
    ]:
        items: list[SourceItem] = self.adapter.items(split)
        units: dict[str, Unit] = {}
        candidates: dict[str, list[tuple[ScenarioConfig, Condition]]] = defaultdict(list)
        skipped: dict[str, list[tuple[ScenarioConfig, str]]] = defaultdict(list)
        for scenario in scenarios:
            composed: dict[int | None, list[Unit]] = {}
            reasons: dict[str, str | None] = {}
            for condition in conditions(scenario):
                target: int | None = self._speech_target(scenario, condition)
                if target not in composed:
                    composed[target] = self.compose(items, scenario, target, split)
                for unit in composed[target]:
                    if unit.id not in reasons:
                        reason: str | None = self._skip_reason(unit, scenario)
                        reasons[unit.id] = reason
                        if reason is not None:
                            skipped[unit.id].append((scenario, reason))
                    if reasons[unit.id] is not None:
                        continue
                    units.setdefault(unit.id, unit)
                    candidates[unit.id].append((scenario, condition))
        return units, candidates, skipped

    def _skip_reason(self, unit: Unit, scenario: ScenarioConfig) -> str | None:
        speech: int = sum(self._block_length(slot, scenario) for slot in unit.slots)
        length = scenario.noise_length
        if length.mode == "fill_to_target":
            assert length.target_duration_s is not None
            if round(length.target_duration_s * self.sr) - speech < length.min_noise_s * self.sr:
                return "too_long"
        loudness = scenario.loudness
        if loudness.measure == "rms" and loudness.speech_only:
            labelled: int = sum(self.speech_label_samples(i) for slot in unit.slots for i in slot)
            if labelled < loudness.min_speech_s * self.sr:
                return "too_little_speech"
        if loudness.measure == "rms":
            powers: list[tuple[float, int]] = [
                self.speech_power(i, loudness.speech_only) for slot in unit.slots for i in slot
            ]
            count: int = sum(n for _, n in powers)
            level: float = math.sqrt(sum(p for p, _ in powers) / count) if count else 0.0
            if level < self.min_speech_rms(scenario):
                return "too_quiet"
        return None

    def _plan_split(self, split: str, scenarios: list[ScenarioConfig]) -> list[FileSpec]:
        units, candidates, skipped = self._candidates(split, scenarios)
        if self.config.population is not None:
            policy = self.config.population.splits.get(split)
            strata: list[str] = self.config.population.strata
        else:
            policy, strata = None, []
        assigned: list[str] = policy.assigned if policy is not None else []

        all_conditions: list[Condition] = [c for s in scenarios for c in conditions(s)]
        projections: list[tuple[Any, ...]] = list(
            dict.fromkeys(tuple(c.factor(f) for f in assigned) for c in all_conditions)
        )
        # A skipped source counts once per output it would have produced: the scenario's
        # conditions spread over the assigned combinations (1 per train file here).
        for reasons in skipped.values():
            for reason in {r for _, r in reasons}:
                self.skipped_sources[f"{split}/{reason}"] += 1
            for scenario, reason in reasons:
                share: Fraction = Fraction(len(list(conditions(scenario))), len(projections))
                self._skipped_outputs[f"{split}/{reason}"] += share

        counters: Counter[tuple[Any, ...]] = Counter()
        orders: dict[tuple[Any, ...], list[tuple[Any, ...]]] = {}

        specs: list[FileSpec] = []
        cursors: dict[str, int] = {}
        group_offsets: dict[str, int] = {}
        for unit_id, unit in units.items():
            options: list[tuple[ScenarioConfig, Condition]] = candidates[unit_id]
            if assigned:
                stratum: tuple[Any, ...] = tuple(unit.first.fields.get(s) for s in strata)
                if stratum not in orders:
                    rng: np.random.Generator = _seeded_rng(
                        self.config.seed, "population", split, *map(str, stratum)
                    )
                    orders[stratum] = [projections[i] for i in rng.permutation(len(projections))]
                available: set[tuple[Any, ...]] = {
                    tuple(c.factor(f) for f in assigned) for _, c in options
                }
                order: list[tuple[Any, ...]] = orders[stratum]
                start: int = counters[stratum]
                chosen: tuple[Any, ...] | None = next(
                    (
                        order[(start + k) % len(order)]
                        for k in range(len(order))
                        if order[(start + k) % len(order)] in available
                    ),
                    None,
                )
                counters[stratum] += 1
                options = [
                    (s, c) for s, c in options if tuple(c.factor(f) for f in assigned) == chosen
                ]
            for scenario, condition in options:
                specs.append(self._spec(split, unit, scenario, condition, cursors, group_offsets))
        return specs

    def _spec(
        self,
        split: str,
        unit: Unit,
        scenario: ScenarioConfig,
        condition: Condition,
        cursors: dict[str, int],
        group_offsets: dict[str, int],
    ) -> FileSpec:
        sr: int = self.sr
        speech_lengths: list[int] = [self._block_length(slot, scenario) for slot in unit.slots]
        speech: int = sum(speech_lengths)
        length = scenario.noise_length
        if length.mode == "fill_to_target":
            assert length.target_duration_s is not None
            noise_len: int = round(length.target_duration_s * sr) - speech
        else:
            noise_len = noise_length(
                speech, condition.values.get("noise_ratio", length.noise_ratio)
            )

        noise_name: str = condition.values.get("noise") or scenario.noise_block.source or ""
        source: NoiseSource = self.noises[noise_name]
        level_free: dict[str, Any] = {k: v for k, v in condition.values.items() if k != "snr_db"}
        group: str = "/".join(
            [split, unit.id, scenario.name, *map(format_value, level_free.values())]
        )

        fade: int = min(round(scenario.noise_block.fade_ms / 1000 * sr), noise_len // 2)
        blocks: list[BlockSpec] = []
        position: int = 0
        slot_index: int = 0
        gap: int = round(scenario.speech_block.join_gap_s * sr)
        for kind in scenario.layout:
            if kind == "speech":
                refs: list[ItemRef] = []
                cursor: int = position
                for item in unit.slots[slot_index]:
                    refs.append(
                        ItemRef(
                            id=item.id, path=str(item.path), n_samples=item.n_samples, offset=cursor
                        )
                    )
                    cursor += item.n_samples + gap
                blocks.append(
                    BlockSpec(
                        kind="speech", start=position, length=speech_lengths[slot_index], items=refs
                    )
                )
                position += speech_lengths[slot_index]
                slot_index += 1
            else:
                blocks.append(
                    BlockSpec(kind="noise", start=position, length=noise_len, noise=noise_name)
                )
                position += noise_len
        for previous, current in pairwise(blocks):
            if previous.kind == "noise":
                previous.fade_out = fade
            if current.kind == "noise":
                current.fade_in = fade

        noise_block: BlockSpec = next(b for b in blocks if b.kind == "noise")
        noise_block.noise_file = str(source.file(split))
        noise_block.noise_offset = self._noise_offset(
            split, unit, scenario, condition, source, noise_len, group, cursors, group_offsets
        )

        context: dict[str, Any] = self._template_context(split, unit, scenario, condition)
        output = self.config.output
        try:
            out_path: str = (scenario.output_path or output.path).format(**context)
            metadata_dir: str | None = (
                output.metadata_dir.format(**context) if output.metadata_dir is not None else None
            )
        except (KeyError, ValueError, IndexError) as exc:
            raise ConfigError(f"cannot fill output templates for {unit.id}: {exc!r}") from exc
        if metadata_dir is not None and not PurePosixPath(out_path).is_relative_to(metadata_dir):
            raise ConfigError(f"{out_path} is not inside metadata_dir {metadata_dir}")
        label_path: str | None = (
            str(PurePosixPath(out_path).with_suffix(".npy")) if output.write_npy_labels else None
        )

        row_id: str = "-".join(
            [unit.id, scenario.name, *(format_value(v) for v in condition.values.values())]
        )
        snr: Any = condition.values.get("snr_db")
        return FileSpec(
            split=split,
            split_folder=self.config.dataset.splits[split],
            scenario=scenario.name,
            conditions=condition.values,
            unit=unit.id,
            group=group,
            total_samples=position,
            noise_samples=noise_len,
            speech_samples=speech,
            blocks=blocks,
            out_path=out_path,
            metadata_dir=metadata_dir,
            label_path=label_path,
            row_id=row_id,
            snr_db=float(snr) if snr is not None else None,
            fixed_gain=scenario.loudness.gain
            if scenario.loudness.measure == "fixed_gain"
            else None,
        )

    def _noise_offset(
        self,
        split: str,
        unit: Unit,
        scenario: ScenarioConfig,
        condition: Condition,
        source: NoiseSource,
        noise_len: int,
        group: str,
        cursors: dict[str, int],
        group_offsets: dict[str, int],
    ) -> int:
        if group in group_offsets:
            return group_offsets[group]
        total: int = source.length(split)
        if noise_len > total:
            self.warnings[f"noise {source.name} {split} shorter than the excerpt, tiled"] += 1
        if scenario.noise_block.slicing == "sequential_wrap":
            offset: int = cursors.get(source.name, 0)
            cursors[source.name] = (offset + noise_len) % total
        else:
            values: dict[str, Any] = {
                **unit.first.fields,
                "id": unit.id,
                "scenario": scenario.name,
                **condition.values,
            }
            key: list[str] = [format_value(values.get(k)) for k in scenario.noise_block.key]
            offset = keyed_offset(self.config.seed, key, max(0, total - noise_len))
        group_offsets[group] = offset
        return offset

    def _template_context(
        self, split: str, unit: Unit, scenario: ScenarioConfig, condition: Condition
    ) -> dict[str, Any]:
        first: SourceItem = unit.first
        last: SourceItem = unit.last
        return {
            **first.fields,
            "split": split,
            "split_folder": self.config.dataset.splits[split],
            "scenario": scenario.name,
            **condition.values,
            "id": unit.id,
            "first": {**first.fields, "stem": first.path.stem},
            "last": {**last.fields, "stem": last.path.stem},
            "last_suffix": last_suffix(first.path.stem, last.path.stem),
        }

    # --- report -------------------------------------------------------------------------

    def report(self, specs: list[FileSpec]) -> dict[str, Any]:
        output = self.config.output
        strata: list[str] = self.config.population.strata if self.config.population else []
        by_split: Counter[str] = Counter()
        by_condition: Counter[str] = Counter()
        by_stratum: Counter[str] = Counter()
        per_folder: Counter[str] = Counter()
        metadata_dirs: set[str] = set()
        total_bytes: int = 0
        n_labels: int = 0
        audio_seconds: float = 0.0
        index: dict[str, SourceItem] = {}
        if strata:
            index = self.adapter.index(sorted({s.split for s in specs}))
        for spec in specs:
            if not spec.selected:
                continue
            by_split[spec.split] += 1
            condition = Condition(scenario=spec.scenario, values=spec.conditions)
            by_condition[f"{spec.split}/{condition.key()}"] += 1
            if strata:
                fields: dict[str, Any] = index[spec.speech_items()[0].id].fields
                by_stratum[f"{spec.split}/" + "/".join(str(fields.get(s)) for s in strata)] += 1
            per_folder[str(PurePosixPath(spec.out_path).parent)] += 1
            total_bytes += spec.total_samples * 2 + WAV_HEADER_BYTES
            audio_seconds += spec.total_samples / self.sr
            if spec.metadata_dir is not None:
                metadata_dirs.add(spec.metadata_dir)
            if spec.label_path is not None:
                n_labels += 1
                total_bytes += spec.total_samples * 2 + NPY_HEADER_BYTES
        for directory in metadata_dirs:
            per_folder[directory] += 1
        n_wavs: int = sum(by_split.values())
        files_total: int = n_wavs + len(metadata_dirs)
        max_folder: tuple[str, int] = per_folder.most_common(1)[0] if per_folder else ("", 0)

        exceeded: list[str] = []
        if (
            output.limits.max_files_total is not None
            and files_total > output.limits.max_files_total
        ):
            exceeded.append(f"files_total {files_total} > {output.limits.max_files_total}")
        if (
            output.limits.max_files_per_folder is not None
            and max_folder[1] > output.limits.max_files_per_folder
        ):
            exceeded.append(
                f"files in {max_folder[0]} {max_folder[1]} > {output.limits.max_files_per_folder}"
            )
        return {
            "outputs": n_wavs,
            "outputs_by_split": dict(by_split),
            "outputs_by_condition": dict(sorted(by_condition.items())),
            "outputs_by_stratum": dict(sorted(by_stratum.items())),
            "metadata_files": len(metadata_dirs),
            "label_files": n_labels,
            "files_total": files_total,
            "max_files_per_folder": {"folder": max_folder[0], "files": max_folder[1]},
            "bytes": total_bytes,
            "audio_hours": round(audio_seconds / 3600, 3),
            "skipped": {
                key: int(n) if n.denominator == 1 else round(float(n), 2)
                for key, n in sorted(self._skipped_outputs.items())
            },
            "skipped_sources": dict(sorted(self.skipped_sources.items())),
            "warnings": dict(self.warnings),
            "limits": {
                "max_files_total": output.limits.max_files_total,
                "max_files_per_folder": output.limits.max_files_per_folder,
                "exceeded": exceeded,
            },
        }
