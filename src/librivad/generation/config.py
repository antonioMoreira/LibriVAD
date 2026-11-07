"""
Loads and validates scenario configs (version 2) for the scenario builder.

A config describes one corpus (`dataset`), the noise recordings (`noise_sources`), where
and how outputs are written (`output`), and a list of `scenarios`. Each scenario is
deep-merged over `defaults` (dicts merge key by key, lists and scalars are replaced), and
the top-level `conditions` are merged the same way. The format is documented in
docs/scenario_builder.md.
"""

import hashlib
import json
import os
from collections.abc import Iterator
from itertools import product
from pathlib import Path
from string import Formatter
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

FACTOR_KEYS: tuple[str, ...] = ("noise", "noise_ratio", "snr_db", "loudness_lu")
"""Condition keys a scenario can sweep. `scenario` is always a factor as well."""

LEVEL_FACTORS: tuple[str, ...] = ("snr_db", "loudness_lu")
"""The condition keys that set the noise level. Outputs that differ only by them form one
group (same sources, same excerpt, same clip guard)."""

SPLITS: tuple[str, ...] = ("train", "val", "test")


class ConfigError(ValueError):
    """Raised when a scenario config is invalid. The CLI exits with code 2."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LabelsConfig(_Model):
    """Where the speech/non-speech labels of each source file come from."""

    source: Literal["textgrid", "npy", "words", "none"] = "none"
    dir: Path | None = None
    path: str | None = None
    """Template relative to `dir`, e.g. `{split_folder}/{speaker}/{chapter}/{stem}.TextGrid`."""
    tier: str = "words"
    silence_token: str = ""
    field: str = "words"
    """Metadata field holding the word list (`source: words`)."""
    bridge_gaps_ms: float = 0.0
    """Gaps between words shorter than this are labelled speech (`source: words`)."""


class ContinuityConfig(_Model):
    group_by: list[str]
    order_by: str


class DatasetConfig(_Model):
    name: str
    adapter: Literal["path_pattern", "audiofolder"]
    root: Path
    sample_rate: int = 16000
    splits: dict[str, str]
    """Generic split name (`train`, `val`, `test`) -> folder name in this corpus."""
    path_pattern: str | None = None
    exclude_list: Path | None = None
    schema_version: int | None = None
    metadata_file: str | None = None
    """`audiofolder`: metadata file name; default `metadata_file` from `.provenance/schema.json`,
    else `metadata.parquet`."""
    metadata_glob: str = "{split}/*/{metadata_file}"
    order: Literal["path", "metadata"] = "path"
    """`audiofolder`: the order of a split's items, which the population's balanced assignment
    follows. `path` sorts them by file path. `metadata` keeps the metadata files' row order (files
    in path order): a corpus that only appends rows then keeps every existing item's assignment
    when it grows."""
    id_field: str = "id"
    passthrough_fields: list[str] = []
    labels: LabelsConfig = LabelsConfig()
    continuity: ContinuityConfig | None = None

    @model_validator(mode="after")
    def _adapter_fields(self) -> Self:
        if self.adapter == "path_pattern" and not self.path_pattern:
            raise ValueError("adapter path_pattern needs `path_pattern`")
        if self.order == "metadata" and self.adapter != "audiofolder":
            raise ValueError("order metadata needs adapter audiofolder")
        if unknown := set(self.splits) - set(SPLITS):
            raise ValueError(f"unknown split names {sorted(unknown)}; use {list(SPLITS)}")
        return self


class NoiseSourceConfig(_Model):
    type: Literal["file_per_split", "extracted_from_dataset"]
    path: str | None = None
    """`file_per_split`: path template containing `{split}`."""
    split: str | None = None
    """`extracted_from_dataset`: generic split the label-0 regions are cut from."""
    cache: Path | None = None

    @model_validator(mode="after")
    def _type_fields(self) -> Self:
        if self.type == "file_per_split" and (not self.path or "{split}" not in self.path):
            raise ValueError("file_per_split needs a `path` containing {split}")
        if self.type == "extracted_from_dataset" and (self.split is None or self.cache is None):
            raise ValueError("extracted_from_dataset needs `split` and `cache`")
        return self


class Limits(_Model):
    max_files_total: int | None = None
    max_files_per_folder: int | None = None


class OutputConfig(_Model):
    root: Path
    path: str
    """Template of each output wav, relative to `root`."""
    metadata_dir: str | None = None
    """Template of the folder holding `metadata.parquet`. None writes no metadata."""
    metadata_format: Literal["parquet"] = "parquet"
    write_npy_labels: bool = False
    labels_root: Path | None = None
    dtype: Literal["int16"] = "int16"
    normalization: Literal["none", "peak"] = "none"
    peak_dbfs: float = 0.0
    clip_guard: Literal["per_group", "per_file", "off"] = "per_group"
    drop_fields: list[str] = []
    limits: Limits = Limits()

    @model_validator(mode="after")
    def _labels_root(self) -> Self:
        if self.write_npy_labels and self.labels_root is None:
            raise ValueError("write_npy_labels needs `labels_root`")
        return self


class SpeechBlockConfig(_Model):
    composition: Literal["single", "concatenate", "contiguous"] = "single"
    max_utterances: int = Field(1, ge=1)
    group_by: list[str] = []
    """`concatenate` only: join only items that share these fields."""
    selection: Literal["sequential", "shuffled"] = "sequential"
    duration_tolerance: float | None = None
    """`single` with a speech target: keep items within this fraction of the target."""
    join_gap_s: float = 0.0
    join_fade_ms: float = 0.0
    edge_fade_ms: float | None = None
    """Fade in and out at the edges of each speech block. Default: 10 ms with
    `noise_block.span: file` (the speech is added over the noise), 0 otherwise."""

    def edge_fade(self, span: str) -> float:
        if self.edge_fade_ms is not None:
            return self.edge_fade_ms
        return 10.0 if span == "file" else 0.0


class NoiseLengthConfig(_Model):
    mode: Literal["ratio", "fill_to_target"] = "ratio"
    noise_ratio: float | None = None
    target_duration_s: float | None = None
    min_noise_s: float = 0.0


class NoiseBlockConfig(_Model):
    source: str | None = None
    """Noise source when `conditions` has no `noise` factor."""
    span: Literal["block", "file"] = "block"
    """`block`: the noise fills the layout's noise block only. `file`: one excerpt runs
    under the whole output at one gain and the speech is added on top; the layout's noise
    block is then the part without speech, and the layout may have none (`[speech]`)."""
    slicing: Literal["random_offset", "sequential_wrap"] = "random_offset"
    key: list[str] = ["id", "scenario", "noise"]
    """Values hashed to choose a random offset. Leave the level out to reuse the excerpt."""
    fade_ms: float = 10.0
    """`span: block` only: fade of the noise block at its edges next to speech."""


class LoudnessConfig(_Model):
    measure: Literal["rms", "lufs", "fixed_gain"] = "rms"
    """`rms` sets the noise by `snr_db`, `lufs` by `loudness_lu` (ITU-R BS.1770 integrated
    loudness of the speech block minus that of the noise), `fixed_gain` by `gain`."""
    speech_only: bool = True
    min_speech_s: float = 0.5
    min_speech_rms_dbfs: float = -60.0
    """Sources whose speech-only RMS is quieter are skipped (`too_quiet`). The planner also
    raises it to the level where the highest `snr_db` or `loudness_lu` still fits in int16
    precision."""
    min_speech_lufs: float = -70.0
    """`lufs`: sources whose gated loudness is quieter are skipped (`too_quiet`)."""
    gain: float | None = None


class ConstraintsConfig(_Model):
    min_snr_db: float | None = None
    min_loudness_lu: float | None = None


class ScenarioConfig(_Model):
    name: str
    layout: list[Literal["speech", "noise"]]
    speech_block: SpeechBlockConfig = SpeechBlockConfig()
    noise_length: NoiseLengthConfig = NoiseLengthConfig()
    noise_block: NoiseBlockConfig = NoiseBlockConfig()
    loudness: LoudnessConfig = LoudnessConfig()
    constraints: ConstraintsConfig = ConstraintsConfig()
    conditions: dict[str, list[Any]] = {}
    output_path: str | None = None
    """Replaces `output.path` for this scenario (its conditions may differ from the others)."""
    metadata_dir: str | None = None
    """Replaces `output.metadata_dir` for this scenario, e.g. when its level factor differs."""

    @property
    def n_speech(self) -> int:
        return self.layout.count("speech")


class SplitPopulation(_Model):
    crossed: list[str] = []
    assigned: list[str] = []


class PopulationConfig(_Model):
    strata: list[str] = []
    splits: dict[str, SplitPopulation] = {}

    @model_validator(mode="before")
    @classmethod
    def _split_keys(cls, data: Any) -> Any:
        if isinstance(data, dict) and "splits" not in data:
            data = dict(data)
            strata = data.pop("strata", [])
            return {"strata": strata, "splits": data}
        return data


class Config(_Model):
    version: Literal[2]
    seed: int = 0
    dataset: DatasetConfig
    noise_sources: dict[str, NoiseSourceConfig] = {}
    output: OutputConfig
    scenarios: list[ScenarioConfig]
    population: PopulationConfig | None = None
    config_hash: str = ""

    def factors(self, scenarios: list[ScenarioConfig] | None = None) -> list[str]:
        """`scenario` plus every condition key used by the given scenarios, in order."""
        keys: list[str] = ["scenario"]
        for scenario in scenarios or self.scenarios:
            keys += [k for k in scenario.conditions if k not in keys]
        return keys

    def scenario(self, name: str) -> ScenarioConfig:
        for scenario in self.scenarios:
            if scenario.name == name:
                return scenario
        raise ConfigError(f"unknown scenario {name!r}")


class Condition(BaseModel):
    """One combination of scenario and condition values, e.g. noise_first/office/snr 5."""

    scenario: str
    values: dict[str, Any]

    def factor(self, name: str) -> Any:
        return self.scenario if name == "scenario" else self.values.get(name)

    def key(self) -> str:
        return "/".join([self.scenario, *(format_value(v) for v in self.values.values())])


def format_value(value: Any) -> str:
    """Folder-friendly text of a condition value: ints zero-padded to two digits, the sign
    outside the padding (`-05`, `05`)."""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return f"-{-value:02d}" if value < 0 else f"{value:02d}"
    return str(value)


class _TemplateFormatter(Formatter):
    """`str.format` where a zero-padded width does not count the minus sign, so
    `{snr_db:02d}` gives `-05` and `05`, like `format_value`."""

    def format_field(self, value: Any, format_spec: str) -> str:
        if (
            isinstance(value, int)
            and not isinstance(value, bool)
            and value < 0
            and format_spec[:1] == "0"
            and format_spec[1:-1].isdigit()
            and format_spec.endswith("d")
        ):
            return "-" + format(-value, format_spec)
        return super().format_field(value, format_spec)


def format_template(template: str, context: dict[str, Any]) -> str:
    """Fill an output path template (see `_TemplateFormatter`)."""
    return _TemplateFormatter().vformat(template, (), context)


def conditions(scenario: ScenarioConfig) -> Iterator[Condition]:
    """Cartesian product of the scenario's condition lists, in key order."""
    keys: list[str] = list(scenario.conditions)
    for values in product(*(scenario.conditions[k] for k in keys)):
        yield Condition(scenario=scenario.name, values=dict(zip(keys, values, strict=True)))


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge dicts key by key; any other value in `override` replaces the base's."""
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def project_root() -> Path:
    """Base for relative paths: `PROJECT_ROOT` from the environment, else the current folder."""
    return Path(os.environ.get("PROJECT_ROOT", ".")).resolve()


def _resolve(path: Path | str, base: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else base / path


def template_fields(template: str) -> set[str]:
    """Base names used by a `str.format` template (`{last[speaker]}` -> `last`)."""
    names: set[str] = set()
    for _, field, _, _ in Formatter().parse(template):
        if field:
            names.add(field.split("[", 1)[0].split(".", 1)[0])
    return names


def load_config(path: str | Path, base: Path | None = None) -> Config:
    """Load a version-2 scenario config, merge each scenario and validate every rule.

    Relative paths are resolved against `base`, or `PROJECT_ROOT` when `base` is None.
    Raises `ConfigError` with a readable message on any problem.
    """
    raw_text: str = Path(path).read_text(encoding="utf-8")
    try:
        raw: dict[str, Any] = yaml.safe_load(raw_text) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    return config_from_dict(raw, base)


def config_from_dict(raw: dict[str, Any], base: Path | None = None) -> Config:
    """Build a `Config` from an already parsed YAML document (see `load_config`)."""
    raw = dict(raw)
    defaults: dict[str, Any] = raw.pop("defaults", {}) or {}
    shared_conditions: dict[str, Any] = raw.pop("conditions", {}) or {}
    scenarios: list[Any] = raw.get("scenarios") or []
    if not isinstance(scenarios, list) or not scenarios:
        raise ConfigError("`scenarios` must be a non-empty list")
    merged_scenarios: list[dict[str, Any]] = []
    for scenario in scenarios:
        if not isinstance(scenario, dict):
            raise ConfigError(f"each scenario must be a mapping, got {scenario!r}")
        merged: dict[str, Any] = deep_merge(defaults, scenario)
        merged["conditions"] = deep_merge(shared_conditions, scenario.get("conditions") or {})
        merged_scenarios.append(merged)
    raw["scenarios"] = merged_scenarios
    raw["config_hash"] = hashlib.sha256(
        json.dumps(raw, sort_keys=True, default=str).encode()
    ).hexdigest()

    try:
        config: Config = Config.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(exc)) from exc

    _resolve_paths(config, base or project_root())
    check_config(config)
    return config


def _format_validation_error(exc: ValidationError) -> str:
    lines: list[str] = []
    for error in exc.errors():
        location: str = ".".join(str(part) for part in error["loc"])
        lines.append(f"{location}: {error['msg']}")
    return "invalid config:\n  " + "\n  ".join(lines)


def _resolve_paths(config: Config, base: Path) -> None:
    dataset: DatasetConfig = config.dataset
    dataset.root = _resolve(dataset.root, base)
    if dataset.exclude_list is not None:
        dataset.exclude_list = _resolve(dataset.exclude_list, base)
    if dataset.labels.dir is not None:
        dataset.labels.dir = _resolve(dataset.labels.dir, base)
    for source in config.noise_sources.values():
        if source.path is not None:
            source.path = str(_resolve(source.path, base))
        if source.cache is not None:
            source.cache = _resolve(source.cache, base)
    config.output.root = _resolve(config.output.root, base)
    if config.output.labels_root is not None:
        config.output.labels_root = _resolve(config.output.labels_root, base)


def item_field_names(dataset: DatasetConfig) -> set[str]:
    """Fields every source item of this corpus provides to templates and grouping."""
    names: set[str] = {"split", "split_folder", "stem", "id"}
    if dataset.adapter == "path_pattern" and dataset.path_pattern:
        names |= template_fields(dataset.path_pattern)
    else:
        names |= {dataset.id_field, *dataset.passthrough_fields}
    return names


def check_config(config: Config) -> None:
    """Enforce the validation rules of plan 2 §5.3. Raises `ConfigError` listing every problem."""
    errors: list[str] = []
    dataset: DatasetConfig = config.dataset
    fields: set[str] = item_field_names(dataset)

    if dataset.labels.source == "words" and dataset.adapter != "audiofolder":
        errors.append("labels.source words needs adapter audiofolder")
    if dataset.labels.source in ("textgrid", "npy") and (
        dataset.labels.dir is None or not dataset.labels.path
    ):
        errors.append(f"labels.source {dataset.labels.source} needs labels.dir and labels.path")
    if dataset.continuity is not None:
        missing: set[str] = {*dataset.continuity.group_by, dataset.continuity.order_by} - fields
        if missing:
            errors.append(f"continuity uses unknown fields {sorted(missing)}")

    for name, source in config.noise_sources.items():
        if source.type == "extracted_from_dataset" and dataset.labels.source == "none":
            errors.append(f"noise source {name}: extracted_from_dataset needs labels")

    names: list[str] = [s.name for s in config.scenarios]
    if len(set(names)) != len(names):
        errors.append("scenario names must be unique")

    for scenario in config.scenarios:
        errors += [f"scenario {scenario.name}: {e}" for e in _check_scenario(config, scenario)]

    errors += _check_population(config)
    errors += _check_templates(config, fields)

    if errors:
        raise ConfigError("invalid config:\n  " + "\n  ".join(errors))


def _check_scenario(config: Config, scenario: ScenarioConfig) -> list[str]:
    errors: list[str] = []
    layout: list[str] = list(scenario.layout)
    has_noise_block: bool = "noise" in layout
    if scenario.noise_block.span == "file":
        if layout.count("noise") > 1:
            errors.append("layout needs at most one `noise`")
    elif layout.count("noise") != 1:
        errors.append("layout needs exactly one `noise` (or noise_block.span file)")
    if scenario.n_speech not in (1, 2):
        errors.append("layout needs `speech` once or twice")

    if unknown := set(scenario.conditions) - set(FACTOR_KEYS):
        errors.append(f"unknown condition keys {sorted(unknown)}; use {list(FACTOR_KEYS)}")
    for key, values in scenario.conditions.items():
        if not isinstance(values, list) or not values:
            errors.append(f"conditions.{key} must be a non-empty list")

    block: SpeechBlockConfig = scenario.speech_block
    if block.composition == "contiguous" and config.dataset.continuity is None:
        errors.append("composition contiguous needs dataset.continuity")
    if unknown_fields := set(block.group_by) - item_field_names(config.dataset):
        errors.append(f"speech_block.group_by uses unknown fields {sorted(unknown_fields)}")

    length: NoiseLengthConfig = scenario.noise_length
    ratios: list[Any] = scenario.conditions.get("noise_ratio") or []
    if not has_noise_block:
        if ratios:
            errors.append("noise_ratio conditions need a `noise` in the layout")
    elif length.mode == "ratio":
        if length.noise_ratio is not None and ratios:
            errors.append("give noise_ratio in noise_length or in conditions, not both")
        if length.noise_ratio is None and not ratios:
            errors.append("noise_length mode ratio needs a noise_ratio")
        for ratio in [length.noise_ratio, *ratios]:
            if ratio is not None and not (isinstance(ratio, int | float) and 0 <= ratio < 1):
                errors.append(f"noise_ratio {ratio!r} must be in [0, 1)")
    else:
        if ratios:
            errors.append("noise_ratio conditions need noise_length mode ratio")
        target: float | None = length.target_duration_s
        if target is None or not target > length.min_noise_s >= 0:
            errors.append("fill_to_target needs target_duration_s > min_noise_s >= 0")

    loudness: LoudnessConfig = scenario.loudness
    wanted: dict[str, str] = {"rms": "snr_db", "lufs": "loudness_lu"}
    for measure, factor in wanted.items():
        if scenario.conditions.get(factor) and loudness.measure != measure:
            errors.append(
                f"loudness.measure {loudness.measure} excludes {factor} (needs {measure})"
            )
    if loudness.measure == "fixed_gain" and loudness.gain is None:
        errors.append("loudness.measure fixed_gain needs loudness.gain")
    elif loudness.measure in wanted and not scenario.conditions.get(wanted[loudness.measure]):
        errors.append(
            f"loudness.measure {loudness.measure} needs conditions.{wanted[loudness.measure]}"
        )
    minimums: dict[str, tuple[str, float | None]] = {
        "snr_db": ("min_snr_db", scenario.constraints.min_snr_db),
        "loudness_lu": ("min_loudness_lu", scenario.constraints.min_loudness_lu),
    }
    for factor, (name, minimum) in minimums.items():
        for level in scenario.conditions.get(factor) or []:
            if not isinstance(level, int | float) or isinstance(level, bool):
                errors.append(f"{factor} {level!r} must be a number")
            elif minimum is not None and level < minimum:
                errors.append(
                    f"{factor} {level} is below constraints.{name} {minimum}"
                    " (the noise would be louder than allowed)"
                )

    noises: list[Any] = scenario.conditions.get("noise") or []
    if noises and scenario.noise_block.source is not None:
        errors.append("give the noise in noise_block.source or in conditions.noise, not both")
    for noise in noises or [scenario.noise_block.source]:
        if noise is None:
            errors.append("no noise: set conditions.noise or noise_block.source")
        elif noise not in config.noise_sources:
            errors.append(f"noise {noise!r} is not in noise_sources")

    key_fields: set[str] = set(scenario.noise_block.key) - {"scenario", "id"}
    key_fields -= set(FACTOR_KEYS) | item_field_names(config.dataset)
    if key_fields:
        errors.append(f"noise_block.key uses unknown names {sorted(key_fields)}")
    return errors


def _check_population(config: Config) -> list[str]:
    if config.population is None:
        return []
    errors: list[str] = []
    factors: set[str] = set(config.factors())
    for split, policy in config.population.splits.items():
        if split not in SPLITS:
            errors.append(f"population.{split}: unknown split")
            continue
        named: list[str] = policy.crossed + policy.assigned
        if unknown := set(named) - factors:
            errors.append(
                f"population.{split}: {sorted(unknown)} are not factors {sorted(factors)}"
            )
        repeated: set[str] = {n for n in named if named.count(n) > 1}
        missing: set[str] = factors - set(named)
        if repeated or missing:
            errors.append(
                f"population.{split}: every factor must appear exactly once in crossed or"
                f" assigned (missing {sorted(missing)}, repeated {sorted(repeated)})"
            )
    if unknown_strata := set(config.population.strata) - item_field_names(config.dataset):
        errors.append(f"population.strata uses unknown fields {sorted(unknown_strata)}")
    return errors


def _check_templates(config: Config, fields: set[str]) -> list[str]:
    errors: list[str] = []
    output: OutputConfig = config.output
    common: set[str] = fields | {
        "scenario",
        "first",
        "last",
        "last_suffix",
        "split",
        "split_folder",
    }
    for scenario in config.scenarios:
        allowed: set[str] = common | set(scenario.conditions)
        path: str = scenario.output_path or output.path
        where: str = f"scenario {scenario.name}:"
        if unknown := template_fields(path) - allowed:
            errors.append(f"{where} output path uses unknown placeholders {sorted(unknown)}")
        if not path.endswith((".wav", ".flac")):
            errors.append(f"{where} output path must end in .wav or .flac, got {path!r}")
        metadata_dir: str | None = scenario.metadata_dir or output.metadata_dir
        if metadata_dir is None:
            continue
        if unknown := template_fields(metadata_dir) - allowed:
            errors.append(
                f"{where} output.metadata_dir uses unknown placeholders {sorted(unknown)}"
            )
        if not path.startswith(metadata_dir.rstrip("/") + "/"):
            errors.append(f"{where} output.metadata_dir must be a parent folder of the output path")
    labels_path: str | None = config.dataset.labels.path
    if labels_path is not None and (unknown := template_fields(labels_path) - fields):
        errors.append(f"labels.path uses unknown placeholders {sorted(unknown)}")
    return errors
