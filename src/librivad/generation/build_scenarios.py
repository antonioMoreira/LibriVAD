"""
Scenario builder CLI: plans and renders the outputs described by a version-2 scenario config.

Usage:
    uv run python -m librivad.generation.build_scenarios --config <scenarios.yaml> \\
        [--dry-run] [--report report.json] [--scenario NAME]... [--split train|val|test]... \\
        [--limit N] [--workers N] [--overwrite] [--check] [--allow-over-limits]

Exit codes: 0 success, 1 other failure (including outputs that failed their checks),
2 invalid config, 3 limits exceeded, 4 the input corpus breaks its contract.
See docs/scenario_builder.md.
"""

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from tqdm import tqdm

from librivad.generation.config import SPLITS, Config, ConfigError, load_config
from librivad.generation.corpus import ContractError, CorpusAdapter, SourceItem, make_adapter
from librivad.generation.labels import load_labels
from librivad.generation.layout import OutputWriter
from librivad.generation.metadata import build_row, output_schema
from librivad.generation.noise import NoiseSource
from librivad.generation.planner import FileSpec, Planner, PlanResult
from librivad.generation.render import LEVEL_TOLERANCE_DB, Rendered, Renderer, RenderError
from librivad.generation.tables import read_rows

EXIT_OK: int = 0
EXIT_FAILED: int = 1
EXIT_CONFIG: int = 2
EXIT_LIMITS: int = 3
EXIT_CONTRACT: int = 4

REPO_ROOT: Path = Path(__file__).resolve().parents[3]


def git_commit() -> dict[str, Any]:
    """LibriVAD commit and whether the working tree has uncommitted changes."""

    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else ""

    return {"commit": run("rev-parse", "HEAD") or None, "dirty": bool(run("status", "--porcelain"))}


def make_noises(config: Config, adapter: CorpusAdapter) -> dict[str, NoiseSource]:
    return {
        name: NoiseSource(name, source, config.dataset.sample_rate, adapter)
        for name, source in config.noise_sources.items()
    }


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan and render scenario outputs.")
    parser.add_argument("--config", required=True, type=Path, help="Version-2 scenario config.")
    parser.add_argument("--dry-run", action="store_true", help="Plan only; render nothing.")
    parser.add_argument("--report", type=Path, help="Write a JSON report here.")
    parser.add_argument("--scenario", action="append", help="Only these scenarios.")
    parser.add_argument("--split", action="append", choices=SPLITS, help="Only these splits.")
    parser.add_argument("--limit", type=int, help="First N outputs per condition.")
    parser.add_argument("--workers", type=int, default=8, help="Rendering threads.")
    parser.add_argument("--overwrite", action="store_true", help="Re-render existing outputs.")
    parser.add_argument("--check", action="store_true", help="Check existing outputs only.")
    parser.add_argument(
        "--allow-over-limits", action="store_true", help="Build even when limits are exceeded."
    )
    return parser.parse_args(argv)


def write_report(path: Path | None, report: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def print_summary(report: dict[str, Any]) -> None:
    plan: dict[str, Any] = report["plan"]
    print(f"\nOutputs: {plan['outputs']} {plan['outputs_by_split']}")
    print(f"Files total: {plan['files_total']} (metadata files: {plan['metadata_files']})")
    folder: dict[str, Any] = plan["max_files_per_folder"]
    print(f"Max files in one folder: {folder['files']} ({folder['folder']})")
    print(f"Disk: {plan['bytes'] / 1e9:.2f} GB, audio: {plan['audio_hours']} h")
    if plan["skipped"]:
        print(f"Skipped: {plan['skipped']}")
    if plan["warnings"]:
        print(f"Warnings: {plan['warnings']}")
    exceeded: list[str] = plan["limits"]["exceeded"]
    print(f"Limits: {'EXCEEDED ' + '; '.join(exceeded) if exceeded else 'ok'}")


def _level_key(rendered: Rendered) -> str:
    snr: float | None = rendered.spec.snr_db
    return "no_level" if snr is None else f"snr_{snr:02.0f}"


def build(
    config: Config,
    result: PlanResult,
    items: dict[str, SourceItem],
    noises: dict[str, NoiseSource],
    workers: int,
    overwrite: bool,
    input_types: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Render every selected, unfinished output. Returns the build section of the report.

    `input_types` are the Arrow types of the input metadata columns; the output metadata
    keeps them for the columns it carries over.
    """
    schema = output_schema(config, input_types or {}) if config.output.metadata_dir else None
    writer = OutputWriter(config, schema)
    renderer = Renderer(config, items, noises)
    selected: list[FileSpec] = [s for s in result.specs if s.selected]
    pending, cleanup = writer.prepare(selected, overwrite)
    pending_ids: set[str] = {s.row_id for s in pending}

    groups: dict[str, list[FileSpec]] = {}
    for spec in result.specs:
        groups.setdefault(spec.group, []).append(spec)
    todo: list[list[FileSpec]] = [
        g for g in groups.values() if any(s.row_id in pending_ids for s in g)
    ]

    stats: Counter[str] = Counter()
    failures: list[str] = []
    clipped: Counter[str] = Counter()
    triggered: Counter[str] = Counter()

    def run_group(group: list[FileSpec]) -> list[Rendered | RenderError]:
        rendered: list[Rendered | RenderError] = renderer.render_group(group)
        for outcome in rendered:
            if isinstance(outcome, Rendered) and outcome.spec.row_id in pending_ids:
                row = build_row(outcome, items, config) if outcome.spec.metadata_dir else None
                writer.write(outcome, row)
        return rendered

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for outcomes in tqdm(executor.map(run_group, todo), total=len(todo), desc="Rendering"):
            for outcome in outcomes:
                if isinstance(outcome, RenderError):
                    stats["failed"] += 1
                    failures.append(str(outcome))
                elif outcome.spec.row_id in pending_ids:
                    stats["written"] += 1
                    if outcome.clip_gain_db < 0:
                        clipped[_level_key(outcome)] += 1
                    if outcome.clip_triggered:
                        triggered[_level_key(outcome)] += 1
    writer.finalize()
    return {
        "written": stats["written"],
        "already_done": len(selected) - len(pending),
        "failed": stats["failed"],
        "failures": failures[:50],
        "clip_guard_by_level": dict(sorted(clipped.items())),
        "clip_guard_triggered_by_level": dict(sorted(triggered.items())),
        "cleanup": {
            "removed_tmp": cleanup.removed_tmp,
            "orphans_moved": len(cleanup.orphans),
            "dropped_rows": cleanup.dropped_rows,
        },
    }


def check(
    config: Config, result: PlanResult, items: dict[str, SourceItem], renderer: Renderer
) -> dict[str, Any]:
    """Re-validate existing outputs against the plan without rendering anything."""
    writer = OutputWriter(config, None)
    problems: list[str] = []
    checked: int = 0
    rows: dict[Path, dict[str, dict[str, Any]]] = {}
    for spec in (s for s in result.specs if s.selected):
        wav: Path = writer.wav_path(spec)
        if not wav.is_file():
            problems.append(f"missing {spec.out_path}")
            continue
        checked += 1
        audio: np.ndarray = sf.read(str(wav), dtype="int16")[0]
        if len(audio) != spec.total_samples:
            problems.append(f"{spec.out_path}: {len(audio)} samples, planned {spec.total_samples}")
            continue
        metadata: Path | None = writer.metadata_path(spec)
        if metadata is not None:
            if metadata not in rows:
                rows[metadata] = {row["id"]: row for row in read_rows(metadata)}
            if spec.row_id not in rows[metadata]:
                problems.append(f"{spec.out_path}: no metadata row")
        label_path: Path | None = writer.label_path(spec)
        if label_path is not None and (
            not label_path.is_file() or len(np.load(label_path)) != spec.total_samples
        ):
            problems.append(f"{spec.out_path}: label missing or of the wrong length")
        if spec.snr_db is not None:
            labels: np.ndarray = np.zeros(spec.total_samples, dtype=np.int8)
            for ref in spec.speech_items():
                labels[ref.offset : ref.offset + ref.n_samples] = load_labels(
                    items[ref.id], config.dataset
                )
            measured: float = renderer.measure_snr(spec, audio, labels)
            if abs(measured - spec.snr_db) > LEVEL_TOLERANCE_DB:
                problems.append(
                    f"{spec.out_path}: measured {measured:.3f} dB, planned {spec.snr_db}"
                )
    return {"checked": checked, "problems": len(problems), "details": problems[:100]}


def main(argv: Sequence[str] | None = None) -> int:
    args: argparse.Namespace = parse_args(argv)
    started: float = time.monotonic()
    report: dict[str, Any] = {"librivad": git_commit(), "config": str(args.config)}
    try:
        config: Config = load_config(args.config)
        report["config_hash"] = config.config_hash
        names: list[str] = [s.name for s in config.scenarios]
        if args.scenario and (unknown := set(args.scenario) - set(names)):
            raise ConfigError(f"unknown scenarios {sorted(unknown)}; config has {names}")
        splits: list[str] = [s for s in SPLITS if s in config.dataset.splits]
        if args.split:
            splits = [s for s in splits if s in args.split]
        adapter: CorpusAdapter = make_adapter(config.dataset)
        noises: dict[str, NoiseSource] = make_noises(config, adapter)
        planner = Planner(config, adapter, noises)
        result: PlanResult = planner.plan(splits, args.scenario, args.limit)
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    except ContractError as exc:
        print(f"Input contract violated: {exc}", file=sys.stderr)
        return EXIT_CONTRACT

    report["plan"] = result.report
    report["splits"] = splits
    print_summary(report)

    if args.dry_run:
        if args.report is not None:
            plan_path: Path = args.report.parent / "plan.jsonl"
            plan_path.parent.mkdir(parents=True, exist_ok=True)
            with open(plan_path, "w", encoding="utf-8") as handle:
                for spec in result.specs:
                    handle.write(spec.model_dump_json() + "\n")
            report["plan_file"] = str(plan_path)
        report["seconds"] = round(time.monotonic() - started, 1)
        write_report(args.report, report)
        return EXIT_LIMITS if result.over_limits() else EXIT_OK

    items: dict[str, SourceItem] = adapter.index(splits)
    if args.check:
        report["check"] = check(config, result, items, Renderer(config, items, noises))
        report["seconds"] = round(time.monotonic() - started, 1)
        write_report(args.report, report)
        print(f"Checked {report['check']['checked']}, problems: {report['check']['problems']}")
        return EXIT_FAILED if report["check"]["problems"] else EXIT_OK

    if result.over_limits() and not args.allow_over_limits:
        print("Limits exceeded; nothing was written (use --allow-over-limits).", file=sys.stderr)
        write_report(args.report, report)
        return EXIT_LIMITS

    try:
        report["build"] = build(
            config, result, items, noises, args.workers, args.overwrite, adapter.field_types
        )
    except ContractError as exc:
        print(f"Input contract violated: {exc}", file=sys.stderr)
        write_report(args.report, report)
        return EXIT_CONTRACT
    report["seconds"] = round(time.monotonic() - started, 1)
    write_report(args.report, report)
    built: dict[str, Any] = report["build"]
    print(
        f"Written {built['written']}, already done {built['already_done']},"
        f" failed {built['failed']}"
    )
    for failure in built["failures"][:10]:
        print(f"  {failure}", file=sys.stderr)
    return EXIT_FAILED if built["failed"] else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
