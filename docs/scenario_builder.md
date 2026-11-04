# Scenario builder

`librivad.generation.build_scenarios` builds datasets where speech and noise are placed
in one file at a chosen level and length: the noise either fills a block next to the
speech, or runs under the whole file with the speech added on top (`noise_block.span:
file`). The level is set by SNR or by loudness. The whole build is described in one YAML
file (version 2). Examples:

| Config | What it builds |
| :--- | :--- |
| [`configs/scenarios/librispeech_middle.yaml`](../configs/scenarios/librispeech_middle.yaml) | The paper's LibriSpeechConcat (`speech + silence + speech`). [`create_LibriSpeechConcat.py`](../src/librivad/generation/creation_scripts/create_LibriSpeechConcat.py) runs it. |
| [`configs/scenarios/skylar_noisy.example.yaml`](../configs/scenarios/skylar_noisy.example.yaml) | SkylarDataset's `noisy` subset from its `clean` subset: 30 s files with the noise under the whole file, speech first or last, 9 noise kinds, levels by SNR and by loudness. |
| [`docs/concat_scenarios.example.yaml`](concat_scenarios.example.yaml) | Commented reference of every option, on LibriSpeech. |

Noise under the whole file and loudness levels are described in
[plans/05-noise-over-speech.md](plans/05-noise-over-speech.md).

## Running it

```bash
uv run python -m librivad.generation.build_scenarios --config configs/scenarios/librispeech_middle.yaml --dry-run
uv run python -m librivad.generation.build_scenarios --config <config.yaml> --report out/report.json
```

| Flag | Purpose |
| :--- | :--- |
| `--dry-run` | Plan only. Prints counts, disk use and limit checks. With `--report`, also writes `plan.jsonl` (one output per line) next to the report. |
| `--report PATH` | JSON summary: LibriVAD commit and whether the tree was dirty, config hash, counts per split, condition and stratum, skips by reason (`skipped`: outputs a skipped source would have produced under the population; `skipped_sources`: distinct sources), files written, clipping-guard counts, bytes, time. |
| `--scenario NAME`, `--split train\|val\|test` | Restrict the build (both repeatable). The other splits' outputs are unaffected. |
| `--limit N` | Only the first N outputs per condition, for listening checks. |
| `--workers N` | Rendering threads (default 8). Output bytes do not depend on it. |
| `--overwrite` | Re-render outputs that already exist. |
| `--check` | Re-check existing outputs against the plan: length, metadata row, measured level, label length. |
| `--allow-over-limits` | Build even when `output.limits` are exceeded. |

| Exit code | Meaning |
| :--- | :--- |
| 0 | Success |
| 1 | Any other failure, including outputs that failed their checks (listed in the report) |
| 2 | Invalid config (every problem is listed) |
| 3 | `output.limits` exceeded (dry run, or a build without `--allow-over-limits`) |
| 4 | The input corpus breaks its contract (schema version, sample rate, missing or invalid rows) |

## Using it from another project

The builder is importable on Python 3.12 or newer. The core dependencies cover it; the
`paper` extra (creation scripts, `validate.py`, `setup.py`) and the `vad` extra (the models)
are not needed. skylar-dataset depends on this checkout as an editable path:

```toml
[dependency-groups]
noisy = ["librivad"]

[tool.uv.sources]
librivad = { path = "../LibriVAD", editable = true }
```

```python
from librivad.generation import ConfigError, ContractError, LimitsExceeded, config_from_dict, run

config = config_from_dict(raw, base=Path("/base/for/relative/paths"))  # or load_config(path)
report = run(config, splits=["val"], workers=8, progress=False)
report["build"]["written"], report["build"]["failed"]
```

`run()` takes the same options as the flags (`scenarios`, `splits`, `limit`, `workers`,
`overwrite`, `dry_run`, `check_only`, `allow_over_limits`, `plan_file`) and returns the
report the CLI writes with `--report`. Instead of exit codes it raises `ConfigError` (2),
`LimitsExceeded` (3, before anything is written) or `ContractError` (4). Failed renders
are counted in `report["build"]`, not raised. `on_plan` is called with the report once the
plan is ready, before rendering. `report["librivad"]` holds the package version, plus the
commit and dirty flag when the code runs from a checkout.

Importing has no side effects. Only the CLI reads this repository's `.env`, so relative
paths in a config resolve against `base`, else `PROJECT_ROOT`, else the current folder.

## Config at a glance

- **`dataset`** describes the corpus. Nothing corpus-specific is in code.
  - `adapter: path_pattern` parses fields from paths, e.g. `{split}/{speaker}/{chapter}/{speaker}-{chapter}-{utt}.wav`.
  - `adapter: audiofolder` reads `metadata.parquet` files; `file_name` is relative to each file. The file name comes from `metadata_file` in `<root>/../.provenance/schema.json` (default `metadata.parquet`), and every file must have the same column types.
  - `labels.source` gives the speech/non-speech labels: `textgrid`, `npy`, `words` (from the metadata row) or `none`.
- **`noise_sources`** are named recordings:
  - `file_per_split` has `{split}` in its path, so train, val and test never share noise audio.
  - `extracted_from_dataset` is the label-0 regions of a split, cached to a WAV.
- **`output`** gives where and how files are written:
  - `path`: a template of each output's path.
  - `metadata_dir`: a template of the folder that gets `metadata.parquet`, or none. A scenario can replace it (`scenarios[].metadata_dir`), as it can `path` (`output_path`), e.g. when its level factor differs. Every output metadata file has one explicit Arrow schema: carried-over columns keep their input types, `snr_db` and `loudness_lu` are int64, other numbers float64, so a column that is null in a whole folder still has its type.
  - In templates, a zero-padded negative integer keeps its width after the sign: `{snr_db:02d}` gives `-05`.
  - `normalization`, `clip_guard` and `limits`.
- **`scenarios`** each give a `layout` (`[noise, speech]`, `[speech, noise]`, `[speech, noise, speech]`, or `[speech]` with `span: file`). Each is deep-merged over `defaults`; `conditions` lists the values it sweeps (`noise`, `noise_ratio`, `snr_db`, `loudness_lu`).
  - `noise_block.span`: `block` (default) puts the excerpt in the layout's noise block; `file` runs one excerpt at one gain under the whole output and adds the speech on top, so the noise block is just the part without speech.
  - `speech_block.edge_fade_ms`: a fade in and out at each speech block's edges; 10 ms by default with `span: file`, 0 otherwise.
  - `loudness.measure`: `rms` with `snr_db`, `lufs` with `loudness_lu`, or `fixed_gain`.
- **`population`** (optional) sets, per split, which factors are **crossed** (every value per source file) and which are **assigned** (one value per file, balanced inside each stratum). Without it, every file gets every condition.

`load_config` rejects invalid configs with every problem listed, for example:
- a noise louder than `constraints.min_snr_db` (or `min_loudness_lu`) allows;
- a level factor that doesn't match `loudness.measure`;
- a placeholder that is not a field;
- a factor missing from `population`.

## Rules

| Rule | Detail |
| :--- | :--- |
| Noise length | `mode: ratio`: `floor(S·p/(1−p))`, exact (`Fraction`). `mode: fill_to_target`: `T·sr − S`; files leaving less than `min_noise_s` are skipped as `too_long`. |
| Quiet sources | A source whose speech-only RMS is below `loudness.min_speech_rms_dbfs` (default −60 dBFS) is skipped as `too_quiet` by the planner, never rendered. The planner raises that floor to the level where the scenario's highest `snr_db` (or `loudness_lu`, as an estimate) still leaves the noise at ≥ 1.89 LSB, below which int16 rounding moves the measured level by more than 0.1 dB (35 dB → −49.8 dBFS). With `lufs`, speech shorter than one 400 ms block is skipped as `too_short_for_loudness`, and speech below `min_speech_lufs` (−70) as `too_quiet`. |
| Level | `rms`: `gain = speech RMS · 10^(−snr_db/20) / noise RMS`. The speech RMS is over speech-only samples when `speech_only` is set. `lufs`: `gain = 10^((L_speech − loudness_lu − L_noise)/20)`, with `L` the integrated loudness (ITU-R BS.1770-4, `generation/loudness.py`) of the speech blocks and of the noise; gating skips the pauses. The noise is measured over the block without its fades (`span: block`) or the whole excerpt (`span: file`). `fixed_gain` multiplies instead (the paper's 0.001). Every output records both `measured_snr_db` and `measured_loudness_lu`, whichever set the gain. |
| Excerpt | `random_offset` is seeded by a hash of `noise_block.key` (default `[id, scenario, noise]`). Every level of a file gets the same excerpt, and the offset never depends on file order. `sequential_wrap` walks a cursor through the file, as the paper did. With `span: file` the excerpt is as long as the output. |
| Fades | `span: block`: a raised-cosine fade of `fade_ms` on the noise block, only at its edges next to speech; the speech samples are copied unchanged. `span: file`: the noise has no fades; each speech block is faded in and out over `edge_fade_ms`, so its own room tone doesn't switch on abruptly over the noise. |
| Clipping | `normalization: none` scales a whole file down only when rounding to int16 would actually clip (a sample above 32767 or below −32768), to exactly full scale, and records `clip_gain_db`. Sources that already peak at full scale are left alone. `per_group` uses one gain for every level of a file, so a file whose 0 dB version clips is scaled at every level (the report lists both `clip_guard_by_level`, applied, and `clip_guard_triggered_by_level`, where the file itself would have clipped). `normalization: peak` reproduces `librispeech_functions.scale_signal`, truncation included. |
| Labels and timestamps | 0 outside the speech blocks, the source labels over them (with `span: file` too: noise under the speech doesn't change the labels). `words` and `segments` are shifted by where the speech starts, and `output.drop_fields` (e.g. Whisper's `tokens`, `seek`) are removed. |
| Resume | Files are written as `*.tmp` and renamed. Parquet cannot be appended to, so each folder's rows are kept in memory and its `metadata.parquet` is rewritten (atomically) when its last output is written, every 500 rows, and at the end; a wav written after the last checkpoint has no row yet if the run is killed. A rerun first cleans up: `.tmp` files are deleted, planned wavs without a row are moved to `.orphans/<time>/`, and rows without a wav are dropped. It then renders only what is missing. Metadata rows are sorted by `file_name`. |

The metadata row of a skylar output holds:
- `file_name`, `id`, `clean_id`, the condition (`scenario`, `noise`, `snr_db` or `loudness_lu`; the factor a scenario doesn't sweep is null);
- `duration`, `source_duration`, `speech_start`/`end` and `noise_start`/`end` (with `span: file`, 0 and the duration);
- `noise_ratio` (the share of the file without speech), `noise_file`, `noise_offset_s`, `noise_gain_db`, `clip_gain_db`, `measured_snr_db` and `measured_loudness_lu`;
- the passthrough fields of the clean row;
- the shifted `words`.

Positions (`speech_start`, `speech_end`, `noise_start`, `noise_end`, `noise_offset_s`) are
written to 7 decimals, so `round(seconds · sr)` gives back the exact sample. Gains are
written to 0.001 dB.

## Tests

```bash
uv run pytest                 # builder tests plus the existing ones
uv run pytest -m slow         # paper parity: LibriSpeechConcat (about 3 min) and LibriVAD's mixing
```

**Parity with the paper's noisy LibriVAD (`create_LibriVAD.py`).** Layout `[speech]` with
`span: file`, `sequential_wrap`, `normalization: peak` and `edge_fade_ms: 0` is the script's
arithmetic: 16 dev-clean outputs at −5 and 20 dB, recomputed with the script's own
`remove_silence`, `RMS` and `scale_signal` from the same excerpt, are **bit-identical**. The
script's excerpts themselves can't be reproduced: it restarts its noise cursor every batch
and walks files in unsorted `glob` order.

**Parity with the paper's LibriSpeechConcat.** Checked on 2,661 dev-clean and test-clean pairs:
- **Audio:** every file has the same name and length and is **bit-identical**.
- **Train:** the 14,267 planned names match the existing files, and the last 40 pairs (deep into the silence cursor) are bit-identical too.
- **Labels:** 123 of the 2,661 files differ from the paper's labels in `files/Labels/LibriSpeechConcat`. These are bugs in `create_labels.py`, which the builder doesn't share:
  - **91 pairs that cross a chapter or speaker.** `create_labels.py` looks for the first file's TextGrid in the second file's folder, so that half of the label comes out empty.
  - **32 same-chapter pairs.** `create_labels.py` sizes each label by the TextGrid end time, which can fall short of the audio length.

  The builder's labels always have the audio's length. `create_LibriSpeechConcat.py` writes no labels (`create_labels.py` still does), so `setup.py` produces the same files as before.
