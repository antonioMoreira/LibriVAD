# Scenario builder

`librivad.generation.build_scenarios` builds datasets where a speech block and a noise
block are placed next to each other, at a chosen level and length. The whole build is
described in one YAML file (version 2). Examples:

| Config | What it builds |
| :--- | :--- |
| [`configs/scenarios/librispeech_middle.yaml`](../configs/scenarios/librispeech_middle.yaml) | The paper's LibriSpeechConcat (`speech + silence + speech`). [`create_LibriSpeechConcat.py`](../src/librivad/generation/creation_scripts/create_LibriSpeechConcat.py) runs it. |
| [`configs/scenarios/skylar_noisy.example.yaml`](../configs/scenarios/skylar_noisy.example.yaml) | SkylarDataset's `noisy` subset from its `clean` subset: 30 s files, noise first or last, 9 noise kinds, 0–35 dB below the speech. |
| [`docs/concat_scenarios.example.yaml`](concat_scenarios.example.yaml) | Commented reference of every option, on LibriSpeech. |

The design and the reasons behind it are in
[plans/02-librivad-generalization.md](plans/02-librivad-generalization.md).

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
  - `metadata_dir`: a template of the folder that gets `metadata.parquet`, or none. Every output metadata file has one explicit Arrow schema: carried-over columns keep their input types, `snr_db` is int64, other numbers float64, so a column that is null in a whole folder still has its type.
  - `normalization`, `clip_guard` and `limits`.
- **`scenarios`** each give a `layout` (`[noise, speech]`, `[speech, noise]`, `[speech, noise, speech]`). Each is deep-merged over `defaults`; `conditions` lists the values it sweeps (`noise`, `noise_ratio`, `snr_db`).
- **`population`** (optional) sets, per split, which factors are **crossed** (every value per source file) and which are **assigned** (one value per file, balanced inside each stratum). Without it, every file gets every condition.

`load_config` rejects invalid configs with every problem listed, for example:
- a noise louder than `constraints.min_snr_db` allows;
- a placeholder that is not a field;
- a factor missing from `population`.

## Rules

| Rule | Detail |
| :--- | :--- |
| Noise length | `mode: ratio`: `floor(S·p/(1−p))`, exact (`Fraction`). `mode: fill_to_target`: `T·sr − S`; files leaving less than `min_noise_s` are skipped as `too_long`. |
| Quiet sources | A source whose speech-only RMS is below `loudness.min_speech_rms_dbfs` (default −60 dBFS) is skipped as `too_quiet` by the planner, never rendered. The planner raises that floor to the level where the scenario's highest `snr_db` still leaves the noise at ≥ 1.89 LSB, below which int16 rounding moves the measured level by more than 0.1 dB (35 dB → −49.8 dBFS). |
| Level | `gain = speech RMS · 10^(−snr_db/20) / noise RMS`. The speech RMS is over speech-only samples when `speech_only` is set; the noise RMS is over the block without its fades. `fixed_gain` multiplies instead (the paper's 0.001). |
| Excerpt | `random_offset` is seeded by a hash of `noise_block.key` (default `[id, scenario, noise]`). Every level of a file gets the same excerpt, and the offset never depends on file order. `sequential_wrap` walks a cursor through the file, as the paper did. |
| Fades | A raised-cosine fade of `fade_ms` on the noise block, only at its edges next to speech. The speech samples are copied unchanged. |
| Clipping | `normalization: none` scales a whole file down only when rounding to int16 would actually clip (a sample above 32767 or below −32768), to exactly full scale, and records `clip_gain_db`. Sources that already peak at full scale are left alone. `per_group` uses one gain for every level of a file, so a file whose 0 dB version clips is scaled at every level (the report lists both `clip_guard_by_level`, applied, and `clip_guard_triggered_by_level`, where the file itself would have clipped). `normalization: peak` reproduces `librispeech_functions.scale_signal`, truncation included. |
| Labels and timestamps | 0 over the noise block, the source labels over the speech. `words` and `segments` are shifted by where the speech starts, and `output.drop_fields` (e.g. Whisper's `tokens`, `seek`) are removed. |
| Resume | Files are written as `*.tmp` and renamed. Parquet cannot be appended to, so each folder's rows are kept in memory and its `metadata.parquet` is rewritten (atomically) when its last output is written, every 500 rows, and at the end; a wav written after the last checkpoint has no row yet if the run is killed. A rerun first cleans up: `.tmp` files are deleted, planned wavs without a row are moved to `.orphans/<time>/`, and rows without a wav are dropped. It then renders only what is missing. Metadata rows are sorted by `file_name`. |

The metadata row of a skylar output holds:
- `file_name`, `id`, `clean_id`, the condition (`scenario`, `noise`, `snr_db`);
- `duration`, `source_duration`, `speech_start`/`end` and `noise_start`/`end`;
- `noise_ratio`, `noise_file`, `noise_offset_s`, `noise_gain_db`, `clip_gain_db` and `measured_snr_db`;
- the passthrough fields of the clean row;
- the shifted `words`.

## Tests

```bash
uv run pytest                 # 69 builder tests plus the existing ones
uv run pytest -m slow         # paper parity: renders dev-clean and test-clean (about 3 min)
```

**Parity with the paper's LibriSpeechConcat.** Checked on 2,661 dev-clean and test-clean pairs:
- **Audio:** every file has the same name and length and is **bit-identical**.
- **Train:** the 14,267 planned names match the existing files, and the last 40 pairs (deep into the silence cursor) are bit-identical too.
- **Labels:** 123 of the 2,661 files differ from the paper's labels in `files/Labels/LibriSpeechConcat`. These are bugs in `create_labels.py`, which the builder doesn't share:
  - **91 pairs that cross a chapter or speaker.** `create_labels.py` looks for the first file's TextGrid in the second file's folder, so that half of the label comes out empty.
  - **32 same-chapter pairs.** `create_labels.py` sizes each label by the TextGrid end time, which can fall short of the audio length.

  The builder's labels always have the audio's length. `create_LibriSpeechConcat.py` writes no labels (`create_labels.py` still does), so `setup.py` produces the same files as before.
