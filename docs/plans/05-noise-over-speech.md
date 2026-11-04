# Plan 5 — Noise over the whole file, levelled by SNR or by loudness

Status: implemented 2026-11-04, uncommitted, pending review. The skylar side is
`skylar-dataset/docs/plans/05-noisy-noise-over-speech.md`. See "Implementation notes" at
the end for what differs from the plan.

## Goal

The scenario builder can lay **one continuous noise excerpt under the whole
output file**, with the speech added on top where the layout puts it. That gives
noisy speech (for ASR) and removes the VAD shortcut of today's layouts, where
the noise starts exactly where the speech ends. The noise level can be set by
SNR, as today, or by **loudness** (LUFS).

```
noise_last, span: file    [ speech + noise ][ noise ]     one excerpt, one gain, 30 s
noise_first, span: file   [ noise ][ noise + speech ]
middle, span: file        [ speech + noise ][ noise ][ speech + noise ]
[speech], span: file      [ speech + noise ]              the paper's create_LibriVAD.py
```

Decided in the discussion:
- The speech position comes from the scenario's `layout`, as today.
- Over the speech and in the noise-only part it is **the same excerpt at the same
  gain**. Two excerpts, or two levels, would change the noise at the join, a
  cue a VAD model can learn.
- Both measures are kept: `rms` with `snr_db` (speech-only RMS, today's), and
  `lufs` with `loudness_lu` (integrated loudness, ITU-R BS.1770). One gain can
  only meet one of them; the other is measured and written.

This is decision G4 (`post_mixing`), plus the `lufs` option that version 1 of
the example config listed and removed.

## Configuration

```yaml
noise_block:
  span: file              # block (today, the default) | file
  slicing: random_offset
  key: [id, scenario, noise]
  fade_ms: 10             # span block: at the block's edges; span file: unused
speech_block:
  edge_fade_ms: 10        # new; span file: fade the speech recording in and out over the noise
loudness:
  measure: lufs           # rms | lufs | fixed_gain
conditions:
  loudness_lu: [-5, 0, 5, 10, 15, 20]
```

Rules the loader adds:
- `span: file` allows a layout of `[speech]` alone. Every other layout still
  has exactly one `noise`.
- `measure: rms` needs `snr_db`; `measure: lufs` needs `loudness_lu` and
  excludes `snr_db`; `fixed_gain` excludes both.
- `constraints.min_snr_db` applies to `rms` only. A new
  `constraints.min_loudness_lu` applies to `lufs`.
- With `span: file`, a noise recording must be at least as long as the output
  file. Shorter ones are tiled with a warning, as today.

## Rendering

1. The planner sizes the file from `layout` and `noise_length` as today.
2. Noise:
   - With `span: file`, one excerpt of the whole output length is cut at the
     keyed random offset.
   - Leave the level out of `key` to share the excerpt across levels, as today.
3. Speech:
   - Each speech block is faded in and out over `edge_fade_ms` (raised cosine).
   - It is added onto the excerpt at its layout position.
   - Labels and `words` are shifted as today.
4. Gain:
   - `rms`: `gain = speech RMS · 10^(−snr_db/20) / noise RMS`. The speech RMS is
     taken over label-1 samples, the noise RMS over the whole excerpt.
   - `lufs`: `gain = 10^((L_speech − loudness_lu − L_noise)/20)`, where `L` is
     integrated loudness. `L_speech` comes from the speech block, whose gating
     skips the pauses. `L_noise` comes from the excerpt.
5. Clipping guard:
   - `per_group` stays, grouping every level of one (file, scenario, noise).
   - It will act more often, because noise now adds to the speech peaks.
   - `normalization: peak` keeps working for paper parity.

## Loudness meter

BS.1770-4 integrated loudness, implemented in `generation/loudness.py`:
- K-weighting: two biquads, with coefficients derived for any sample rate;
- 400 ms blocks with 75 % overlap;
- an absolute gate at −70 LUFS and a relative gate at −10 LU.

No new core dependency. A dev-only test compares it with `pyloudnorm` on noise,
tones and LibriSpeech files, within 0.1 LU.

A file shorter than one 400 ms block has no integrated loudness and is skipped,
as `too_short_for_loudness`. A speech block whose gated loudness is below
−70 LUFS is skipped as `too_quiet`, like the RMS floor.

## Metadata

Every output row gets, whichever measure set the gain:
- `measured_snr_db`: speech-only RMS of the speech minus RMS of the noise, as
  mixed, in dB;
- `measured_loudness_lu`: integrated loudness of the speech minus that of the
  noise, as mixed.

A condition the scenario doesn't sweep is absent, or null in a shared Arrow
schema: `snr_db` in `lufs` scenarios, `loudness_lu` in `rms` ones.

With `span: file`:
- `noise_start = 0` and `noise_end = duration`;
- `noise_ratio` is the share of the file with no speech block, as before.

## Folder names

`format_value` pads integers to two digits, but `f"{-5:02d}"` is `-5`. Negative
integers become `-05`, so `snr_-05`, `snr_00`, `snr_05`. Positive values keep
today's names, so existing outputs keep their paths.

## Population

Scenarios now sweep different level factors (`snr_db` or `loudness_lu`).
`factors()` already joins them, and `Condition.factor` returns None for a
factor a scenario lacks. Check, with a test, what that means for `crossed` and
`assigned`:
- crossing `[snr_db, loudness_lu, scenario]` must give each scenario only its
  own levels;
- assigning them must balance within each scenario.

Fix the planner if it doesn't.

## Tests

- `span: file`: the output minus `gain × excerpt` equals the faded speech at its
  position, within 1 LSB per sample, before the clip guard.
- The excerpt is continuous over the file. Every level of a file shares it.
- `rms` levels within 0.1 dB, `lufs` levels within 0.1 LU, measured
  independently in the test.
- Labels and `words` are the same as with `span: block`.
- The loader rules above, and negative levels in folder names.
- Population with mixed level factors.
- Paper parity, stage 2: layout `[speech]`, `span: file`, `sequential_wrap`,
  `normalization: peak`, SNR −5…20. The aim is to match `create_LibriVAD.py`
  on a few dev-clean files, rendered by the old script into a temporary
  folder.
  - Its noise cursor is shared by every file of a batch. If that makes
    bit-identity impractical, accept agreement within int16 rounding, and
    record why.
- `span: block` output stays bit-identical (the existing parity tests).

## Docs

- `docs/scenario_builder.md`: `span`, `edge_fade_ms`, `lufs`, the new
  columns and rules. Fix its link to the deleted Plan 2.
- `docs/concat_scenarios.example.yaml`: a `span: file` scenario and a `lufs`
  one. Drop "post_mixing … lufs are not implemented" from the header.
- `configs/scenarios/skylar_noisy.example.yaml`: follow skylar's new config.
- `generation/config.py` module docstring: replace the Plan 2 reference.

## Open

| ID | Question | Recommendation |
|---|---|---|
| P5-1 | `edge_fade_ms` default with `span: file` | 10 ms, like `fade_ms` |
| P5-2 | Name of the loudness factor | `loudness_lu`, the same sign as `snr_db` (positive = noise quieter) |

## Not in scope

- Different noise excerpts or levels over the speech and the noise-only part
  (rejected: a cue at the join).
- Loudness measured over the speech only. Gating already skips the pauses.
- Mixing on the fly in a DataLoader.

## Implementation notes (2026-11-04)

| Item | Plan | Done | Why |
|---|---|---|---|
| Loudness meter | No new core dependency | **scipy moved from the `paper` and `vad` extras into the core** | The K-weighting is an IIR filter; a numpy loop is far too slow for tens of thousands of files. `scipy.signal.lfilter` does it. Within 0.1 LU of `pyloudnorm` on noise, speech and a tone. |
| P5-1 `edge_fade_ms` | 10 ms with `span: file` | As recommended (`None` means 10 ms with `span: file`, 0 otherwise) | |
| P5-2 factor name | `loudness_lu` | As recommended | |
| Per-scenario folders | Not planned | **`scenarios[].metadata_dir`** replaces `output.metadata_dir`, like `output_path` | A `snr_{snr_db}` folder template can't be filled in a loudness scenario. |
| Excerpt | On the noise block | **On `FileSpec`** (`noise`, `noise_file`, `noise_offset`, `excerpt_samples`, `noise_span`); `noise_block()` may return None | A `[speech]` layout has no noise block. `plan.jsonl` changes accordingly. |
| Metadata positions | Not planned | `speech_start`, `speech_end`, `noise_start`, `noise_end`, `noise_offset_s` written to **7 decimals** (were 4) | 4 decimals is 0.1 ms, 1.6 samples: a validator could not find the exact excerpt or speech start. `duration` stays at 4, like the clean subset's. |
| Population with two level factors | Check, fix if needed | Works as is: a factor a scenario lacks is None, so each scenario crosses only its own levels. Tested. | |
| `--check` with `span: file` | Not planned | Rebuilds the speech from the sources and the row's clip gain to measure the level | The noise and speech overlap, so the level can't be read from the audio alone. Needs the metadata row and `normalization: none`. |
| Paper parity, stage 2 | Bit-identical if possible | **Bit-identical** (0 LSB) on 16 dev-clean outputs at −5 and 20 dB, per file from the same excerpt | The script's excerpts can't be planned the same way: it restarts its cursor every batch and walks files in `glob` order. `tests/test_generation_paper_mixing.py` (slow). |
| Old planner bug | — | **Fixed**: units are keyed by their slots (`Unit.key`) | `docs/concat_scenarios.example.yaml` crashed the planner before this plan: a contiguous `[a, b]` and the `middle` pair `[a]`, `[b]` had the same `unit.id`. Regression test added. |

Verification: `uv run pytest` 105 passed; `uv run pytest -m slow` both parity tests pass
(LibriSpeechConcat bit-identical as before; noisy LibriVAD arithmetic bit-identical).
`ruff` and `ty` clean. The example config plans 27,827 val outputs without errors.
