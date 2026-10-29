# Open decisions: LibriVAD (Plan 2)

These are the gaps in [Plan 2](02-librivad-generalization.md) that only
concern LibriVAD's builder.

Decisions about the SkylarDataset design are kept in one place:
`/mnt/HD1TB/skylar-dataset/docs/plans/00-open-decisions.md`. Plan 2 implements
some of them, listed in §4.

IDs match Plan 2's decision table. The "Recommendation" column is what Plan 2
is written against. To answer, reply with the ID and your choice, and the
answer gets recorded here and in the plan.

---

## 1. Builder design

| ID | Question | Options | Recommendation | Blocks |
|---|---|---|---|---|
| G1 | Library for the config schema | `pydantic`, or `dataclasses` with hand-written checks | pydantic | Phase 1 |
| G2 | How sweep folders are named | readable names from path templates, or a hash per variant | readable | Phase 5 |
| G3 | How LibriSpeech's `contiguous` composition groups files | `[speaker, chapter]` or `[speaker]` | `[speaker, chapter]` | Phase 4 |
| G5 | Level key in the config | `snr_db` (≥ 0 means the noise is below the speech), or v1's `noise_level_db` (= −SNR) | `snr_db`, which matches the `snr_XX` folders. The v1 key was never implemented. | Phase 1 |
| G7 | `.npy` label files for skylar outputs | write them, or rely on the shifted `words` in the metadata | Don't write them. LibriSpeech scenarios keep writing `.npy` for the existing VAD scripts. | Phase 5 |
| G9 | Name of the corpus adapter module | `datasets.py` (v1) or `corpus.py` | `corpus.py`, so it isn't confused with Hugging Face `datasets` | Phase 2 |

## 2. LibriSpeech scenarios: left open since the scenario YAML

| ID | Question | Options | Recommendation | Blocks |
|---|---|---|---|---|
| G8 | Where LibriSpeech scenarios take their noise block from | `dataset_silence` (cut from LibriSpeech pauses, as the paper does) or `files/Noises` | Still open. It doesn't block skylar. | Phase 3 for LibriSpeech |
| G4 | Does `create_LibriVAD.py` still mix noise over the new scenario outputs (`post_mixing`)? If it does, over the whole file or skipping the inserted block? | off, whole file, skip the block | Off for now. Whole file, as the paper does, if it is ever turned on. | Phase 8 |
| G10 | Are the LibriSpeech ratio-sweep scenarios in the YAML still wanted? They use `noise_ratio` 0.25 / 0.5 / 0.75 × 4 levels, `contiguous`. skylar now uses a fixed 30 s instead. | keep as LibriSpeech examples; switch them to the fixed 30 s too; drop them | Keep `middle` plus one small example per noise-length mode, for tests. The rest of the YAML stays as documentation. | Phase 9 |
| G11 | When to implement `contiguous` and `concatenate` | with the skylar path, or after it | After. Only LibriSpeech uses them, and they are the most complex part of the planner. `middle` needs `concatenate` for Phase 6, so build that one first. | Phase 4 order |

## 3. Scope and order

| ID | Question | Options | Recommendation |
|---|---|---|---|
| G12 | Must Phase 6 (paper parity) pass before Plan 3 starts? This is skylar's N12. | gate or recommendation | Gate |
| G13 | Phase 8 (refactor `create_LibriVAD.py`, `post_mixing`): in this effort or later? | now, or later | Later. skylar doesn't need it. |
| G14 | Phase 10 issues that predate this work: fix them in this effort? | all; some; separately | Now: `ruff format` and the undeclared `scikit-learn`/`joblib`. Separately: `create_city_noise.py` on 3.14 (`pydub`), the duplicated framing helpers, and the stale VAD docs. |
| G15 | In `docs/images/librispeechconcat.svg`, rename "Noise (Silence)" to "Silence", and add the subtitle "next utterance (sorted order)" | yes / no | Yes. It was proposed and never confirmed. |
| G16 | Does LibriVAD stay Python ≥ 3.14 only? | keep; or make it importable on 3.12/3.13 by rewriting the two PEP 758 `except A, B:` lines ([librispeech_functions.py:172](../../src/librivad/generation/librispeech_functions.py#L172), [create_labels.py:186](../../src/librivad/generation/creation_scripts/create_labels.py#L186)) and lowering `requires-python` | Keep, and talk to skylar through the CLI (skylar's N6) |

## 4. Shared with SkylarDataset (decided in skylar's list)

Plan 2 implements these. Their status is tracked only in
`/mnt/HD1TB/skylar-dataset/docs/plans/00-open-decisions.md`:

- L1 folder order of `noisy`.
- L2 `metadata.jsonl` in each language folder.
- L3 `snr_XX` folder names.
- L5 split inheritance.
- L6 the same excerpt across levels.
- L7 no peak normalization, with a clipping guard per group.
- L8 speech-only RMS.
- L9 fade on the noise edge only.
- L10 noisy metadata details.
- L11 validation tolerances.
- N5 / G6 `min_noise_s`.
- N6 how skylar calls LibriVAD.
