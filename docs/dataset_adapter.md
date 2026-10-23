# Using Datasets Other Than LibriSpeech

The generation scripts (`create_LibriVAD.py`) expect 16 kHz wavs organised like LibriSpeech, plus
one binary label `.npy` per wav (the labels give the speech-only RMS used to set the SNR):

```text
files/datasets/<Name>/<split dir>/<speaker>/<chapter>/<id>.wav
files/Labels/<Name>/<split dir>/<speaker>/<chapter>/<id>.npy
```

with `<split dir>` being `train-clean-100`, `dev-clean` and `test-clean` (the LibriSpeech names, kept
so the existing code works unchanged).

[`DatasetAdapter`](../src/librivad/dataset_adapter.py) describes this layout. Its defaults are the
LibriSpeech behaviour; a new dataset subclasses it and overrides only what differs, then registers
itself in `ADAPTERS`. No per-dataset file is needed.

| Member | Default | Override when |
| :--- | :--- | :--- |
| `name` | `LibriSpeech` | always (folder name under `files/datasets/`) |
| `list_samples()` | not implemented | always: returns `Sample(id, source, group, subgroup)` |
| `fetch(sample)` | not implemented | always: returns the raw audio bytes |
| `assign_split(group)` | 80/10/10 by hash of the group | the split proportions or rule differ |
| `split_dirs`, `sample_rate` | LibriSpeech names, 16 kHz | the layout differs |

`Sample.id` must be unique in the dataset: the generated mixtures are saved as
`Results/<Name>/<split dir>/<noise>/<snr>/<id>.wav`, without the speaker/chapter folders.
`Sample.group` is the speaker-like key; all its samples go to the same split.

## Skylar

[`SkylarAdapter`](../src/librivad/dataset_adapter.py) covers
[skylar-ai-hf/skylar-dataset](https://huggingface.co/datasets/skylar-ai-hf/skylar-dataset)
(English + Spanish, ~30 h, only a `train` split, no speech/silence annotation, English at 24 kHz).

```bash
# Run from the repository root (PROJECT_ROOT in .env is relative)

# 1. Download, convert to 16 kHz mono and split (resumable; ~4.3 GB download)
uv run src/librivad/creation_scripts/prepare_dataset.py skylar

# 2. Labels with rVADfast
uv run src/librivad/creation_scripts/create_labels_rvad.py --dataset Skylar

# 3. Mix with the noises
uv run src/librivad/creation_scripts/create_LibriVAD.py medium --datasets Skylar --no-disk-check
```

`--noises` and `--snrs` of `create_LibriVAD.py` reduce the output: the full set is 9 noises x 6 SNRs
of ~30 h of speech (~185 GB with `large`).

Notes:

- **Splits:** there is no official split, so groups are assigned 80/10/10 by hash. The group is the
  speaker (`<language>:<source dataset>:<speaker>`), or the sentence id for FLEURS, which has no speaker.
- **Labels are an estimate.** LibriSpeech labels come from forced alignment; here they come from an
  unsupervised VAD, so results measure agreement with rVADfast and are not directly comparable with
  LibriVAD. Files where no speech is detected get no label and are skipped by `create_LibriVAD.py`.
- **No Concat variant:** it needs forced alignments to extract real silence.
- The Hugging Face token (`HF_TOKEN`) is used if set; the dataset was downloadable without it.
