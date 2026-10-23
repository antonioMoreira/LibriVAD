# Composed Clips: Speech, Pauses and Noise Layers

[`create_LibriVAD.py`](../src/librivad/creation_scripts/create_LibriVAD.py) mixes every utterance
with every noise on a fixed grid (9 noises x 6 SNRs). [`create_composed_dataset.py`](../src/librivad/creation_scripts/create_composed_dataset.py)
is the flexible alternative: each clip draws its duration, speech segments, pauses and noises
from ranges chosen by the user, and its VAD label is created together with the audio. It works
with any dataset prepared with labels (LibriSpeech, Skylar, ...) and leaves the existing scripts untouched.

```bash
# Speech with two noises, each at its own level
uv run src/librivad/creation_scripts/create_composed_dataset.py --dataset Skylar \
    --noise Street_noise:-35 --noise Babble_noise:-42 --duration 10 30

# Clips with only one or two noises
uv run src/librivad/creation_scripts/create_composed_dataset.py --dataset Skylar \
    --mode noise_only --noise-count 1 2 --noise-db -50 -30 --duration 5 8

# Parameters from a file; the command line wins over the file
uv run src/librivad/creation_scripts/create_composed_dataset.py --dataset Skylar \
    --config my_config.toml --seed 3
```

## Levels

Every layer has its own level in **dBFS**: the RMS in dB relative to full scale (+-1.0).

- **Speech:** measured only over the speech samples of the label, over the whole clip. Each
  segment is first brought to that level, so they are consistent regardless of their source, and the
  finished track is then adjusted so the clip as a whole has exactly the requested level, even when
  its last segment is cut.
- **Noise:** measured over the whole clip.
- The **SNR** of a noise is `speech_db - noise_db`, saved in the manifest.
- Clips with no speech have no SNR; their noise levels are absolute.

With `--normalize limit` (default) a clip that would clip is attenuated as a whole, so the levels keep their
relation to each other; `gain_db` in the manifest says by how much (add it to the levels to get
the final ones). `none` hard clips instead. Unlike `create_LibriVAD.py`, peaks are not
normalised, because that would erase the absolute levels.

## Parameters

All of them have a default and are optional. The same names are the keys of the TOML file
(`--noise-only-ratio` is `noise_only_ratio`, ...).

| Flag | Meaning | Default |
| :--- | :--- | :--- |
| `--mode` | `speech` (speech + noise) or `noise_only` | `speech` |
| `--noise-only-ratio` | fraction of the clips without speech (in `speech` mode) | `0` |
| `--duration MAX` or `MIN MAX` | clip length in seconds, drawn per clip; up to 30 | `10 30` |
| `--speech-db LEVEL` or `MIN MAX` | speech level (dBFS) | `-25` |
| `--noise NAME:DB` | fixed noise layer; repeat once for two noises | none |
| `--noise-pool NAMES...` | noises to draw from when `--noise` is not used | all available |
| `--noise-count N` or `MIN MAX` | how many noises a drawn clip has (0 to 2) | `1` |
| `--noise-db MIN MAX` | level range of each drawn noise (dBFS) | `-45 -30` |
| `--pause-range MIN MAX` | length of a pause in seconds; only noise is heard | `0.5 3` |
| `--pause-prob` | chance of a pause before each segment (the start and between segments) | `0.5` |
| `--num-clips TRAIN VAL TEST` | clips per split | `1000 100 100` |
| `--seed` | random seed | `0` |
| `--normalize` | `limit` or `none` | `limit` |
| `--dataset_dir`, `--labels_dir`, `--noises_dir`, `--save_dir` | locations | from `.env` |

`--noise` fixes the layers of every clip; without it they are drawn from `--noise-pool`.
A `noise_only` clip always has at least one noise. The noise of a split is only used with the
speech of the same split, and any folder `<name>/<name>_<split>.wav` in the noises directory counts as a noise.

Example TOML:

```toml
duration = [8, 12]
noise_only_ratio = 0.2
noises = [{ name = "City_noise", db = -38 }, { name = "Babble_noise", db = -45 }]
pause_prob = 0.7
num_clips = [500, 50, 50]
seed = 5
```

## How a clip is built

1. A duration is drawn and the speech track is filled, from the start, with segments of
   random files of the split. A pause may be added before each segment (digital silence, so only
   the noise remains), which also covers the start of the clip. The last piece, a segment or a
   pause, is cut at the target duration.
2. The label is the concatenation of the labels of the segments (so their natural pauses stay
   at 0) and zeros for the pauses. Labels that differ in length from their audio are trimmed or padded.
3. Each noise is taken from a random position of its file, wrapping around its end, and scaled to its level.
4. The layers are summed and, if needed, attenuated (see Levels).

Every clip has its own random generator, derived from the seed, the split and the clip index, so a
clip does not depend on the others and the same seed reproduces the same dataset.

## Output

Under `--save_dir` (default `<RESULTS_DIR>_composed`, apart from `Results/`):

```text
<Dataset>/<split>/<clip id>.wav          16 kHz, 16-bit
Labels/<Dataset>/<split>/<clip id>.npy   one 0/1 value per sample
<Dataset>/manifest.csv                   the recipe of every clip
```

The manifest has, per clip: duration, mode, speech level, `gain_db`, whether it clipped, and
JSON columns with the noises (name, level, SNR and position in the file), the speech segments
(source, start, end, whether cut) and the pauses. Positions come in seconds (rounded to 1 ms) and
in samples (`offset_samples`, `start_sample`, `end_sample`, and the `pauses_samples` column), which
are exact: with them, and the noise and speech files, a clip can be rebuilt from its manifest row
alone. Running again with the same `--save_dir`
overwrites the clips with the same ids; use a new directory when changing the parameters.

## Limitations

- Speech sources need a label; files without one are ignored.
- At most two noise layers, with a constant level during the whole clip.
- Not implemented: noise starting or ending inside a clip, reverberation, overlapping speakers.
