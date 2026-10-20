# `setup.py` Execution Guide & Expected Directory Tree

This document describes the URLs fetched, the preprocessing steps executed, and the resulting directory tree produced by running [`src/librivad/setup.py`](../src/librivad/setup.py):

```bash
uv run src/librivad/setup.py <US|EU|CN>
```

---

## 1. URLs Fetched by `setup.py`

### Hugging Face Dataset Archives
Configured via `FORCED_ALIGNMENTS_URL` and `NOISES_URL` in [`.env`](../.env):

* **Forced Alignments (`Forced_alignments.zip`)**
  * **URL:** `https://huggingface.co/datasets/LibriVAD/LibriVAD/resolve/main/Files/Forced_alignments.zip`
  * **Target Directory:** `files/Forced_alignments/` (`FORCED_ALIGNMENTS_DIR`)
  * **Behavior:** Downloaded temporarily to `files/Forced_alignments.zip`, extracted into `files/Forced_alignments/`, and deleted after extraction. Skipped if `files/Forced_alignments/` is already non-empty.
* **Pre-generated Noises (`Noises.zip`)**
  * **URL:** `https://huggingface.co/datasets/LibriVAD/LibriVAD/resolve/main/Files/Noises.zip`
  * **Target Directory:** `files/Noises/` (`NOISES_DIR`)
  * **Behavior:** Downloaded temporarily to `files/Noises.zip`, extracted into `files/Noises/`, and deleted after extraction. Skipped if `files/Noises/` is already non-empty.

### OpenSLR LibriSpeech Mirrors & Splits
Configured via `URL_MIRROR_US`, `URL_MIRROR_EU`, `URL_MIRROR_CN`, and `FILES_TO_DOWNLOAD` in [`.env`](../.env):

* **Regional Base Mirrors (`<base_url>`)**
  * **US (`URL_MIRROR_US`):** `https://www.openslr.org/resources/12`
  * **EU (`URL_MIRROR_EU`):** `https://openslr.elda.org/resources/12`
  * **CN (`URL_MIRROR_CN`):** `https://openslr.magicdatatech.com/resources/12`
* **LibriSpeech Archives (`FILES_TO_DOWNLOAD`)**
  * **`train-clean-100.tar.gz`:** `<base_url>/train-clean-100.tar.gz`
  * **`dev-clean.tar.gz`:** `<base_url>/dev-clean.tar.gz`
  * **`test-clean.tar.gz`:** `<base_url>/test-clean.tar.gz`
  * **Behavior:** Each archive is downloaded to `files/datasets/flac/`, extracted into `files/datasets/flac/LibriSpeech/<split>/`, and immediately removed (`os.remove`). After [`create_LibriSpeech_wav.py`](../src/librivad/creation_scripts/create_LibriSpeech_wav.py) converts the `.flac` audio to `.wav`, the entire temporary `files/datasets/flac/` directory is removed (`shutil.rmtree`).

---

## 2. Pre-trained Models & Evaluation Assets (Hugging Face Repository)

Configured via `HF_MODELS_BASE_URL` and `MODELS_TO_DOWNLOAD` in [`.env`](../.env), `setup.py` automatically downloads all pre-trained `.model` and `.zip` files from the original Hugging Face repository (`https://huggingface.co/datasets/LibriVAD/LibriVAD/resolve/main/Eval-ViT-MFCC`) into `files/` (overwriting any Git LFS pointer stubs `< 1 KB` and skipping files already downloaded):

* **Small Models (`files/small-models/`)**
  * `LibriSpeech_bDNN_MFCC/model_epoch_50.model` — `https://huggingface.co/datasets/LibriVAD/LibriVAD/resolve/main/Eval-ViT-MFCC/small-models/LibriSpeech_bDNN_MFCC/model_epoch_50.model`
  * `LibriSpeechConcat_bDNN_MFCC/model_epoch_50.model` — `https://huggingface.co/datasets/LibriVAD/LibriVAD/resolve/main/Eval-ViT-MFCC/small-models/LibriSpeechConcat_bDNN_MFCC/model_epoch_50.model`
  * `LibriSpeech_vit_MFCC/model_epoch_50.model` — `https://huggingface.co/datasets/LibriVAD/LibriVAD/resolve/main/Eval-ViT-MFCC/small-models/LibriSpeech_vit_MFCC/model_epoch_50.model`
  * `LibriSpeechConcat_vit_MFCC/model_epoch_50.model` — `https://huggingface.co/datasets/LibriVAD/LibriVAD/resolve/main/Eval-ViT-MFCC/small-models/LibriSpeechConcat_vit_MFCC/model_epoch_50.model`
* **Medium Models (`files/medium-models/`)**
  * `LibriSpeech_vit_MFCC/model_epoch_50.model` — `https://huggingface.co/datasets/LibriVAD/LibriVAD/resolve/main/Eval-ViT-MFCC/medium-models/LibriSpeech_vit_MFCC/model_epoch_50.model`
  * `LibriSpeechConcat_vit_MFCC/model_epoch_50.model` — `https://huggingface.co/datasets/LibriVAD/LibriVAD/resolve/main/Eval-ViT-MFCC/medium-models/LibriSpeechConcat_vit_MFCC/model_epoch_50.model`
* **Large Models (`files/large-models/`)**
  * `LibriSpeech_vit_MFCC/model_epoch_49.model` — `https://huggingface.co/datasets/LibriVAD/LibriVAD/resolve/main/Eval-ViT-MFCC/large-models/LibriSpeech_vit_MFCC/model_epoch_49.model`
  * `LibriSpeechConcat_vit_MFCC/model_epoch_49.model` — `https://huggingface.co/datasets/LibriVAD/LibriVAD/resolve/main/Eval-ViT-MFCC/large-models/LibriSpeechConcat_vit_MFCC/model_epoch_49.model`
  * `Best_models.zip` — `https://huggingface.co/datasets/LibriVAD/LibriVAD/resolve/main/Eval-ViT-MFCC/large-models/Best_models.zip`
  * `final_models.zip` — `https://huggingface.co/datasets/LibriVAD/LibriVAD/resolve/main/Eval-ViT-MFCC/large-models/final_models.zip`
* **VOiCES Forced Alignments (`files/Voices-force-alignment/`)**
  * `Voices-force-alignment/source-16k/{train,test}/*.TextGrid` — `https://huggingface.co/datasets/LibriVAD/LibriVAD/tree/main/Eval-ViT-MFCC/Voices-force-alignment` (plain text files tracked directly in Git)

---

## 3. External Datasets Referenced by Optional Scripts

Because `setup.py` downloads the pre-compiled `Noises.zip` archive from Hugging Face, you **do not** need to download the raw noise corpora for standard LibriVAD generation. However, if you want to regenerate the noise files from scratch using the standalone scripts in `src/librivad/creation_scripts/` or evaluate on VOiCES, the following external datasets are required:

* **DEMAND (Diverse Environments Multichannel Acoustic Noise Database)** — used by [`create_DEMAND_noises.py`](../src/librivad/creation_scripts/create_DEMAND_noises.py)
  * **Expected Path:** `files/datasets/DEMAND/` (`DEMAND_DIR`)
  * **URL:** `https://zenodo.org/records/1227121` (16 kHz zip archives for Domestic, Nature, Office, Public, Street, and Transport categories; `SCAFE` 48 kHz is downsampled by the script)
* **WHAM! Noise Dataset** — used by [`create_city_noise.py`](../src/librivad/creation_scripts/create_city_noise.py)
  * **Expected Path:** Passed as CLI argument (`uv run src/librivad/creation_scripts/create_city_noise.py /path/to/wham_noise`)
  * **URL:** `https://wham.whisper.ai/` (`https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/wham_noise.zip`)
* **Libri-Light (`small` & `medium` splits)** — used by [`create_SSN.py`](../src/librivad/creation_scripts/create_SSN.py) and [`create_babble_noise.py`](../src/librivad/creation_scripts/create_babble_noise.py)
  * **URL (GitHub):** `https://github.com/facebookresearch/libri-light`
  * **Direct Downloads:**
    * `small.tar` (for `create_SSN.py`): `https://dl.fbaipublicfiles.com/librilight/data/small.tar`
    * `medium.tar` (for `create_babble_noise.py`): `https://dl.fbaipublicfiles.com/librilight/data/medium.tar`
* **VOiCES (Voices Obscured in Complex Environmental Settings)** — used with `files/Voices-force-alignment/` for out-of-distribution (OOD) evaluation
  * **URL:** `https://openslr.org/82/` (`https://us.openslr.org/resources/82/VOiCES_devkit.tar.gz`)

---

## 4. Preprocessing Pipeline Triggered by `setup.py`

After downloading and extracting the archives, `setup.py` runs three scripts sequentially from `src/librivad/creation_scripts/`:

1. **[`create_LibriSpeech_wav.py`](../src/librivad/creation_scripts/create_LibriSpeech_wav.py)**
   * Converts `.flac` files in `files/datasets/flac/LibriSpeech/{train-clean-100,dev-clean,test-clean}` to `.wav` files in `files/datasets/LibriSpeech/{train-clean-100,dev-clean,test-clean}`.
   * `setup.py` then deletes `files/datasets/flac/`.
2. **[`create_LibriSpeechConcat.py`](../src/librivad/creation_scripts/create_LibriSpeechConcat.py)**
   * Extracts silent segments using the forced alignments in `files/Forced_alignments/librispeech_alignments/train-clean-100` and writes `files/datasets/LibriSpeech/train100_silence.wav`.
   * Concatenates pairs of utterances with intermediate silence to build `files/datasets/LibriSpeechConcat/{train-clean-100,dev-clean,test-clean}`.
3. **[`create_labels.py`](../src/librivad/creation_scripts/create_labels.py)**
   * Reads the `.TextGrid` forced alignments and generates binary frame-level `.npy` VAD labels (16 kHz) for both `LibriSpeech` and `LibriSpeechConcat` across `train-clean-100`, `dev-clean`, and `test-clean` under `files/Labels/`.

---

## 5. Expected `tree` After `setup.py` Execution

```text
files/
├── Forced_alignments/
│   └── librispeech_alignments/
│       ├── unaligned.txt
│       ├── dev-clean/
│       │   └── <speaker_id>/
│       │       └── <chapter_id>/
│       │           └── <speaker_id>-<chapter_id>-<utterance_id>.TextGrid
│       ├── test-clean/
│       │   └── <speaker_id>/
│       │       └── <chapter_id>/
│       │           └── <speaker_id>-<chapter_id>-<utterance_id>.TextGrid
│       └── train-clean-100/
│           └── <speaker_id>/
│               └── <chapter_id>/
│                   └── <speaker_id>-<chapter_id>-<utterance_id>.TextGrid
├── Noises/
│   ├── Babble_noise/
│   │   ├── Babble_noise_test.wav
│   │   ├── Babble_noise_train.wav
│   │   ├── Babble_noise_val.wav
│   │   └── first_speaker_id.txt
│   ├── City_noise/
│   │   ├── City_noise_test.wav
│   │   ├── City_noise_train.wav
│   │   └── City_noise_val.wav
│   ├── Domestic_noise/
│   │   ├── Domestic_noise_test.wav
│   │   ├── Domestic_noise_train.wav
│   │   ├── Domestic_noise_val.wav
│   │   └── domestic_noise.txt
│   ├── Nature_noise/
│   │   ├── Nature_noise_test.wav
│   │   ├── Nature_noise_train.wav
│   │   └── Nature_noise_val.wav
│   ├── Office_noise/
│   │   ├── Office_noise_test.wav
│   │   ├── Office_noise_train.wav
│   │   └── Office_noise_val.wav
│   ├── Public_noise/
│   │   ├── Public_noise_test.wav
│   │   ├── Public_noise_train.wav
│   │   └── Public_noise_val.wav
│   ├── SSN_noise/
│   │   ├── SSN_noise_test.wav
│   │   ├── SSN_noise_train.wav
│   │   ├── SSN_noise_val.wav
│   │   └── ssn.txt
│   ├── Street_noise/
│   │   ├── Street_noise_test.wav
│   │   ├── Street_noise_train.wav
│   │   └── Street_noise_val.wav
│   └── Transport_noise/
│       ├── Transport_noise_test.wav
│       ├── Transport_noise_train.wav
│       └── Transport_noise_val.wav
├── small-models/
│   ├── LibriSpeech_bDNN_MFCC/
│   │   └── model_epoch_50.model
│   ├── LibriSpeechConcat_bDNN_MFCC/
│   │   └── model_epoch_50.model
│   ├── LibriSpeech_vit_MFCC/
│   │   └── model_epoch_50.model
│   └── LibriSpeechConcat_vit_MFCC/
│       └── model_epoch_50.model
├── medium-models/
│   ├── LibriSpeech_vit_MFCC/
│   │   └── model_epoch_50.model
│   └── LibriSpeechConcat_vit_MFCC/
│       └── model_epoch_50.model
├── large-models/
│   ├── Best_models.zip
│   ├── final_models.zip
│   ├── LibriSpeech_vit_MFCC/
│   │   └── model_epoch_49.model
│   └── LibriSpeechConcat_vit_MFCC/
│       └── model_epoch_49.model
├── datasets/
│   ├── LibriSpeech/
│   │   ├── train100_silence.wav
│   │   ├── dev-clean/
│   │   │   └── <speaker_id>/
│   │   │       └── <chapter_id>/
│   │   │           └── <speaker_id>-<chapter_id>-<utterance_id>.wav
│   │   ├── test-clean/
│   │   │   └── <speaker_id>/
│   │   │       └── <chapter_id>/
│   │   │           └── <speaker_id>-<chapter_id>-<utterance_id>.wav
│   │   └── train-clean-100/
│   │       └── <speaker_id>/
│   │           └── <chapter_id>/
│   │               └── <speaker_id>-<chapter_id>-<utterance_id>.wav
│   └── LibriSpeechConcat/
│       ├── dev-clean/
│       │   └── <speaker_id>/
│       │       └── <chapter_id>/
│       │           └── <part1>_+_<part2>.wav
│       ├── test-clean/
│       │   └── <speaker_id>/
│       │       └── <chapter_id>/
│       │           └── <part1>_+_<part2>.wav
│       └── train-clean-100/
│           └── <speaker_id>/
│               └── <chapter_id>/
│                   └── <part1>_+_<part2>.wav
└── Labels/
    ├── LibriSpeech/
    │   ├── dev-clean/
    │   │   └── <speaker_id>/
    │   │       └── <chapter_id>/
    │   │           └── <speaker_id>-<chapter_id>-<utterance_id>.npy
    │   ├── test-clean/
    │   │   └── <speaker_id>/
    │   │       └── <chapter_id>/
    │   │           └── <speaker_id>-<chapter_id>-<utterance_id>.npy
    │   └── train-clean-100/
    │       └── <speaker_id>/
    │           └── <chapter_id>/
    │               └── <speaker_id>-<chapter_id>-<utterance_id>.npy
    └── LibriSpeechConcat/
        ├── dev-clean/
        │   └── <speaker_id>/
        │       └── <chapter_id>/
        │           └── <part1>_+_<part2>.npy
        ├── test-clean/
        │   └── <speaker_id>/
        │       └── <chapter_id>/
        │           └── <part1>_+_<part2>.npy
        └── train-clean-100/
            └── <speaker_id>/
                └── <chapter_id>/
                    └── <part1>_+_<part2>.npy
```

> [!NOTE]
> Running [`create_LibriVAD.py`](../src/librivad/creation_scripts/create_LibriVAD.py) afterwards (`uv run src/librivad/creation_scripts/create_LibriVAD.py <small|medium|large>`) populates `Results/LibriVAD/` (`RESULTS_DIR`) with noise-mixed `.wav` files structured by `<dataset>/<split>/<noise_type>/<SNR>/<speaker_id>/<chapter_id>/`.
