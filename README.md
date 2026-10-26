---
license: cc-by-nc-sa-4.0
language: en
pretty_name: "LibriVAD"
datasets:
- librispeech
- DEMAND
- WHAM!
- librivox
tags:
- audio
- voice-activity-detection
- vad
- librispeech
- noise-augmented
---

> [!NOTE]
> This repository was cloned from Hugging Face using `git clone https://huggingface.co/datasets/LibriVAD/LibriVAD` and some refactorings were made. Reference paper: [LibriVAD Paper (PDF)](docs/LibriVAD_2512.17281v1.pdf).

# LibriVAD Dataset Generation

This repository contains the scripts necessary to generate the LibriVAD dataset, a large-scale, noise-augmented dataset for Voice Activity Detection (VAD) based on the LibriSpeech corpus.

## Setup LibriVAD Requirements

1.  **Install Dependencies:** Navigate to the `LibriVAD` directory and synchronize the virtual environment using [`uv`](https://docs.astral.sh/uv/):

    ```bash
    uv sync
    ```

2.  **Configure Environment Variables:** Review and adjust the paths and parameters in `.env` if needed. Run all the commands from the repository root: `PROJECT_ROOT` in `.env` is relative (`.`).

3.  **Download Source Data:** Run the `setup.py` script with `uv run` and choose the LibriSpeech mirror closest to your location from `[EU, US, CN]`. This will download the necessary splits of the LibriSpeech dataset.

    *EU Example:*
    ```bash
    uv run src/librivad/setup.py EU
    ```
    After downloading `train-clean-100`, `dev-clean`, and `test-clean`, the local setup should take a few minutes.

## Creating the LibriVAD Dataset

4.  **Generate the Dataset:** After the setup is complete, run the `create_LibriVAD.py` script. Choose the size of the generated dataset: `"small"`, `"medium"`, or `"large"`.

    The approximate sizes of the final dataset are **15GB**, **150GB**, and **1.5TB**, respectively.

    *Small Dataset Example:*
    ```bash
    uv run src/librivad/creation_scripts/create_LibriVAD.py small
    ```
    The generation process can take a few minutes to several hours, depending on the chosen size and your machine's performance.

> [!WARNING]
> The initial space requirement for the setup process (before generating the final dataset) is approximately **61GB**. Please ensure you have sufficient disk space.

## Using Other Datasets

Datasets other than LibriSpeech are brought in through a `DatasetAdapter`, which describes how to list, download and organise a dataset. The [skylar-ai-hf/skylar-dataset](https://huggingface.co/datasets/skylar-ai-hf/skylar-dataset) (English and Spanish) is the first one:

```bash
# 1. Download, convert to 16 kHz mono and split into train, val and test
uv run src/librivad/creation_scripts/prepare_dataset.py skylar

# 2. Create the VAD labels with rVADfast
uv run src/librivad/creation_scripts/create_labels_rvad.py --dataset Skylar

# 3. Mix with the noises
uv run src/librivad/creation_scripts/create_LibriVAD.py medium --datasets Skylar --no-disk-check
```

> [!NOTE]
> These labels come from an unsupervised VAD, not from forced alignment like the LibriSpeech ones, so they are an estimate. See [`docs/dataset_adapter.md`](docs/dataset_adapter.md).

## Composed Clips (Speech, Pauses and Noise Layers)

`create_composed_dataset.py` creates clips of up to 30 seconds made of speech segments, pauses and up to two noise layers, each one with its own level in dBFS, plus clips with noise only. Every parameter has a default and can be set by the command line or by a YAML (or TOML) file, such as [`configs/composer_example.yaml`](configs/composer_example.yaml). It works with any dataset prepared with labels:

```bash
uv run src/librivad/creation_scripts/create_composed_dataset.py --dataset Skylar \
    --noise Street_noise:-35 --noise Babble_noise:-42 --duration 10 30
```

Each clip gets its own VAD label, and a `manifest.csv` records the recipe of every clip. A seed makes the result reproducible. See [`docs/composer.md`](docs/composer.md).

## Documentation

- [`docs/setup.md`](docs/setup.md): what `setup.py` downloads and the resulting directory tree.
- [`docs/parameters_configuration.md`](docs/parameters_configuration.md): every configurable parameter.
- [`docs/librispeech_vs_librispeechconcat.md`](docs/librispeech_vs_librispeechconcat.md): the two LibriVAD variants.
- [`docs/vad_models_and_evaluation.md`](docs/vad_models_and_evaluation.md): features, models and metrics.
- [`docs/dataset_adapter.md`](docs/dataset_adapter.md): using datasets other than LibriSpeech.
- [`docs/composer.md`](docs/composer.md): composed clips with pauses and noise layers.
