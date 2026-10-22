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

2.  **Configure Environment Variables:** Review and adjust the paths and parameters in `.env` if needed.

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
