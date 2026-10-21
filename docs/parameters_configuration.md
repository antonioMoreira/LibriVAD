# LibriVAD Parameters & Configuration Report

This report documents all configurable parameters across environment variables, dataset setup, dataset generation, silence/SSR calibration, noise synthesis, and VAD model evaluation in the LibriVAD repository, grounded in the [LibriVAD paper (PDF)](LibriVAD_2512.17281v1.pdf).

---

## 1. Environment Variables ([`.env`](../.env))

Loaded automatically via `python-dotenv` across all setup and generation scripts.

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `FORCED_ALIGNMENTS_URL` | `.../Files/Forced_alignments.zip` | Hugging Face URL for Montreal Forced Aligner `.TextGrid` files. |
| `NOISES_URL` | `.../Files/Noises.zip` | Hugging Face URL for the 9 pre-generated noise categories. |
| `HF_MODELS_BASE_URL` | `.../Eval-ViT-MFCC` | Base Hugging Face URL for pre-trained bDNN and ViT checkpoints. |
| `MODELS_TO_DOWNLOAD` | Comma-separated list (10 items) | Relative paths of `.model` and `.zip` checkpoints downloaded into `files/`. |
| `URL_MIRROR_US` | `https://www.openslr.org/resources/12` | US OpenSLR mirror for LibriSpeech `.tar.gz` archives. |
| `URL_MIRROR_EU` | `https://openslr.elda.org/resources/12` | EU OpenSLR mirror for LibriSpeech `.tar.gz` archives. |
| `URL_MIRROR_CN` | `https://openslr.magicdatatech.com/resources/12` | CN OpenSLR mirror for LibriSpeech `.tar.gz` archives. |
| `FILES_TO_DOWNLOAD` | `train-clean-100.tar.gz,dev-clean.tar.gz,test-clean.tar.gz` | Comma-separated LibriSpeech splits to download and process. |
| `PROJECT_ROOT` | `/mnt/HD1TB/LibriVAD` | Absolute root path of the project workspace. |
| `FILES_DIR` | `files` | Root directory for downloaded assets, source datasets, and labels. |
| `DATASETS_DIR` | `files/datasets` | Parent directory of clean `LibriSpeech` and `LibriSpeechConcat` WAVs. |
| `FLAC_DIR` | `files/datasets/flac` | Temporary staging directory for `.tar.gz` archives and `.flac` files. |
| `LIBRISPEECH_FLAC_DIR` | `files/datasets/flac/LibriSpeech` | Extracted `.flac` splits directory (removed after WAV conversion). |
| `LIBRISPEECH_WAV_DIR` | `files/datasets/LibriSpeech` | Converted 16 kHz `.wav` splits + `train100_silence.wav`. |
| `LIBRISPEECH_CONCAT_DIR` | `files/datasets/LibriSpeechConcat` | Silence-injected paired `.wav` splits (`<part1>_+_<part2>.wav`). |
| `DEMAND_DIR` | `files/datasets/DEMAND` | Optional path to raw DEMAND corpus (only for `create_DEMAND_noises.py`). |
| `FORCED_ALIGNMENTS_DIR` | `files/Forced_alignments` | Destination directory where `Forced_alignments.zip` is extracted. |
| `LIBRISPEECH_ALIGNMENTS_DIR` | `files/Forced_alignments/librispeech_alignments` | Contains `{train-clean-100,dev-clean,test-clean}/*.TextGrid` and `unaligned.txt`. |
| `NOISES_DIR` | `files/Noises` | Contains the 9 noise folders (`<Noise>_noise_{train,val,test}.wav`). |
| `LABELS_DIR` | `files/Labels` | Binary 16 kHz `.npy` VAD label arrays for `LibriSpeech` and `LibriSpeechConcat`. |
| `RESULTS_DIR` | `Results` | Default output directory for generated noisy LibriVAD mixtures. |
| `SCRIPTS_DIR` | `src/librivad/creation_scripts` | Directory containing the dataset creation and preprocessing scripts. |

---

## 2. Setup Script Parameters ([`src/librivad/setup.py`](../src/librivad/setup.py))

```bash
uv run src/librivad/setup.py <region>
```

| Parameter / Check | Type | Options / Threshold | Behavior |
| :--- | :--- | :--- | :--- |
| `region` | Positional CLI arg | `US`, `EU`, `CN` | Selects the OpenSLR base URL from `URL_MIRRORS`. |
| Model stub threshold | Internal constant | `1024` bytes (`1 KB`) | Files in `MODELS_TO_DOWNLOAD` smaller than 1 KB (e.g., Git LFS pointer stubs) are overwritten with full weights from Hugging Face. |
| Idempotency / Skip checks | Directory verification | Non-empty `.wav` / `.npy` | Automatically skips OpenSLR downloads, `create_LibriSpeech_wav.py`, `create_LibriSpeechConcat.py`, and `create_labels.py` when their respective output directories under `files/` are already populated. |

---

## 3. Dataset Generation Parameters ([`src/librivad/creation_scripts/create_LibriVAD.py`](../src/librivad/creation_scripts/create_LibriVAD.py))

### A. Command-Line Arguments (CLI)

```bash
uv run src/librivad/creation_scripts/create_LibriVAD.py <size> \
    [--dataset_dir FILES/DATASETS] \
    [--snrs -5 0 5 10 15 20] \
    [--batch_size 2900] \
    [--save_dir RESULTS]
```

| CLI Argument | Default | Allowed Values | Effect on Generation |
| :--- | :--- | :--- | :--- |
| `size` *(required)* | — | `"small"`, `"medium"`, `"large"` | Controls deterministic sub-sampling (`file_paths_all[::step]`):<br/>• `"small"` (`step = 100`): retains every 100th file (~15 GB, ~140 h)<br/>• `"medium"` (`step = 10`): retains every 10th file (~150 GB, ~1,400 h)<br/>• `"large"` (`step = 1`): retains all files (~1.5 TB, ~14,000 h) |
| `--snrs` | `[-5, 0, 5, 10, 15, 20]` | One or more integers (dB) | Signal-to-Noise Ratios (in dB) used to scale the additive noise relative to the speech-active RMS of each utterance. |
| `--batch_size` | `2900` | Positive integer | Number of clean utterances pre-loaded into RAM in parallel (`ThreadPoolExecutor`) per batch. Lowering this reduces peak memory usage. |
| `--dataset_dir` | `files/datasets` | Path string | Root folder containing the clean `LibriSpeech` and `LibriSpeechConcat` splits. |
| `--save_dir` | `None` (`Results/`) | Path string | Root output folder where `<dataset>/<split>/<noise>/<snr>/<utt>.wav` files are written. |

### B. Pre-Execution Disk Space Thresholds (`check_disk_space`)

Before generating any files, [`check_disk_space`](../src/librivad/creation_scripts/create_LibriVAD.py#L47-L76) verifies free disk space on the target drive:

| `size` Argument | Minimum Free Disk Space Required | Exact Byte Threshold |
| :--- | :--- | :--- |
| `"small"` | **15 GB** | `15 * 1024**3` (`16,106,127,360` bytes) |
| `"medium"` | **150 GB** | `150 * 1024**3` (`161,061,273,600` bytes) |
| `"large"` | **1.5 TB** | `int(1.5 * 1024**4)` (`1,649,267,441,664` bytes) |

### C. Script-Level Lists & Signal Mixing Constants

Defined in [`create_LibriVAD.py`](../src/librivad/creation_scripts/create_LibriVAD.py#L140-L330):

| Symbol | Default Value | Description |
| :--- | :--- | :--- |
| `Datasets` | `["LibriSpeech", "LibriSpeechConcat"]` | Controls which SSR variants are generated (`LibriVAD-NonConcat` vs. `LibriVAD-Concat`). |
| `Noises` | `["Babble_noise", "SSN_noise", "Domestic_noise", "Nature_noise", "Office_noise", "Public_noise", "Street_noise", "Transport_noise", "City_noise"]` | The 9 noise categories mixed with each clean utterance (6 seen in training: Nature, Office, Public, Street, Transport, City; 3 unseen in training: Babble, SSN, Domestic). |
| `Splits` | `["train", "val", "test"]` | Dataset splits mapped to `train-clean-100`, `dev-clean`, and `test-clean`. |
| `sr` | `16000` Hz | Output sampling rate for all generated `.wav` mixtures (16-bit PCM). |
| `scale` | `rms_speech / (rms_noise * 10**(snr / 20))` | Noise scaling factor computed **only over speech-active samples** (`label == 1`) via [`remove_silence`](../src/librivad/librispeech_functions.py#L148-L186) so silent regions do not bias SNR. |

---

## 4. Silence Injection, Transcriptions & Label Parameters

Configured in [`create_LibriSpeechConcat.py`](../src/librivad/creation_scripts/create_LibriSpeechConcat.py), [`create_labels.py`](../src/librivad/creation_scripts/create_labels.py), and [`librispeech_functions.py`](../src/librivad/librispeech_functions.py):

| Parameter / Mechanism | Value / Formula | Location | Description |
| :--- | :--- | :--- | :--- |
| **Inserted Silence Length** | `int((len(wav1) + len(wav2)) / 4)` | [`process_pair`](../src/librivad/creation_scripts/create_LibriSpeechConcat.py#L98), [`create_label_for_file`](../src/librivad/creation_scripts/create_labels.py#L170-L172) | Inserts silence equal to **25% of the combined duration** of two consecutive utterances, raising the Silence-to-Speech Ratio (SSR) from ~21% (17.6% silence) to **~34%**. |
| **Inserted Silence Gain** | `0.001 * inbetween_silence` | [`process_pair`](../src/librivad/creation_scripts/create_LibriSpeechConcat.py#L115) | Attenuates the extracted natural silence buffer (`train100_silence.wav`) before concatenating `wav1 -> silence -> wav2`. |
| **Forced Alignment Tiers** | `grid["words"]` (`xmin`, `xmax`, `text`) | [`find_timeframes`](../src/librivad/librispeech_functions.py#L59-L86) | Reads word-level transcriptions and timestamps from `.TextGrid` files (`""` indicates silence; non-empty strings contain the spoken word). |
| **Unaligned Utterance Filter** | `unaligned.txt` (4 files in `train-clean-100`) | [`remove_unaligned_paths`](../src/librivad/librispeech_functions.py#L23-L34) | Excludes the 4 utterances in `train-clean-100` (`1992-141719-0042`, `2289-152257-0025`, `5456-62014-0010`, `7067-76048-0021`) that lack forced alignments, ensuring **100%** of LibriVAD segments have valid transcriptions and `.npy` labels. |
| **Label Resolution** | `16000` samples/sec (`np.int16`) | [`_generate_label_from_timeframes`](../src/librivad/creation_scripts/create_labels.py#L37-L80) | Binary sample-level mask (`1 = speech`, `0 = silence`) saved as `.npy` under `files/Labels/`. |

---

## 5. Custom Noise Generation Parameters (Optional Standalone Scripts)

> [!NOTE]
> `setup.py` downloads pre-built noises via `Noises.zip`, so these scripts are only needed if regenerating noises from raw external corpora (DEMAND, WHAM!, Libri-Light).

| Script | Noise Type | Key Parameters & Thresholds |
| :--- | :--- | :--- |
| [`create_DEMAND_noises.py`](../src/librivad/creation_scripts/create_DEMAND_noises.py) | `Domestic`, `Nature`, `Office`, `Public`, `Street`, `Transport` | • **Categories:** `["D", "N", "O", "P", "S", "T"]` (3 environments × 16 recordings each)<br/>• **Split Modulo (`i % 16`):** indices `0–11` $\rightarrow$ `train` (3 h), `12–13` $\rightarrow$ `val` (30 m), `14–15` $\rightarrow$ `test` (30 m)<br/>• **Downsampling:** `SCAFE` downsampled from 48 kHz to `sr = 16000` Hz |
| [`create_city_noise.py`](../src/librivad/creation_scripts/create_city_noise.py) | `City_noise` (WHAM!) | • **Grouping:** Grouped by `"Location ID"` in `noise_meta_{tr,cv,tt}.csv` to preserve location distribution<br/>• **Durations:** `train = 10800s` (3 h), `val = 1800s` (30 m), `test = 1800s` (30 m) at `16000` Hz, converted from stereo to mono (`channels = 1`) |
| [`create_SSN.py`](../src/librivad/creation_scripts/create_SSN.py) | `SSN_noise` (Speech-Shaped Noise) | • **Speaker Selection Criteria:** `SNR > 15 dB`, `Silence Ratio < 0.1` (`SSR > 90%`), `speech duration > 10 min`, `Genre = Literature` (18 train, 2 val, 2 test speakers from Libri-Light `small`)<br/>• **LPC Filter Order:** `LPC_ORDER = 12` (`SAMPLE_RATE = 16000` Hz)<br/>• **Durations:** `SEGMENT_DURATIONS_SEC = {"train": 3600, "val": 600, "test": 600}` (total: 1 h train, 10 m val, 10 m test) |
| [`create_babble_noise.py`](../src/librivad/creation_scripts/create_babble_noise.py) | `Babble_noise` | • **Channels mixed:** `num_voices = 6` energy-standardized (`unit RMS`) speech streams from Libri-Light `medium` (disjoint speakers per stream)<br/>• **Quality Filter:** `snr_threshold = 15` dB, `sr_threshold = 0.15`, silence removed via `rVADfast` (`remove_silence = True`)<br/>• **Intro/Outro Trim:** Removes first `30s` and last `10s` for signals longer than `40s`<br/>• **Durations:** `train = 10800s` (3 h), `val = 1800s` (30 m), `test = 1800s` (30 m) |

---

## 6. Feature Extraction, Model & Evaluation Parameters

Detailed explanations of these modules are available in [`docs/vad_models_and_evaluation.md`](vad_models_and_evaluation.md). Below is a reference summary of their default parameters:

| Module | Parameter | Value | Description |
| :--- | :--- | :--- | :--- |
| **[`mfcc.py`](../src/librivad/mfcc.py)** | `fs` / `winlen` / `ovrlen` | `16000` Hz / `0.025` s / `0.01` s | 16 kHz audio, 25 ms Hamming window (400 samples), 10 ms frame shift (160 samples). |
| | `preemph` / `nfilter` / `ndft` | `0.97` / `24` / `512` | Pre-emphasis coefficient, 24 Mel filter banks, 512-point FFT. |
| | `ncep` / Feature dim | `12` (+ log $E$, $\Delta$, $\Delta\Delta$ = `39`) | 12 static cepstral coefficients + log-energy + first and second derivatives = 39-D feature vector. |
| **[`auc_bdnn.py`](../src/librivad/auc_bdnn.py)** | `u` / `w` / `step` | `9` / `19` / `3` | bDNN left/right context window parameters (19 context frames per window). |
| | `thres` (`vad_for_w`) | `0.5` | Frame is labeled speech (`1`) if $\ge 50\%$ of its 400 samples are speech. |
| **[`KWT3_vad.py`](../src/librivad/KWT3_vad.py)** | `input_res` / `patch_res` | `[100, 39]` / `[1, 39]` | 100-frame sequence (1.0 s) where each 10 ms frame is a `[1, 39]` patch token. |
| | `dim` / `depth` / `heads` / `mlp_dim` | `192` / `12` / `3` / `768` | `kwt-3` Vision Transformer encoder dimensions (`dim_head = 64`, `num_classes = 2`). |
| | `emb_dropout` / `lr` / `batch_size` | `0.1` / `0.001` / `32` | Dropout rate, AdamW learning rate (`weight_decay = 0.1`), and training batch size (`50` epochs). |
| **[`auc_Vit.py`](../src/librivad/auc_Vit.py)** | `sequence_length` | `100` | Number of frames per ViT sequence (residual frames $< 100$ are zero-padded and trimmed after inference). |
| **[`eer.py`](../src/librivad/eer.py)** | `c_miss` / `c_fa` / `p_target` | `10` / `1` / `0.01` | NIST 2008 SRE Detection Cost Function (MinDCF) weights (Paper §3.3, Eq. 3). |
