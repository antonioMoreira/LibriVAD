# Difference Between `Results/LibriSpeech` and `Results/LibriSpeechConcat`

When [`create_LibriVAD.py`](../src/librivad/creation_scripts/create_LibriVAD.py) runs, it generates two noisy dataset variants inside `Results/`, corresponding to **LibriVAD-NonConcat** and **LibriVAD-Concat** in the [LibriVAD paper](LibriVAD_2512.17281v1.pdf) (§2.1 & §2.3).

---

## Core Difference at a Glance

The difference is **how much silence (non-speech)** exists inside each audio clip and **where that silence is located**:

```text
Results/LibriSpeech (LibriVAD-NonConcat):
┌───────────┬──────────────────────────────────────┬───────────┐
│ Short Sil │            Utterance 1               │ Short Sil │  (~17.6% silence)
└───────────┴──────────────────────────────────────┴───────────┘

Results/LibriSpeechConcat (LibriVAD-Concat):
┌───────────┬─────────────────┬────────────────────┬─────────────────┬───────────┐
│ Short Sil │   Utterance 1   │ Inserted Silence   │   Utterance 2   │ Short Sil │  (~34% SSR)
└───────────┴─────────────────┴────────────────────┴─────────────────┴───────────┘
                               (25% of U1 + U2 len)
```

1. **`Results/LibriSpeech/` (`LibriVAD-NonConcat` in the paper)**
   - **What it is:** Individual LibriSpeech utterances mixed directly with the 9 noise types across 6 SNR levels (`-5, 0, 5, 10, 15, 20 dB`).
   - **Silence profile:** Contains only the natural pauses from the original audiobook recordings (~**17.6%** silence overall), mostly located briefly at the beginning and end of each clip.

2. **`Results/LibriSpeechConcat/` (`LibriVAD-Concat` in the paper)**
   - **What it is:** Created by [`create_LibriSpeechConcat.py`](../src/librivad/creation_scripts/create_LibriSpeechConcat.py), which takes **two consecutive LibriSpeech utterances** and inserts a segment of real extracted silence between them before [`create_LibriVAD.py`](../src/librivad/creation_scripts/create_LibriVAD.py) mixes in the noise.
   - **Silence profile:** The inserted middle silence is calibrated to **25% of the combined length of the two utterances** (`(len(wav1) + len(wav2)) / 4`), raising the Silence-to-Speech Ratio (SSR) to **~34%**.

---

## Why Both Variants Exist

According to Section 2.1 and Section 4 of the [LibriVAD paper](LibriVAD_2512.17281v1.pdf):

- **Preventing Structural Bias:** In standard speech corpora, almost every file starts and ends with short silence and has continuous speech in the middle. Supervised VAD models can cheat by learning file position rather than actual voice activity. Inserting a long silent gap in the middle breaks this bias.
- **Balancing Speech vs. Non-Speech Classes:** VAD is a binary classification task (`1 = speech`, `0 = non-speech`). Increasing the proportion of non-speech in `Results/LibriSpeechConcat` reduces class imbalance during training.
- **Better Real-World Generalization:** In real-world audio, people speak intermittently with pauses in between. In the paper's benchmarks (Tables 6–12), models trained on `LibriSpeechConcat` consistently achieve **higher AUC**, **lower EER/MinDCF**, and **stronger out-of-distribution generalization** (e.g., on the reverberant VOiCES dataset).

---

## Side-by-Side Comparison (Paper Tables 1 & 5)

| Characteristic | `Results/LibriSpeech`<br/>*(LibriVAD-NonConcat)* | `Results/LibriSpeechConcat`<br/>*(LibriVAD-Concat)* |
| :--- | :--- | :--- |
| **Utterances per file** | 1 utterance (`<speaker>-<chapter>-<utt>.wav`) | 2 merged utterances (`<part1>_+_<part2>.wav`) |
| **Inserted middle silence** | None | 25% of `(len(utt1) + len(utt2))` |
| **Silence-to-Speech Ratio (SSR)** | ~21% (17.6% of total audio is silence) | ~34% |
| **Average clip length (`train`)** | 12.71 s | 31.72 s |
| **Source utterance count (`Large`)** | 28,535 train / 2,703 val / 2,620 test | 14,267 train / 1,351 val / 1,310 test *(half as many files, ~2.5× longer each)* |
| **Final size (`Small`, step=100)** | ~6.6 GB (60.8 hours) | ~8.2 GB (76.3 hours) |
| **Final size (`Medium`, step=10)** | ~65.3 GB (603.5 hours) | ~80.7 GB (750.9 hours) |
| **Final size (`Large`, step=1)** | ~648 GB (6,023 hours) | ~809 GB (7,517 hours) |
