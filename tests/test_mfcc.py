from pathlib import Path

import numpy as np
import pytest
import scipy.io.wavfile as wav
from numpy.typing import NDArray

from librivad.mfcc import cmvn, deltas, mfcc, speech_wave, trimf


def test_trimf() -> None:
    x: NDArray[np.float64] = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = trimf(x, 1.0, 2.0, 3.0)
    np.testing.assert_allclose(y, np.array([0.0, 0.0, 1.0, 0.0, 0.0]))

    with pytest.raises(SystemExit):
        trimf(x, 3.0, 2.0, 4.0)

    with pytest.raises(SystemExit):
        trimf(x, 1.0, 4.0, 3.0)


def test_deltas() -> None:
    feat: NDArray[np.float64] = np.arange(30, dtype=np.float64).reshape(10, 3)
    d = deltas(feat, N=2)
    assert d.shape == (10, 3)

    with pytest.raises(ValueError, match="N must be an integer >= 1"):
        deltas(feat, N=0)


def test_cmvn() -> None:
    arr: NDArray[np.float32] = np.array(
        [[1.0, 5.0], [2.0, 5.0], [3.0, 5.0], [4.0, 5.0]], dtype=np.float32
    )
    norm = cmvn(arr)
    assert norm.shape == (4, 2)
    assert np.abs(np.mean(norm[:, 0])) < 1e-5
    assert np.isclose(np.std(norm[:, 0], ddof=1), 1.0, atol=1e-5)
    # Second column has zero std, so std is replaced by 1 and mean-subtracted values are 0
    np.testing.assert_allclose(norm[:, 1], np.zeros(4))


def test_speech_wave_and_mfcc(tmp_path: Path) -> None:
    fs: int = 16000
    duration_sec: float = 0.2
    n_samples: int = int(fs * duration_sec)
    t: NDArray[np.float64] = np.linspace(0, duration_sec, n_samples, endpoint=False)
    sig: NDArray[np.int16] = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)

    wav_path: Path = tmp_path / "test.wav"
    label_path: Path = tmp_path / "test.npy"
    wav.write(wav_path, fs, sig)

    labels: NDArray[np.int16] = np.ones(n_samples, dtype=np.int16)
    labels[:800] = 0
    np.save(label_path, labels)

    sr_out, norm_sig = speech_wave(wav_path)
    assert sr_out == fs
    assert norm_sig.shape == (n_samples,)
    assert np.isclose(np.max(np.abs(norm_sig)), 1.0)

    scp_entry: str = f"{wav_path},{label_path},{tmp_path / 'out'}"
    feat, frame_labels = mfcc(
        scp_entry,
        winlen=0.025,
        ovrlen=0.01,
        pre_coef=0.97,
        nfilter=24,
        nftt=512,
        no_coef=12,
    )
    # 12 cepstral + 1 energy = 13, plus delta (13) and delta-delta (13) = 39 features
    assert feat.shape[1] == 39
    assert feat.shape[0] == frame_labels.shape[0]
    assert feat.shape[0] > 0
