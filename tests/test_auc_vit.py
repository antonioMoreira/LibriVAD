from pathlib import Path

import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from torch import nn

from librivad.auc_Vit import (
    LRcontext,
    auc_acc,
    auc_acc_bcres,
    enframe,
    split_data_into_sequences,
    uW_context,
)


class DummyViTModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # ty: ignore[invalid-mutable-override]
        # x has shape [1, seq_len, feat_dim]
        seq_len: int = int(x.shape[1])
        logits = torch.zeros((1, seq_len, 2), dtype=torch.float32)
        logits[:, :, 1] = 0.9
        logits[:, :, 0] = 0.1
        return logits


class DummyBCResModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # ty: ignore[invalid-mutable-override]
        frames: int = int(x.shape[0])
        logits = torch.zeros((frames, 2), dtype=torch.float32)
        logits[:, 1] = 1.0
        return logits


def test_context_and_enframe() -> None:
    x_batch: NDArray[np.float64] = np.arange(30, dtype=np.float64).reshape(10, 3)
    uw = uW_context(x_batch, w=3, u=1, bDnnWin=7)
    assert uw.shape == (10, 21)

    lr = LRcontext(x_batch, Lt=-1, Rt=1, bDnnWin=3)
    assert lr.shape == (10, 9)

    speech: NDArray[np.float64] = np.ones(1000, dtype=np.float64)
    frames, fovr = enframe(speech, fs=16000, winlen=0.025, ovrlen=0.01)
    assert fovr == 160
    assert frames.shape[1] == 400


def test_split_data_into_sequences_exact_and_remainder() -> None:
    features: NDArray[np.float64] = np.ones((20, 4), dtype=np.float64)
    labels: NDArray[np.float64] = np.ones((20, 1), dtype=np.float64)
    seq_feat, seq_lab, padded_feat, last_lab = split_data_into_sequences(
        features, labels, sequence_length=10
    )
    assert seq_feat.shape == (2, 10, 4)
    assert seq_lab.shape == (2, 10)
    assert padded_feat.shape == (10, 4)
    assert last_lab.shape == (10, 4)

    features_rem: NDArray[np.float64] = np.arange(100, dtype=np.float64).reshape(25, 4)
    labels_rem: NDArray[np.float64] = np.arange(25, dtype=np.float64).reshape(25, 1) % 2
    seq_feat_r, seq_lab_r, padded_feat_r, last_lab_r = split_data_into_sequences(
        features_rem, labels_rem, sequence_length=10
    )
    assert seq_feat_r.shape == (2, 10, 4)
    assert seq_lab_r.shape == (2, 10)
    assert padded_feat_r.shape == (10, 4)
    assert last_lab_r.shape == (5, 1)


def test_auc_acc_vit_and_bcres(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    feat_prefix: Path = tmp_path / "vit_sample"
    feats: torch.Tensor = torch.randn(15, 4, dtype=torch.float32)
    labels: torch.Tensor = torch.tensor(
        [[0], [1], [0], [1], [1], [0], [1], [0], [1], [1], [0], [1], [0], [1], [1]],
        dtype=torch.float32,
    )
    torch.save(feats, f"{feat_prefix}.pt")
    torch.save(labels, f"{feat_prefix}.lab.pt")

    scp: list[str] = [f"wav.wav,lab.npy,{feat_prefix}"]
    vit_model = DummyViTModel()

    muc, acc, n_samples, _thresholds, _fpr, _tpr, scores_out, labels_out = auc_acc(
        scp, vit_model, "cpu", sequence_length=10, cutThres="True"
    )
    assert n_samples == 15
    assert scores_out.shape[0] == 15
    assert labels_out.shape[0] == 15
    assert 0.0 <= muc <= 1.0
    assert 0.0 <= acc <= 1.0
    assert (tmp_path / "frame_info.txt").is_file()

    muc2, acc2, n_samples2, thres2, fpr2, tpr2, _scores2, _labels2 = auc_acc(
        scp, vit_model, "cpu", sequence_length=10, cutThres=0.5
    )
    assert n_samples2 == 15
    assert thres2 == 0.5
    assert fpr2 == 0 and tpr2 == 0
    assert 0.0 <= muc2 <= 1.0
    assert 0.0 <= acc2 <= 1.0

    bcres_model = DummyBCResModel()
    muc_b, acc_b, n_b, _ = auc_acc_bcres(
        scp, bcres_model, "cpu", tw=-1, tu=1, bDnnWin=3, featdim=4, cutThres="True"
    )
    assert n_b == 15
    assert 0.0 <= muc_b <= 1.0
    assert 0.0 <= acc_b <= 1.0
