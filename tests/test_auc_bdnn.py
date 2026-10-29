from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from torch import nn

from librivad.vad.auc_bdnn import LRcontext, auc_acc, auc_acc_bcres, enframe, uW_context


class DummyBDNNModel(nn.Module):
    def __init__(self, b_dnn_win: int) -> None:
        super().__init__()
        self.b_dnn_win: int = b_dnn_win

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # ty: ignore[invalid-mutable-override]
        batch_size, frames, _ = x.shape
        logits = torch.full((batch_size, frames, self.b_dnn_win), 0.8, dtype=torch.float32)
        return logits, logits, logits


class DummyBCResModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # ty: ignore[invalid-mutable-override]
        frames = x.shape[0]
        logits = torch.zeros((frames, 2), dtype=torch.float32)
        logits[:, 1] = 1.5
        return logits


def test_uw_context_valid() -> None:
    x_batch: NDArray[np.float64] = np.arange(40, dtype=np.float64).reshape(10, 4)
    out = uW_context(x_batch, w=3, u=1, bDnnWin=7)
    assert out.shape == (10, 28)


def test_uw_context_errors() -> None:
    single_frame: NDArray[np.float64] = np.ones((1, 4), dtype=np.float64)
    with pytest.raises(Exception, match="one frame only"):
        uW_context(single_frame, w=3, u=1, bDnnWin=7)

    x_batch: NDArray[np.float64] = np.ones((10, 4), dtype=np.float64)
    with pytest.raises(Exception, match="mismatch contextual bDnnWin dimension"):
        uW_context(x_batch, w=3, u=1, bDnnWin=5)


def test_lr_context_valid_and_errors() -> None:
    x_batch: NDArray[np.float64] = np.arange(40, dtype=np.float64).reshape(10, 4)
    out = LRcontext(x_batch, Lt=-2, Rt=2, bDnnWin=5)
    assert out.shape == (10, 20)

    with pytest.raises(Exception, match="one frame only"):
        LRcontext(np.ones((1, 4)), Lt=-2, Rt=2, bDnnWin=5)

    with pytest.raises(Exception, match="mismatch contextual bDnnWin dimension"):
        LRcontext(x_batch, Lt=-2, Rt=2, bDnnWin=3)


def test_enframe() -> None:
    speech: NDArray[np.float64] = np.sin(np.linspace(0, 10, 1600))
    frames, fovr = enframe(speech, fs=16000, winlen=0.025, ovrlen=0.01)
    assert fovr == 160
    assert frames.shape[1] == 400
    assert frames.shape[0] >= 8

    short_speech: NDArray[np.float64] = np.zeros(10)
    with pytest.raises(SystemExit):
        enframe(short_speech, fs=16000, winlen=0.025, ovrlen=0.01)


def test_auc_acc_and_bcres(tmp_path: Path) -> None:
    feat_prefix: Path = tmp_path / "sample1"
    feats: torch.Tensor = torch.randn(12, 4, dtype=torch.float32)
    labels: torch.Tensor = torch.tensor(
        [[0], [1], [1], [0], [1], [0], [1], [1], [0], [0], [1], [1]],
        dtype=torch.float32,
    )
    torch.save(feats, f"{feat_prefix}.pt")
    torch.save(labels, f"{feat_prefix}.lab.pt")

    scp: list[str] = [f"wav.wav,lab.npy,{feat_prefix}"]
    bdnn_model = DummyBDNNModel(b_dnn_win=7)

    muc_true, acc_true, n_samples, _thresholds, _fpr, _tpr = auc_acc(
        scp, bdnn_model, "cpu", tw=3, tu=1, bDnnWin=7, cutThres="True"
    )
    assert n_samples == 12
    assert 0.0 <= muc_true <= 1.0
    assert 0.0 <= acc_true <= 1.0

    muc_num, acc_num, n_samples_num, thres_out, fpr_int, tpr_int = auc_acc(
        scp, bdnn_model, "cpu", tw=3, tu=1, bDnnWin=7, cutThres=0.5
    )
    assert n_samples_num == 12
    assert 0.0 <= muc_num <= 1.0
    assert 0.0 <= acc_num <= 1.0
    assert thres_out == 0.5
    assert fpr_int == 0 and tpr_int == 0

    bcres_model = DummyBCResModel()
    muc_b, acc_b, n_b, _ = auc_acc_bcres(
        scp, bcres_model, "cpu", tw=-2, tu=2, bDnnWin=5, featdim=4, cutThres="True"
    )
    assert n_b == 12
    assert 0.0 <= muc_b <= 1.0
    assert 0.0 <= acc_b <= 1.0

    muc_b2, acc_b2, n_b2, thres_b2 = auc_acc_bcres(
        scp, bcres_model, "cpu", tw=-2, tu=2, bDnnWin=5, featdim=4, cutThres=0.5
    )
    assert n_b2 == 12
    assert 0.0 <= muc_b2 <= 1.0
    assert 0.0 <= acc_b2 <= 1.0
    assert thres_b2 == 0.5
    _ = Any
