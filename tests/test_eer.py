import sys
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from librivad.vad.eer import compute_mindcf, main


def test_compute_mindcf() -> None:
    scores: NDArray[np.float64] = np.array([0.05, 0.1, 0.2, 0.8, 0.9, 0.95], dtype=np.float64)
    labels: NDArray[np.intp] = np.array([0, 0, 0, 1, 1, 1], dtype=np.intp)
    priors: list[float] = [0.01, 0.99]
    costs: list[float] = [10.0, 1.0]

    min_dcf, fpr, fnr, eer = compute_mindcf(scores, labels, priors, costs)
    assert min_dcf >= 0.0
    assert 0.0 <= eer <= 1.0
    assert fpr.shape == fnr.shape
    assert eer == pytest.approx(0.0, abs=1e-3)


def test_eer_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    score_file: Path = tmp_path / "scores.txt"
    label_file: Path = tmp_path / "labels.txt"
    np.savetxt(score_file, np.array([0.1, 0.2, 0.8, 0.9]), fmt="%.4f")
    np.savetxt(label_file, np.array([0, 0, 1, 1]), fmt="%d")

    monkeypatch.setattr(sys, "argv", ["eer.py", str(score_file), str(label_file)])
    main()

    bad_label_file: Path = tmp_path / "bad_labels.txt"
    np.savetxt(bad_label_file, np.array([0, 1]), fmt="%d")
    monkeypatch.setattr(sys, "argv", ["eer.py", str(score_file), str(bad_label_file)])
    with pytest.raises(ValueError, match="mismatch number of score and labels"):
        main()
