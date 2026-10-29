import sys
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve


def compute_mindcf(
    score: NDArray[Any],
    label: NDArray[Any],
    priors: Sequence[float],
    costs: Sequence[float],
) -> tuple[float, NDArray[Any], NDArray[Any], float]:
    """Compute the minimum detection cost function (minDCF) for a given classifier."""
    fpr, tpr, thresholds = roc_curve(label, score, pos_label=1)  # Specify target class label
    # False Negative Rate (FNR) = 1 - True Positive Rate (TPR)
    fnr: NDArray[Any] = 1 - tpr
    C_fn, C_fp = costs
    P_target, P_non_target = priors
    min_dcf: float = float(np.min(C_fn * P_target * fnr + C_fp * P_non_target * fpr))
    eer: float = float(brentq(lambda x: 1.0 - x - float(interp1d(fpr, tpr)(x)), 0.0, 1.0))
    _thresh: NDArray[Any] = interp1d(fpr, thresholds)(eer)
    return min_dcf, fpr, fnr, eer


def main() -> None:
    score_arg: str = sys.argv[1]
    label_arg: str = sys.argv[2]

    # Define the prior probabilities for the target and non-target class
    priors: list[float] = [0.01, 1 - 0.01]  # Equal priors for simplicity
    print("priors", priors)

    # Define the cost of false negative and false positive (C_fn, C_fp)
    costs: list[float] = [10, 1]  # Equal cost for false negative and false positive
    print("costs", costs)

    score: NDArray[np.float64] = np.loadtxt(str(score_arg)).astype(float)
    label: NDArray[np.intp] = np.loadtxt(str(label_arg)).astype(int)
    print(score)
    print(label)
    if score.shape[0] != len(label):
        raise ValueError("mismatch number of score and labels!")
    min_dcf_value_mfcc_vit, _fpr, _fnr, eer = compute_mindcf(score, label, priors, costs)

    print("MinDCF", min_dcf_value_mfcc_vit, "EER(%)", eer * 100)


if __name__ == "__main__":
    main()
