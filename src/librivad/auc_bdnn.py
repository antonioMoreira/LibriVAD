import copy
import sys
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn import metrics
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import nn

warnings.filterwarnings("ignore")


def uW_context(
    X_batch: NDArray[Any],
    w: int,
    u: int,
    bDnnWin: int,
) -> NDArray[Any]:
    if X_batch.shape[0] < 2:
        raise Exception("data is not matrix or having one frame only")

    index: NDArray[np.intp] = np.arange(X_batch.shape[0])  # [frame x feat]
    Xarray: NDArray[Any] = copy.deepcopy(X_batch)  # [0] current location

    # generate left context   #w=19, u=9
    L1: NDArray[Any] = np.sort(np.concatenate((np.arange(-1 - u, -w, -u), np.array([-w]))))
    L: NDArray[Any] = np.sort(
        np.unique(np.concatenate((L1, np.array([-1]))))
    )  # array([-19, -10,  -1])

    # right context
    R1: NDArray[Any] = np.concatenate((np.array([1]), np.arange(1 + u, w, u)))  #
    R: NDArray[Any] = np.sort(
        np.unique(np.concatenate((R1, np.array([w]))))
    )  # array([1, 10, 19])

    # both L and R are symmetric
    for i in np.arange(len(L) - 1, -1, -1):  # negative indices
        x: int = int(np.abs(L[i]))
        t: NDArray[np.intp] = np.pad(
            index[:-x],
            pad_width=(len(index) - len(index[:-x]), 0),
            mode="constant",
            constant_values=index[0],
        )
        Xarray = np.hstack((X_batch[t, :], Xarray))

    for i in np.arange(0, len(R)):  # positive indic
        x = int(np.abs(R[i]))
        t = np.pad(
            index[x:],
            pad_width=(0, len(index) - len(index[x:])),
            mode="constant",
            constant_values=index[-1],
        )
        Xarray = np.hstack((Xarray, X_batch[t, :]))

    Xarray = np.sort(Xarray)

    if bDnnWin * X_batch.shape[1] != Xarray.shape[1]:
        raise Exception("mismatch contextual bDnnWin dimension")

    return Xarray


def LRcontext(
    X_batch: NDArray[Any],
    Lt: int,
    Rt: int,
    bDnnWin: int,
) -> NDArray[Any]:
    if X_batch.shape[0] < 2:
        raise Exception("data is not matrix or having one frame only")

    index: NDArray[np.intp] = np.arange(X_batch.shape[0])  # [frame x feat]
    Xarray: NDArray[Any] = copy.deepcopy(X_batch)  # [0] current location

    # generate left context   #w=19, u=9
    L1: NDArray[Any] = np.sort(np.concatenate((np.arange(Lt, 0, 1), np.array([Lt]))))
    L: NDArray[Any] = np.sort(np.unique(np.concatenate((L1, np.array([Lt])))))

    # right context
    R1: NDArray[Any] = np.concatenate((np.array([1]), np.arange(1, Rt, 1)))  #
    R: NDArray[Any] = np.sort(np.unique(np.concatenate((R1, np.array([Rt])))))

    # both L and R are symmetric
    for i in np.arange(len(L) - 1, -1, -1):  # negative indices (read like that -1, -2, ,,,)
        x: int = int(np.abs(L[i]))
        t: NDArray[np.intp] = np.sort(
            np.pad(
                index[:-x],
                pad_width=(len(index) - len(index[:-x]), 0),
                mode="constant",
                constant_values=index[0],
            )
        )
        Xarray = np.hstack((X_batch[t, :], Xarray))

    for i in np.arange(0, len(R)):  # positive indic
        x = int(np.abs(R[i]))
        t = np.pad(
            index[x:],
            pad_width=(0, len(index) - len(index[x:])),
            mode="constant",
            constant_values=index[-1],
        )
        Xarray = np.hstack((Xarray, X_batch[t, :]))

    if bDnnWin * X_batch.shape[1] != Xarray.shape[1]:
        raise Exception("mismatch contextual bDnnWin dimension")

    return Xarray


def enframe(
    speech: NDArray[Any],
    fs: int | float,
    winlen: float,
    ovrlen: float,
) -> tuple[NDArray[Any], int]:
    # split the speech data into frames
    N: int = len(speech)
    flth: int = int(np.trunc(fs * winlen))
    foVr: int = int(np.trunc(fs * ovrlen))

    if len(speech) < flth:
        print("speech file length shorter than 1-frame")
        sys.exit()

    frames: int = int(np.ceil((N - flth) / foVr)) + 1
    slen: int = (frames - 1) * foVr + flth
    signal: NDArray[Any]
    if len(speech) < slen:
        signal = np.concatenate((speech, np.zeros(slen - N)))

    else:
        signal = copy.deepcopy(speech)

    idx: NDArray[np.int64] = np.array(
        np.tile(np.arange(0, flth), (frames, 1))
        + np.tile(np.arange(0, (frames) * foVr, foVr), (flth, 1)).T,
        dtype=np.int64,
    )

    return signal[idx], foVr


def auc_acc(
    tstscp: Sequence[str] | NDArray[Any],
    model: nn.Module,
    device: torch.device | str,
    tw: int,
    tu: int,
    bDnnWin: int,
    cutThres: str | float,
) -> tuple[float, float, int, Any, NDArray[Any] | int, NDArray[Any] | int]:
    score_list: list[Any] = []
    lab_list: list[Any] = []
    for fe in tstscp:
        try:
            sgn_tensor: torch.Tensor = torch.load(
                fe.split(",")[2].strip("\n") + ".pt", weights_only=True
            )
            sgn_arr: NDArray[Any] = sgn_tensor.detach().cpu().numpy()
            sgn_arr = copy.deepcopy(uW_context(sgn_arr, tw, tu, bDnnWin))
            sgn: torch.Tensor = torch.from_numpy(sgn_arr)

            max_voters_tensor: torch.Tensor = torch.load(
                fe.split(",")[2].strip("\n") + ".lab.pt", weights_only=True
            )
            max_voters_pred: NDArray[Any] = max_voters_tensor.detach().cpu().numpy()

            if sgn.shape[0] == max_voters_pred.shape[0]:
                sgn = sgn.to(device)
                sgn = sgn.unsqueeze(0)
                _bn1, _bn2, logit_tensor = model(sgn.float())
                logit_tensor = logit_tensor.squeeze(0)
                conTxt: int = int(logit_tensor.shape[1])
                if conTxt != bDnnWin:
                    raise Exception("mismatch score and frame context bDnnWin")
                logit: Any = np.asmatrix(np.sum(logit_tensor.cpu().detach().numpy(), 1) / conTxt)
                logit = logit.T
                logit = np.concatenate((1 - logit, logit), axis=1)

                if logit.shape[0] == max_voters_pred.shape[0]:
                    score_list.append(logit)
                    lab_list.append(np.asmatrix(max_voters_pred))

        except ValueError:
            pass
    score: NDArray[Any] = np.squeeze(np.concatenate(score_list, axis=0))
    Lab: Any = np.squeeze(np.concatenate(lab_list, axis=0))
    Lab = Lab.T
    tot: int = 0
    corr: int = 0
    ACC1: float = 0.0
    if cutThres == "True":
        fpr, tpr, thresholds = metrics.roc_curve(
            np.squeeze(np.asarray(Lab)), np.asarray(score[:, 1]), pos_label=1
        )
        muc1: float = float(metrics.auc(fpr, tpr))
        pred_arr: Any = np.argmax(score, axis=1)  # ML
        if pred_arr.shape[0] == Lab.shape[0]:
            pred_diff: Any = pred_arr - Lab
            pred_zero: NDArray[Any] = np.array(np.where(pred_diff == 0))
            corr = int(pred_zero.shape[1])
            tot = int(score.shape[0])
            ACC1 = corr / tot
        muc1 = round(muc1, 4)
        return muc1, round(ACC1, 4), int(score.shape[0]), thresholds, fpr, tpr

    else:
        thres_val: float = float(cutThres)
        pred_list: list[int] = [1 if s >= thres_val else 0 for s in np.asarray(score[:, 1])]
        muc1 = float(roc_auc_score(np.squeeze(np.asarray(Lab)), pred_list))
        pred_mat: Any = np.matrix(np.array(pred_list)).T
        fpr_int: int = 0
        tpr_int: int = 0
        if pred_mat.shape[0] == Lab.shape[0]:
            pred_diff = pred_mat - Lab
            pred_zero = np.array(np.where(pred_diff == 0))
            corr = int(pred_zero.shape[1])
            tot = int(score.shape[0])
            ACC1 = corr / tot

        muc1 = round(muc1, 4)
        return muc1, round(ACC1, 4), int(score.shape[0]), cutThres, fpr_int, tpr_int


def auc_acc_bcres(
    tstscp: Sequence[str] | NDArray[Any],
    model: nn.Module,
    device: torch.device | str,
    tw: int,
    tu: int,
    bDnnWin: int,
    featdim: int,
    cutThres: str | float,
) -> tuple[float, float, int, Any]:
    score_list: list[Any] = []
    lab_list: list[Any] = []
    ACC1: float = 0.0
    for fe in tstscp:
        try:
            sgn_tensor: torch.Tensor = torch.load(
                fe.split(",")[2].strip("\n") + ".pt", weights_only=True
            )
            sgn_arr: NDArray[Any] = sgn_tensor.detach().cpu().numpy()
            sgn_arr = copy.deepcopy(LRcontext(sgn_arr, tw, tu, bDnnWin))  # context
            sgn: torch.Tensor = torch.from_numpy(sgn_arr)

            max_voters_tensor: torch.Tensor = torch.load(
                fe.split(",")[2].strip("\n") + ".lab.pt", weights_only=True
            )
            max_voters_pred: NDArray[Any] = max_voters_tensor.detach().cpu().numpy()

            if sgn.shape[0] == max_voters_pred.shape[0]:
                sgn = sgn.to(device)
                sgn = torch.reshape(sgn, [sgn.shape[0], -1, featdim])
                sgn = sgn.unsqueeze(0)  # [1, 931, 11, 39]
                sgn = sgn.permute(1, 0, 3, 2)
                logit_tensor: torch.Tensor = model(sgn.float())  # [BZ, 1, 40, #Frames]
                logit_tensor = torch.softmax(logit_tensor, dim=1)
                logit_np: NDArray[Any] = logit_tensor.cpu().detach().numpy()
                logit: Any = np.asmatrix(logit_np)
                score_list.append(logit)
                lab_list.append(np.asmatrix(max_voters_pred))

        except ValueError:
            pass
        score: NDArray[Any] = np.squeeze(np.concatenate(score_list, axis=0))
        Lab: NDArray[Any] = np.squeeze(np.asarray(np.squeeze(np.concatenate(lab_list, axis=0))))

        if cutThres == "True":
            fpr, tpr, thresholds = metrics.roc_curve(Lab, np.asarray(score[:, 1]), pos_label=1)
            muc1: float = float(metrics.auc(fpr, tpr))
            pred: Any = np.argmax(score, axis=1)  # ML
            if pred.shape[0] == Lab.shape[0]:
                ACC1 = float(accuracy_score(Lab, np.asarray(pred)))
            muc1 = round(muc1, 4)
            return muc1, round(ACC1, 4), int(score.shape[0]), thresholds
        else:
            thres_val: float = float(cutThres)
            pred_list: list[int] = [1 if s >= thres_val else 0 for s in np.asarray(score[:, 1])]
            pred_arr: NDArray[Any] = np.squeeze(np.asarray(pred_list))
            muc1 = float(roc_auc_score(Lab, pred_arr))
            if pred_arr.shape[0] == Lab.shape[0]:
                ACC1 = float(accuracy_score(Lab, pred_arr))

            muc1 = round(muc1, 4)
            return muc1, round(ACC1, 4), int(score.shape[0]), cutThres
    raise ValueError("tstscp cannot be empty")
