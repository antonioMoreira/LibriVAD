import copy
import sys
from pathlib import Path
from typing import Any

import numpy
import numpy as np
import scipy
import scipy.io.wavfile as wav
import torch
from numpy.typing import NDArray
from scipy.fftpack import dct

from librivad.vad.auc_bdnn import enframe


def mfcc(
    file_wav: str,
    winlen: float,
    ovrlen: float,
    pre_coef: float,
    nfilter: int,
    nftt: int,
    no_coef: int,
) -> tuple[NDArray[Any], Any]:
    # S. Davis ; P. Mermelstein, "Comparison of parametric representations for monosyllabic word
    # recognition in continuously spoken sentences", IEEE Transactions on Acoustics, Speech, and
    # Signal Processing ( Volume: 28, Issue: 4, Aug 1980 )

    fs: int
    speech: NDArray[Any]
    fs, speech = speech_wave(file_wav.split(",")[0].strip("\n"))

    eps: float = float(numpy.finfo(float).eps)
    #
    flen: int = int(numpy.trunc(fs * winlen))
    fsh10: int = int(numpy.trunc(fs * ovrlen))
    nfr10: int = int(numpy.floor((len(speech) - (flen - fsh10)) / fsh10))

    # for simple  - enegry threshold vad
    Espeech: NDArray[Any]
    foVr: int
    Espeech, foVr = enframe(speech, fs, winlen, ovrlen)  # framing (before pre-emphasis)
    Espeech = 20 * numpy.log10(numpy.std(Espeech, axis=1, ddof=1) + eps)
    # ====

    speech = numpy.append(speech[0], speech[1:] - pre_coef * speech[:-1])  # pre-emphasis
    speech, foVr = enframe(speech, fs, winlen, ovrlen)  # framing

    # framing and it's label
    y: NDArray[Any] = np.load(file_wav.split(",")[1].strip("\n"))
    # y = (np.loadtxt(os.path.splitext(file_wav.split(",")[1].strip('\n'))[0] + '.txt')).astype(int)
    # when label in txt format
    y, foVr = enframe(y, fs, winlen, ovrlen)

    # speech/non-speech votting samples
    max_votes: Any = scipy.stats.mode(y, axis=1)
    max_voters_pred: Any = max_votes[0]
    max_voters_pred = np.asmatrix(max_voters_pred).T

    if numpy.size(speech, axis=0) != numpy.size(Espeech, axis=0):
        print("Mismatch frame numbers for  enegry and feature vectors")

    if max_voters_pred.shape[0] != y.shape[0]:
        print("Mismatch Label and feature frames")

    w: Any = numpy.matrix(numpy.hamming(int(fs * winlen)))
    w = numpy.tile(w, (numpy.size(speech, axis=0), 1))

    speech = numpy.multiply(speech, w)  # apply window

    ff: NDArray[np.float64] = (fs / 2) * (numpy.linspace(0, 1, int(nftt / 2 + 1)))
    fmel: NDArray[np.float64] = 2595 * numpy.log10(1 + ff / 700)  # mel-scale
    fmelmax: float = float(numpy.max(fmel))
    fmelmin: float = float(numpy.min(fmel))

    filtbankMel: NDArray[np.float64] = numpy.linspace(
        fmelmin, fmelmax, nfilter + 2
    )  # define filter in mel domain
    filbankF: NDArray[np.float64] = 700 * (numpy.power(10, (filtbankMel / 2595)) - 1)

    # fft
    ffy: NDArray[Any] = numpy.abs(numpy.fft.fft(speech, nftt))
    idx: range = range(1, int(nftt / 2) + 1)
    ffy = numpy.power(ffy, 2)
    ffy = ffy[:, idx]

    BB: tuple[int, int] = (len(ff), nfilter)
    fbank: NDArray[np.float64] = numpy.zeros(BB)

    for nf in range(0, nfilter):
        fbank[:, nf] = trimf(
            ff, float(filbankF[nf]), float(filbankF[nf + 1]), float(filbankF[nf + 2])
        )

    fbank = fbank.T
    fbank = fbank[:, idx].T
    fbnkSum: Any = numpy.matrix(ffy) * numpy.matrix(fbank)

    # dct
    fbankSum_eps: Any = numpy.log10(fbnkSum.T + eps)
    t: NDArray[Any] = (dct(fbankSum_eps.T, norm="ortho")).T
    t = t[1:]  # dicard "c0" #(19, 291)
    t = t[:no_coef, :]

    # append energy
    t = np.vstack((t, Espeech))  # [feat x frames]
    numpy_array: NDArray[Any] = np.asarray(t.T)  # (frames x feat)

    # d, dd
    # delta -delta
    array_delta: NDArray[Any] = copy.deepcopy(deltas(numpy_array, 2))
    array_delta_delta: NDArray[Any] = copy.deepcopy(deltas(array_delta, 2))
    numpy_array = copy.deepcopy(np.hstack((numpy_array, array_delta, array_delta_delta)))

    return numpy_array[:nfr10, :], max_voters_pred[:nfr10, :]


def speech_wave(fileName_: str | Path) -> tuple[int, NDArray[Any]]:
    fs: int
    sig: NDArray[Any]
    (fs, sig) = wav.read(fileName_)
    sig = sig / numpy.amax(numpy.abs(sig))  # normalize the signal (for the feature extraction)
    return fs, sig


def trimf(
    Xx: NDArray[Any],
    aA_: float,
    bB_: float,
    cC_: float,
) -> NDArray[np.float64]:
    if aA_ > bB_:
        print("Parameter: a > b")
        sys.exit()
    elif bB_ > cC_:
        print("Parameter: b > c")
        sys.exit()
    elif aA_ > cC_:
        print("Parameter: a > c")
        sys.exit()

    BB: int = len(Xx)  ## ff
    ky: NDArray[np.float64] = numpy.zeros(BB)

    index: tuple[NDArray[np.intp], ...] = numpy.where((Xx <= aA_) | (cC_ <= Xx))
    ky[index] = numpy.zeros(len(index[0]))

    # slope 1
    if aA_ != bB_:
        index = numpy.where((aA_ < Xx) & (Xx < bB_))
        ky[index] = (Xx[index] - aA_) / (bB_ - aA_)

    # slope 2
    if bB_ != cC_:
        index = numpy.where((bB_ < Xx) & (Xx < cC_))
        ky[index] = (cC_ - Xx[index]) / (cC_ - bB_)

    # Center
    index = numpy.where(Xx == bB_)
    ky[index] = numpy.ones(len(index[0]))

    return ky


def deltas(feat: NDArray[Any], N: int) -> NDArray[Any]:
    if N < 1:
        raise ValueError("N must be an integer >= 1")
    denominator: int = 2 * sum([i**2 for i in range(1, N + 1)])
    NUMFRAMES: int = len(feat)
    delta_feat: NDArray[Any] = np.empty_like(feat)
    padded: NDArray[Any] = np.concatenate(
        (np.tile(feat[0, :], (N, 1)), feat, np.tile(feat[-1, :], (N, 1))), axis=0
    )
    for t in range(NUMFRAMES):
        delta_feat[t] = np.dot(np.arange(-N, N + 1), padded[t : t + 2 * N + 1])
    delta_feat /= denominator
    return delta_feat


def cmvn(numpy_array: NDArray[Any]) -> NDArray[Any]:
    # zero-mean std
    _mean: NDArray[np.float32] = np.mean(numpy_array, axis=0, dtype="float32")
    _std: NDArray[np.float32] = np.std(numpy_array, axis=0, ddof=1, dtype="float32")
    # if any element in _std value "0" replace by 1
    _std = np.where(_std == 0, 1, _std)
    Z: NDArray[Any] = (numpy_array - _mean) / _std

    return Z


def main() -> None:
    winlen: float = 0.025
    ovrlen: float = 0.01
    pre_coef: float = 0.97
    nfilter: int = 24
    nftt: int = 512
    no_coef: int = 12
    # [window size (sec)], [frame shift(sec)], [pre-emp coeff],

    scp: NDArray[np.str_] = np.genfromtxt(sys.argv[1], dtype="str")
    for x in scp:
        # print(x)
        feat_np, label_np = mfcc(str(x), winlen, ovrlen, pre_coef, nfilter, nftt, no_coef)
        feat_norm: NDArray[Any] = cmvn(feat_np)  # zero-mean unit variance normalization
        feat_tensor: torch.Tensor = torch.FloatTensor(np.asarray(feat_norm))
        label_tensor: torch.Tensor = torch.FloatTensor(np.asarray(label_np))
        # write in torch format
        torch.save(feat_tensor, str(x).split(",")[2].strip("\n") + ".pt")
        torch.save(label_tensor, str(x).split(",")[2].strip("\n") + ".lab.pt")


if __name__ == "__main__":
    main()
