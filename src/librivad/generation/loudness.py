"""
Integrated loudness of a mono signal (ITU-R BS.1770-4), for `loudness.measure: lufs`.

K-weighting (a high shelf of +4 dB above about 1.5 kHz, then a high-pass at 38 Hz),
designed for any sample rate, then gated 400 ms blocks with 75 % overlap: blocks below
−70 LUFS are dropped, then blocks more than 10 LU below the mean of the rest. Gating
skips pauses, so the loudness of a speech recording is that of its speech.

Signals are in int16 units (as `noise.read_int16_units` returns them); full scale is
32768, so the result is in LUFS.
"""

import math
from functools import cache

import numpy as np
from numpy.typing import NDArray
from scipy.signal import lfilter

FULL_SCALE: float = 32768.0
BLOCK_S: float = 0.4
STEP: float = 0.25
"""Block hop as a fraction of the block (75 % overlap)."""
ABSOLUTE_GATE_LUFS: float = -70.0
RELATIVE_GATE_LU: float = -10.0


@cache
def k_weighting(sample_rate: int) -> tuple[tuple[NDArray[np.float64], NDArray[np.float64]], ...]:
    """The two biquads of the K-weighting filter, as (b, a) pairs."""
    # Stage 1: high shelf, gain 4 dB, 1500 Hz, Q 1/sqrt(2).
    gain: float = 10 ** (4.0 / 40)
    w0: float = 2 * math.pi * 1500.0 / sample_rate
    alpha: float = math.sin(w0) / (2 / math.sqrt(2))
    cos: float = math.cos(w0)
    root: float = 2 * math.sqrt(gain) * alpha
    shelf_b = np.array(
        [
            gain * ((gain + 1) + (gain - 1) * cos + root),
            -2 * gain * ((gain - 1) + (gain + 1) * cos),
            gain * ((gain + 1) + (gain - 1) * cos - root),
        ]
    )
    shelf_a = np.array(
        [
            (gain + 1) - (gain - 1) * cos + root,
            2 * ((gain - 1) - (gain + 1) * cos),
            (gain + 1) - (gain - 1) * cos - root,
        ]
    )
    # Stage 2: high-pass, 38 Hz, Q 0.5.
    w0 = 2 * math.pi * 38.0 / sample_rate
    alpha = math.sin(w0) / (2 * 0.5)
    cos = math.cos(w0)
    pass_b = np.array([(1 + cos) / 2, -(1 + cos), (1 + cos) / 2])
    pass_a = np.array([1 + alpha, -2 * cos, 1 - alpha])
    return (shelf_b / shelf_a[0], shelf_a / shelf_a[0]), (pass_b / pass_a[0], pass_a / pass_a[0])


def integrated_loudness(signal: NDArray[np.floating], sample_rate: int) -> float:
    """Gated loudness in LUFS; −inf when the signal is shorter than one block or every
    block is below the absolute gate."""
    block: int = round(BLOCK_S * sample_rate)
    hop: int = round(BLOCK_S * STEP * sample_rate)
    if len(signal) < block:
        return float("-inf")
    weighted: NDArray[np.float64] = np.asarray(signal, dtype=np.float64) / FULL_SCALE
    for b, a in k_weighting(sample_rate):
        weighted = lfilter(b, a, weighted)
    squares: NDArray[np.float64] = np.concatenate([[0.0], np.cumsum(weighted**2)])
    starts: NDArray[np.int64] = np.arange(0, len(weighted) - block + 1, hop)
    powers: NDArray[np.float64] = (squares[starts + block] - squares[starts]) / block
    with np.errstate(divide="ignore"):
        levels: NDArray[np.float64] = -0.691 + 10 * np.log10(powers)
    kept: NDArray[np.float64] = powers[levels > ABSOLUTE_GATE_LUFS]
    if kept.size == 0:
        return float("-inf")
    relative: float = -0.691 + 10 * math.log10(float(kept.mean())) + RELATIVE_GATE_LU
    with np.errstate(divide="ignore"):
        kept = kept[-0.691 + 10 * np.log10(kept) > relative]
    return -0.691 + 10 * math.log10(float(kept.mean()))
