"""
Renders planned outputs: places the speech and noise, sets the noise level, guards against
clipping and converts to int16.

With `noise_span: block` the excerpt fills the layout's noise block. With `noise_span:
file` it runs under the whole output at one gain and the speech is added on top, faded in
and out at its edges.

The level is set by `snr_db` (speech-only RMS over the noise RMS), by `loudness_lu`
(integrated loudness of the speech blocks over that of the noise, ITU-R BS.1770) or by a
fixed gain. Every output measures both, whichever set the gain.

Outputs that differ only by level (one `group`) are rendered together: they share the same
sources and noise excerpt, and with `clip_guard: per_group` the same guard gain, so the
speech sits at the same absolute level in every one of them.
"""

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from librivad.generation.config import Config, ScenarioConfig
from librivad.generation.corpus import SourceItem
from librivad.generation.labels import load_labels
from librivad.generation.loudness import integrated_loudness
from librivad.generation.noise import NoiseSource, raised_cosine, read_int16_units
from librivad.generation.planner import LEVEL_TOLERANCE_DB, BlockSpec, FileSpec

FULL_SCALE: float = 32767.0
INT16_MAX: float = 32767.0
INT16_MIN: float = -32768.0


class RenderError(RuntimeError):
    """An output failed one of its checks; it is reported and not written."""


@dataclass
class Rendered:
    spec: FileSpec
    audio: NDArray[np.int16]
    labels: NDArray[np.int8]
    noise_gain_db: float
    clip_gain_db: float
    measured_snr_db: float | None
    measured_loudness_lu: float | None = None
    clip_triggered: bool = False
    """This output alone would have clipped (with `per_group`, others in its group are
    scaled with it)."""


def rms(signal: NDArray[np.floating] | NDArray[np.integer]) -> float:
    if signal.size == 0:
        return 0.0
    values: NDArray[np.float64] = np.asarray(signal, dtype=np.float64)
    return math.sqrt(float(np.mean(values**2)))


def to_db(ratio: float) -> float:
    return 20 * math.log10(ratio) if ratio > 0 else float("-inf")


def clip_gain(signal: NDArray[np.float64]) -> float:
    """1.0, or the gain that makes the rounded signal fit int16 exactly when it would not."""
    if signal.size == 0:
        return 1.0
    gain: float = 1.0
    high: float = float(signal.max())
    low: float = float(signal.min())
    if round(high) > INT16_MAX:
        gain = min(gain, INT16_MAX / high)
    if round(low) < INT16_MIN:
        gain = min(gain, INT16_MIN / low)
    return gain


@dataclass
class _Draft:
    spec: FileSpec
    signal: NDArray[np.float64]
    speech: NDArray[np.float64]
    """The speech alone, as placed (before the noise is added)."""
    labels: NDArray[np.int8]
    noise_gain: float
    clip_triggered: bool = False


def finite(value: float) -> float | None:
    return value if math.isfinite(value) else None


class Renderer:
    def __init__(
        self, config: Config, items: dict[str, SourceItem], noises: dict[str, NoiseSource]
    ) -> None:
        self.config = config
        self.items = items
        self.noises = noises

    def render_group(self, specs: list[FileSpec]) -> list[Rendered | RenderError]:
        """Render every output of one group. A failed output comes back as its `RenderError`."""
        audio: dict[str, NDArray[np.float64]] = {}
        labels: dict[str, NDArray[np.int8]] = {}
        excerpts: dict[tuple[str, str, int, int], NDArray[np.float64]] = {}
        for ref in specs[0].speech_items():
            item: SourceItem = self.items[ref.id]
            audio[ref.id] = read_int16_units(item.path)
            labels[ref.id] = load_labels(item, self.config.dataset)

        drafts: list[_Draft | RenderError] = []
        for spec in specs:
            try:
                drafts.append(self._draft(spec, audio, labels, excerpts))
            except RenderError as exc:
                drafts.append(exc)
        return self._finish(drafts)

    def place_speech(
        self,
        spec: FileSpec,
        audio: dict[str, NDArray[np.float64]],
        labels: dict[str, NDArray[np.int8]],
    ) -> tuple[NDArray[np.float64], NDArray[np.int8]]:
        """The speech of one output at its positions (zeros elsewhere), and its labels.

        Sources joined inside a block are faded at the joins (`join_fade_ms`); each speech
        block is faded at its edges (`edge_fade_ms`, by default only with `noise_span:
        file`, where the speech is added over the noise).
        """
        scenario: ScenarioConfig = self.config.scenario(spec.scenario)
        sr: int = self.config.dataset.sample_rate
        signal: NDArray[np.float64] = np.zeros(spec.total_samples)
        label: NDArray[np.int8] = np.zeros(spec.total_samples, dtype=np.int8)
        join_fade: int = round(scenario.speech_block.join_fade_ms / 1000 * sr)
        edge_fade: int = round(scenario.speech_block.edge_fade(spec.noise_span) / 1000 * sr)

        for block in spec.blocks:
            if block.kind != "speech":
                continue
            for index, ref in enumerate(block.items):
                piece: NDArray[np.float64] = audio[ref.id].copy()
                if join_fade > 0 and len(block.items) > 1:
                    fade: int = min(join_fade, len(piece) // 2)
                    if index > 0:
                        piece[:fade] *= raised_cosine(fade)
                    if index < len(block.items) - 1:
                        piece[len(piece) - fade :] *= raised_cosine(fade)[::-1]
                signal[ref.offset : ref.offset + ref.n_samples] = piece
                label[ref.offset : ref.offset + ref.n_samples] = labels[ref.id]
            if edge_fade > 0:
                fade = min(edge_fade, block.length // 2)
                end: int = block.start + block.length
                signal[block.start : block.start + fade] *= raised_cosine(fade)
                signal[end - fade : end] *= raised_cosine(fade)[::-1]
        return signal, label

    def speech_mask(self, spec: FileSpec, labels: NDArray[np.int8]) -> NDArray[np.bool_]:
        """Samples the speech RMS is measured over: the speech blocks, label 1 only when
        `loudness.speech_only`."""
        mask: NDArray[np.bool_] = np.zeros(spec.total_samples, dtype=bool)
        for block in spec.blocks:
            if block.kind == "speech":
                mask[block.start : block.start + block.length] = True
        if self.config.scenario(spec.scenario).loudness.speech_only:
            mask &= labels == 1
        return mask

    @staticmethod
    def speech_blocks(spec: FileSpec, signal: NDArray[np.floating]) -> NDArray[np.float64]:
        """The speech blocks of `signal` joined, for their loudness (gating skips pauses)."""
        parts = [signal[b.start : b.start + b.length] for b in spec.blocks if b.kind == "speech"]
        return np.concatenate(parts).astype(np.float64)

    @staticmethod
    def noise_region(spec: FileSpec, noise: NDArray[np.floating]) -> NDArray[np.floating]:
        """Where the noise level is measured: the whole excerpt with `noise_span: file`,
        the noise block without its fades otherwise. `noise` spans the excerpt."""
        if spec.noise_span == "file":
            return noise
        block: BlockSpec | None = spec.noise_block()
        assert block is not None
        return noise[block.fade_in : block.length - block.fade_out]

    def _draft(
        self,
        spec: FileSpec,
        audio: dict[str, NDArray[np.float64]],
        labels: dict[str, NDArray[np.int8]],
        excerpts: dict[tuple[str, str, int, int], NDArray[np.float64]],
    ) -> _Draft:
        sr: int = self.config.dataset.sample_rate
        speech, label = self.place_speech(spec, audio, labels)

        cache_key: tuple[str, str, int, int] = (
            spec.noise,
            spec.split,
            spec.noise_offset,
            spec.excerpt_samples,
        )
        if cache_key not in excerpts:
            source: NoiseSource = self.noises[spec.noise]
            excerpts[cache_key] = source.read(spec.split, spec.noise_offset, spec.excerpt_samples)
        excerpt: NDArray[np.float64] = excerpts[cache_key]
        measured_region: NDArray[np.floating] = self.noise_region(spec, excerpt)

        if spec.fixed_gain is not None:
            gain: float = spec.fixed_gain
        elif spec.loudness_lu is not None:
            speech_lufs: float = integrated_loudness(self.speech_blocks(spec, speech), sr)
            noise_lufs: float = integrated_loudness(measured_region, sr)
            if not (math.isfinite(speech_lufs) and math.isfinite(noise_lufs)):
                raise RenderError(f"{spec.out_path}: speech or noise below the loudness gate")
            gain = 10 ** ((speech_lufs - spec.loudness_lu - noise_lufs) / 20)
        else:
            assert spec.snr_db is not None
            reference: float = rms(speech[self.speech_mask(spec, label)])
            noise_rms: float = rms(measured_region)
            if reference == 0 or noise_rms == 0:
                raise RenderError(f"{spec.out_path}: silent speech or noise, cannot set the level")
            gain = reference * 10 ** (-spec.snr_db / 20) / noise_rms

        scaled: NDArray[np.float64] = excerpt * gain
        if spec.noise_span == "file":
            signal: NDArray[np.float64] = speech + scaled
        else:
            block: BlockSpec | None = spec.noise_block()
            assert block is not None
            if block.fade_in:
                scaled[: block.fade_in] *= raised_cosine(block.fade_in)
            if block.fade_out:
                scaled[block.length - block.fade_out :] *= raised_cosine(block.fade_out)[::-1]
            signal = speech.copy()
            signal[block.start : block.start + block.length] = scaled
        return _Draft(spec=spec, signal=signal, speech=speech, labels=label, noise_gain=gain)

    def _finish(self, drafts: list[_Draft | RenderError]) -> list[Rendered | RenderError]:
        ok: list[_Draft] = [d for d in drafts if isinstance(d, _Draft)]
        guards: list[float] = self._guards(ok)
        results: list[Rendered | RenderError] = []
        index: int = 0
        for draft in drafts:
            if isinstance(draft, RenderError):
                results.append(draft)
                continue
            try:
                results.append(self._quantize(draft, guards[index]))
            except RenderError as exc:
                results.append(exc)
            index += 1
        return results

    def _guards(self, drafts: list[_Draft]) -> list[float]:
        """Gain (<= 1) per draft that keeps it inside int16 after rounding.

        1.0 unless rounding would push a sample above 32767 or below -32768; then the
        largest gain that brings the peak exactly to the limit. Sources that already sit
        at full scale are left alone as long as the noise does not push past it.
        """
        output = self.config.output
        if output.normalization != "none" or output.clip_guard == "off":
            return [1.0] * len(drafts)
        guards: list[float] = [clip_gain(draft.signal) for draft in drafts]
        for draft, guard in zip(drafts, guards, strict=True):
            draft.clip_triggered = guard < 1.0
        if output.clip_guard == "per_group" and guards:
            guards = [min(guards)] * len(guards)
        return guards

    def _quantize(self, draft: _Draft, guard: float) -> Rendered:
        output = self.config.output
        spec: FileSpec = draft.spec
        signal: NDArray[np.float64] = draft.signal
        applied: float = guard  # the gain from `signal` to the written audio
        if output.normalization == "peak":
            max_abs: float = float(np.max(np.abs(signal)))
            if max_abs == 0:
                audio: NDArray[np.int16] = signal.astype(np.int16)
            else:
                # Same arithmetic and truncation as librispeech_functions.scale_signal.
                target: float = FULL_SCALE * 10 ** (output.peak_dbfs / 20)
                audio = np.asarray(signal / max_abs * target, dtype=np.int16)
                applied = target / max_abs
        else:
            rounded: NDArray[np.float64] = np.round(signal * guard)
            if rounded.size and (rounded.max() > INT16_MAX or rounded.min() < INT16_MIN):
                raise RenderError(f"{spec.out_path}: would clip in int16")
            audio = rounded.astype(np.int16)

        if len(audio) != spec.total_samples or len(draft.labels) != spec.total_samples:
            raise RenderError(f"{spec.out_path}: length differs from the plan")
        guarded: bool = guard < 1.0

        snr, loudness = self.measure(spec, audio, draft.labels, draft.speech * applied)
        checks = ((spec.snr_db, snr, "dB"), (spec.loudness_lu, loudness, "LU"))
        for wanted, measured, unit in checks:
            if wanted is None:
                continue
            if measured is None or abs(measured - wanted) > LEVEL_TOLERANCE_DB:
                shown: str = "nothing" if measured is None else f"{measured:.3f} {unit}"
                raise RenderError(f"{spec.out_path}: measured {shown}, target {wanted} {unit}")
        return Rendered(
            spec=spec,
            audio=audio,
            labels=draft.labels,
            noise_gain_db=to_db(draft.noise_gain),
            clip_gain_db=to_db(guard) if guarded else 0.0,
            measured_snr_db=snr,
            measured_loudness_lu=loudness,
            clip_triggered=draft.clip_triggered,
        )

    def measure(
        self,
        spec: FileSpec,
        audio: NDArray[np.int16],
        labels: NDArray[np.int8],
        speech: NDArray[np.float64] | None = None,
    ) -> tuple[float | None, float | None]:
        """SNR (dB) and loudness difference (LU) of a written output.

        With `noise_span: block` both come from the audio: the speech blocks and the noise
        block without its fades. With `noise_span: file` the speech and noise overlap, so
        `speech` (the speech as written, before rounding) is needed: the noise is the audio
        minus it, over the whole file.
        """
        sr: int = self.config.dataset.sample_rate
        mask: NDArray[np.bool_] = self.speech_mask(spec, labels)
        values: NDArray[np.float64] = audio.astype(np.float64)
        if spec.noise_span == "file":
            if speech is None:
                raise ValueError("noise_span file: measuring needs the speech as written")
            speech_signal: NDArray[np.float64] = speech
            noise_region: NDArray[np.floating] = values - speech
        else:
            block: BlockSpec | None = spec.noise_block()
            assert block is not None
            speech_signal = values
            noise_region = self.noise_region(spec, values[block.start : block.start + block.length])
        snr: float = to_db(rms(speech_signal[mask]) / max(rms(noise_region), 1e-12))
        loudness: float = integrated_loudness(
            self.speech_blocks(spec, speech_signal), sr
        ) - integrated_loudness(noise_region, sr)
        return finite(snr), finite(loudness)
