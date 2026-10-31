"""
Renders planned outputs: places the speech and noise blocks, sets the noise level, guards
against clipping and converts to int16.

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
    labels: NDArray[np.int8]
    noise_gain: float
    clip_triggered: bool = False


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

    def _draft(
        self,
        spec: FileSpec,
        audio: dict[str, NDArray[np.float64]],
        labels: dict[str, NDArray[np.int8]],
        excerpts: dict[tuple[str, str, int, int], NDArray[np.float64]],
    ) -> _Draft:
        scenario: ScenarioConfig = self.config.scenario(spec.scenario)
        sr: int = self.config.dataset.sample_rate
        signal: NDArray[np.float64] = np.zeros(spec.total_samples)
        label: NDArray[np.int8] = np.zeros(spec.total_samples, dtype=np.int8)
        join_fade: int = round(scenario.speech_block.join_fade_ms / 1000 * sr)

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

        speech_mask: NDArray[np.bool_] = np.zeros(spec.total_samples, dtype=bool)
        for block in spec.blocks:
            if block.kind == "speech":
                speech_mask[block.start : block.start + block.length] = True
        if scenario.loudness.speech_only:
            speech_mask &= label == 1

        noise: BlockSpec = spec.noise_block()
        assert noise.noise is not None and noise.noise_file is not None
        assert noise.noise_offset is not None
        cache_key: tuple[str, str, int, int] = (
            noise.noise,
            spec.split,
            noise.noise_offset,
            noise.length,
        )
        if cache_key not in excerpts:
            source: NoiseSource = self.noises[noise.noise]
            excerpts[cache_key] = source.read(spec.split, noise.noise_offset, noise.length)
        excerpt: NDArray[np.float64] = excerpts[cache_key]
        measured_region: NDArray[np.float64] = excerpt[
            noise.fade_in : noise.length - noise.fade_out
        ]

        if spec.fixed_gain is not None:
            gain: float = spec.fixed_gain
        else:
            assert spec.snr_db is not None
            reference: float = rms(signal[speech_mask])
            noise_rms: float = rms(measured_region)
            if reference == 0 or noise_rms == 0:
                raise RenderError(f"{spec.out_path}: silent speech or noise, cannot set the level")
            gain = reference * 10 ** (-spec.snr_db / 20) / noise_rms

        scaled: NDArray[np.float64] = excerpt * gain
        if noise.fade_in:
            scaled[: noise.fade_in] *= raised_cosine(noise.fade_in)
        if noise.fade_out:
            scaled[noise.length - noise.fade_out :] *= raised_cosine(noise.fade_out)[::-1]
        signal[noise.start : noise.start + noise.length] = scaled
        return _Draft(spec=spec, signal=signal, labels=label, noise_gain=gain)

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
        if output.normalization == "peak":
            max_abs: float = float(np.max(np.abs(signal)))
            if max_abs == 0:
                audio: NDArray[np.int16] = signal.astype(np.int16)
            else:
                # Same arithmetic and truncation as librispeech_functions.scale_signal.
                target: float = FULL_SCALE * 10 ** (output.peak_dbfs / 20)
                audio = np.asarray(signal / max_abs * target, dtype=np.int16)
        else:
            rounded: NDArray[np.float64] = np.round(signal * guard)
            if rounded.size and (rounded.max() > INT16_MAX or rounded.min() < INT16_MIN):
                raise RenderError(f"{spec.out_path}: would clip in int16")
            audio = rounded.astype(np.int16)

        if len(audio) != spec.total_samples or len(draft.labels) != spec.total_samples:
            raise RenderError(f"{spec.out_path}: length differs from the plan")
        guarded: bool = guard < 1.0

        measured: float | None = None
        if spec.snr_db is not None:
            measured = self.measure_snr(spec, audio, draft.labels)
            if abs(measured - spec.snr_db) > LEVEL_TOLERANCE_DB:
                raise RenderError(
                    f"{spec.out_path}: measured {measured:.3f} dB, target {spec.snr_db} dB"
                )
        return Rendered(
            spec=spec,
            audio=audio,
            labels=draft.labels,
            noise_gain_db=to_db(draft.noise_gain),
            clip_gain_db=to_db(guard) if guarded else 0.0,
            measured_snr_db=measured,
            clip_triggered=draft.clip_triggered,
        )

    def measure_snr(
        self, spec: FileSpec, audio: NDArray[np.int16], labels: NDArray[np.int8]
    ) -> float:
        """Speech RMS (speech-only when configured) over the noise block's RMS without fades."""
        scenario: ScenarioConfig = self.config.scenario(spec.scenario)
        speech_mask: NDArray[np.bool_] = np.zeros(spec.total_samples, dtype=bool)
        for block in spec.blocks:
            if block.kind == "speech":
                speech_mask[block.start : block.start + block.length] = True
        if scenario.loudness.speech_only:
            speech_mask &= labels == 1
        noise: BlockSpec = spec.noise_block()
        region: NDArray[np.int16] = audio[
            noise.start + noise.fade_in : noise.start + noise.length - noise.fade_out
        ]
        return to_db(rms(audio[speech_mask]) / max(rms(region), 1e-12))
