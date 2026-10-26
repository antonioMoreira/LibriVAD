# Description: Composes audio clips out of speech segments, pauses and up to two noise layers,
# each one with its own level, together with the matching VAD label.
#
# Levels are RMS values in dB relative to full scale (dBFS, with full scale being +-1.0):
#  - the speech level is measured only over the speech-active samples (label == 1);
#  - the level of a noise layer is measured over the whole clip.
# The SNR of a noise layer is therefore `speech_db - noise_db`.

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

SAMPLE_RATE: int = 16000
MAX_DURATION_SECONDS: float = 30.0
MAX_NOISE_LAYERS: int = 2
MODES: tuple[str, ...] = ("speech", "noise_only")
NORMALIZATIONS: tuple[str, ...] = ("limit", "none")

# Samples scaled this close to full scale are considered not to clip.
PEAK_CEILING: float = 0.999


@dataclass(frozen=True)
class NoiseLayer:
    """A noise and the level (dBFS) it is mixed at."""

    name: str
    db: float


@dataclass(frozen=True)
class ComposeConfig:
    """
    Every parameter of the composition, with its default. Ranges are (min, max) and are
    sampled uniformly for each clip.

    - mode             : "speech" (speech + noise) or "noise_only".
    - noise_only_ratio : Fraction of the clips that carry no speech (only in "speech" mode).
    - duration         : Duration range of a clip in seconds (up to 30).
    - speech_db        : Speech level range (dBFS).
    - noises           : Fixed noise layers. When empty, layers are sampled with the
                         parameters below instead.
    - noise_pool       : Noises to sample from. None means all the available ones.
    - noise_count      : How many sampled noise layers a clip has (min, max).
    - noise_db         : Level range (dBFS) of each sampled noise layer.
    - pause_range      : Length range (s) of a pause, during which only noise is heard.
    - pause_prob       : Chance of a pause before each segment (the start and between segments).
    - num_clips        : Clips to create for the train, val and test splits.
    - seed             : Seed of the random generator.
    - normalize        : "limit" only attenuates a clip that would clip (levels keep their
                         relation), "none" hard clips.
    """

    mode: str = "speech"
    noise_only_ratio: float = 0.0
    duration: tuple[float, float] = (10.0, 30.0)
    speech_db: tuple[float, float] = (-25.0, -25.0)
    noises: tuple[NoiseLayer, ...] = ()
    noise_pool: tuple[str, ...] | None = None
    noise_count: tuple[int, int] = (1, 1)
    noise_db: tuple[float, float] = (-45.0, -30.0)
    pause_range: tuple[float, float] = (0.5, 3.0)
    pause_prob: float = 0.5
    num_clips: tuple[int, int, int] = (1000, 100, 100)
    seed: int = 0
    normalize: str = "limit"

    def validate(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}.")
        if self.normalize not in NORMALIZATIONS:
            raise ValueError(f"normalize must be one of {NORMALIZATIONS}.")
        if not 0.0 <= self.noise_only_ratio <= 1.0:
            raise ValueError("noise_only_ratio must be between 0 and 1.")
        if not 0.0 <= self.pause_prob <= 1.0:
            raise ValueError("pause_prob must be between 0 and 1.")
        for name, (low, high) in (
            ("duration", self.duration),
            ("speech_db", self.speech_db),
            ("noise_db", self.noise_db),
            ("pause_range", self.pause_range),
            ("noise_count", self.noise_count),
        ):
            if low > high:
                raise ValueError(f"{name}: min ({low}) is greater than max ({high}).")
        if not self.duration[0] > 0.0 or self.duration[1] > MAX_DURATION_SECONDS:
            raise ValueError(f"duration must be within (0, {MAX_DURATION_SECONDS}] seconds.")
        if self.pause_range[0] < 0:
            raise ValueError("pause_range must not be negative.")
        if self.noise_count[0] < 0 or self.noise_count[1] > MAX_NOISE_LAYERS:
            raise ValueError(f"noise_count must be within [0, {MAX_NOISE_LAYERS}].")
        if len(self.noises) > MAX_NOISE_LAYERS:
            raise ValueError(f"At most {MAX_NOISE_LAYERS} noise layers are supported.")
        if any(n < 0 for n in self.num_clips):
            raise ValueError("num_clips must not be negative.")

    def replace(self, **changes: Any) -> ComposeConfig:
        return dataclasses.replace(self, **changes)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> ComposeConfig:
        """Builds a config from plain values, e.g. a parsed TOML file."""
        known: set[str] = {f.name for f in dataclasses.fields(cls)}
        unknown: set[str] = set(values) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")

        converted: dict[str, Any] = {}
        for key, value in values.items():
            if value is None:  # an empty or null entry (YAML) keeps the default
                continue
            if key == "noises":
                converted[key] = tuple(NoiseLayer(str(n["name"]), float(n["db"])) for n in value)
            elif key == "noise_pool" or isinstance(value, list):
                converted[key] = tuple(value)
            else:
                converted[key] = value

        config: ComposeConfig = cls(**converted)
        config.validate()

        return config


@dataclass
class Segment:
    """A speech source with its per-sample label (1 = speech)."""

    source: str
    audio: NDArray[Any]
    label: NDArray[Any]


@dataclass
class Clip:
    audio: NDArray[np.int16]
    label: NDArray[np.int16]
    recipe: dict[str, Any] = field(default_factory=dict)


def to_float(signal: NDArray[Any]) -> NDArray[np.float32]:
    """Converts integer PCM to floats in [-1, 1); floats are returned unchanged."""
    if np.issubdtype(signal.dtype, np.integer):
        return np.asarray(signal, dtype=np.float32) / 32768.0

    return np.asarray(signal, dtype=np.float32)


def rms(signal: NDArray[Any]) -> float:
    return float(np.sqrt(np.mean(np.square(signal, dtype=np.float64))))


def db_to_amplitude(db: float) -> float:
    return float(10 ** (db / 20))


def amplitude_to_db(amplitude: float) -> float:
    return float(20 * np.log10(amplitude))


def fit_label(label: NDArray[Any], n_samples: int) -> NDArray[np.int16]:
    """Trims or zero-pads a label to `n_samples` (forced alignments may end a bit off)."""
    label = np.asarray(label, dtype=np.int16)
    if len(label) >= n_samples:
        return label[:n_samples]

    return np.pad(label, (0, n_samples - len(label)))


def take_wrapped(signal: NDArray[Any], offset: int, n_samples: int) -> NDArray[np.float32]:
    """Takes `n_samples` from `offset`, wrapping around the end of the signal."""
    if len(signal) == 0:
        raise ValueError("Cannot take samples from an empty signal.")

    parts: list[NDArray[np.float32]] = []
    position: int = offset % len(signal)
    remaining: int = n_samples
    while remaining > 0:
        chunk: NDArray[Any] = signal[position : position + remaining]
        parts.append(to_float(np.asarray(chunk)))
        remaining -= len(chunk)
        position = 0

    return np.concatenate(parts)


def choose_noise_layers(
    rng: np.random.Generator,
    config: ComposeConfig,
    noise_pool: Sequence[str],
    at_least_one: bool,
) -> list[NoiseLayer]:
    """The fixed layers of the config or, if there are none, a random draw from the pool."""
    if config.noises:
        return list(config.noises)

    low, high = config.noise_count
    if at_least_one:
        low = max(low, 1)
        high = max(high, low)
    count: int = int(rng.integers(low, high + 1))
    if count == 0:
        return []
    if not noise_pool:
        raise ValueError("There are no noises to sample from.")

    count = min(count, len(noise_pool))
    names: NDArray[Any] = rng.choice(np.array(noise_pool), size=count, replace=False)

    return [NoiseLayer(str(name), float(rng.uniform(*config.noise_db))) for name in names]


def build_speech_track(
    rng: np.random.Generator,
    config: ComposeConfig,
    next_segment: Callable[[np.random.Generator], Segment],
    n_samples: int,
    sample_rate: int,
) -> tuple[
    NDArray[np.float32],
    NDArray[np.int16],
    float,
    list[dict[str, Any]],
    list[list[float]],
    list[list[int]],
]:
    """
    Fills `n_samples` with speech segments and pauses and cuts the last piece at the end.
    Returns the track, its label, the speech level, the segments, the pauses in seconds and
    the pauses in samples (exact).
    """
    speech_db: float = float(rng.uniform(*config.speech_db))
    target_amplitude: float = db_to_amplitude(speech_db)

    track: NDArray[np.float32] = np.zeros(n_samples, dtype=np.float32)
    label: NDArray[np.int16] = np.zeros(n_samples, dtype=np.int16)
    segments: list[dict[str, Any]] = []
    pauses: list[list[float]] = []
    pauses_samples: list[list[int]] = []

    position: int = 0
    attempts: int = 0
    while position < n_samples:
        attempts += 1
        if attempts > 10_000:
            raise RuntimeError("Could not fill the clip with speech segments.")

        if rng.random() < config.pause_prob:
            length: int = int(rng.uniform(*config.pause_range) * sample_rate)
            end: int = min(position + length, n_samples)
            if end > position:
                pauses.append([round(position / sample_rate, 3), round(end / sample_rate, 3)])
                pauses_samples.append([position, end])
            position = end
            if position >= n_samples:
                break

        segment: Segment = next_segment(rng)
        audio: NDArray[np.float32] = to_float(segment.audio)
        seg_label: NDArray[np.int16] = fit_label(segment.label, len(audio))

        active: NDArray[np.float32] = audio[seg_label == 1]
        if active.size == 0 or rms(active) == 0.0:
            continue  # no usable speech in this source

        audio = audio * np.float32(target_amplitude / rms(active))
        end = min(position + len(audio), n_samples)
        used: int = end - position
        track[position:end] = audio[:used]
        label[position:end] = seg_label[:used]
        segments.append(
            {
                "source": segment.source,
                "start": round(position / sample_rate, 3),
                "end": round(end / sample_rate, 3),
                "start_sample": position,
                "end_sample": end,
                "truncated": used < len(audio),
            }
        )
        position = end

    # Each segment was levelled before being cut to the clip, so the speech of the clip as a
    # whole is brought to the requested level here.
    active_track: NDArray[np.float32] = track[label == 1]
    if active_track.size and rms(active_track) > 0.0:
        track *= np.float32(target_amplitude / rms(active_track))

    return track, label, speech_db, segments, pauses, pauses_samples


def compose_clip(
    rng: np.random.Generator,
    config: ComposeConfig,
    next_segment: Callable[[np.random.Generator], Segment],
    get_noise: Callable[[str], NDArray[Any]],
    noise_pool: Sequence[str],
    sample_rate: int = SAMPLE_RATE,
) -> Clip:
    """
    Composes one clip. All the randomness comes from `rng`, so a clip is reproducible from
    its generator.
    """
    duration: float = float(rng.uniform(*config.duration))
    n_samples: int = max(1, int(duration * sample_rate))

    noise_only: bool = config.mode == "noise_only" or (
        config.noise_only_ratio > 0 and rng.random() < config.noise_only_ratio
    )
    layers: list[NoiseLayer] = choose_noise_layers(rng, config, noise_pool, noise_only)

    mix: NDArray[np.float32]
    label: NDArray[np.int16]
    speech_db: float | None
    segments: list[dict[str, Any]]
    pauses: list[list[float]]
    pauses_samples: list[list[int]]
    if noise_only:
        mix = np.zeros(n_samples, dtype=np.float32)
        label = np.zeros(n_samples, dtype=np.int16)
        speech_db, segments, pauses, pauses_samples = None, [], [], []
    else:
        mix, label, speech_db, segments, pauses, pauses_samples = build_speech_track(
            rng, config, next_segment, n_samples, sample_rate
        )

    noises_info: list[dict[str, Any]] = []
    for layer in layers:
        source: NDArray[Any] = get_noise(layer.name)
        offset: int = int(rng.integers(0, len(source)))
        noise: NDArray[np.float32] = take_wrapped(source, offset, n_samples)
        noise_rms: float = rms(noise)
        if noise_rms == 0.0:
            raise ValueError(f"The noise {layer.name} is silent at the chosen position.")

        mix += noise * np.float32(db_to_amplitude(layer.db) / noise_rms)
        noises_info.append(
            {
                "name": layer.name,
                "db": round(layer.db, 3),
                "snr_db": None if speech_db is None else round(speech_db - layer.db, 3),
                "offset": round(offset / sample_rate, 3),
                "offset_samples": offset,
            }
        )

    # Attenuating every layer by the same amount keeps their levels relative to each other.
    peak: float = float(np.max(np.abs(mix))) if n_samples else 0.0
    gain_db: float = 0.0
    clipped: bool = False
    if peak > PEAK_CEILING:
        if config.normalize == "limit":
            mix *= np.float32(PEAK_CEILING / peak)
            gain_db = amplitude_to_db(PEAK_CEILING / peak)
        else:
            clipped = True
    audio: NDArray[np.int16] = np.round(np.clip(mix, -1.0, 1.0) * 32767).astype(np.int16)

    recipe: dict[str, Any] = {
        "duration": round(n_samples / sample_rate, 3),
        "mode": "noise_only" if noise_only else "speech",
        "speech_db": None if speech_db is None else round(speech_db, 3),
        "gain_db": round(gain_db, 3),
        "clipped": clipped,
        "noises": noises_info,
        "segments": segments,
        "pauses": pauses,
        "pauses_samples": pauses_samples,
    }

    return Clip(audio=audio, label=label, recipe=recipe)
