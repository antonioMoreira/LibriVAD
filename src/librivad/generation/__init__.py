"""The scenario builder, importable from other projects.

    from librivad.generation import config_from_dict, run

    config = config_from_dict(raw_yaml_dict, base=Path("/where/relative/paths/start"))
    report = run(config, splits=["val"], progress=False)

Only the core dependencies are needed (`pip install librivad`); the paper scripts and the
VAD models need the `paper` and `vad` extras. Importing this package has no side effects:
the `.env` of this repository is read by the command line only, so pass `base` (or set
`PROJECT_ROOT`) for relative paths. The names below are loaded on first use, so
`python -m librivad.generation.build_scenarios` does not import that module twice.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from librivad.generation.build_scenarios import LimitsExceeded, run
    from librivad.generation.config import Config, ConfigError, config_from_dict, load_config
    from librivad.generation.corpus import ContractError

_EXPORTS: dict[str, str] = {
    "run": "librivad.generation.build_scenarios",
    "LimitsExceeded": "librivad.generation.build_scenarios",
    "Config": "librivad.generation.config",
    "ConfigError": "librivad.generation.config",
    "config_from_dict": "librivad.generation.config",
    "load_config": "librivad.generation.config",
    "ContractError": "librivad.generation.corpus",
}

__all__ = [
    "Config",
    "ConfigError",
    "ContractError",
    "LimitsExceeded",
    "config_from_dict",
    "load_config",
    "run",
]


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(_EXPORTS[name]), name)
