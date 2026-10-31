"""Shared fixtures for the scenario builder tests."""

from pathlib import Path
from typing import Any

import pytest
from generation_helpers import skylar_raw_config, write_clean_corpus, write_noises


@pytest.fixture
def skylar_setup(tmp_path: Path) -> dict[str, Any]:
    clean: Path = write_clean_corpus(tmp_path / "SkylarDataset")
    noises: dict[str, str] = write_noises(tmp_path / "noises")
    output: Path = tmp_path / "SkylarDataset" / "noisy"
    return {
        "clean": clean,
        "noises": noises,
        "output": output,
        "raw": skylar_raw_config(clean, noises, output),
        "tmp": tmp_path,
    }
