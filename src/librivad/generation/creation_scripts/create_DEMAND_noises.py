import glob
import os
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import scipy.io.wavfile as wav
import soundfile as sf
from dotenv import load_dotenv
from numpy.typing import NDArray

load_dotenv()

PROJECT_ROOT: Path = Path(os.environ["PROJECT_ROOT"])
DEMAND_dir: str = str(PROJECT_ROOT / os.environ["DEMAND_DIR"]) + "/"
noises_dir: str = str(PROJECT_ROOT / os.environ["NOISES_DIR"]) + "/"


def downsample_cafe() -> None:
    paths: list[str] = glob.glob(DEMAND_dir + "SCAFE/*.wav")

    print("Downsampling Cafe files to 16000 Hz")
    for path in paths:
        _file: str = path.split("/")[-1]

        sgn: NDArray[Any]
        sr: int | float
        sgn, sr = librosa.load(path, sr=16000)

        sf.write(path, sgn, int(sr))


def create_DEMAND_noises() -> None:
    downsample_cafe()

    Categories: list[str] = ["D", "N", "O", "P", "S", "T"]
    Categories_full: list[str] = [
        "Domestic_noise",
        "Nature_noise",
        "Office_noise",
        "Public_noise",
        "Street_noise",
        "Transport_noise",
    ]

    for j in range(len(Categories)):
        category: str = Categories[j]

        paths: list[str] = glob.glob(DEMAND_dir + f"{category}*/*.wav")
        paths.sort()

        train_paths: list[str] = []
        val_paths: list[str] = []
        test_paths: list[str] = []

        for i in range(len(paths)):
            path: str = paths[i]
            mod: int = i % 16

            if mod < 12:
                train_paths.append(path)
            elif mod < 14:
                val_paths.append(path)
            else:
                test_paths.append(path)

        sr: int = 16000
        sgn: NDArray[Any]
        train_sgn_list: list[NDArray[Any]] = []
        for path in train_paths:
            sgn, sr = sf.read(path)
            train_sgn_list.append(sgn)

        val_sgn_list: list[NDArray[Any]] = []
        for path in val_paths:
            sgn, sr = sf.read(path)
            val_sgn_list.append(sgn)

        test_sgn_list: list[NDArray[Any]] = []
        for path in test_paths:
            sgn, sr = sf.read(path)
            test_sgn_list.append(sgn)

        train_sgn_concat: NDArray[Any] = np.hstack(train_sgn_list)
        val_sgn_concat: NDArray[Any] = np.hstack(val_sgn_list)
        test_sgn_concat: NDArray[Any] = np.hstack(test_sgn_list)

        train_sgn: NDArray[np.int16] = np.asarray(
            train_sgn_concat / np.max(np.abs(train_sgn_concat)) * 32767, dtype=np.int16
        )
        val_sgn: NDArray[np.int16] = np.asarray(
            val_sgn_concat / np.max(np.abs(val_sgn_concat)) * 32767, dtype=np.int16
        )
        test_sgn: NDArray[np.int16] = np.asarray(
            test_sgn_concat / np.max(np.abs(test_sgn_concat)) * 32767, dtype=np.int16
        )

        os.makedirs(noises_dir + f"{Categories_full[j]}/", exist_ok=True)

        wav.write(
            noises_dir + f"{Categories_full[j]}/{Categories_full[j]}_train.wav", sr, train_sgn
        )
        wav.write(noises_dir + f"{Categories_full[j]}/{Categories_full[j]}_val.wav", sr, val_sgn)
        wav.write(noises_dir + f"{Categories_full[j]}/{Categories_full[j]}_test.wav", sr, test_sgn)
