# This script was developed in order to generate the city noise signal.
# Due to the large size of the WHAM! noise dataset, it was decided to reduce
# the number of files that contribute to the city noise used in this project

import os
import sys
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.io.wavfile as wav
from dotenv import load_dotenv
from numpy.typing import NDArray
from pydub import AudioSegment

from librivad import librispeech_functions as lf

load_dotenv()

scripts_path: str = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(scripts_path)

PROJECT_ROOT: Path = Path(os.getenv("PROJECT_ROOT", str(Path(__file__).resolve().parents[3])))
NOISES_DIR: Path = PROJECT_ROOT / os.getenv("NOISES_DIR", "files/Noises")


def categorize(df: pd.DataFrame) -> list[NDArray[Any]]:
    """
    Utilizing WHAM! metadata to split them into sub-dataframes based on the
    recording location

    Input:
      - df     : The dataframe containing the metadata

    Output:
      - subsets: A list of numpy arrays each containing the file paths of a
                 unique location
    """

    subsets: list[NDArray[Any]] = [
        np.array(group["utterance_id"]) for _, group in df.groupby("Location ID")
    ]

    return subsets


def extract_info(file: str) -> tuple[str, NDArray[np.int16], int]:
    """
    Function for reading, scaling and returning the audio file, along with its
    length and title. This is used for accelerating the generation process.

    Input:
      - file          : The path to the audio file (same for output)

    Outputs:
      - file
      - scaled_signal : The scaled audio waveform (to 16-bit)
      - sgn_len       : The length of the signal

    """

    _sr: int
    sgn: NDArray[Any]
    _sr, sgn = wav.read(file)

    scaled_signal: NDArray[np.int16] = lf.scale_signal(sgn)

    sgn_len: int = len(sgn)

    return file, scaled_signal, sgn_len


def process_subsets(
    df: pd.DataFrame, wham_folder: str
) -> list[dict[str, tuple[NDArray[np.int16], int]]]:
    """
    Utilizing the categorize function, the extract_info function and threading,
    to accelerate the processing of the files.

    Inputs:
      - df               : The dataframe containing the metadata
      - wham_folder      : The path to the WHAM noise folder

    Output:
      - processed_subsets: A list of dictionaries containing the audio files and
                           other relevant information regarding the noise
                           generation process
    """

    subsets: list[NDArray[Any]] = categorize(df)
    processed_subsets: list[dict[str, tuple[NDArray[np.int16], int]]] = []
    m: int = len(subsets)

    for j, subset in enumerate(subsets, start=1):
        n: int = len(subset)
        processed_subset: dict[str, tuple[NDArray[np.int16], int]] = {}

        # Utilising asynchronous execution with threads for the files
        with ThreadPoolExecutor() as executor:
            futures: dict[Future[tuple[str, NDArray[np.int16], int]], str] = {
                executor.submit(extract_info, os.path.join(wham_folder, str(file))): str(file)
                for file in subset
            }

            for i, future in enumerate(as_completed(futures)):
                try:
                    file: str
                    signal: NDArray[np.int16]
                    length: int
                    file, signal, length = future.result()

                    processed_subset[file] = (signal, length)

                    # Calculate and print progress
                    percent: float = float(np.round(100 * (i + 1) / n, 2))
                    print(f"{percent}%", end="\r")

                except Exception as exc:
                    print(f"Error processing file: {futures[future]}, Error: {exc}")

        print(f"[{j}/{m}]")
        processed_subsets.append(dict(sorted(processed_subset.items())))

    return processed_subsets


def generate_city_noise(df: pd.DataFrame, wham_folder: str, split: str, duration: int) -> None:
    """
    Generates the city noise signal by concatenating audio files until the desired
    duration is reached.

    Inputs:
      - df          : The dataframe containing the metadata
      - wham_folder : The path to the WHAM noise folder
      - split       : The dataset split (train/val/test)
      - duration    : The desired duration of the city noise in seconds

    Saves the generated city noise in the appropriate directory.
    """

    processed_subsets: list[dict[str, tuple[NDArray[np.int16], int]]] = process_subsets(
        df, wham_folder
    )
    subsets: list[list[tuple[NDArray[np.int16], int]]] = [
        list(d.values()) for d in processed_subsets
    ]
    m: int = len(processed_subsets)
    len_sum: int = 0
    i: int = 0
    j: int = 0
    city_noise: list[NDArray[np.int16]] = []
    while len_sum < duration * 16000:
        signal: NDArray[np.int16] = subsets[i][j][0]
        len_sum += subsets[i][j][1]
        i += 1

        if i == m:
            j += 1
            i %= m

        city_noise.append(signal)

    city_noise_flat: NDArray[Any] = np.vstack(city_noise)
    city_noise_flat = city_noise_flat[: duration * 16000]

    save_dir: Path = NOISES_DIR / "City_noise" / f"City_noise_{split}.wav"
    save_dir.parent.mkdir(parents=True, exist_ok=True)

    wav.write(save_dir, 16000, city_noise_flat)


if __name__ == "__main__":
    wham_folder: str = sys.argv[1]
    metadata: str = os.path.join(wham_folder, "metadata")
    noise_meta_tr: str = os.path.join(metadata, "noise_meta_tr.csv")
    noise_meta_cv: str = os.path.join(metadata, "noise_meta_cv.csv")
    noise_meta_tt: str = os.path.join(metadata, "noise_meta_tt.csv")

    df_tr: pd.DataFrame = pd.read_csv(noise_meta_tr)
    df_cv: pd.DataFrame = pd.read_csv(noise_meta_cv)
    df_tt: pd.DataFrame = pd.read_csv(noise_meta_tt)

    generate_city_noise(df_cv, os.path.join(wham_folder, "cv"), "val", 1800)
    generate_city_noise(df_tt, os.path.join(wham_folder, "tt"), "test", 1800)
    generate_city_noise(df_tr, os.path.join(wham_folder, "tr"), "train", int(3 * 3600))

    # Converting from stereo to mono
    for split in ["val", "test", "train"]:
        loc_dir: Path = NOISES_DIR / "City_noise" / f"City_noise_{split}.wav"

        sound: Any = AudioSegment.from_wav(str(loc_dir))
        sound = sound.set_channels(1)
        sound.export(str(loc_dir), format="wav")
