import os
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import soundfile as sf
from dotenv import load_dotenv
from numpy.typing import NDArray
from tqdm import tqdm

load_dotenv()


def flac_to_wav(flac_path: Path, flac_root_dir: Path, wav_root_dir: Path) -> None:
    """
    Converts a single .flac file to .wav, preserving the directory structure.

    Args:
        flac_path (Path): Path to the source .flac file.
        flac_root_dir (Path): Root directory of the flac dataset (for path calculations).
        wav_root_dir (Path): Root directory where the .wav file will be saved.
    """
    try:
        # Calculate the destination path for the .wav file.
        # This keeps the same subfolder structure as the source.
        relative_path: Path = flac_path.relative_to(flac_root_dir)
        wav_path: Path = wav_root_dir / relative_path.with_suffix(".wav")

        # Create parent directories for the output file if they don't exist.
        wav_path.parent.mkdir(parents=True, exist_ok=True)

        # Use 'with' statements for safe and automatic file handling.
        # This guarantees files are closed even if errors occur.
        with sf.SoundFile(str(flac_path), "r") as f_in:
            data: NDArray[Any] = f_in.read()
            samplerate: int = int(f_in.samplerate)
            channels: int = f_in.channels
            subtype: str = f_in.subtype

            with sf.SoundFile(
                str(wav_path), mode="w", samplerate=samplerate, channels=channels, subtype=subtype
            ) as f_out:
                f_out.write(data)

    except Exception as e:
        # Report any errors for a specific file without stopping the whole process.
        print(f"Error processing {flac_path}: {e}")


def process_files_in_parallel(
    flac_files: Sequence[Path], flac_root_dir: Path, wav_root_dir: Path
) -> None:
    """
    Uses a thread pool to convert a list of .flac files in parallel.

    Args:
        flac_files (list[Path]): A list of file paths to convert.
        flac_root_dir (Path): The root directory of the flac dataset.
        wav_root_dir (Path): The root directory for wav output.
    """
    # ThreadPoolExecutor is ideal for I/O-bound tasks like reading/writing files.
    with ThreadPoolExecutor() as executor:
        # Submit all conversion tasks to the thread pool.
        futures: list[Future[None]] = [
            executor.submit(flac_to_wav, path, flac_root_dir, wav_root_dir) for path in flac_files
        ]

        # Display a progress bar that updates as tasks are completed.
        for future in tqdm(as_completed(futures), total=len(futures), desc="Converting files"):
            try:
                # Retrieve the result to raise any exceptions from the thread.
                future.result()
            except Exception as exc:
                print(f"A conversion task generated an exception: {exc}")


if __name__ == "__main__":
    # Get the project root directory
    project_root: Path = Path(os.environ["PROJECT_ROOT"])

    flac_root_dir: Path = project_root / os.environ["LIBRISPEECH_FLAC_DIR"]
    wav_root_dir: Path = project_root / os.environ["LIBRISPEECH_WAV_DIR"]

    # List the dataset splits to be processed.
    splits: list[str] = ["train-clean-100", "test-clean"]

    for split in splits:
        print(f"\nConverting LibriSpeech {split} from flac to wav format")

        # Recursively find all .flac files within the current split's directory.
        split_dir: Path = flac_root_dir / split
        flac_files: list[Path] = sorted(list(split_dir.rglob("*.flac")))

        if not flac_files:
            print(f"Warning: No .flac files found in {split_dir}")
            continue

        # Run the parallel conversion process for the found files.
        process_files_in_parallel(flac_files, flac_root_dir, wav_root_dir)
