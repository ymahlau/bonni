import datetime
import os
from pathlib import Path
import shutil


def change_to_timestamped_dir(
    file: Path | str | None = None,
    fixed_time_folder: str | None = None,
):
    """
    Creates a new directory with a timestamp-based path structure and changes
    the working directory to it, similar to Hydra's approach.

    Returns:
        str: Path to the newly created directory
    """
    # Get current timestamp
    now = datetime.datetime.now()

    # Format the date and time components
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H-%M-%S-%f")[
        :12
    ]  # Includes milliseconds (truncated to 3 digits)

    # Construct the path
    output_base = "outputs"
    date_dir = os.path.join(output_base, date_str)
    if fixed_time_folder is None:
        timestamp_dir = os.path.join(date_dir, time_str)
    else:
        timestamp_dir = os.path.join(date_dir, fixed_time_folder)

    # Create directories if they don't exist
    os.makedirs(timestamp_dir, exist_ok=False)

    # Copy file if provided
    if file is not None:
        file_path = Path(file)
        if file_path.exists():
            new_filename = file_path.stem + ".txt"
            destination = Path(timestamp_dir) / new_filename
            shutil.copy2(file_path, destination)
            print(f"Copied {file_path} to {destination}")
        else:
            print(f"Warning: File {file_path} does not exist and was not copied")

    # Change working directory
    os.chdir(timestamp_dir)

    print(f"Working directory changed to: {timestamp_dir}")

    return Path(timestamp_dir)
