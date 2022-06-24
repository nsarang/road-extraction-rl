import os
from datetime import datetime


def create_auto_file_name(filename_prefix: str, ext: str, timestamp_format: str = "%Y%m%d_%H%M%S") -> str:
    timestamp = datetime.now().strftime(timestamp_format)
    filename = filename_prefix + timestamp + ext
    return filename


def check_path(path: str, auto_create: bool = True) -> None:
    if not path or os.path.exists(path):
        return

    if auto_create:
        os.makedirs(path)
    else:
        raise OSError(f"Path '{path}' not found.")
