from pathlib import Path
import os
from datetime import datetime


def get_base_dir_path():
    """
    Returns local system path to the directory where the generative social choice package is.
    So, this is the directory where utils/, test/, etc are.
    """
    base_dir_name = "chatbot_personalization"

    path = Path(os.path.abspath(os.path.dirname(__file__)))
    current_path_parts = list(path.parts)
    base_dir_idx = (
        len(current_path_parts) - current_path_parts[::-1].index(base_dir_name) - 1
    )

    base_dir_path = Path(*current_path_parts[: 1 + base_dir_idx])
    return base_dir_path


def get_time_string():
    now = datetime.now()
    time_string = now.strftime("%Y-%m-%d-%H:%M:%S")

    return time_string
