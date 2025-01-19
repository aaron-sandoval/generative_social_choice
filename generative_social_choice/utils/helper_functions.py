from collections.abc import Sequence
import math
from pathlib import Path
import os
from datetime import datetime, timezone
import re


def get_base_dir_path() -> Path:
    """
    Returns local system path to the directory where the generative social choice package is.
    So, this is the directory where utils/, test/, etc are.
    """
    base_dir_name = "generative_social_choice"

    path = Path(os.path.abspath(os.path.dirname(__file__)))
    current_path_parts = list(path.parts)
    base_dir_idx = (
        len(current_path_parts) - current_path_parts[::-1].index(base_dir_name) - 1
    )

    base_dir_path = Path(*current_path_parts[: 1 + base_dir_idx])
    return base_dir_path


def get_time_string() -> str:
    now = datetime.now(timezone.utc)
    time_string = now.strftime("%Y-%m-%d-%H%M%S")

    return time_string

def sanitize_name(name: str) -> str:
    """
    Sanitize a name to be compatible as a Python function name.
    """
    if name[0] in ["0123456789"]:
        name = "_" + name
    name = name.replace('.', 'p')
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)  
    name = name.rstrip('_')  # Remove trailing underscores only
    return name


def geq_lib(
    a: float | int | Sequence[float | int],
    b: float | int | Sequence[float | int],
    rel_tol: float = 1e-9,
    abs_tol: float = 0.0,
) -> bool:
    """
    Returns True if a is greater than or close to b.
    Supports both individual numbers and sequences of numbers.
    """
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        for x, y in zip(a, b):
            if not math.isclose(x, y, rel_tol=rel_tol, abs_tol=abs_tol):
                return x > y
        return len(a) >= len(b)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol) or a >= b
    else:
        raise TypeError("Both arguments must be numbers or sequences of numbers")

def leq_lib(
    a: float | int | Sequence[float | int],
    b: float | int | Sequence[float | int],
    rel_tol: float = 1e-9,
    abs_tol: float = 0.0,
) -> bool:
    """
    Returns True if a is less than or close to b.
    Supports both individual numbers and sequences of numbers.
    """
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        for x, y in zip(a, b):
            if not math.isclose(x, y, rel_tol=rel_tol, abs_tol=abs_tol):
                return x < y
        return len(a) <= len(b)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol) or a <= b
    else:
        raise TypeError("Both arguments must be numbers or sequences of numbers")
