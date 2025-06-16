from collections.abc import Sequence
import math
from pathlib import Path
import os
from datetime import datetime, timezone
import re
from typing import Literal

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


def get_results_paths(labelling_model: str, embedding_type: Literal["llm", "seed_statement"], baseline: bool=False,
                      base_dir: Path | None = None, run_id: str | None = None) -> dict[str, Path]:
    """Get directories given a hierarchy of directories.
    
    The following structure is assumed:
    base_dir/
        baseline/[run_id]/
            4o_for_labelling/
                baseline_utility_matrix.csv
                baseline_utility_matrix_statements.csv
            4o-mini_for_labelling/
                baseline_utility_matrix.csv
                baseline_utility_matrix_statements.csv
        statements/[run_id]/
            generated_with_4o_using_llm_embeddings/
                4o_for_labelling/
                    utility_matrix.csv
                    utility_matrix_statements.csv
                4o-mini_for_labelling/
                    utility_matrix.csv
                    utility_matrix_statements.csv
            generated_with_4o_using_seed_statement_embeddings/
                4o_for_labelling/
                    utility_matrix.csv
                    utility_matrix_statements.csv
                4o-mini_for_labelling/
                    utility_matrix.csv
                    utility_matrix_statements.csv
    """
    if base_dir is None:
        base_dir = get_base_dir_path() / "data/results/"

    assert labelling_model in ["4o", "4o-mini"]
    assert embedding_type in ["seed_statement", "llm"]

    # Now figure out which directory to use
    selected_dir = base_dir / ("baseline" if baseline else "statements")
    if run_id is not None:
        selected_dir = selected_dir / str(run_id)
    main_dir = selected_dir[:]

    if baseline:
        selected_dir = selected_dir / f"{labelling_model}_for_labelling"
    else:
        selected_dir = selected_dir / f"generated_with_4o_using_{embedding_type}_embeddings" / f"{labelling_model}_for_labelling"

    utility_matrix_file = selected_dir / f"{'baseline_' if baseline else ''}utility_matrix.csv"
    statement_id_file = selected_dir / f"{'baseline_' if baseline else ''}utility_matrix_statements.csv"

    # Output
    if baseline:
        assignments = selected_dir / "baseline_assignments.json"
    else:
        assignments = selected_dir / "assignments/"

    return {
        "utility_matrix_file": utility_matrix_file,
        "statement_id_file": statement_id_file,
        "assignments": assignments,
        "base_dir": main_dir,
    }


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
