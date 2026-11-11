# %% [markdown]
# # Stepwise-Monroe Voting Ablation
#
# Isolates the **voting rule** as the only variable. We take the greedy-Monroe rule used by
# `slate_generation.generate_slate_ensemble_greedy` -- which normally *interleaves* utility
# computation with slate selection and only ever queries the currently-unmatched agents -- and
# instead run it on the **complete, precomputed utility matrix** (every agent x every generated
# statement, all available from step 1). We compare it against the standard scoring pipeline
# (`ExactTotalUtilityMaximization`, as used by `scripts/compute_assignments.py`).
#
# Four methods are compared, in this cluster order:
#   1. `Monroe (constrained)` -- true Monroe: each voter kept in the equal-sized coalition
#      (~N/K voters) it was greedily matched into.
#   2. `Monroe (constrained, optimal)` -- same greedy slate, but the equal-sized-coalition
#      assignment is recomputed optimally via `optimize_monroe_matching` (ILP maximizing total
#      utility s.t. each slate member gets exactly N/K voters).
#   3. `Monroe (unconstrained)` -- stepwise Monroe selection, then each voter reassigned to its
#      favorite member of the final slate (equal-coalition constraint dropped).
#   4. `Exact` -- the baseline utility-maximizing voting rule.
#
# Both methods consume the *identical* loaded utility matrices; every statement-generation step is
# skipped. Output: a single scalar-metric confidence-interval plot, styled like the
# `scalar_confidence_intervals_plot` figures in `new_pipeline_results.ipynb`.
#
# Runnable **both** from the command line (`MPLBACKEND=Agg poetry run python <this file>`, which
# writes the figure to ./figures/) **and** cell-by-cell in the VS Code interactive window / Jupyter.

# %%
# Enable autoreload only inside an IPython shell; harmless (and skipped) under `python file.py`.
try:
    from IPython import get_ipython

    _ipy = get_ipython()
    if _ipy is not None:
        _ipy.run_line_magic("load_ext", "autoreload")
        _ipy.run_line_magic("autoreload", "2")
except Exception:
    pass

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

from generative_social_choice.utils.helper_functions import get_results_paths
from generative_social_choice.slates.voting_algorithms import (
    ExactTotalUtilityMaximization,
    StepwiseMonroe,
)
from generative_social_choice.slates.voting_utils import voter_utilities
from generative_social_choice.utils.postprocessing import (
    scalar_utility_metrics,
    bootstrap_df_rows,
    plot_scalar_clustered_confidence_intervals,
    save_figure,
)


# %% [markdown]
# ## Config & load the complete utility matrices (no statement generation)

# %%
LABELLING_MODEL = "4o-mini"
EMBEDDING_TYPE = "llm"
RUN_IDS = list(range(10))
SLATE_SIZE = 5
# Match compute_assignments default: drop the first 6 (seed) statement columns so all methods see
# only the generated candidates.
IGNORE_INITIAL_STATEMENTS = 6
CONFIDENCE_LEVEL = 0.95
N_BOOTSTRAP = 400
SEED = 1612

FIG_FORMAT = "pdf"
# Save next to this script so a headless/CLI run always produces an inspectable figure.
try:
    HERE = Path(__file__).parent
except NameError:  # running cell-by-cell with no __file__
    HERE = Path.cwd()
SAVE_DIR = HERE / "figures"

utility_matrix_files = {
    run_id: get_results_paths(
        labelling_model=LABELLING_MODEL,
        baseline=False,
        run_id=run_id,
        embedding_type=EMBEDDING_TYPE,
    )["utility_matrix_file"]
    for run_id in RUN_IDS
}

utility_matrices: dict[int, pd.DataFrame] = {}
for run_id, path in utility_matrix_files.items():
    df = pd.read_csv(path, index_col=0)
    if IGNORE_INITIAL_STATEMENTS:
        df = df.drop(columns=df.columns[:IGNORE_INITIAL_STATEMENTS])
    utility_matrices[run_id] = df

print(f"Loaded {len(utility_matrices)} utility matrices.")
_example = utility_matrices[RUN_IDS[0]]
print(f"Each matrix: {_example.shape[0]} voters x {_example.shape[1]} candidates "
      f"(candidates {list(_example.columns[:3])} ... {list(_example.columns[-1:])}).")


# %% [markdown]
# ## Run the three voting rules on the loaded matrices

# %%
# The plot label for each method is its algorithm's `display_name` -- the single source of truth.
# To rename a series, change the algorithm's `display_name`; nothing here needs touching.
ALGORITHMS = [
    StepwiseMonroe(final_assignment="greedy_equal"),
    StepwiseMonroe(final_assignment="optimal_equal"),
    StepwiseMonroe(final_assignment="free"),
    ExactTotalUtilityMaximization(),
]
METHOD_ORDER = [alg.display_name for alg in ALGORITHMS]

utility_series: dict[tuple[str, int], pd.Series] = {}
slates: dict[tuple[str, int], list[str]] = {}
assignment_series: dict[tuple[str, int], pd.Series] = {}
for alg in ALGORITHMS:
    method_name = alg.display_name
    for run_id in RUN_IDS:
        matrix = utility_matrices[run_id]
        slate, assignments = alg.vote(matrix.copy(), SLATE_SIZE)

        assert (assignments["candidate_id"] != "NULL_CAND").all(), (
            f"{method_name} run {run_id}: some voters left unassigned"
        )

        realized = voter_utilities(matrix, assignments["candidate_id"])
        utility_series[(method_name, run_id)] = realized
        slates[(method_name, run_id)] = slate
        assignment_series[(method_name, run_id)] = assignments["candidate_id"]

# Assemble a (voters x (method, run_id)) DataFrame -- the shape scalar_utility_metrics expects.
utilities_multidf = pd.DataFrame(utility_series)
utilities_multidf.columns = pd.MultiIndex.from_tuples(
    utilities_multidf.columns, names=["method", "run_id"]
)

# Guard against np.log(<=0) -> -inf/NaN downstream in scalar_utility_metrics (Mean Log / Gini).
assert np.isfinite(utilities_multidf.to_numpy()).all(), "Non-finite realized utilities"
assert (utilities_multidf.to_numpy() > 0).all(), (
    "Non-positive realized utility found; Mean Log / Gini metrics require positive utilities"
)

print(f"utilities_multidf: {utilities_multidf.shape[0]} voters x "
      f"{utilities_multidf.shape[1]} (method, run) columns")
display(utilities_multidf.head())


# %% [markdown]
# ## Scalar metrics, bootstrap CIs, and the comparison plot
#
# The plotted metrics are grouped onto three axes by their natural scale. Metric columns are
# selected **by name** (never positionally): commit `a148da6` inserted `"2*Mean Log"` at position 3
# of the default metric order, so any `.iloc`-based slicing (e.g. in `new_pipeline_results.ipynb`)
# now grabs the wrong columns -- name-based selection avoids that bug. (`"2*Mean Log"` itself is
# intentionally not plotted.)

# %%
scalar_metrics_per_run = scalar_utility_metrics(utilities_multidf)
scalar_cis = bootstrap_df_rows(
    scalar_metrics_per_run,
    confidence_level=CONFIDENCE_LEVEL,
    n_bootstrap=N_BOOTSTRAP,
    seed=SEED,
)
# bootstrap_df_rows groups via groupby (which sorts method names alphabetically); restore the
# intended cluster order: constrained -> optimal-constrained -> unconstrained -> Exact.
scalar_cis = scalar_cis.reindex(METHOD_ORDER, level=0)
display(scalar_cis)

primary_metrics = scalar_cis[["Mean", "Mean of\nBottom 50%", "Minimum"]]
log_metrics = scalar_cis[["Mean Log"]]
gini_metric = scalar_cis[["Gini"]]

fig = plot_scalar_clustered_confidence_intervals(
    primary_metrics,
    y_label="Agreement",
    legend_loc="lower left",
    fig_size=(8.5, 3.5),
    secondary_axis_df=log_metrics,
    secondary_y_label="log(Agreement)",
    tertiary_axis_df=gini_metric,
    tertiary_y_label="Gini",
    font_size=12,
)
display(fig)

SAVE_DIR.mkdir(parents=True, exist_ok=True)
save_figure(fig, SAVE_DIR / "stepwise_monroe_ablation", FIG_FORMAT)
print(f"Saved figure to {SAVE_DIR / ('stepwise_monroe_ablation.' + FIG_FORMAT)}")


# %% [markdown]
# ## Winning candidates and coalition sizes per algorithm
#
# For each voting algorithm, show its winning slate per run and how many voters were assigned to
# each slate member. Slate members are listed in selection order; a member with zero assigned
# voters (possible for the unconstrained rule) still appears, with a count of 0.

# %%
for alg in ALGORITHMS:
    method_name = alg.display_name
    rows = []
    for run_id in RUN_IDS:
        counts = assignment_series[(method_name, run_id)].value_counts()
        for candidate in slates[(method_name, run_id)]:
            rows.append(
                {
                    "run_id": run_id,
                    "candidate": candidate,
                    "num_voters": int(counts.get(candidate, 0)),
                }
            )
    coalition_sizes = pd.DataFrame(rows)
    print(f"\n=== {method_name}: winning candidates and voters assigned ===")
    display(coalition_sizes)

# %%
