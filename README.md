# GenerativeCC: Chamberlin–Courant Elections with Generative Social Choice

Code and data for the paper *GenerativeCC: Chamberlin–Courant Elections with Generative Social Choice*.

## Abstract

Large language models (LLMs) present new opportunities for virtual democracy systems to faithfully condense and represent populous and diverse constituencies.
However, prior systems have limitations such as losing diversity by selecting a single statement, only testing on very small groups, or enforcing excessive constraints that limit representativeness.
We build GenerativeCC, an LLM-based system that compresses many free-form participant opinions into a small weighted slate of representative statements.
The pipeline (i) summarizes and embeds individual opinions, (ii) generates candidate representative statements, (iii) estimates participant agreement with statements using virtual voters, and (iv) selects a slate using a multiwinner range voting algorithm, where each voter is assigned to one slate member, and each weight is proportional to the quantity of assigned voters.
On a dataset of 100 opinions, GenerativeCC outperforms the most closely related method from previous literature across all egalitarian and utilitarian social welfare functions.
Performance is improved mostly because more voters evaluate more candidates, enabling broader optimization.
While the pipeline runs in exponential time, we demonstrate that multiple polynomial-time alternatives achieve similar performance.
GenerativeCC is a practical starting point for virtual democracy systems that represent diverse opinions.

## Relation to prior work

This repo builds on the code of *Generative Social Choice* by Fish et al. ([paper](https://arxiv.org/abs/2309.01291)). Their chatbot personalization survey data is our dataset, and their slate generation pipeline is our baseline.


# Setup instructions

Requirements: Python 3.12 and [Poetry](https://python-poetry.org/).

1. In the folder where this `README.md` file is located, run `poetry install`. This installs the dependencies and the repo itself as a package (in editable mode).
2. Run all commands below inside this environment (`poetry run python ...`, or `poetry shell`). For the notebooks, select the Poetry environment as the Jupyter kernel (`ipykernel` is included).
3. Only for steps that call the OpenAI API (generating statements, rating, re-running the baseline, the model-drift checks): create a file `generative_social_choice/utils/OPENAI_API_KEY` containing your API key, or set the environment variable `OPENAI_API_KEY`. An optional file `generative_social_choice/utils/OPENAI_ORGANIZATION` sets the organization. Both files are gitignored.

All figures of the paper can be regenerated from the data in this repo **without an API key** (see [Minimal replication](#minimal-replication)).


# Overview of repo

All paths are relative to `generative_social_choice/`.

- `data/`: inputs and results of all experiments in the paper (see [How the data was produced](#how-the-data-was-produced)).
    - `chatbot_personalization_data.csv`: the cleaned and anonymized survey data of Fish et al., also available [at their dedicated repo](https://github.com/generative-social-choice/chatbot_personalization_data).
    - `user_summaries_generation.csv`: their summaries of each participant's survey responses.
    - `results/statements/`: the GenerativeCC runs.
    - `demo_data/generate_slate_results_*/`: the baseline runs.
    - `model_drift/`: scripts and LLM outputs of the confounding-variable analysis.
    - `voting_algorithm_evals/`: logs of the voting algorithm axiom tests.
- `scripts/`: the GenerativeCC pipeline and its steps, and the statement ablation.
- `paper_replication/`: the baseline pipeline of Fish et al. (`generate_slate.py`) and the generation of the summaries.
- `statements/`, `ratings/`, `slates/`: statement generation, rating (discriminative queries and utility matrices), and voting algorithms.
- `queries/`: the LLM queries for chatbot personalization.
- `plots/`: the notebooks that produce the paper's figures.
- `test/`: unit tests.
- `utils/`: LLM calls (`gpt_wrapper.py`), paths (`helper_functions.py`), and the plotting and analysis functions used by the notebooks (`postprocessing.py`).

`paper/figures/` (at the repo root) has the figures as included in the paper.


# Running the pipeline

The pipeline uses the survey dataset and the summaries of Fish et al., both included in `data/`.

From the directory where this README is located, you can run the whole pipeline by calling `python generative_social_choice/scripts/full_pipeline.py --generation_model=4o`. This script has additional arguments in case you want to use other embeddings, fix a seed etc.

You can also run the different steps of the pipeline individually, using the scripts

- `generative_social_choice/scripts/generate_statements.py`, which creates a file with statements
- `generative_social_choice/scripts/rate_statements.py`, which creates a file with the utility matrix and another file which maps the statement IDs used in the utility matrix back to statements
- `generative_social_choice/scripts/compute_assignments.py`, which creates final assignments for each method and writes these assignments to JSON files

To run our pipeline with embeddings from Fish et al., first compute an embedding matrix using the script `generative_social_choice/scripts/compute_fish_embeddings.py`, then move the file with the embeddings matrix that is created by that script to a folder for the run results (`generative_social_choice/data/results/statements/[run_id]/`) and then call the pipeline using arguments `--embedding_type=fish` and `--run_id=[run_id]`.

You can use command line arguments to use other models or subsample the agents for debugging purposes.


## Statement Ablation

The new pipeline generates more statements by default than the original one. To compute assignments with a subset of statements, you can use the `run_statement_ablation.py` script.

Command used for results in the paper (it makes no LLM calls: it subsamples the existing statements and reuses their ratings):
```
python -m generative_social_choice.scripts.run_statement_ablation \
  --run_ids 0,1,2,3,4,5,6,7,8,9 \
  --generation_model 4o \
  --embedding_type llm \
  --labelling_model 4o-mini \
  --ablation_name llm_subsample_3 \
  --llm_subsample_per_condition 3 \
  --random_seed 0 \
  --ignore_initial
```
Its output (`data/results/statements/[run_id]/generated_with_4o_using_llm_embeddings/ablations/llm_subsample_3/`) is not committed, so run this command before regenerating Figure 8.


# Traceability of paper results

Paths are relative to `generative_social_choice/`. Notebooks are in `plots/`; they save their figures relative to the notebook's directory, i.e., to `plots/figures/` unless noted otherwise. Cells are identified by their content:

- `new_pipeline_results.ipynb`: **(A)** the first `collect_results_and_plot` call (series "Ours" and "Baseline"); **(B)** the next cell, which saves `scalar_confidence_intervals`; **(C)** the second `collect_results_and_plot` call (four series with "LLM Embs." / "Their Embs."); **(D)** the next cell, which saves `scalar_confidence_intervals_all_stacked`; **(E)** the cell defining `ABLATION_NAME`.
- Cells (A) and (C) both save `sorted_utility_CIs.pdf`, so rename it after each.

| Paper figure | File in `paper/figures/` | Produced by | Data |
|---|---|---|---|
| Fig. 1 (system overview) | `Opinion_agg_functionality*.pdf` | Diagram drawn by hand | – |
| Fig. 2 (SWFs, GenerativeCC vs. baseline) | `scalar_confidence_intervals.pdf` | `new_pipeline_results.ipynb`, cells (A) and (B) | `data/results/statements/{0..9}/` (GenerativeCC); `data/demo_data/generate_slate_results_baseline_persistent_pool/` (baseline) |
| Fig. 3 (sorted utilities, bottom 50%) | `sorted_utility_CIs_main.pdf` | `new_pipeline_results.ipynb`, cell (A); saved as `sorted_utility_CIs.pdf`, renamed | same as Fig. 2 |
| Fig. 4 (voting algorithms on GenerativeCC profiles) | `voting_alg_scalar_metrics_v8.png` | `compare_voting_algs_on_chatbot_data.ipynb`, output of the slate-size-5 plotting cell (`create_all_plots_from_utilities(slate5_...)`); the notebook displays it but does not save it, so it was saved from the output | `data/results/statements/{0..9}/`; baseline series as in Fig. 2 |
| Fig. 5 (voting algorithms on baseline profiles) | `voting_alg_scalar_metrics_fish_candidates.pdf` | `compare_voting_algs_on_chatbot_data.ipynb`, last cells (`FIG5_PATH`) | `data/demo_data/generate_slate_results_baseline_persistent_pool/`, including the ratings added by `scripts/complete_baseline_ratings.py` (`completion_ratings.jsonl` in each run directory) |
| Fig. 6 (embedding ablation, SWFs) | `scalar_confidence_intervals_all_stacked.pdf` | `new_pipeline_results.ipynb`, cells (C) and (D) | `data/results/statements/{0..9}/` (LLM embeddings), `data/results/statements/fish_{0..9}/` (embeddings of Fish et al.); `data/demo_data/generate_slate_results_openai_embeddings/` (baseline, LLM embeddings), `data/demo_data/generate_slate_results_baseline/` (baseline, their embeddings) |
| Fig. 7 (embedding ablation, sorted utilities) | `sorted_utility_CIs_all.pdf` | `new_pipeline_results.ipynb`, cell (C); saved as `sorted_utility_CIs.pdf`, renamed | same as Fig. 6 |
| Fig. 8 (statement count ablation) | `statement_ablation_comparison.pdf` | `new_pipeline_results.ipynb`, cell (E) (needs `results` from cell (C)); saved to `plots/` | ablation output of `scripts/run_statement_ablation.py` (see [Statement Ablation](#statement-ablation)); GenerativeCC and baseline as in Fig. 2 |

Other results:

- **Model drift and line endings (confounding-variable appendix, its tables and numbers):** `data/model_drift/measure_model_drift.py` (write-up in `data/model_drift/RESULT_model_drift.md`), `data/model_drift/check_fish_embeddings_line_endings.py`, and `data/model_drift/line_endings_generation/{run_calls,analyze}.py`. The LLM outputs they collected are committed next to them (`queries/*.jsonl`, `line_endings_generation/calls.jsonl`, `results.json`), and the scripts only send calls whose outputs are missing; `measure_model_drift.py --analyze_only` reruns its analysis from the committed outputs without new LLM calls (embeddings are read from the committed cache).
- **Contributions of generators (statement-origin appendix):** counted from the slates in `data/results/statements/{0..9}/generated_with_4o_using_llm_embeddings/4o-mini_for_labelling/assignments/exact.json` and the `generation_method` column of the run's `statement_generation_raw_output.csv`.
- **Axiom results and counterexamples (voting-axioms appendix):** proved in the paper. The axiom checks are implemented in `slates/voting_algorithm_axioms.py` and run by `test/test_voting_algorithms.py`, which logs to `data/voting_algorithm_evals/`; `plots/analyze_voting_algorithm_axiom_results.ipynb` analyzes these logs.

## How the data was produced

- **GenerativeCC runs** (`data/results/statements/{0..9}` and `fish_{0..9}`): `scripts/full_pipeline.py` (see [Running the pipeline](#running-the-pipeline)), with `--embedding_type=fish` and the embeddings of `scripts/compute_fish_embeddings.py` for the `fish_*` runs. Assignments can be recomputed for all runs, without LLM calls, with `scripts/generate_all_assignments.py`.
- **Baseline runs** (`data/demo_data/generate_slate_results_*`): `paper_replication/generate_slate.py`, the pipeline of Fish et al. `--embedding_type=llm` (default; nearest neighbors found by discriminative queries, as in Fish et al.) writes to `generate_slate_results_baseline`, `--embedding_type=seed_statement` (precomputed OpenAI embeddings, `data/demo_data/baseline_embeddings.json`) to `generate_slate_results_openai_embeddings`, and `--candidates_persist` (keep earlier rounds' candidates, as described by Fish et al.) adds the suffix `_persistent_pool`. The paper's main baseline is `--embedding_type=llm --candidates_persist`. The ratings needed to complete their ballot profiles (Fig. 5) come from `scripts/complete_baseline_ratings.py`. The loaders identify baseline runs by their position among the sorted run directories, so don't add other directories next to them.
- **Inputs:** `data/chatbot_personalization_data.csv` (survey data of Fish et al.) and `data/user_summaries_generation.csv` (their summaries of it, from `paper_replication/generate_summaries.py`).

## Minimal replication

All of the following uses only committed data and makes no LLM calls.

1. `poetry install` (see [Setup instructions](#setup-instructions)).
2. Run the statement ablation command in [Statement Ablation](#statement-ablation) (only needed for Fig. 8).
3. Run `generative_social_choice/plots/new_pipeline_results.ipynb` from top to bottom: Figs. 2, 3, 6, 7 and 8. (Its last two cells, which plot a single run, refer to a series name that no longer exists and fail; they don't produce paper figures.)
4. Run `generative_social_choice/plots/compare_voting_algs_on_chatbot_data.ipynb` from top to bottom: Figs. 4 and 5. This runs every voting algorithm on every ballot profile, including exact integer programs, so it takes longer than the other notebook.
5. Offline unit tests: `python -m unittest generative_social_choice.test.test_voting_utils generative_social_choice.test.test_optimize_monroe_matching generative_social_choice.test.test_utilities_for_testing -v`.


# Testing instructions

The offline unit tests are listed in [Minimal replication](#minimal-replication). To also run the tests that make LLM calls with `gpt-4o-mini` (requires an API key), run
```
python -m unittest -k fast -v
```
`test_voting_algorithms` writes a log of the axiom tests to `data/voting_algorithm_evals/`.
