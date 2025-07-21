# Generative Social Choice: early chatbot personalization experiment

This repo contains the code and data associated with an early pilot experiment on chatbot personalization from the project *Generative Social Choice* ([paper](https://arxiv.org/abs/2309.01291), [general audience report](https://procaccia.info/GenSoc_OpenAI_report.pdf)). This experiment was conducted in November 2023 as part of OpenAI's [Democratic Inputs to AI program](https://openai.com/index/democratic-inputs-to-ai-grant-program-update/). **We've since updated and improved our entire experimental pipeline, and conducted a follow-up experiment. If you want to build on our framework, we strongly recommend you use our new code and data** (public link forthcoming). This repo only contains the necessary code to replicate the early pilot experiment on chatbot personalization.

Authors of *Generative Social Choice*: [Sara Fish](https://sara-fish.github.io/), [Paul Gölz](https://paulgoelz.de/), [David Parkes](https://parkes.seas.harvard.edu/), [Ariel Procaccia](https://procaccia.info/), [Gili Rusak](https://gilirusak.github.io/), [Itai Shapira](https://ishapira1.github.io/), and [Manuel Wüthrich](https://scholar.google.de/citations?user=7EWrVYIAAAAJ&hl=en). 


# Setup instructions 

1. In the folder where this `README.md` file is located, call `pip install -e .`
2. Install dependencies: `pipenv install`
3. Create a file `OPENAI_API_KEY` in `utils/`, and write in it your (personal) API key. 

TODO Adjust instructions to use poetry instead of pipenv

# Overview of repo 

- `data/` has all cleaned and anonymized data associated with the experiments in the paper:
    - `chatbot_personalization_data.csv`: our cleaned and anonymized survey data, collected on Prolific. Also available [at the dedicated repo](https://github.com/generative-social-choice/chatbot_personalization_data)
    - `validate_disc_query_logs.csv`: logs from our discriminative query validation experiment (Figure 1, replicate with `paper_replication/validate_discriminative_query.py`)
    - `gen_query_eval/`: logs from our generative query evaluation experiment (Figure 2, replicate with `paper_replication/gen_query_eval.py`)
    - `user_summaries_generation.csv` and `user_summaries_generation_raw_output.csv`: the user summaries (and logs) used in our slate generation (replicate with `paper_replication/generate_summaries.py`)
    - `ratings_and_matching.csv`: assignments of validation users to statements (Figures 4-5, replicate with `paper_replication/compute_matching.py`)
- `paper_replication/` has scripts for replicating the experiments in the paper:
    - `validate_discriminative_query.py`: validating discriminative queries (Figure 1)
    - `gen_query_eval.py`: evaluating generative queries (Figure 2)
    - `generate_summaries.py`: generating user summaries 
    - `generate_slate.py`: generating slate
    - `compute_matching.py`: compute assignment of users to statements in slate (Figures 4-5)
- `plots/` has code for generating each of the plots in the paper, and the plots themselves
- `queries/` has implementation of the queries:
    - `query_chatbot_personalization.py` contains all of the chatbot personalization specific implementation 
    - `query_interface.py` describes the interface for agents and generators. Anything that implements this interface should automatically work with our slate generation code. 
- `slates/` has our implementation of the slate generation algorithm in `generate_slate_ensemble_greedy.py`
- `test/` has unit tests
- `utils/` has miscellaneous tools
    - `gpt_wrapper.py` contains code for making LLM calls 
    - `helper_functions.py` has `get_base_dir_path()` and `get_time_string()`
    - `dataframe_completion.py` contains code for df-completion style LLM calls, used for our summary generation and generative query.

# Paper Extension by Sandoval & Blandfort

## Running the pipeline

The pipeline uses the survey dataset from Fish et al. We assume that the summaries have been generated already (as included in the repo or explained below).

From the directory where this README is located, you can run the whole pipeline by calling `python generative_social_choice/scripts/full_pipeline.py --generation_model=4o`. This script has additional arguments in case you want to use other embeddings, fix a seed etc.

You can also run the different steps of the pipeline individually, using the scripts

- `generative_social_choice/scripts/generate_statements.py`, which creates a file with statements
- `generative_social_choice/scripts/rate_statements.py`, which creates a file with the utility matrix and another file which maps the statement IDs used in the utility matrix back to statements
- `generative_social_choice/scripts/compute_assignments.py`, which creates final assignments for each method and writes these assignments to JSON files

To run our pipeline with embeddings from Fish et al., first compute an embedding matrix using the script `generative_social_choice/scripts/compute_fish_embeddings.py`, then move the file with the embeddings matrix that is created by that script to a folder for the run results (`generative_social_choice/data/results/statements/[run_id]/`) and then call the pipeline using arguments `--embeddings=fish` and `--run_id=[run_id]`.

You can then use the jupyter notebooks in `generative_social_choice/plots/` to analyze the results.

Notes:

- Make sure that this repo is installed as package (`pip install -e .`) before running the scripts.
- You can use command line arguments to use other models or subsample the agents for debugging purposes.

You can also convert the results from Fish et al. into our result format (including calling DISC on all their statements) by calling `python generative_social_choice/scripts/compute_baseline_results.py --model=gpt-4o-mini`.


# Generating plots from paper 

Each figure in the paper can be generated using a dedicated notebook:
* Figure 1: `plots/fig1_disc_query_eval.ipynb`
* Figure 2: `plots/fig2_slate_composition.ipynb`
* Figure 3: N/A
* Figure 4: `plots/fig4_assigned_utilities_pie_chart.ipynb`
* Figure 5: `plots/fig5_assigned_utilities_histogram.ipynb`

# Testing instructions 

## Quick testing

To run unit tests with `gpt-4o-mini`, run the following command. 
```
python -m unittest -k fast -v 
``` 

## Slower testing for replication

To run unit tests using the exact LLMs used in the paper (for replication purposes), run the following command. This requires access to `gpt-4-base` and `gpt-4-32k-0613`.
```
python -m unittest -k replication -v
```

To run all unit tests, run the following command. This requires access to `gpt-4-base` and `gpt-4-32k-0613`.
```
python -m unittest -v
```

# Replicating paper instructions 

## Quick replication

The quickest and cheapest way to rerun our experiments is to use a more modern LLM such as `gpt-4o`. The below commands run the exact experiments from our paper, except `gpt-4o` is used in place of `gpt-4-base` and `gpt-4-32k-0613`. 

### Generate summaries of users
Generate summaries of all users:
```
python paper_replication/generate_summaries.py --model gpt-4o
```

Generate summary for a single user (for testing):
```
python paper_replication/generate_summaries.py --model gpt-4o --num_agents 1
```


### Empirical validation of discriminative query
To run the full experiment empirically validating the discriminative query (600 LLM calls):
```
python paper_replication/validate_discriminative_query.py --model gpt-4o
```
To empirically validate a single discriminative query (for testing):
```
python paper_replication/validate_discriminative_query.py --model gpt-4o --num_samples 1
```

### Empirical evaluation of generative query
To run the full experiment empirically evaluating the generative query:
```
python paper_replication/gen_query_eval.py --model gpt-4o
```
To evaluate a single ensemble round (for testing):
```
python paper_replication/gen_query_eval.py --model gpt-4o --num_rounds 1
```

### Generating slate
To generate a slate for all users:
```
python paper_replication/generate_slate.py --model gpt-4o
```
To generate a slate for only 10 users (for testing):
```
python paper_replication/generate_slate.py --model gpt-4o --num_agents 10
```

## "Exact" reproduction (subject to LLM stochasticity)

To "exactly" (subject to inherent LLM stochasticity) reproduce our experiments, run the below commands. These require access to `gpt-4-base` and `gpt-4-32k-0613`. These will write logs to `data/chatbot_personalization/demo_data/`. To test on smaller sample sizes, use the `--num_agents` and `--num_samples` arguments (usage demonstrated above). 


### Generate summaries of users
```
python paper_replication/generate_summaries.py --model default
```

### Empirical validation of discriminative query
```
python paper_replication/validate_discriminative_query.py --model default
```

### Empirical evaluation of generative query
```
python paper_replication/gen_query_eval.py --model default
```

### Generating slate
```
python paper_replication/generate_slate.py --model default
```

### Match validation users to slate statements 

This step uses Gurobi, but no LLM calls.

```
python paper_replication/compute_matching.py
```
