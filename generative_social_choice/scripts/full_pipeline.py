import argparse

from generative_social_choice.scripts.generate_statements import run as generate_statements
from generative_social_choice.scripts.rate_statements import run as rate_statements
from generative_social_choice.scripts.compute_assignments import run as compute_assignments
from generative_social_choice.scripts.compute_assignments import VOTING_ALGORITHMS
from generative_social_choice.utils.helper_functions import get_results_paths


def run_pipeline(
    generation_model: str,
    rating_model: str,
    embedding_type: str,
    run_id: str | None = None,
    num_agents: int | None = None,
    num_clusters: int = 5,
    slate_size: int = 5,
    seed: int = 0,
    ignore_initial: bool = False,
    verbose: bool = True,
    steps_to_run: list[str] = ["generate_statements", "rate_statements", "compute_assignments"],
):
    """Run the full pipeline:
    1. Generate statements
    2. Rate statements
    3. Compute assignments
    """
    if "generate_statements" in steps_to_run:
        print("\n=== Step 1: Generating Statements ===")
        generate_statements(
            embedding_method=embedding_type,
        num_agents=num_agents,
        num_clusters=num_clusters,
        model=generation_model,
        seed=seed,
        run_id=run_id,
    )

    if "rate_statements" in steps_to_run:
        print("\n=== Step 2: Rating Statements ===")
        rate_statements(
            model=rating_model,
            num_agents=num_agents,
            run_id=run_id,
            generation_model=generation_model,
            verbose=verbose,
        )


    if "compute_assignments" in steps_to_run:
        print("\n=== Step 3: Computing Assignments ===")
        results_paths = get_results_paths(run_id=run_id, embedding_type=embedding_type, generation_model=generation_model, labelling_model=rating_model)
        # Run each voting algorithm
        for name, algo in VOTING_ALGORITHMS.items():
            print(f"\nRunning algorithm '{algo.name}' ...")
            compute_assignments(
                slate_size=slate_size,
                voting_algotirhm=algo,
                ignore_initial_statements=ignore_initial,
                verbose=verbose,
                utility_matrix_file=results_paths["utility_matrix_file"],
                statement_id_file=results_paths["statement_id_file"],
                assignment_file=results_paths["assignments"] / f"{name}.json",
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full pipeline for statement generation, rating, and assignment computation.")

    parser.add_argument(
        "--generation_model",
        type=str,
        default="4o-mini",
        help="Model to use for statement generation. Default is 4o-mini.",
    )

    parser.add_argument(
        "--rating_model",
        type=str,
        default="4o-mini",
        help="Model to use for rating statements. Default is 4o-mini.",
    )

    parser.add_argument(
        "--embedding_type",
        type=str,
        choices=["llm", "seed_statement", "fish"],
        default="llm",
        help="Type of embeddings to use. Default is llm.",
    )

    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Optional run ID to use for organizing results in a specific directory.",
    )

    parser.add_argument(
        "--num_agents",
        type=int,
        default=None,
        help="Number of agents to consider. If not provided, all agents will be used.",
    )

    parser.add_argument(
        "--num_clusters",
        type=int,
        default=5,
        help="Number of clusters to use in partitioning methods. Default is 5.",
    )

    parser.add_argument(
        "--slate_size",
        type=int,
        default=5,
        help="Number of statements to include in the slate. Default is 5.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for random number generator. Default is 0.",
    )

    parser.add_argument(
        "--ignore_initial",
        action="store_true",
        help="If set, the first 6 statements in the utility matrix will be ignored.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="If set, print detailed progress information.",
    )

    args = parser.parse_args()

    run_pipeline(
        generation_model=args.generation_model,
        rating_model=args.rating_model,
        embedding_type=args.embedding_type,
        run_id=args.run_id,
        num_agents=args.num_agents,
        num_clusters=args.num_clusters,
        slate_size=args.slate_size,
        seed=args.seed,
        ignore_initial=args.ignore_initial,
        verbose=args.verbose,
    ) 