from typing import List, Tuple
from gurobipy import Model, GRB, quicksum
import pandas as pd
from generative_social_choice.utils.helper_functions import get_base_dir_path
import random


def optimize_monroe_matching(utilities: List[List[int]]) -> Tuple[int]:
    """Function to compute the optimal assignment of agents to statements using integer linear programming, to maximize
    the Monroe value.

    Args:
    utilities (List[List[int]]): n x m Utility-Matrix. `utilities[i][j]` indicate the utility of agent `i` for statement `j`.

    Returns:
    Tuple[int]: A tuple of length n, whose i’th entry is a number 0≤j<m indicating which statement agent i is assigned to.
    """

    n = len(utilities)
    m = len(utilities[0])
    assert (
        n % m == 0
    ), "The number of agents (rows) must be a multiple of the number of statements (columns)."

    k = n // m
    model = Model()
    model.Params.LogToConsole = 0

    # a two-dimensional list of binary variables, where variables[i][j] will indicate whether agent i is assigned to statement j.
    variables = [[model.addVar(vtype=GRB.BINARY) for _ in range(m)] for _ in range(n)]

    model.update()

    # objective function
    model.setObjective(
        quicksum(utilities[i][j] * variables[i][j] for i in range(n) for j in range(m)),
        GRB.MAXIMIZE,
    )

    # constraints: each agent is assigned to exactly one statement.
    for i in range(n):
        model.addConstr(quicksum(variables[i]) == 1)

    # constraints: each statement is assigned to exactly k agents.
    for j in range(m):
        model.addConstr(quicksum(variables[i][j] for i in range(n)) == k)

    model.optimize()
    assert model.status == GRB.OPTIMAL or model.status == GRB.SUBOPTIMAL

    assignments = [-1] * n
    for i in range(n):
        for j in range(m):
            if round(variables[i][j].X) == 1:
                assert assignments[i] == -1
                assignments[i] = j
    assert all(x != -1 for x in assignments)

    return tuple(assignments)


def compute_matching():
    df = pd.read_csv(get_base_dir_path() / "data/chatbot_personalization_data.csv")
    df = df[df["sample_type"] == "validation"]
    df = df[df["detailed_question_type"] == "rating statement"]

    pivot_df = df.pivot(index="user_id", columns="statement", values="choice_numeric")
    pivot_df = pivot_df.astype(int)

    # Round number of agents down so it's a multiple of the number of statements
    random.seed(0)
    num_all_agents = pivot_df.shape[0]
    num_statements = pivot_df.shape[1]
    num_sampled_agents = num_statements * (num_all_agents // num_statements)
    sampled_agents = random.sample(list(pivot_df.index), num_sampled_agents)
    pivot_df = pivot_df[pivot_df.index.isin(sampled_agents)]

    # Compute optimal Monroe matching
    utilities = pivot_df.values.astype(int).tolist()
    matching = optimize_monroe_matching(utilities=utilities)
    statements = list(pivot_df.columns)
    matching = [statements[i] for i in matching]

    # Compute utilities of agents under matching
    pivot_df["assignments"] = matching
    pivot_df["utility"] = pivot_df.apply(lambda row: row[row["assignments"]], axis=1)

    pivot_df["user_num"] = pivot_df.index.str.replace("validation", "").astype(int)
    pivot_df.sort_values("user_num", inplace=True)
    pivot_df.drop(columns=["user_num"], inplace=True)

    pivot_df.to_csv(get_base_dir_path() / "data/ratings_and_matching.csv")


if __name__ == "__main__":
    compute_matching()
