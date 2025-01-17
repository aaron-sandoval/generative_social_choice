from dataclasses import dataclass
from typing import Optional

import re
import pandas as pd

@dataclass
class RatedVoteCase:
    """
    A voting case with rated votes and sets of possible results which satisfy various properties.

    # Arguments
    - `rated_votes: pd.DataFrame | list[list[int | float]]`: Utility of each voter (rows) for each candidate (columns)
      - If passed as a nested list, it's converted to a DataFrame with columns named `s1`, `s2`, etc.
    - `slate_size: int`: The number of candidates to be selected
    - `pareto_efficient_slates: Optional[Sequence[list[int]]] = None`: Slates that are Pareto efficient on the egalitarian-utilitarian trade-off parameter.
      - Egalitarian objective: Maximize the minimum utility among all individual voters
      - Utilitarian objective: Maximize the total utility among all individual voters
    - `non_extremal_pareto_efficient_slates: Optional[Sequence[list[int]]] = None`: Slates that are non-extremal Pareto efficient on the egalitarian-utilitarian trade-off parameter.
        - Subset of `pareto_efficient_slates` which don't make arbitrarily large egalitarian-utilitarian sacrifices in either direction.
        - Ex: For Example Alg2.1, s1 is Pareto efficient, but not non-extremal Pareto efficient because it makes an arbitrarily large egalitarian sacrifice for an incremental utilitarian gain.
    - `expected_assignments: Optional[pd.DataFrame] = None`: An expected assignment of voters to candidates with the following columns:
        - `candidate_id`: The candidate to which the voter is assigned
        - Other columns not guaranteed to always be present, used for functional testing only. They should always be checked in the unit tests
    """
    rated_votes: pd.DataFrame | list[list[int | float]]
    slate_size: int
    pareto_efficient_slates: Optional[set[frozenset[str]]] = None
    non_extremal_pareto_efficient_slates: Optional[set[frozenset[str]]] = None
    expected_assignments: Optional[pd.DataFrame] = None
    name: Optional[str] = None

    def __post_init__(self):
        if isinstance(self.rated_votes, list):
            self.rated_votes = pd.DataFrame(self.rated_votes, columns=[f"s{i}" for i in range(1, len(self.rated_votes[0]) + 1)])

        if self.name is None:
            cols_str = "_".join(str(col) + "_" + "_".join(str(x).replace(".", "p") for x in self.rated_votes[col]) 
                              for col in self.rated_votes.columns)
            self.name = f"k_{self.slate_size}_{cols_str}"
        elif self.name is not None:
            # Format name to be compatible as a Python function name
            self.name = self.name.replace('.', 'p')
            self.name = re.sub(r'[^a-zA-Z0-9_]', '_', self.name)
            self.name = re.sub(r'^[^a-zA-Z_]+', '', self.name)  # Remove leading non-letters
            # self.name = re.sub(r'_+', '_', self.name)  # Collapse multiple underscores
            # self.name = self.name.strip('_')  # Remove trailing underscores


# The voting cases to test, please add more as needed
_rated_vote_cases: tuple[RatedVoteCase, ...] = (
    RatedVoteCase(
        rated_votes=[[1, 2, 3], [1, 2, 3], [1, 2, 3]],
        slate_size=1,
        pareto_efficient_slates=[["s3"]],
        non_extremal_pareto_efficient_slates=[["s3"]],
        # expected_assignments=pd.DataFrame(["s3"]*3, columns=["candidate_id"])
    ),
    RatedVoteCase(
        rated_votes=[[4, 2, 3], [4, 2, 3], [4, 2, 3]],
        slate_size=1,
        pareto_efficient_slates=[["s1"]],
        non_extremal_pareto_efficient_slates=[["s1"]],
        # expected_assignments=pd.DataFrame(["s1"]*3, columns=["candidate_id"])
    ),
    RatedVoteCase(
        rated_votes=[[1, 1] , [1.1, 1], [1, 1]],
        slate_size=1,
        pareto_efficient_slates=[["s1"]],
        non_extremal_pareto_efficient_slates=[["s1"]],
        # expected_assignments=pd.DataFrame(["s1"]*3, columns=["candidate_id"])
    ),
    RatedVoteCase(
        name="Ex 1.1",
        rated_votes=[
            [3, 2, 0, 0],
            [0, 2, 0, 0],
            [0, 2, 0, 0],
            [0, 0, 3, 2],
            [0, 0, 0, 2],
            [0, 0, 0, 2],
        ],
        slate_size=2,
    ),
    RatedVoteCase(
        name="Ex 1.1 modified",
        rated_votes=[
            [6, 2, 0, 0],
            [0, 2, 0, 0],
            [0, 2, 0, 0],
            [0, 0, 6, 2],
            [0, 0, 0, 2],
            [0, 0, 0, 2],
        ],
        slate_size=2,
    ),
    RatedVoteCase(
        name="Ex 1.2",
        rated_votes=[
            [1.01, 1, 0],
            [1.01, 0, 1],
            [0, 5, 0],
            [0, 0, 5],
        ],
        slate_size=2,
    ),
    RatedVoteCase(
        name="Ex A.1",
        rated_votes=[
            [2, 0, 1, 1],
            [2, 2, 1, 0],
            [0, 2, 1, 0],
            [0, 0, 0, 2],
        ],
        slate_size=2,
    ),
    RatedVoteCase(
        name="Ex 1.3",
        rated_votes=[
            [2, 0, 0, 0],
            [2, 2, 1, 0],
            [0, 2, 1, 1],
            [0, 0, 1, 2],
        ],
        slate_size=2,
    ),
    RatedVoteCase(
        name="Ex 2.1",
        rated_votes=[
            [2, 0, 0],
            [2, 2, 1],
            [0, 2, 0],
            [0, 0, 1],
        ],
        slate_size=2,
    ),
    RatedVoteCase(
        name="Ex 2.2",
        rated_votes=[
            [5, 0, 0, 1],
            [0, 5, 0, 1],
            [0, 5, 2, 1],
            [0, 0, 1, 1],
        ],
        slate_size=2,
    ),
    RatedVoteCase(
        name="Ex 3.1",
        rated_votes=[
            [2, 0, 1, 0, 0, 0],
            [1, 2, 0, 0, 0, 0],
            [0, 1, 2, 0, 0, 0],
            [0, 0, 0, 2, 0, 1],
            [0, 0, 0, 1, 2, 0],
            [0, 0, 0, 0, 1, 2],
        ],
        slate_size=3,
    ),
    RatedVoteCase(
        name="Ex 4.1",
        rated_votes=[
            [2, 0, 0, 1],
            [0, 5, 0, 1],
            [0, 5, 2, 1],
            [0, 0, 1, 1],
        ],
        slate_size=2,
    ),
    RatedVoteCase(
        # Bad egalitarian tradeoff
        name="Ex 4.2",
        rated_votes=[
            [5, 0, 1],
            [5, 0, 1],
            [5, 0, 1],
            [5, 0, 1],
            [5, 0, 1],
            [5, 0, 1],
            [5, 0, 1],
            [5, 0, 1],
            [5, 0, 1],
            [5, 5, 1],
            [5, 5, 1],
            [0, 5, 1],
            [0, 5, 1],
            [0, 5, 1],
            [0, 5, 1],
            [0, 5, 1],
            [0, 5, 1],
            [0, 5, 1],
            [0, 5, 1],
            [0, 0, 1],
        ],
        slate_size=2,
    ),
    RatedVoteCase(
        name="Ex 4.3",
        rated_votes=[
            [5, 0, 0, 0, 1],
            [5, 5, 0, 0, 1],
            [0, 5, 0, 0, 1],
            [0, 0, 5, 0, 1],
            [0, 0, 5, 5, 1],
            [0, 0, 0, 5, 1],
        ],
        slate_size=3,
    ),
    RatedVoteCase(
        name="Ex 4.4",
        rated_votes=[
            [3, 2, 0, 0, 0, 1],
            [0, 2, 0, 0, 0, 0],
            [0, 2, 0, 0, 0, 1],
            [0, 0, 3, 2, 0, 0],
            [0, 0, 0, 2, 3, 0],
            [0, 0, 0, 2, 0, 1],
        ],
        slate_size=2,
    ),
    RatedVoteCase(
        name="Ex B.1",
        rated_votes=[
            [2, 0, 0],
            [2, 0, 2],
            [0, 2, 0],
            [0, 2, 1],
        ],
        slate_size=2,
    ),
    RatedVoteCase(
        name="Ex B.2",
        rated_votes=[
            [9, 0, 0, 1],
            [9, 0, 1, 0],
            [0, 9, 0, 1],
            [0, 9, 1, 0],
        ],
        slate_size=2,
    ),
    RatedVoteCase(
        name="Ex B.3",
        rated_votes=[
            [3, 0, 2],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 0],
        ],
        slate_size=2,
    ),
    RatedVoteCase(
        name="Ex C.1",
        rated_votes=[
            [1, 0, 3],
            [1, 0, 1],
            [0, 2, 0],
            [0, 2, 0],
        ],
        slate_size=2,
    ),
    RatedVoteCase(
        name="Ex C.2",
        rated_votes=[
            [3, 0, 0, 2],
            [3, 0, 2, 0],
            [0, 1, 2, 0],
            [0, 1, 0, 2],
        ],
        slate_size=2,
    ),
    RatedVoteCase(
        name="Ex D.1",
        rated_votes=[
            [4, 0, 0, 0, 0],
            [4, 3, 0, 0, 0],
            [4, 3, 2, 0, 0],
            [0, 3, 2, 1, 0],
            [0, 0, 2, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
        ],
        slate_size=3,
    ),
    RatedVoteCase(
        name="Ex Alg1.3",
        rated_votes=[
            [1, 1],
            [1, 1],
            [1, 4],
        ],
        slate_size=1,
    ),
    RatedVoteCase(
        # Bad egalitarian tradeoff
        name="Ex Alg1.4",
        rated_votes=[
            [1.01, 1],
            [1.01, 1],
            [1.01, 4],
        ],
        slate_size=1,
    ),
    RatedVoteCase(
        name="Ex Alg1.5",
        rated_votes=[
            [1, 1],
            [1, 1],
            [1, 2],
        ],
        slate_size=1,
    ),
    RatedVoteCase(
        # Bad utilitarian tradeoff
        name="Ex Alg2.1",
        rated_votes=[
            [.01, 1],
            [.01, 1],
            [3, 1],
        ],
        slate_size=1,
    ),
    RatedVoteCase(
        name="Ex Alg A.1",
        rated_votes=[
            [1.01, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [1, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
        ],
        slate_size=3,
    ),
        RatedVoteCase(
        name="Ex Alg A.2",
        rated_votes=[
            [1.01, 1, 0, 0, 0],
            [0, 1, 0, 0, 1.01],
            [0, 1, 0, 0, 0],
            [1, 0, 1, 0, 0],
            [0, 0, 1, 0, 1],
            [0, 0, 1, 0, 0],
            [1, 0, 0, 1, 0],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 0],
        ],
        slate_size=3,
    ),
)

rated_vote_cases: dict[str, RatedVoteCase] = {case.name: case for case in _rated_vote_cases}