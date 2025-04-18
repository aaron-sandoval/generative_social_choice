from dataclasses import dataclass
from typing import Optional

import re
import pandas as pd

from generative_social_choice.slates.voting_algorithms import RatedVoteCase


# The voting cases to test, please add more as needed
_rated_vote_cases: tuple[RatedVoteCase, ...] = (
    RatedVoteCase(
        name="Simple 1",
        rated_votes=[[1, 2, 3], [1, 2, 3], [1, 2, 3]],
        slate_size=1,
    ),
    RatedVoteCase(
        name="Simple 2",
        rated_votes=[[4, 2, 3], [4, 2, 3], [4, 2, 3]],
        slate_size=1,
    ),
    RatedVoteCase(
        name="Simple 3",
        rated_votes=[[1, 1] , [1.1, 1], [1, 1]],
        slate_size=1,
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
            [5, 0, .1],
            [5, 0, .1],
            [5, 0, .1],
            [5, 0, .1],
            [5, 0, .1],
            [5, 0, .1],
            [5, 0, .1],
            [5, 0, .1],
            [5, 0, .1],
            [5, 5, .1],
            [5, 5, .1],
            [0, 5, .1],
            [0, 5, .1],
            [0, 5, .1],
            [0, 5, .1],
            [0, 5, .1],
            [0, 5, .1],
            [0, 5, .1],
            [0, 5, .1],
            [0, 0, .1],
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
            [0,    1, 0, 0],
            [0,    1, 0, 0],
            [1,    0, 1, 0],
            [0,    0, 1, 0],
            [0,    0, 1, 0],
            [1,    0, 0, 1],
            [0,    0, 0, 1],
            [0,    0, 0, 1],
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