import pandas as pd
import numpy as np

from generative_social_choice.slates.voting_algorithm_axioms import (
    IndividualParetoAxiom,
    HappiestParetoAxiom,
    CoverageAxiom,
    MinimumAndTotalUtilityParetoAxiom,
)
from generative_social_choice.slates.voting_algorithms import (
    GreedyTotalUtilityMaximization,
    GeometricTransformation,
)
from generative_social_choice.test.utilities_for_testing import _rated_vote_cases

if __name__=="__main__":
    #cases = [case for case in _rated_vote_cases if case.name == "Ex B.3"]
    cases = _rated_vote_cases[:]

    for test_case in cases:
        results = []
        all_assignments = []
        for i in range(50):
            #print(f"\nIteration {i+1}")

            # Compute solution
            #algorithm = GreedyTotalUtilityMaximization(utility_transform=GeometricTransformation(p=1.5))
            algorithm = GreedyTotalUtilityMaximization()
            slate, assignments = algorithm.vote(
                test_case.rated_votes.copy(),  # Voting algorithms might append columns
                test_case.slate_size,
            )
            all_assignments.append(assignments["candidate_id"].to_numpy())

            result = [Axiom().evaluate_assignment(rated_votes=test_case.rated_votes, slate_size=test_case.slate_size, assignments=assignments)
                        for Axiom in  [IndividualParetoAxiom, HappiestParetoAxiom, CoverageAxiom, MinimumAndTotalUtilityParetoAxiom]]
            results.append(result)
        
        results = np.array(results)
        all_assignments = np.array(all_assignments)

        # Check if there is any position for which assignments changed
        for i in range(all_assignments.shape[1]):
            num_assignments = len(set(list(all_assignments[:, i])))
            if num_assignments>1:
                print("Variation found for test case {test_case.name} member {i}: {num_assignments} different assignments")

        # Check if there are any differences across runs
        # (If exactly 0 or 1 it means there was no variation!)
        avg_res = np.mean(results, axis=0)
        print(f"Average results for test case {test_case.name}:", avg_res)