"""
Temporary pseudocode for brainstorming methods for generating a slate of statements
"""

from math import ceil


# Assume these functions are defined somewhere
Constituent = str
def k_means(N: list[Constituent], j: int) -> list[set[Constituent]]:
    pass

def GEN(S: list[Constituent], j: int) -> str:
    pass

def DISC(c: Constituent, statement: str) -> float:
    pass

def generate_slate(N: list[Constituent], u_min, k_max):
    """
    N: the set of survey respondents/constituents/agents
    n: number of survey respondents/constituents/agents
    u_i: The utility function mapping statements to their utility according to constituent i
    u_min: Minimum utility guaranteed for any (n/k_max)-cohesive group
    k: size of final slate
    k_min: Possible user-defined constraint on the minimum size of slate
    k_max: Possible user-defined constraint on the maximum size of slate
    k_means(N, j) -> list[set[Constituent]]: function which clusters constituents into j sets

    """
    # Calculate the minimum group size that must be represented
    rep_guarantee_group_size = ceil(N // k_max)
    
    # Initialize the set of constituents who are not yet represented
    unrepresented_constituents = set(N)
    n_unrepresented = len(N)
    
    # Start with a small cluster size
    cur_cluster_cardinality = 2
    
    # List to store the final statement assignments
    statement_assignments: list[tuple[str, list[Constituent]]] = []
    
    # Initial clustering of constituents, sorted by cluster size
    clusters: list[set[Constituent]] = sorted(k_means(unrepresented_constituents, cur_cluster_cardinality), key=lambda x: len(x))  # Sort by size of cluster
    
    # Continue until all constituents are represented or the maximum number of statements is reached
    while n_unrepresented > rep_guarantee_group_size and len(statement_assignments) < k_max:
        # If the smallest cluster is too small, increase the cluster size
        if len(clusters) < 1 or len(clusters[0]) < N / cur_cluster_cardinality:
            cur_cluster_cardinality += 1
            clusters.append(k_means(unrepresented_constituents, cur_cluster_cardinality))
            clusters.sort(key=lambda x: len(x))
        
        # Generate a statement for the smallest cluster
        statement_cand = GEN(clusters[0], len(clusters[0]))
        
        # Check if the statement meets the minimum utility for all constituents in the cluster
        if min([DISC(c, statement_cand) for c in clusters[0]]) >= u_min:
            # Assign the statement to the cluster
            statement_assignments.append((statement_cand, clusters[0]))
            unrepresented_constituents.difference_update(clusters[0])
            n_unrepresented -= len(clusters[0])
            
            # Remove the cluster from the list and its members from all other clusters
            clusters.pop(0)
            for cluster in clusters:
                cluster.difference_update(clusters[0])
        else: # This cluster was too big to be represented by a single statement, try for a smaller cluster on the next iteration
            clusters.pop(0)
    
    # Assign remaining unrepresented constituents to the statement that maximizes their utility
    for constituent in unrepresented_constituents:
        best_statement = argmax(statement_assignments, key=lambda x: DISC(constituent, x[0]))
        statement_assignments[best_statement][1].add(constituent)
    return statement_assignments
