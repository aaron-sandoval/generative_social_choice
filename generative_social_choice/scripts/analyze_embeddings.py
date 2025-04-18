# %%
from generative_social_choice.statements.partitioning import PrecomputedPartition
from generative_social_choice.utils.helper_functions import get_base_dir_path
from generative_social_choice.statements.statement_generation import get_simple_agents
import random
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

num_clusters = 5
partitioning_file=get_base_dir_path() / f"data/demo_data/kmeans_partitioning_openai_small_nosummary_{num_clusters}.json"

print("Reading partitioning from existing file ...")
partitioning = PrecomputedPartition(filepath=partitioning_file)

# Get all agents
print("Loading agents...")
agents = get_simple_agents()

# Get cluster assignments for all agents
print("Getting cluster assignments...")
assignments = partitioning.assign(agents=agents)
agent_to_cluster = {agent.id: cluster_id for agent, cluster_id in zip(agents, assignments)}

# Get initial statements from the survey
print("\nGetting initial statements...")
initial_statements = set()
for agent in agents:
    for statement in agent.survey_responses["statement"].to_list():
        if statement == statement and statement not in initial_statements:  # Check for NaN
            initial_statements.add(statement)
initial_statements = sorted(list(initial_statements))

# Create rating vectors for each agent
print("\nCreating rating vectors...")
rating_map = {
    "not at all": 1,
    "poorly": 2,
    "somewhat": 3,
    "mostly": 4,
    "perfectly": 5
}

agent_vectors = {}
for agent in agents:
    vector = []
    for statement in initial_statements:
        rating = agent.survey_responses[agent.survey_responses["statement"] == statement]["choice"].iloc[0]
        if rating == rating:  # Check for NaN
            vector.append(rating_map.get(rating, float('nan')))
        else:
            vector.append(float('nan'))
    agent_vectors[agent.id] = vector

# Convert to numpy array for cosine similarity computation
agent_ids = list(agent_vectors.keys())
vectors = np.array([agent_vectors[agent_id] for agent_id in agent_ids])

# Compute cosine similarities
print("\nComputing cosine similarities...")
similarities = cosine_similarity(vectors)

# Create similarity matrix
similarity_df = pd.DataFrame(similarities, index=agent_ids, columns=agent_ids)

# Compute average similarities within and between clusters
print("\nComputing average similarities within and between clusters...")
cluster_similarities = np.zeros((num_clusters, num_clusters))
cluster_counts = np.zeros((num_clusters, num_clusters))

for i, agent1_id in enumerate(agent_ids):
    cluster1 = agent_to_cluster[agent1_id]
    for j, agent2_id in enumerate(agent_ids):
        if i < j:  # Only consider each pair once
            cluster2 = agent_to_cluster[agent2_id]
            similarity = similarity_df.loc[agent1_id, agent2_id]
            cluster_similarities[cluster1, cluster2] += similarity
            cluster_similarities[cluster2, cluster1] += similarity
            cluster_counts[cluster1, cluster2] += 1
            cluster_counts[cluster2, cluster1] += 1

# Compute averages
cluster_avg_similarities = cluster_similarities / cluster_counts

# Create a nice table
print("\nAverage cosine similarities between clusters:")
similarity_table = pd.DataFrame(
    cluster_avg_similarities,
    index=[f"Cluster {i+1}" for i in range(num_clusters)],
    columns=[f"Cluster {i+1}" for i in range(num_clusters)]
)
print(similarity_table.round(3))

# Analysis of clustering results:
# 1. Cosine Similarity Analysis:
#    - High similarity values (all above 0.87) suggest users' rating patterns are quite similar overall
#    - Highest within-cluster similarity in Cluster 2 (0.958), indicating most cohesive group
#    - Lowest within-cluster similarity in Cluster 5 (0.877), suggesting more diversity
#    - High between-cluster similarities (0.897-0.939) indicate clusters aren't very distinct
#
# 2. Statement Analysis by Cluster:
#    - Cluster 1: Strongly supports opt-out (mean 4.76) and factual responses (mean 4.60)
#                 Strongly opposes hyper-personalization (mean 1.92)
#    - Cluster 2: Generally moderate views, highest support for opt-out (mean 4.07)
#    - Cluster 3: More skeptical of personalization, lowest support for most statements
#    - Cluster 4: Most opposed to complete avoidance (mean 2.74)
#                 Most supportive of hyper-personalization (mean 2.79)
#    - Cluster 5: Strong support for opt-out (mean 4.25) and factual responses (mean 3.55)
#
# 3. Sample Summaries Analysis:
#    - Cluster 1: Focus on privacy and control, strong emphasis on user autonomy
#    - Cluster 2: Balanced view, considering both benefits and risks
#    - Cluster 3: More skeptical of personalization, emphasizing potential risks
#    - Cluster 4: More open to personalization but with clear boundaries
#    - Cluster 5: Strong focus on privacy and ethical considerations
#
# Overall, while clusters aren't extremely distinct (high between-cluster similarities),
# they capture different attitudes towards:
# 1. Privacy and data control
# 2. Level of personalization desired
# 3. Trust in AI systems
# 4. Balance between convenience and privacy

# Analyze rating distributions for each cluster and statement
print("\nAnalyzing rating distributions by cluster...")
for statement_idx, statement in enumerate(initial_statements):
    print(f"\nStatement {statement_idx + 1}: {statement}")
    print("-" * 80)
    
    # Group ratings by cluster
    cluster_ratings = defaultdict(list)
    for agent in agents:
        if agent.id in agent_to_cluster:
            cluster_id = agent_to_cluster[agent.id]
            # Get the rating for this statement from the agent's survey responses
            rating = agent.survey_responses[agent.survey_responses["statement"] == statement]["choice"].iloc[0]
            if rating == rating:  # Check for NaN
                # Convert rating to numeric value
                numeric_rating = rating_map.get(rating, float('nan'))
                if numeric_rating == numeric_rating:  # Check for NaN
                    cluster_ratings[cluster_id].append(numeric_rating)
    
    # Print rating distributions for each cluster
    for cluster_id in range(num_clusters):
        ratings = cluster_ratings[cluster_id]
        if ratings:
            print(f"\nCluster {cluster_id + 1}:")
            print(f"  Number of ratings: {len(ratings)}")
            print(f"  Mean rating: {np.mean(ratings):.2f}")
            print(f"  Rating distribution: {np.histogram(ratings, bins=5, range=(1, 6))[0]}")

# Show sample summaries from each cluster
print("\nSample summaries from each cluster:")
for cluster_id in range(num_clusters):
    print(f"\nCluster {cluster_id + 1}:")
    # Get all agents in this cluster
    cluster_agents = [agent for agent in agents if agent_to_cluster[agent.id] == cluster_id]
    sample_agents = random.sample(cluster_agents, min(5, len(cluster_agents)))
    
    for agent in sample_agents:
        print(f"\nAgent {agent.id}:")
        print(f"Summary: {agent.summary}")
        print("Ratings for initial statements:")
        for statement in initial_statements:
            rating = agent.survey_responses[agent.survey_responses["statement"] == statement]["choice"].iloc[0]
            if rating == rating:  # Check for NaN
                print(f"  {statement}: {rating}")

# %%


# %%
