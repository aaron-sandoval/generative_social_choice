import json
import numpy as np
from pathlib import Path
from generative_social_choice.queries.query_chatbot_personalization import ChatbotPersonalizationAgent
import argparse
import pandas as pd
from generative_social_choice.utils.helper_functions import get_base_dir_path


def load_or_init_matrix(filepath: Path, agent_ids: list[str]):
    if filepath.exists():
        with open(filepath, "r") as f:
            data = json.load(f)
        if data["agent_ids"] != agent_ids:
            raise ValueError("Agent IDs in file do not match current agents. Please delete or update the file.")
        approval_matrix = np.array(data["embeddings"], dtype=float)
    else:
        n = len(agent_ids)
        approval_matrix = np.full((n, n), np.nan)
    return approval_matrix


def save_as_precomputed_embedding(filepath: Path, agent_ids: list[str], approval_matrix: np.ndarray):
    data = {
        "agent_ids": agent_ids,
        "embeddings": approval_matrix.tolist()
    }
    with open(filepath, "w") as f:
        json.dump(data, f)


def compute_approval_matrix_incremental(agents: list[ChatbotPersonalizationAgent], output_path: Path, k: int = 10):
    agent_ids = [agent.get_id() for agent in agents]
    approval_matrix = load_or_init_matrix(output_path, agent_ids)
    update_count = 0
    total_to_fill = np.sum(np.isnan(approval_matrix))
    print(f"Total missing entries to fill: {int(total_to_fill)}")

    for i, agent_i in enumerate(agents):
        for j, agent_j in enumerate(agents):
            if i == j:
                if np.isnan(approval_matrix[i, j]):
                    approval_matrix[i, j] = agent_i.approval_levels["perfectly"]
                    update_count += 1
            else:
                if np.isnan(approval_matrix[i, j]):
                    approval, _ = agent_j.get_approval(agent_i.get_description())
                    approval_matrix[i, j] = approval
                    update_count += 1
            if update_count >= k:
                save_as_precomputed_embedding(output_path, agent_ids, approval_matrix)
                print(f"Saved after {k} updates...")
                update_count = 0
    # Final save
    save_as_precomputed_embedding(output_path, agent_ids, approval_matrix)
    print("Final save complete. Matrix is now up to date.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=10, help="Save after every k updates (default: 10)")
    parser.add_argument("--model", type=str, default="gpt-4o-mini-2024-07-18", help="Model to use for approval queries")
    args = parser.parse_args()

    df = pd.read_csv(get_base_dir_path() / "data/chatbot_personalization_data.csv")
    df = df[df["sample_type"] == "generation"]
    agent_id_to_summary = (
        pd.read_csv(get_base_dir_path() / "data/user_summaries_generation.csv")
        .set_index("user_id")["summary"]
        .to_dict()
    )

    disc_query_model_arg = {"model": args.model} if args.model else {}
    agents = []
    for id in df.user_id.unique():
        agent = ChatbotPersonalizationAgent(
            id=id,
            survey_responses=df[df.user_id == id],
            summary=agent_id_to_summary[id],
            **disc_query_model_arg,
        )
        agents.append(agent)
    output_path = get_base_dir_path() / "data/demo_data/fish_embeddings.json"
    compute_approval_matrix_incremental(agents, output_path, k=args.k)