"""
Which line endings did the prompts of the GenerativeCC runs with Fish's embeddings (statements/fish_*,
the "Ours (Their Embs.)" series of the embedding ablation) use? Their ratings left no logs. Same
pilot as the R2 pilot of measure_model_drift.py, which found LF for the runs with LLM embeddings:
rate the same cells with LF and CRLF prompts and see which reproduces the 2025 values.
"""
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import measure_model_drift as md
from measure_model_drift import RatingTask, value_bin, stratified_sample, read_jsonl, run_threaded, make_rating_fn
from generative_social_choice.ratings.rating_generation import get_agents

OUT_FILE = md.OUT_DIR / "ratings_fish_embeddings_pilot.jsonl"
N_PER_BIN = 30
VARIANTS = ["survey_crlf+statement_crlf", "survey_lf+statement_lf"]


def fish_embeddings_cells():
    """Like md.ours_cells, for the runs with Fish's embeddings"""
    rows, statements = [], {}
    for run_id in [f"fish_{i}" for i in range(10)]:
        run_dir = md.OURS_DIR / run_id / "generated_with_4o_using_fish_embeddings"
        raw_statements = pd.read_csv(run_dir / "statement_generation_raw_output.csv")["statement"].to_list()
        ids = pd.read_csv(run_dir / "4o-mini_for_labelling/utility_matrix_statements.csv", index_col=0)
        matrix = pd.read_csv(run_dir / "4o-mini_for_labelling/utility_matrix.csv", index_col=0)
        key = lambda s: " ".join(s.split())
        raw_by_text = {key(statement): statement for statement in raw_statements}
        for statement_id, statement in ids["statement"].items():
            if statement_id in md.SEED_STATEMENT_IDS:
                continue
            statements[(run_id, statement_id)] = raw_by_text[key(statement)]
            for agent_id, value in matrix[statement_id].items():
                rows.append({"run": run_id, "statement_id": statement_id, "agent_id": agent_id, "value": value})
    return pd.DataFrame(rows), statements


if __name__ == "__main__":
    cells, statements = fish_embeddings_cells()
    cells["bin"] = cells["value"].map(value_bin)
    has_newline = cells.apply(lambda row: "\r\n" in statements[(row.run, row.statement_id)], axis=1)
    pilot = stratified_sample(cells[has_newline], N_PER_BIN, random.Random(md.SEED + 5))
    tasks = [
        RatingTask("R2", v, 0, row.agent_id, md.r2_variant_statement(statements[(row.run, row.statement_id)], v), [row.value],
                   {"key": f"fishpilot:{row.run}:{row.statement_id}", "bin": row.bin, "pilot": True})
        for v in VARIANTS for row in pilot.itertuples()
    ]
    agents_ours = {a.get_id(): a for a in get_agents(model=md.OURS_RATER_MODEL)}
    run_threaded(tasks, make_rating_fn({}, agents_ours), OUT_FILE, 8, "Fish-embeddings pilot")

    df = pd.DataFrame(read_jsonl(OUT_FILE))
    df["diff"] = df["new_value"] - df["old_values"].map(lambda v: v[0])
    print(f"{df['task_id'].nunique()} ratings, {pilot.shape[0]} cells, {(has_newline.mean()):.0%} of cells have CRLF statements")
    for v, g in df.groupby("variant"):
        boot = [g["diff"].sample(len(g), replace=True, random_state=i).mean() for i in range(2000)]
        print(f"{v}: mean abs diff {g['diff'].abs().mean():.3f}, bias {g['diff'].mean():+.3f} "
              f"[{np.percentile(boot, 2.5):+.3f}, {np.percentile(boot, 97.5):+.3f}]")
    # Paired: the same cells under both variants
    wide = df.assign(cell=df["meta"].map(lambda m: m["key"]) + ":" + df["agent_id"]).pivot(index="cell", columns="variant", values="new_value")
    paired = wide[VARIANTS[0]] - wide[VARIANTS[1]]
    boot = [paired.sample(len(paired), replace=True, random_state=i).mean() for i in range(2000)]
    print(f"CRLF - LF, paired over {len(paired)} cells: {paired.mean():+.3f} [{np.percentile(boot, 2.5):+.3f}, {np.percentile(boot, 97.5):+.3f}]")
