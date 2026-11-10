"""CRLF-vs-LF test for the round-0 generator calls (slots 0-2) of the Fish baseline.

Sends, now, the exact 2025 round-0 messages (CRLF) and the exact Nov-3-2026 round-0 messages (LF) to
gpt-4o-2024-08-06, 15 times each per slot, with the logged params, through the repo's GPT.call path.
Before sending anything it verifies that the messages are identical within each era, differ only in
line endings across eras, and that the params match. Results are appended to calls.jsonl as they
arrive; a rerun only sends calls that are missing.

Run from the repo root: poetry run python generative_social_choice/data/model_drift/line_endings_generation/run_calls.py [--verify_only]
"""
import ast
import json
import random
import sys
import threading
import argparse
import concurrent.futures
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import measure_model_drift as md  # noqa: E402
from generative_social_choice.utils.gpt_wrapper import GPT  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
CALLS_FILE = OUT_DIR / "calls.jsonl"
N_REPS = 15
MAX_CALLS = 100
SEED = 0
NEW_DIR = md.BASELINE_DIR.with_name(md.BASELINE_DIR.name + "_persistent_pool")


def new_run_calls() -> list[pd.DataFrame]:
    """Generator calls of the finished new runs (same selection as md.new_runs_round0_outputs)"""
    out = []
    for run_dir in sorted(d for d in NEW_DIR.iterdir() if (d / "committee.txt").exists()):
        logs = pd.read_csv(run_dir / "logs.csv", index_col=0, low_memory=False)
        out.append(logs[logs["model"] == md.GENERATOR_MODEL].reset_index(drop=True))
    return out


def to_lf(messages: list[dict]) -> list[dict]:
    return [{k: (v.replace("\r\n", "\n").replace("\\r\\n", "\\n") if isinstance(v, str) else v) for k, v in m.items()}
            for m in messages]


def verify() -> dict:
    """Returns {slot: {"crlf": (messages, params), "lf": (messages, params)}}; raises on any failed check"""
    old_runs = md.load_baseline_runs()
    old_calls = [md.generator_calls(run) for run in old_runs]
    new_calls = new_run_calls()
    print(f"2025 runs: {len(old_calls)}, 2026 runs: {len(new_calls)}")
    assert len(old_calls) == 15 and len(new_calls) == 15
    prompts = {}
    for slot in range(3):
        result = {}
        for era, calls in (("crlf", old_calls), ("lf", new_calls)):
            raw_messages = {c.iloc[slot]["messages"] for c in calls}
            raw_params = {c.iloc[slot]["params"] for c in calls}
            parsed = [ast.literal_eval(c.iloc[slot]["messages"]) for c in calls]
            # Check 1: identical across runs within the era (both the logged repr and the parsed value)
            assert len(raw_messages) == 1, f"slot {slot} {era}: {len(raw_messages)} distinct logged messages"
            assert all(p == parsed[0] for p in parsed), f"slot {slot} {era}: parsed messages differ"
            assert len(raw_params) == 1, f"slot {slot} {era}: {len(raw_params)} distinct params"
            result[era] = (parsed[0], ast.literal_eval(raw_params.pop()))
        crlf_msgs, crlf_params = result["crlf"]
        lf_msgs, lf_params = result["lf"]
        content = lambda msgs: "".join(m["content"] for m in msgs)
        n_real = content(crlf_msgs).count("\r\n")
        n_escaped = content(crlf_msgs).count("\\r\\n")
        # Check 2: differ only in line endings
        assert crlf_msgs != lf_msgs, f"slot {slot}: 2025 and 2026 messages are already identical"
        assert to_lf(crlf_msgs) == lf_msgs, f"slot {slot}: messages differ beyond line endings"
        assert "\r" not in content(lf_msgs) and "\\r" not in content(lf_msgs), f"slot {slot}: CR in the LF messages"
        assert [m["role"] for m in crlf_msgs] == [m["role"] for m in lf_msgs]
        # Check 3: params match
        assert crlf_params == lf_params, f"slot {slot}: params differ: {crlf_params} vs {lf_params}"
        print(f"slot {slot}: OK. {len(crlf_msgs)} messages, roles {[m['role'] for m in crlf_msgs]}; 2025 content has "
              f"{n_real} real CRLF and {n_escaped} escaped CRLF; {len(content(lf_msgs))} chars LF vs "
              f"{len(content(crlf_msgs))} CRLF; params {crlf_params}")
        prompts[slot] = result
    # The old runs logged the alias "gpt-4o"; check it resolved to the snapshot we call now
    resolved = {str(c.iloc[s]["completion"]).split("model='")[1].split("'")[0] for c in old_calls for s in range(3)}
    print(f"2025 logged model 'gpt-4o' resolved to: {resolved}; 2026 logged model {md.GENERATOR_MODEL}")
    assert resolved == {md.GENERATOR_MODEL}
    return prompts


def read_done() -> dict:
    if not CALLS_FILE.exists():
        return {}
    with open(CALLS_FILE, "r", encoding="utf-8") as f:
        return {r["task_id"]: r for r in map(json.loads, f)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify_only", action="store_true")
    parser.add_argument("--num_threads", type=int, default=2)
    args = parser.parse_args()
    prompts = verify()
    print("All checks passed.")
    if args.verify_only:
        return

    tasks = [{"task_id": f"{variant}:{slot}:{rep}", "variant": variant, "slot": slot, "rep": rep}
             for slot in range(3) for variant in ("crlf", "lf") for rep in range(N_REPS)]
    random.Random(SEED).shuffle(tasks)  # Interleave the variants and slots in time
    done = read_done()
    todo = [t for t in tasks if t["task_id"] not in done]
    print(f"{len(tasks) - len(todo)} calls done, {len(todo)} to send")
    assert len(tasks) <= MAX_CALLS
    if not todo:
        return

    lock = threading.Lock()

    def call(task):
        messages, params = prompts[task["slot"]][task["variant"]]
        llm = GPT(model=md.GENERATOR_MODEL)  # Same construction as md.generate (openai_seed defaults to 0, as in the runs)
        response, completion, log = llm.call(messages=messages, **params)
        record = {**task, "response": response, "system_fingerprint": log["system_fingerprint"],
                  "resolved_model": getattr(completion, "model", None), "finish_reason": completion.choices[0].finish_reason,
                  "params": params, "timestamp": log["timestamp"]}
        with lock, open(CALLS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        return record

    # Submit in the shuffled order; with 2 workers, calls run roughly in that order
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        for i, future in enumerate(concurrent.futures.as_completed([executor.submit(call, t) for t in todo])):
            r = future.result()
            print(f"[{i + 1}/{len(todo)}] {r['task_id']} {r['system_fingerprint']} {r['timestamp']}")


if __name__ == "__main__":
    main()
