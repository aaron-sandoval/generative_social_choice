"""Analysis of the CRLF-vs-LF test (see run_calls.py). Four groups of round-0 outputs per slot:
A = 2025 CRLF (logged), B = Nov-3-2026 LF (logged), C = now CRLF, D = now LF.

Distances are cosine distances between text-embedding-3-small embeddings. Per group: variance = mean pairwise
distance within the group, and the number of distinct outputs. Per pair (X, Y), with X the reference group:
cross = mean distance between X and Y outputs, and a two-sided permutation test (20,000 permutations of the group
labels of the pooled 30 outputs, fixed seed) of whether cross differs from the mean pairwise distance within X,
as in round0_tests.py. The same test with Y as the reference is reported as a secondary column.

Embeddings are cached in the shared cache of the drift study (../queries/embeddings.jsonl).

Run from the repo root: poetry run python generative_social_choice/data/model_drift/line_endings_generation/analyze.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import measure_model_drift as md  # noqa: E402
from run_calls import CALLS_FILE, new_run_calls  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
RESULTS_JSON = OUT_DIR / "results.json"
N_PERM = 20000
SEED = 0
PAIRS = [("C", "A", "drift, prompts fixed (CRLF)"), ("D", "B", "2026 stability, LF, one week"),
         ("C", "D", "line-ending effect, same time"), ("D", "A", "now LF vs 2025 CRLF"), ("B", "A", "original comparison")]


def norm(text: str) -> str:
    return str(text).replace("\r\n", "\n")


def cross_vs_within_test(M, lab, rng):
    """M: distance matrix of the pooled outputs; lab: True for the reference group.
    Statistic: cross mean distance - mean pairwise distance within the reference group."""
    def stat(lab):
        ref, other = np.where(lab)[0], np.where(~lab)[0]
        within = M[np.ix_(ref, ref)][np.triu_indices(len(ref), 1)].mean()
        return M[np.ix_(other, ref)].mean() - within, within
    obs, within = stat(lab)
    null = np.array([stat(rng.permutation(lab))[0] for _ in range(N_PERM)])
    p_two = min(1.0, 2 * min((null <= obs + 1e-12).mean(), (null >= obs - 1e-12).mean()))
    return float(obs), float(within), p_two


def main():
    runs = md.load_baseline_runs()
    A_texts = md.round0_references(runs)
    B_texts = md.new_runs_round0_outputs()
    calls = pd.DataFrame([json.loads(line) for line in open(CALLS_FILE, encoding="utf-8")])
    assert calls["task_id"].is_unique and len(calls) == 90
    print(f"{len(calls)} new calls; raw responses containing CR: {int(calls['response'].str.contains(chr(13)).sum())}; "
          f"finish reasons {calls['finish_reason'].value_counts().to_dict()}; resolved models {calls['resolved_model'].value_counts().to_dict()}")
    groups = {}
    for slot in range(3):
        c = calls[calls.slot == slot]
        groups[slot] = {"A": A_texts[slot], "B": B_texts[slot],
                        "C": [norm(t) for t in c[c.variant == "crlf"].sort_values("rep")["response"]],
                        "D": [norm(t) for t in c[c.variant == "lf"].sort_values("rep")["response"]]}
        assert all(len(v) == 15 for v in groups[slot].values()), {k: len(v) for k, v in groups[slot].items()}

    emb = md.embed_all(sorted({t for g in groups.values() for v in g.values() for t in v}))

    old_calls = [md.generator_calls(run) for run in runs]
    fps = {
        "A (2025 logs)": Counter(c.iloc[s]["system_fingerprint"] for c in old_calls for s in range(3)),
        "B (Nov 3 2026 logs)": Counter(c.iloc[s]["system_fingerprint"] for c in new_run_calls() for s in range(3)),
        "C (now, CRLF)": Counter(calls[calls.variant == "crlf"]["system_fingerprint"]),
        "D (now, LF)": Counter(calls[calls.variant == "lf"]["system_fingerprint"]),
    }
    fps_by_slot = {f"{v} slot {s}": dict(Counter(calls[(calls.variant == v) & (calls.slot == s)]["system_fingerprint"]))
                   for v in ("crlf", "lf") for s in range(3)}

    rng = np.random.default_rng(SEED)
    results = {"fingerprints": {k: dict(v) for k, v in fps.items()}, "fingerprints_now_by_slot": fps_by_slot,
               "timestamps": {"first": calls["timestamp"].min(), "last": calls["timestamp"].max()}, "slots": {}}
    for slot in range(3):
        g = groups[slot]
        var = {k: float(np.mean([md.cosine_distance(emb[x], emb[y]) for i, x in enumerate(v) for y in v[i + 1:]])) for k, v in g.items()}
        distinct = {k: len(set(v)) for k, v in g.items()}
        modal = {k: Counter(v).most_common(1)[0][1] for k, v in g.items()}
        exact = {f"{x} in {y}": sum(t in set(g[y]) for t in g[x]) for x, y in [("C", "A"), ("D", "B"), ("C", "D"), ("D", "A"), ("B", "A")]}
        pairs = {}
        for x, y, label in PAIRS:
            texts = g[x] + g[y]
            M = np.array([[md.cosine_distance(emb[a], emb[b]) for b in texts] for a in texts])
            lab_x = np.array([True] * 15 + [False] * 15)
            cross = float(M[:15, 15:].mean())
            diff_x, within_x, p_x = cross_vs_within_test(M, lab_x, rng)
            diff_y, within_y, p_y = cross_vs_within_test(M, ~lab_x, rng)
            pairs[f"{x}-{y}"] = {"label": label, "cross": cross, "ref": x, "within_ref": within_x, "cross_minus_within_ref": diff_x,
                                 "p_two_sided": p_x, "alt_ref": y, "within_alt_ref": within_y,
                                 "cross_minus_within_alt_ref": diff_y, "p_two_sided_alt_ref": p_y}
        results["slots"][slot] = {"name": md.GENERATOR_SLOTS[slot], "var": var, "distinct": distinct, "modal_count": modal,
                                  "exact_matches": exact, "pairs": pairs}

        print(f"\n=== slot {slot}: {md.GENERATOR_SLOTS[slot]}")
        print("  variance:", {k: round(v, 4) for k, v in var.items()}, " distinct:", distinct, " modal count:", modal)
        print("  outputs identical to one in the other group:", exact)
        for key, r in pairs.items():
            print(f"  {key:4s} {r['label']:32s} cross {r['cross']:.4f} | ref {r['ref']}: cross-within {r['cross_minus_within_ref']:+.4f} "
                  f"p {r['p_two_sided']:.4f} | ref {r['alt_ref']}: cross-within {r['cross_minus_within_alt_ref']:+.4f} p {r['p_two_sided_alt_ref']:.4f}")
    print("\nFingerprints:", results["fingerprints"])
    print("Fingerprints now by slot:", fps_by_slot)
    print("New calls made between", results["timestamps"])
    RESULTS_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {RESULTS_JSON}")


if __name__ == "__main__":
    main()
