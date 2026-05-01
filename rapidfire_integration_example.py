"""Drop-in compute_metrics_fn and accumulate_metrics_fn for Project 1.

Uses project1_eval so scores match evaluate_retrieval.py and run_judge.py.

Your preprocess/postprocess must populate these batch fields:
    retrieved_spans      list of lists of (file, start, end) tuples per query, in rank order
    serialized_context   list of str per query: the text passed to the generator
    ground_truth_spans   list of lists of (file, start, end) tuples per query
    reference_answer     list of str per query

Configure the judge via env vars before experiment.run_evals():
    os.environ["OPENAI_API_KEY"]  = "..."
    os.environ["JUDGE_MODEL"]     = "gpt-4o-mini"  # or a TritonAI chat model
    os.environ["JUDGE_BASE_URL"]  = ""             # empty for OpenAI
"""

import os
from typing import Any, Dict
from typing import List as listtype

from project1_eval import call_judge, f1_at_k, precision_at_k, recall_at_k


def sample_compute_metrics_fn(batch: Dict[str, listtype]) -> Dict[str, Dict[str, Any]]:
    total = len(batch["query"])
    mean = lambda xs: sum(xs) / total

    # Retrieval (top-5)
    f1s = [f1_at_k(r, g) for r, g in zip(batch["retrieved_spans"], batch["ground_truth_spans"])]
    ps = [precision_at_k(r, g) for r, g in zip(batch["retrieved_spans"], batch["ground_truth_spans"])]
    rs = [recall_at_k(r, g) for r, g in zip(batch["retrieved_spans"], batch["ground_truth_spans"])]
    mean_f1, mean_p, mean_r = mean(f1s), mean(ps), mean(rs)

    metrics: Dict[str, Dict[str, Any]] = {
        "Total":           {"value": total},
        "F1@5":            {"value": mean_f1},
        "Precision@5":     {"value": mean_p},
        "Recall@5":        {"value": mean_r},
        "Retrieval Score": {"value": (mean_f1 + mean_p + mean_r) / 3},
    }

    # Generation (LLM judge) -- only when the generator produced answers
    if "generated_text" in batch:
        model = os.environ.get("JUDGE_MODEL", "gpt-4o-mini")
        base_url = os.environ.get("JUDGE_BASE_URL") or None
        corr, faith, comp, failures = [], [], [], 0

        for q, ref, ctx, ans in zip(batch["query"], batch["reference_answer"],
                                     batch["serialized_context"], batch["generated_text"]):
            r = call_judge(q, ref, ctx, ans, model=model, base_url=base_url)
            if r.get("failed"):
                failures += 1
            # Correctness and Faithfulness are binary (0/1); Completeness is 0-5,
            # normalized to [0, 1] to match the leaderboard formula.
            corr.append(r["correctness"])
            faith.append(r["faithfulness"])
            comp.append(r["completeness"] / 5.0)

        mean_c, mean_fa, mean_cp = mean(corr), mean(faith), mean(comp)
        # Partial Generation Score: mean of 3 normalized released metrics.
        # Leaderboard Generation Score includes 2 hidden binary metrics (not computed here).
        metrics.update({
            "Correctness (pass rate)":        {"value": mean_c},
            "Faithfulness (pass rate)":       {"value": mean_fa},
            "Completeness (normalized)":      {"value": mean_cp},
            "Generation Score (3 released)":  {"value": (mean_c + mean_fa + mean_cp) / 3},
            "Judge Failures":                 {"value": failures},
        })

    return metrics


def sample_accumulate_metrics_fn(aggregated: Dict[str, listtype]) -> Dict[str, Dict[str, Any]]:
    """Weighted averages over batches, weighted by query count. Failures are summed."""
    ns = [m.get("value", 0) for m in aggregated.get("Total", [])]
    total = sum(ns)
    out: Dict[str, Dict[str, Any]] = {"Total": {"value": total}}

    # If no queries were processed (e.g., all batches failed), return zeros
    # rather than dividing by zero.
    if total == 0:
        for metric in ["F1@5", "Precision@5", "Recall@5", "Retrieval Score",
                       "Correctness (pass rate)", "Faithfulness (pass rate)",
                       "Completeness (normalized)", "Generation Score (3 released)"]:
            if metric in aggregated:
                out[metric] = {"value": 0.0, "is_algebraic": True, "value_range": (0, 1)}
        if "Judge Failures" in aggregated:
            out["Judge Failures"] = {"value": sum(m["value"] for m in aggregated["Judge Failures"])}
        return out

    for metric in ["F1@5", "Precision@5", "Recall@5", "Retrieval Score",
                   "Correctness (pass rate)", "Faithfulness (pass rate)",
                   "Completeness (normalized)", "Generation Score (3 released)"]:
        if metric in aggregated:
            out[metric] = {
                "value": sum(m["value"] * n for m, n in zip(aggregated[metric], ns)) / total,
                "is_algebraic": True,
                "value_range": (0, 1),
            }

    if "Judge Failures" in aggregated:
        out["Judge Failures"] = {"value": sum(m["value"] for m in aggregated["Judge Failures"])}

    return out
