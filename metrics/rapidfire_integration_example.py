"""Drop-in compute_metrics_fn and accumulate_metrics_fn for Project 1.

Uses project1_eval so scores match evaluate_retrieval.py and run_judge.py.

Your preprocess/postprocess must populate these batch fields:
    retrieved_spans      list of lists of (file, start, end) tuples per query, in rank order
    serialized_context   list of str per query: the text passed to the generator
    ground_truth_spans   list of lists of (file, start, end) tuples per query
    reference_answer     list of str per query

The official judge for this project is `claude-sonnet-4-6` on the TritonAI
gateway. Configure via env vars before experiment.run_evals():
    os.environ["OPENAI_API_KEY"] = "<TritonAI key from api-key.txt>"
    os.environ["JUDGE_BASE_URL"] = "https://tritonai-api.ucsd.edu/v1"
    os.environ["JUDGE_MODEL"]    = "claude-sonnet-4-6"

See https://tritonai-api.ucsd.edu/ui/model_hub_table/ for available models.
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
        "F1_at_5":         {"value": mean_f1},
        "Precision_at_5":  {"value": mean_p},
        "Recall_at_5":     {"value": mean_r},
        "Retrieval Score": {"value": (mean_f1 + mean_p + mean_r) / 3},
    }

    # Generation (LLM judge) -- only when the generator produced answers
    if "generated_text" in batch:
        model = os.environ.get("JUDGE_MODEL", "claude-sonnet-4-6")
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
        # Metric names use only alphanumerics, underscores, and spaces so they
        # are accepted by MLflow's metric-name validation (parentheses and `@`
        # cause MLflow to silently drop the metric).
        metrics.update({
            "Correctness_pass_rate":         {"value": mean_c},
            "Faithfulness_pass_rate":        {"value": mean_fa},
            "Completeness_normalized":       {"value": mean_cp},
            "Generation_Score_3_released":   {"value": (mean_c + mean_fa + mean_cp) / 3},
            "Judge Failures":                {"value": failures},
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
        for metric in ["F1_at_5", "Precision_at_5", "Recall_at_5", "Retrieval Score",
                       "Correctness_pass_rate", "Faithfulness_pass_rate",
                       "Completeness_normalized", "Generation_Score_3_released"]:
            if metric in aggregated:
                out[metric] = {"value": 0.0, "is_algebraic": True, "value_range": (0, 1)}
        if "Judge Failures" in aggregated:
            out["Judge Failures"] = {"value": sum(m["value"] for m in aggregated["Judge Failures"])}
        return out

    for metric in ["F1_at_5", "Precision_at_5", "Recall_at_5", "Retrieval Score",
                   "Correctness_pass_rate", "Faithfulness_pass_rate",
                   "Completeness_normalized", "Generation_Score_3_released"]:
        if metric in aggregated:
            out[metric] = {
                "value": sum(m["value"] * n for m, n in zip(aggregated[metric], ns)) / total,
                "is_algebraic": True,
                "value_range": (0, 1),
            }

    if "Judge Failures" in aggregated:
        out["Judge Failures"] = {"value": sum(m["value"] for m in aggregated["Judge Failures"])}

    return out
