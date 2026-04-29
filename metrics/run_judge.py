#!/usr/bin/env python3
"""CSE/DSC 234 Project 1 LLM-as-judge CLI.

Scores a student output JSON against a validation set using the released
3-metric rubric: Correctness (binary Pass/Fail), Faithfulness (binary Pass/Fail),
and Completeness (0-5).

The official judge for this project is `claude-sonnet-4-6` on the TritonAI
gateway. The hidden test set will be graded with this model. Run as:

    export OPENAI_API_KEY=<TritonAI key from api-key.txt>
    python3 run_judge.py --output OUTPUT.json --validation VAL.json \\
        --base-url https://tritonai-api.ucsd.edu/v1 \\
        --model claude-sonnet-4-6

See https://tritonai-api.ucsd.edu/ui/model_hub_table/ for the TritonAI model
hub. The CLI uses an OpenAI-compatible client, so any OpenAI-compatible
endpoint also works as long as --base-url and --model are set correctly.

Output JSON entries must include 'retrieved_context' (the text the generator saw);
it is used to score Faithfulness. Emits a JSON report to stdout (redirect with
`> report.json`). A one-line summary is printed to stderr.

The report's "Partial Generation Score" uses only the 3 released metrics and
applies the leaderboard formula's normalization (Completeness/5) but omits the
2 hidden metrics. Because hidden metrics are missing, the partial score is
computed as the mean of the 3 normalized released metrics; the leaderboard
Generation Score averages all 5 normalized metrics.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from project1_eval import call_judge


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", required=True)
    ap.add_argument("--validation", required=True)
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--max-attempts", type=int, default=3)
    args = ap.parse_args()

    val = {int(e["question_id"]): e for e in json.load(open(args.validation))}
    out = {int(e["question_id"]): e for e in json.load(open(args.output))}

    missing = sorted(set(val) - set(out))
    extras = sorted(set(out) - set(val))

    per_question = []
    n_missing_context = 0
    n_judge_failures = 0

    for qid in sorted(val):
        v = val[qid]
        if qid not in out:
            per_question.append({"question_id": qid, "correctness": 0, "faithfulness": 0,
                                 "completeness": 0, "status": "missing"})
            continue
        ctx = out[qid].get("retrieved_context")
        if not ctx:
            n_missing_context += 1
            per_question.append({"question_id": qid, "correctness": 0, "faithfulness": 0,
                                 "completeness": 0, "status": "no_context"})
            continue
        r = call_judge(
            query=v["question"],
            reference_answer=v["reference_answer"],
            retrieved_context=ctx,
            system_answer=out[qid].get("answer", ""),
            model=args.model,
            base_url=args.base_url,
            max_attempts=args.max_attempts,
        )
        entry = {
            "question_id": qid,
            "correctness":  int(r["correctness"]),
            "faithfulness": int(r["faithfulness"]),
            "completeness": int(r["completeness"]),
        }
        if r.get("failed"):
            n_judge_failures += 1
            entry["status"] = "judge_failed"
            entry["error"] = r.get("error", "")
        else:
            entry["status"] = "ok"
        per_question.append(entry)

    n = len(per_question)
    if n == 0:
        print("[judge] Empty validation set; nothing to score.", file=sys.stderr)
        json.dump({"meta": {"n_questions": 0}, "summary": {}, "per_question": []},
                  sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 1

    mean_c = sum(row["correctness"]  for row in per_question) / n
    mean_f = sum(row["faithfulness"] for row in per_question) / n
    mean_p = sum(row["completeness"] for row in per_question) / n

    # Leaderboard formula normalizes Completeness by 5; Correctness and
    # Faithfulness are already in [0, 1] (binary). The partial generation
    # score averages the three normalized released metrics; the leaderboard
    # score averages all five (including two hidden binary metrics).
    mean_p_norm = mean_p / 5.0
    partial_gen = (mean_c + mean_f + mean_p_norm) / 3.0

    report = {
        "meta": {
            "model": args.model,
            "base_url": args.base_url,
            "n_questions": n,
            "n_missing_from_output": len(missing),
            "n_extra_in_output": len(extras),
            "n_missing_context": n_missing_context,
            "n_judge_failures": n_judge_failures,
            "output_file": args.output,
            "validation_file": args.validation,
        },
        "summary": {
            "Correctness (pass rate)":   mean_c,
            "Faithfulness (pass rate)":  mean_f,
            "Completeness (raw, 0-5)":   mean_p,
            "Completeness (normalized)": mean_p_norm,
            "Partial Generation Score":  partial_gen,
            "Partial Generation Score note":
                "Mean of 3 normalized released metrics. The leaderboard Generation Score averages all 5 normalized metrics (the 3 released + 2 hidden binary).",
        },
        "per_question": per_question,
    }

    json.dump(report, sys.stdout, indent=2)
    sys.stdout.write("\n")

    print(f"[judge] Corr={mean_c:.2f} (pass rate) Faith={mean_f:.2f} (pass rate) "
          f"Comp={mean_p:.2f}/5 Partial Gen Score={partial_gen:.4f} "
          f"[{len(missing)} missing, {n_missing_context} no-ctx, {n_judge_failures} judge-fail]",
          file=sys.stderr)

    return 1 if (n_judge_failures or n_missing_context) else 0


if __name__ == "__main__":
    sys.exit(main())
