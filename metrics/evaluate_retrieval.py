#!/usr/bin/env python3
"""CSE/DSC 234 Project 1 retrieval metric CLI.

Scores a student output JSON against a validation set using span-based retrieval
metrics. A retrieved chunk hits a ground-truth span if they share a filename and
their inclusive line intervals overlap.

    python3 evaluate_retrieval.py --output OUTPUT.json --validation VAL.json [--k 5]

Emits a JSON report to stdout (redirect with `> report.json`). A one-line
summary is printed to stderr for quick eyeballing.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from project1_eval import f1_at_k, precision_at_k, recall_at_k, to_spans


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", required=True)
    ap.add_argument("--validation", required=True)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    val = {int(e["question_id"]): e for e in json.load(open(args.validation))}
    out = {int(e["question_id"]): e for e in json.load(open(args.output))}

    missing = sorted(set(val) - set(out))
    extras = sorted(set(out) - set(val))

    per_question = []
    n_malformed = 0
    for qid in sorted(val):
        gt = to_spans(val[qid]["source_evidence"])
        if qid not in out:
            per_question.append({"question_id": qid, f"F1@{args.k}": 0.0,
                                 f"Precision@{args.k}": 0.0, f"Recall@{args.k}": 0.0,
                                 "status": "missing"})
            continue
        if "sources" not in out[qid]:
            n_malformed += 1
            per_question.append({"question_id": qid, f"F1@{args.k}": 0.0,
                                 f"Precision@{args.k}": 0.0, f"Recall@{args.k}": 0.0,
                                 "status": "malformed", "error": "missing 'sources' key"})
            continue
        try:
            retrieved = to_spans(out[qid]["sources"])
        except ValueError as e:
            n_malformed += 1
            per_question.append({"question_id": qid, f"F1@{args.k}": 0.0,
                                 f"Precision@{args.k}": 0.0, f"Recall@{args.k}": 0.0,
                                 "status": "malformed", "error": str(e)})
            continue
        per_question.append({
            "question_id": qid,
            f"F1@{args.k}":        f1_at_k(retrieved, gt, args.k),
            f"Precision@{args.k}": precision_at_k(retrieved, gt, args.k),
            f"Recall@{args.k}":    recall_at_k(retrieved, gt, args.k),
            "status": "ok",
        })

    n = len(per_question)
    if n == 0:
        print("[retrieval] Empty validation set; nothing to score.", file=sys.stderr)
        json.dump({"meta": {"n_questions": 0}, "summary": {}, "per_question": []},
                  sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 1

    mean_f = sum(row[f"F1@{args.k}"]        for row in per_question) / n
    mean_p = sum(row[f"Precision@{args.k}"] for row in per_question) / n
    mean_r = sum(row[f"Recall@{args.k}"]    for row in per_question) / n

    report = {
        "meta": {
            "k": args.k,
            "n_questions": n,
            "n_missing_from_output": len(missing),
            "n_extra_in_output": len(extras),
            "n_malformed_sources": n_malformed,
            "output_file": args.output,
            "validation_file": args.validation,
        },
        "summary": {
            f"F1@{args.k}":        mean_f,
            f"Precision@{args.k}": mean_p,
            f"Recall@{args.k}":    mean_r,
            "Retrieval Score":     (mean_f + mean_p + mean_r) / 3,
        },
        "per_question": per_question,
    }

    json.dump(report, sys.stdout, indent=2)
    sys.stdout.write("\n")

    print(f"[retrieval] F1@{args.k}={mean_f:.4f} P@{args.k}={mean_p:.4f} "
          f"R@{args.k}={mean_r:.4f} Retrieval Score={(mean_f+mean_p+mean_r)/3:.4f} "
          f"[{len(missing)} missing, {n_malformed} malformed]", file=sys.stderr)

    return 1 if n_malformed else 0


if __name__ == "__main__":
    sys.exit(main())
