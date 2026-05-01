"""Shared evaluation primitives for CSE/DSC 234 Project 1.

Imported by evaluate_retrieval.py, run_judge.py, and rapidfire_integration_example.py.
Do not modify: the TAs will re-run the CLIs with the original module.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

# A span is (filename, start_line, end_line). Line ranges are inclusive on both ends.
Span = Tuple[str, int, int]


# --- Span matching and retrieval metrics (top-k) --------------------------
#
# Scoring semantics:
#   - A retrieved span is a "hit" if it shares a filename with some ground-truth
#     span AND their inclusive line intervals overlap.
#   - Precision@k uses a fixed denominator of k (standard IR convention);
#     missing slots (when fewer than k spans are retrieved) count as non-hits.
#   - Precision counts each retrieved span independently: multiple retrieved
#     spans overlapping the same ground-truth span each count as hits.
#   - Recall counts each ground-truth span independently: a gt span is recalled
#     if any retrieved span overlaps it.
#   - Span length is capped (see _MAX_SPAN_LINES in to_spans) to prevent gaming
#     precision by returning oversized spans.

def _overlap(a: Span, b: Span) -> bool:
    return a[0] == b[0] and a[1] <= b[2] and b[1] <= a[2]


def _count_hits(xs: List[Span], ys: List[Span]) -> int:
    """Number of spans in xs that overlap at least one span in ys."""
    return sum(1 for x in xs if any(_overlap(x, y) for y in ys))


def precision_at_k(retrieved: List[Span], gt: List[Span], k: int = 5) -> float:
    """Standard fixed-denominator Precision@k. If fewer than k spans are
    retrieved, missing slots are counted as non-hits."""
    return _count_hits(retrieved[:k], gt) / k if k > 0 else 0.0


def recall_at_k(retrieved: List[Span], gt: List[Span], k: int = 5) -> float:
    return _count_hits(gt, retrieved[:k]) / len(gt) if gt else 0.0


def f1_at_k(retrieved: List[Span], gt: List[Span], k: int = 5) -> float:
    p = precision_at_k(retrieved, gt, k)
    r = recall_at_k(retrieved, gt, k)
    return 2 * p * r / (p + r) if (p + r) else 0.0


# --- Span coercion --------------------------------------------------------

# Maximum allowed span length (inclusive lines). Prevents gaming precision by
# returning very large spans that trivially overlap ground truth. In practice,
# RAG chunkers produce spans much smaller than this. Spans exceeding this limit
# are rejected as malformed.
_MAX_SPAN_LINES = 500


def to_spans(items) -> List[Span]:
    """Convert a list of {file, lines} dicts or (file, start, end) triples to
    Span tuples. Deduplicates. Raises ValueError on any malformed entry."""
    out: List[Span] = []
    seen = set()
    for item in items:
        try:
            if isinstance(item, dict) and "file" in item and "lines" in item:
                lines = item["lines"]
                if not isinstance(lines, (list, tuple)) or len(lines) != 2:
                    raise ValueError(f"'lines' must be a 2-element list [start, end]; got {lines!r}")
                span = (str(item["file"]), int(lines[0]), int(lines[1]))
            elif isinstance(item, (list, tuple)) and len(item) == 3:
                span = (str(item[0]), int(item[1]), int(item[2]))
            else:
                raise ValueError(f"Expected (file, start, end) or {{file, lines: [start, end]}}; got {item!r}")
        except (IndexError, TypeError, KeyError) as e:
            raise ValueError(f"Malformed source entry {item!r}: {e}")
        if span[1] > span[2]:
            raise ValueError(f"Span start > end: {span!r}")
        if span[2] - span[1] + 1 > _MAX_SPAN_LINES:
            raise ValueError(f"Span too large (>{_MAX_SPAN_LINES} lines): {span!r}")
        if span not in seen:
            seen.add(span)
            out.append(span)
    return out


# --- LLM-as-judge ---------------------------------------------------------

_PROMPT_PATH = Path(__file__).parent / "judge_prompt.txt"


def call_judge(
    query: str,
    reference_answer: str,
    retrieved_context: str,
    system_answer: str,
    model: str = "gpt-4o-mini",
    base_url: str | None = None,
    api_key: str | None = None,
    max_attempts: int = 3,
) -> Dict[str, float]:
    """Score one answer on Correctness (binary 0/1), Faithfulness (binary 0/1),
    and Completeness (integer 0-5).

    Returns {'correctness', 'faithfulness', 'completeness', 'failed': bool,
    'error': str (if failed)}. On repeated failure scores 0/0/0 -- a silent
    failure should hurt, not inflate, a config's score.
    """
    from openai import OpenAI

    prompt = _PROMPT_PATH.read_text()
    for placeholder, value in [
        ("{question}", query),
        ("{reference_answer}", reference_answer),
        ("{retrieved_context}", retrieved_context),
        ("{system_answer}", system_answer),
    ]:
        prompt = prompt.replace(placeholder, value)

    client = OpenAI(
        api_key=api_key or os.environ.get("OPENAI_API_KEY"),
        base_url=base_url or os.environ.get("JUDGE_BASE_URL") or None,
    )

    last_err: Exception | None = None
    for _ in range(max_attempts):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=600,
            )
            text = resp.choices[0].message.content.strip()
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
            parsed = json.loads(text)

            scores: Dict[str, float] = {}
            # Correctness and Faithfulness are binary (0 or 1); Completeness is 0-5.
            ranges = {"correctness": (0, 1), "faithfulness": (0, 1), "completeness": (0, 5)}
            for key, (lo, hi) in ranges.items():
                entry = parsed[key]
                s = int(entry["score"] if isinstance(entry, dict) else entry)
                if not lo <= s <= hi:
                    raise ValueError(f"{key} out of range [{lo},{hi}]: {s}")
                scores[key] = float(s)
            scores["failed"] = False
            return scores
        except Exception as e:
            last_err = e

    return {
        "correctness": 0.0, "faithfulness": 0.0, "completeness": 0.0,
        "failed": True, "error": str(last_err) if last_err else "unknown",
    }
