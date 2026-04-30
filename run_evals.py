import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure repo root is importable
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Ensure `project1_eval` is importable as a top-level module in Ray workers.
METRICS_DIR = ROOT_DIR / "metrics"
if str(METRICS_DIR) not in sys.path:
    sys.path.insert(0, str(METRICS_DIR))

existing_pythonpath = os.environ.get("PYTHONPATH", "")
pythonpath_parts = [p for p in existing_pythonpath.split(os.pathsep) if p]
if str(METRICS_DIR) not in pythonpath_parts:
    os.environ["PYTHONPATH"] = os.pathsep.join([str(METRICS_DIR), *pythonpath_parts])

from datasets import Dataset
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rapidfireai import Experiment
from rapidfireai.automl import RFOpenAIAPIModelConfig, RFLangChainRagSpec

from metrics.project1_eval import to_spans
from metrics.rapidfire_integration_example import (
    sample_accumulate_metrics_fn,
    sample_compute_metrics_fn,
)


# Triton API key
TRITON_API_KEY = os.getenv("TRITON_API_KEY")
if not TRITON_API_KEY:
    TRITON_API_KEY = (
        Path("~/api-key.txt")
        .expanduser()
        .read_text(encoding="utf-8")
        .splitlines()[0]
        .strip()
    )

# Keep judge key in sync by default unless user explicitly set OPENAI_API_KEY.
if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = TRITON_API_KEY

DOCS_DIR = "data/sourcedocs"
MODEL_NAME = "api-mistral-small-3.2-2506"
JUDGE_MODEL = os.getenv("JUDGE_MODEL", MODEL_NAME)
JUDGE_BASE_URL = os.getenv("JUDGE_BASE_URL", "https://tritonai-api.ucsd.edu/v1")

os.environ["JUDGE_MODEL"] = JUDGE_MODEL
os.environ["JUDGE_BASE_URL"] = JUDGE_BASE_URL

# 8 configs: vary chunking, top-k, and embedding model.
CONFIGS: List[Dict[str, Any]] = [
    {"chunk_size": 256, "chunk_overlap": 32, "k": 5, "embed_model": "sentence-transformers/all-MiniLM-L6-v2"},
    {"chunk_size": 512, "chunk_overlap": 32, "k": 5, "embed_model": "sentence-transformers/all-MiniLM-L6-v2"},
    {"chunk_size": 1024, "chunk_overlap": 32, "k": 5, "embed_model": "sentence-transformers/all-MiniLM-L6-v2"},
    {"chunk_size": 512, "chunk_overlap": 64, "k": 5, "embed_model": "sentence-transformers/all-MiniLM-L6-v2"},
    {"chunk_size": 512, "chunk_overlap": 32, "k": 3, "embed_model": "sentence-transformers/all-MiniLM-L6-v2"},
    {"chunk_size": 512, "chunk_overlap": 32, "k": 10, "embed_model": "sentence-transformers/all-MiniLM-L6-v2"},
    {"chunk_size": 512, "chunk_overlap": 32, "k": 5, "embed_model": "BAAI/bge-small-en-v1.5"},
    {"chunk_size": 256, "chunk_overlap": 64, "k": 10, "embed_model": "BAAI/bge-small-en-v1.5"},
]

_FILE_CACHE: dict[str, str] = {}


def _get_text(path: str) -> str:
    if path not in _FILE_CACHE:
        _FILE_CACHE[path] = Path(path).read_text(encoding="utf-8")
    return _FILE_CACHE[path]


def docs_to_spans(docs: List[Any]) -> List[tuple[str, int, int]]:
    spans: List[tuple[str, int, int]] = []
    seen = set()
    for doc in docs:
        source_path = doc.metadata.get("source", "")
        if not source_path:
            continue
        text = _get_text(source_path)
        start_char = int(doc.metadata.get("start_index", 0))
        end_char = min(start_char + len(doc.page_content), len(text))
        startline = text.count("\n", 0, start_char) + 1
        endline = text.count("\n", 0, end_char) + 1
        if startline > endline:
            startline, endline = endline, startline
        key = (Path(source_path).name, startline, endline)
        if key not in seen:
            seen.add(key)
            spans.append(key)
    return spans


def preprocess_fn(batch: Dict[str, list], rag: RFLangChainRagSpec, prompt_manager=None) -> Dict[str, list]:
    all_docs = rag.get_context(batch_queries=batch["query"], serialize=False)
    serialized_context = rag.serialize_documents(all_docs)
    retrieved_spans = [docs_to_spans(docs) for docs in all_docs]

    # Convert JSON-safe spans back to tuple spans for metric functions.
    ground_truth_spans = [
        [tuple(span) for span in json.loads(s)]
        for s in batch["ground_truth_spans_json"]
    ]

    prompts = [
        [
            {"role": "system", "content": "Answer the question using only the provided context."},
            {"role": "user", "content": f"Question:\n{q}\n\nContext:\n{ctx}\n\nAnswer:"},
        ]
        for q, ctx in zip(batch["query"], serialized_context)
    ]

    return {
        "prompts": prompts,
        "serialized_context": serialized_context,
        "retrieved_spans": retrieved_spans,
        "ground_truth_spans": ground_truth_spans,
        **batch,
    }


def load_eval_dataset(
    query_path: str = "data/evaluation-set-queries.json",
    validation_path: str = "data/evaluation-set-golden-qa-pairs.json",
) -> Dataset:
    queries = json.loads(Path(query_path).read_text(encoding="utf-8"))
    val = {int(x["question_id"]): x for x in json.loads(Path(validation_path).read_text(encoding="utf-8"))}

    rows = []
    for q in queries:
        qid = int(q.get("question_id", q.get("id")))
        if qid not in val:
            continue
        rows.append(
            {
                "question_id": qid,
                "query": q["question"],
                "reference_answer": val[qid]["reference_answer"],
                "ground_truth_spans_json": json.dumps(to_spans(val[qid]["source_evidence"])),
            }
        )

    return Dataset.from_list(rows)


def make_eval_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    rag = RFLangChainRagSpec(
        document_loader=DirectoryLoader(
            path=DOCS_DIR,
            glob="*.rst",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            sample_seed=42,
        ),
        text_splitter=RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="gpt2",
            chunk_size=cfg["chunk_size"],
            chunk_overlap=cfg["chunk_overlap"],
            add_start_index=True,
        ),
        embedding_cfg={
            "class": HuggingFaceEmbeddings,
            "model_name": cfg["embed_model"],
            "model_kwargs": {"device": "cpu"},
            "encode_kwargs": {"normalize_embeddings": True},
        },
        vector_store_cfg={"type": "faiss"},
        search_cfg={"type": "similarity", "k": cfg["k"]},
        enable_gpu_search=False,
    )

    openai_config = RFOpenAIAPIModelConfig(
        client_config={
            "api_key": TRITON_API_KEY,
            "base_url": "https://tritonai-api.ucsd.edu",
            "max_retries": 2,
        },
        model_config={
            "model": MODEL_NAME,
            "max_completion_tokens": 512,
        },
        rpm_limit=120,
        tpm_limit=1_000_000,
        rag=rag,
        prompt_manager=None,
    )

    return {
        "pipeline": openai_config,
        "batch_size": 8,
        "preprocess_fn": preprocess_fn,
        "compute_metrics_fn": sample_compute_metrics_fn,
        "accumulate_metrics_fn": sample_accumulate_metrics_fn,
        "online_strategy_kwargs": {
            "strategy_name": "normal",
            "confidence_level": 0.95,
            "use_fpc": True,
        },
    }


def _metric_value(row: Dict[str, Any], key: str) -> Any:
    value = row.get(key)
    if isinstance(value, dict):
        return value.get("value")
    return value


def main() -> None:
    dataset = load_eval_dataset()
    config_group = [make_eval_config(cfg) for cfg in CONFIGS]

    experiment = Experiment(
        experiment_name="p1-rag-evals",
        mode="evals",
    )

    results = experiment.run_evals(
        config_group=config_group,
        dataset=dataset,
        num_shards=4,
        num_actors=8,
        seed=42,
    )

    Path("logs").mkdir(exist_ok=True)
    Path("logs/run_evals_results.json").write_text(
        json.dumps(results, indent=2, default=str),
        encoding="utf-8",
    )

    # Compact metrics-only summary (easy to read without CSV tooling)
    if isinstance(results, dict):
        result_rows = list(results.values())
    elif isinstance(results, list):
        result_rows = results
    else:
        result_rows = [results]

    summary_rows = []
    for r in result_rows:
        if not isinstance(r, dict):
            continue
        summary_rows.append(
            {
                "run_id": _metric_value(r, "run_id"),
                "chunk_size": _metric_value(r, "chunk_size"),
                "chunk_overlap": _metric_value(r, "chunk_overlap"),
                "search_cfg": _metric_value(r, "search_cfg"),
                "embedding_model": (
                    (_metric_value(r, "embedding_cfg") or {}).get("model_name")
                ),
                "F1@5": _metric_value(r, "F1@5"),
                "Precision@5": _metric_value(r, "Precision@5"),
                "Recall@5": _metric_value(r, "Recall@5"),
                "Retrieval Score": _metric_value(r, "Retrieval Score"),
                "Correctness (pass rate)": _metric_value(r, "Correctness (pass rate)"),
                "Faithfulness (pass rate)": _metric_value(r, "Faithfulness (pass rate)"),
                "Completeness (normalized)": _metric_value(r, "Completeness (normalized)"),
                "Generation Score (3 released)": _metric_value(r, "Generation Score (3 released)"),
            }
        )

    Path("logs/run_evals_summary.json").write_text(
        json.dumps(summary_rows, indent=2),
        encoding="utf-8",
    )

    experiment.end()

    print("Completed run_evals for 8 configs.")
    print("Saved logs/run_evals_results.json and logs/run_evals_summary.json")


if __name__ == "__main__":
    main()
