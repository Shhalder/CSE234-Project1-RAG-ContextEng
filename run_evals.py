import json
import os
import sys
import importlib
from pathlib import Path
from typing import Any, Dict, List, Optional

import tiktoken
from datasets import Dataset
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from rapidfireai import Experiment
from rapidfireai.automl import RFOpenAIAPIModelConfig, RFLangChainRagSpec
from sentence_transformers import CrossEncoder

from metrics.project1_eval import to_spans

# Ensure modules that import top-level `project1_eval` resolve correctly.
sys.modules["project1_eval"] = importlib.import_module("metrics.project1_eval")

from metrics.rapidfire_integration_example import (
    sample_accumulate_metrics_fn,
    sample_compute_metrics_fn,
)

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

# Mirror main.py defaults for context construction.
TOTAL_BUDGET_TOKENS = 2000
RESERVED_NON_CONTEXT_TOKENS = 400
MAX_CONTEXT_TOKENS = TOTAL_BUDGET_TOKENS - RESERVED_NON_CONTEXT_TOKENS
TOKENIZER = tiktoken.get_encoding("gpt2")
INSTRUCTIONS = (
    "You are a technical documentation assistant.\n"
    "Answer the user's question using only the provided context chunks.\n\n"
    "Rules:\n"
    "- If the context is insufficient, say you do not know based on the provided context.\n"
    "- Keep the answer concise and factual.\n"
    "- Do not invent APIs, arguments, or behaviors not supported by context."
)

# Configurable eval designs:
# - retrieval_backend: "faiss" or "hnsw"
# - use_hyde: generate hypothetical answer for retrieval query
# - reranker_model: None to disable, or CrossEncoder model string
# - use_score_fusion: combine retriever score + reranker score
CONFIGS = [
    # Best so far (baseline)
    {"chunk_size": 288, "chunk_overlap": 72, "k": 40, "top_n": 8, "embed_model": "BAAI/bge-small-en-v1.5", "search_type": "mmr", "retrieval_backend": "faiss", "use_hyde": False, "reranker_model": "cross-encoder/ms-marco-MiniLM-L6-v2", "use_score_fusion": False},

    # GTE large embedding
    {"chunk_size": 288, "chunk_overlap": 72, "k": 40, "top_n": 8, "embed_model": "thenlper/gte-large", "search_type": "mmr", "retrieval_backend": "faiss", "use_hyde": False, "reranker_model": "cross-encoder/ms-marco-MiniLM-L6-v2", "use_score_fusion": False},
    # BGE large + HyDE
    {"chunk_size": 288, "chunk_overlap": 72, "k": 40, "top_n": 8, "embed_model": "BAAI/bge-large-en-v1.5", "search_type": "mmr", "retrieval_backend": "faiss", "use_hyde": True, "reranker_model": "cross-encoder/ms-marco-MiniLM-L6-v2", "use_score_fusion": False}
]

_FILE_CACHE: dict[str, str] = {}
_RERANKERS: dict[str, CrossEncoder] = {}
_HYDE_CLIENT: Optional[OpenAI] = None


def _get_text(path: str) -> str:
    if path not in _FILE_CACHE:
        _FILE_CACHE[path] = Path(path).read_text(encoding="utf-8")
    return _FILE_CACHE[path]


def _get_reranker(model_name: str) -> CrossEncoder:
    if model_name not in _RERANKERS:
        _RERANKERS[model_name] = CrossEncoder(model_name)
    return _RERANKERS[model_name]


def _get_hyde_client() -> OpenAI:
    global _HYDE_CLIENT
    if _HYDE_CLIENT is None:
        _HYDE_CLIENT = OpenAI(
            api_key=TRITON_API_KEY,
            base_url="https://tritonai-api.ucsd.edu",
            max_retries=2,
        )
    return _HYDE_CLIENT


def custom_template(doc: Document) -> str:
    return f"{doc.metadata.get('source', '')}: {doc.page_content}"


def count_tokens(text: str) -> int:
    return len(TOKENIZER.encode(text))


def clip_docs_to_context_budget(docs: List[Document]) -> tuple[str, List[Document]]:
    selected_docs: List[Document] = []
    context_parts: List[str] = []
    used_tokens = 0
    sep_tokens = count_tokens("\n\n")

    for doc in docs:
        part = custom_template(doc)
        part_tokens = count_tokens(part)
        additional = part_tokens + (sep_tokens if context_parts else 0)

        if used_tokens + additional > MAX_CONTEXT_TOKENS:
            break

        context_parts.append(part)
        selected_docs.append(doc)
        used_tokens += additional

    if not context_parts and docs:
        first = custom_template(docs[0])
        clipped = TOKENIZER.decode(TOKENIZER.encode(first)[:MAX_CONTEXT_TOKENS])
        context_parts = [clipped]
        selected_docs = [docs[0]]

    return "\n\n".join(context_parts), selected_docs


def _retriever_score(doc: Document, fallback_rank: int) -> float:
    md = doc.metadata or {}
    for key in ("score", "relevance_score", "similarity"):
        if key in md:
            try:
                return float(md[key])
            except Exception:
                pass
    if "distance" in md:
        try:
            return -float(md["distance"])
        except Exception:
            pass
    return 1.0 / (1 + fallback_rank)


def _apply_rerank(
    query: str,
    docs: List[Document],
    reranker_model: Optional[str],
    top_n: int,
    use_score_fusion: bool,
    fusion_rerank_weight: float = 0.7,
) -> List[Document]:
    if not docs:
        return docs
    if not reranker_model:
        return docs[:top_n]

    reranker = _get_reranker(reranker_model)
    pairs = [[query, d.page_content] for d in docs]
    rr_scores = list(reranker.predict(pairs))

    if not use_score_fusion:
        ranked = sorted(zip(rr_scores, docs), key=lambda x: x[0], reverse=True)
        return [d for _, d in ranked[:top_n]]

    # Simple min-max normalized fusion.
    retr_scores = [_retriever_score(d, i) for i, d in enumerate(docs)]

    def normalize(vals: List[float]) -> List[float]:
        vmin, vmax = min(vals), max(vals)
        if vmax - vmin < 1e-12:
            return [0.5] * len(vals)
        return [(v - vmin) / (vmax - vmin) for v in vals]

    n_rr = normalize([float(x) for x in rr_scores])
    n_rt = normalize([float(x) for x in retr_scores])
    rw = max(0.0, min(1.0, float(fusion_rerank_weight)))
    tw = 1.0 - rw
    fused = [rw * rr + tw * rt for rr, rt in zip(n_rr, n_rt)]

    ranked = sorted(zip(fused, docs), key=lambda x: x[0], reverse=True)
    return [d for _, d in ranked[:top_n]]


def _hyde_query(query: str, cfg: Dict[str, Any]) -> str:
    if not cfg.get("use_hyde", False):
        return query

    prompt = (
        "Write a concise hypothetical answer passage that would likely contain facts "
        "needed to answer the question. Do not mention uncertainty.\n\n"
        f"Question: {query}\n\nPassage:"
    )
    try:
        client = _get_hyde_client()
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You produce retrieval-oriented hypothetical passages."},
                {"role": "user", "content": prompt},
            ],
            max_completion_tokens=180,
        )
        hyde_text = (response.choices[0].message.content or "").strip()
        return hyde_text or query
    except Exception:
        return query


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


def make_preprocess_fn(cfg: Dict[str, Any]):
    def preprocess_fn(batch: Dict[str, list], rag: RFLangChainRagSpec, prompt_manager=None) -> Dict[str, list]:
        queries = batch["query"]
        retrieval_queries = [_hyde_query(q, cfg) for q in queries]

        all_docs = rag.get_context(batch_queries=retrieval_queries, serialize=False)
        reranked_docs = [
            _apply_rerank(
                query=q,
                docs=docs,
                reranker_model=cfg.get("reranker_model"),
                top_n=int(cfg.get("top_n", 5)),
                use_score_fusion=bool(cfg.get("use_score_fusion", False)),
                fusion_rerank_weight=float(cfg.get("fusion_rerank_weight", 0.7)),
            )
            for q, docs in zip(queries, all_docs)
        ]

        serialized_context: List[str] = []
        kept_docs_per_query: List[List[Document]] = []
        for docs in reranked_docs:
            clipped_context, kept_docs = clip_docs_to_context_budget(docs)
            serialized_context.append(clipped_context)
            kept_docs_per_query.append(kept_docs)

        retrieved_spans = [docs_to_spans(docs) for docs in kept_docs_per_query]

        ground_truth_spans = [[tuple(span) for span in json.loads(s)] for s in batch["ground_truth_spans_json"]]

        prompts = [
            [
                {"role": "system", "content": INSTRUCTIONS},
                {"role": "user", "content": f"Question:\n{q}\n\nContext:\n{ctx}\n\nAnswer:"},
            ]
            for q, ctx in zip(queries, serialized_context)
        ]

        return {
            "prompts": prompts,
            "serialized_context": serialized_context,
            "retrieved_spans": retrieved_spans,
            "ground_truth_spans": ground_truth_spans,
            **batch,
        }

    return preprocess_fn


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


def _vector_store_cfg_for(cfg: Dict[str, Any]) -> Dict[str, Any]:
    if cfg.get("retrieval_backend") == "hnsw":
        # Best-effort HNSW config; unsupported fields are ignored by some backends.
        return {"type": "faiss", "index_factory": "HNSW32", "metric": "ip"}
    return {"type": "faiss"}


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
        vector_store_cfg=_vector_store_cfg_for(cfg),
        search_cfg={"type": cfg.get("search_type", "mmr"), "k": int(cfg.get("k", 20))},
        enable_gpu_search=False,
        document_template=custom_template,
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
        "preprocess_fn": make_preprocess_fn(cfg),
        "compute_metrics_fn": sample_compute_metrics_fn,
        "accumulate_metrics_fn": sample_accumulate_metrics_fn,
        "online_strategy_kwargs": {
            "strategy_name": "normal",
            "confidence_level": 0.95,
            "use_fpc": True,
        },
    }


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

    summary_rows: List[Dict[str, Any]] = []
    metric_rows = []
    if isinstance(results, dict):
        run_items = sorted(results.items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else str(kv[0]))
        for run_key, run_val in run_items:
            if isinstance(run_val, list) and len(run_val) >= 2 and isinstance(run_val[1], dict):
                metric_rows.append((run_key, run_val[1]))

    for run_key, r in metric_rows:
        summary_rows.append(
            {
                "run_id": (r.get("run_id", {}) or {}).get("value", run_key),
                "chunk_size": (r.get("chunk_size", {}) or {}).get("value"),
                "chunk_overlap": (r.get("chunk_overlap", {}) or {}).get("value"),
                "search_cfg": (r.get("search_cfg", {}) or {}).get("value"),
                "embedding_model": ((r.get("embedding_cfg", {}) or {}).get("value") or {}).get("model_name"),
                "vector_store_cfg": (r.get("vector_store_cfg", {}) or {}).get("value"),
                "F1@5": (r.get("F1@5", {}) or {}).get("value"),
                "Precision@5": (r.get("Precision@5", {}) or {}).get("value"),
                "Recall@5": (r.get("Recall@5", {}) or {}).get("value"),
                "Retrieval Score": (r.get("Retrieval Score", {}) or {}).get("value"),
                "Correctness (pass rate)": (r.get("Correctness (pass rate)", {}) or {}).get("value"),
                "Faithfulness (pass rate)": (r.get("Faithfulness (pass rate)", {}) or {}).get("value"),
                "Completeness (normalized)": (r.get("Completeness (normalized)", {}) or {}).get("value"),
                "Generation Score (3 released)": (r.get("Generation Score (3 released)", {}) or {}).get("value"),
                "use_hyde": CONFIGS[int(run_key) - 1].get("use_hyde") if str(run_key).isdigit() else None,
                "reranker_model": CONFIGS[int(run_key) - 1].get("reranker_model") if str(run_key).isdigit() else None,
                "use_score_fusion": CONFIGS[int(run_key) - 1].get("use_score_fusion") if str(run_key).isdigit() else None,
                "retrieval_backend": CONFIGS[int(run_key) - 1].get("retrieval_backend") if str(run_key).isdigit() else None,
            }
        )

    Path("logs/run_evals_summary.json").write_text(
        json.dumps(summary_rows, indent=2),
        encoding="utf-8",
    )

    if not summary_rows:
        print("Warning: no run rows extracted for summary; check run_evals_results.json structure.")
    else:
        print(f"Extracted {len(summary_rows)} run rows for summary.")

    experiment.end()

    print(f"Completed run_evals for {len(CONFIGS)} configs.")
    print("Saved logs/run_evals_results.json and logs/run_evals_summary.json")


if __name__ == "__main__":
    main()
