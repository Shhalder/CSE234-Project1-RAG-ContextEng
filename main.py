import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import tiktoken
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from rapidfireai.automl import RFLangChainRagSpec, RFOpenAIAPIModelConfig
from sentence_transformers import CrossEncoder

# Triton API key
TRITON_API_KEY = (
    os.getenv("TRITON_API_KEY")
    or os.getenv("OPENAI_API_KEY")
    or Path("~/api-key.txt")
    .expanduser()
    .read_text(encoding="utf-8")
    .splitlines()[0]
    .strip()
)

# load input file
def load_input_questions(input_path: Path) -> List[Dict[str, Any]]:
    data = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list.")
    for row in data:
        if not isinstance(row, dict):
            raise ValueError("Each item must be an object.")
        if "id" in row and "question_id" not in row:
            row["question_id"] = row["id"]
        if "question_id" not in row or "question" not in row:
            raise ValueError("Each item must have question_id and question.")
    return data

ROOT_DIR = Path(__file__).resolve().parent
DOCS_DIR = str(ROOT_DIR / "data" / "sourcedocs")
MODEL_NAME = "api-mistral-small-3.2-2506"
MAX_COMPLETION_TOKENS = 512
TOTAL_BUDGET_TOKENS = 2000
RESERVED_NON_CONTEXT_TOKENS = 400
MAX_CONTEXT_TOKENS = TOTAL_BUDGET_TOKENS - RESERVED_NON_CONTEXT_TOKENS
TOKENIZER = tiktoken.get_encoding("gpt2")
RERANK_TOP_N = 8
RETRIEVAL_K = 40
BATCH_SIZE = 32
CHUNK_SIZE = 288
CHUNK_OVERLAP = 72
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
cross_encoder = CrossEncoder(RERANKER_MODEL)

def custom_template(doc: Document) -> str:
    return f"{doc.metadata.get('source', '')}: {doc.page_content}"

def count_tokens(text: str) -> int:
    return len(TOKENIZER.encode(text))

def rerank_docs(query: str, docs: List[Document], top_n: int = RERANK_TOP_N) -> List[Document]:
    if not docs:
        return docs
    pairs = [[query, doc.page_content] for doc in docs]
    scores = cross_encoder.predict(pairs)
    scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_n]]

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
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    ),
    embedding_cfg={
        "class": HuggingFaceEmbeddings,
        "model_name": EMBED_MODEL,
        "model_kwargs": {"device": "cpu"},
        "encode_kwargs": {"normalize_embeddings": True, "batch_size": BATCH_SIZE},
    },
    vector_store_cfg={"type": "faiss", "batch_size": BATCH_SIZE},
    search_cfg={"type": "mmr", "k": RETRIEVAL_K},
    enable_gpu_search=False,
    document_template=custom_template,
)

INSTRUCTIONS = """
You are a technical documentation assistant.
Answer the user's question using only the provided context chunks.

Rules:
- If the context is insufficient, say you do not know based on the provided context.
- Keep the answer concise and factual.
- Do not invent APIs, arguments, or behaviors not supported by context.
"""

def preprocess_fn(
    batch: Dict[str, List[str]], rag: RFLangChainRagSpec, prompt_manager=None
) -> Dict[str, List]:
    all_context_docs = rag.get_context(batch_queries=batch["question"], serialize=False)
    all_context_docs = [
        rerank_docs(question, docs) for question, docs in zip(batch["question"], all_context_docs)
    ]

    serialized_context = []
    kept_docs_per_query = []

    for docs in all_context_docs:
        clipped_context, kept_docs = clip_docs_to_context_budget(docs)
        serialized_context.append(clipped_context)
        kept_docs_per_query.append(kept_docs)

    return {
        "prompts": [
            [
                {"role": "system", "content": INSTRUCTIONS},
                {
                    "role": "user",
                    "content": f"Question:\n{question}\n\nContext:\n{context}\n\nAnswer:",
                },
            ]
            for question, context in zip(batch["question"], serialized_context)
        ],
        "retrieved_docs": kept_docs_per_query,
        "retrieved_contexts": serialized_context,  # added
        **batch,
    }

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project 1 RAG pipeline entrypoint")
    parser.add_argument("--input", required=True, help="Input JSON path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    return parser.parse_args()

def make_openai_client() -> OpenAI:
    return OpenAI(
        api_key=TRITON_API_KEY,
        base_url="https://tritonai-api.ucsd.edu",
        max_retries=2,
    )

_file_cache: dict[str, str] = {}

def _get_text(path: str) -> str:
    if path not in _file_cache:
        _file_cache[path] = Path(path).read_text(encoding="utf-8")
    return _file_cache[path]

def extract_sources_from_docs(docs: List) -> List[Dict[str, Any]]:
    seen = set()
    sources = []

    for doc in docs:
        source_path = doc.metadata.get("source", "")
        if not source_path:
            continue

        text = _get_text(source_path)
        start_char = int(doc.metadata.get("start_index", 0))
        end_char = min(start_char + len(doc.page_content), len(text))
        startline = text.count("\n", 0, start_char) + 1
        endline = text.count("\n", 0, end_char) + 1
        filename = Path(source_path).name

        key = (filename, startline, endline)
        if key not in seen:
            seen.add(key)
            if startline > endline:
                startline, endline = endline, startline
            sources.append({"file": filename, "lines": [startline, endline]})

    return sources

def generate_answer(client: OpenAI, question: str) -> Dict[str, Any]:
    batch = {"question": [question]}
    prepped = preprocess_fn(batch=batch, rag=rag, prompt_manager=None)

    messages = prepped["prompts"][0]
    retrieved_docs = prepped["retrieved_docs"][0]
    retrieved_context = prepped["retrieved_contexts"][0]  # added

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_completion_tokens=MAX_COMPLETION_TOKENS,
    )
    answer = response.choices[0].message.content or ""
    sources = extract_sources_from_docs(retrieved_docs)

    return {  # added retrieved_context
        "answer": answer.strip(),
        "sources": sources,
        "retrieved_context": retrieved_context,
    }

def write_output(output_path: Path, rows: List[Dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
        f.write("\n")

def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    questions = load_input_questions(input_path)
    client = make_openai_client()
    rag.build_pipeline()

    rows = []
    for row in questions:
        if "id" in row and "question_id" not in row:
            row["question_id"] = row["id"]

        result = generate_answer(client=client, question=row["question"])

        rows.append(
            {
                "question_id": row["question_id"],
                "answer": result["answer"],
                "sources": result["sources"],
                "retrieved_context": result["retrieved_context"],  # added
            }
        )

    write_output(output_path=output_path, rows=rows)

if __name__ == "__main__":
    main()

