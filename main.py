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
from openai import RateLimitError
from rapidfireai.automl import RFLangChainRagSpec, RFOpenAIAPIModelConfig
from sentence_transformers import CrossEncoder

# load and validate input questions from a json file (path to json question lists to list of dict rows with question_id and question keys)
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
DEFAULT_MODEL_NAME = "api-mistral-small-3.2-2506"
FALLBACK_MODEL_NAME = "api-gpt-oss-120b"
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
rag = None

# format a retrieved document into a single context string
# doc (Document) with metadata and page_content to string output in "<source>: <content>" format
def custom_template(doc: Document) -> str:
    return f"{doc.metadata.get('source', '')}: {doc.page_content}"

# count tokenizer tokens for a text string.
def count_tokens(text: str) -> int:
    return len(TOKENIZER.encode(text))

# rerank retrieved documents for a query using a cross-encoder (input: query (str), docs (list of Document), top_n (int))
# output: reranked top_n documents as a list
def rerank_docs(query: str, docs: List[Document], top_n: int = RERANK_TOP_N) -> List[Document]:
    if not docs:
        return docs
    pairs = [[query, doc.page_content] for doc in docs]
    scores = cross_encoder.predict(pairs)
    scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_n]]

# clip documents so total serialized context stays within token budget
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

# construct and return a configured rag specification for a corpus directory
# input: corpus_dir (Path) to rst source documents
# output: RFLangChainRagSpec instance
def build_rag(corpus_dir: Path) -> RFLangChainRagSpec:
    return RFLangChainRagSpec(
        document_loader=DirectoryLoader(
            path=str(corpus_dir),
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

# stitch together prompt and retrieval artifacts (input batch dict with "question" list, rag spec. output dict containing prompts, retrieved docs, retrieved contexts, and batch fields)
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

# parse required command-line arguments for this script
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project 1 RAG pipeline entrypoint")
    parser.add_argument("--input", required=True, help="Input JSON path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--corpus-dir", required=True, help="Path to sourcedocs directory")
    parser.add_argument("--apikey-txt", required=True, help="Path to Triton API key text file")
    parser.add_argument(
        "--generation-model",
        required=True,
        help="Model id for generation (used as fallback on rate limit).",
    )

    return parser.parse_args()

# read the api key from a text file.
def read_api_key(api_key_path: Path) -> str:
    first_line = api_key_path.read_text(encoding="utf-8").splitlines()[0].strip()
    if not first_line:
        raise ValueError("API key file is empty.")
    return first_line

# create an openai client configured for tritonai gateway
def make_openai_client(api_key: str) -> OpenAI:
    return OpenAI(
        api_key=api_key,
        base_url="https://tritonai-api.ucsd.edu",
        max_retries=2,
    )

_file_cache: dict[str, str] = {}

# load and cache file text for later line-number mapping
def _get_text(path: str) -> str:
    if path not in _file_cache:
        _file_cache[path] = Path(path).read_text(encoding="utf-8")

    return _file_cache[path]


# convert retrieved docs into unique source file/line spans (input: docs (list of retrieved document chunks, output: list of source dicts in format)
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


# run retrieval plus llm generation for one question.
# input is client (OpenAI), question (str), generation_model
# output is dict with answer, sources, and retrieved_context
def generate_answer(client: OpenAI, question: str, generation_model: str) -> Dict[str, Any]:

    batch = {"question": [question]}
    prepped = preprocess_fn(batch=batch, rag=rag, prompt_manager=None)

    messages = prepped["prompts"][0]
    retrieved_docs = prepped["retrieved_docs"][0]
    retrieved_context = prepped["retrieved_contexts"][0]  # added

    try:
        response = client.chat.completions.create(
            model=DEFAULT_MODEL_NAME,
            messages=messages,
            max_completion_tokens=MAX_COMPLETION_TOKENS,
        )
    except RateLimitError:
        fallback_model = generation_model or FALLBACK_MODEL_NAME
        response = client.chat.completions.create(
            model=fallback_model,
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

# write prediction rows to the output json file
def write_output(output_path: Path, rows: List[Dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
        f.write("\n")


# main function
# orchestrate cli parsing, retrieval/generation, and output writing
def main() -> None:
    global rag
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    corpus_dir = Path(args.corpus_dir)
    api_key_path = Path(args.apikey_txt).expanduser()
    generation_model = args.generation_model

    questions = load_input_questions(input_path)
    api_key = read_api_key(api_key_path)
    client = make_openai_client(api_key=api_key)

    rag = build_rag(corpus_dir=corpus_dir)
    rag.build_pipeline()

    rows = []
    for row in questions:
        if "id" in row and "question_id" not in row:
            row["question_id"] = row["id"]

        result = generate_answer(
            client=client,
            question=row["question"],
            generation_model=generation_model,
        )

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
