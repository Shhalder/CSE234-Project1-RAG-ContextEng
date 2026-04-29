import argparse
import os
from pathlib import Path
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# Triton API key
TRITON_API_KEY = os.getenv("TRITON_API_KEY")
if not TRITON_API_KEY:
    TRITON_API_KEY = Path("~/api-key.txt").expanduser().read_text(encoding="utf-8").splitlines()[0].strip()


# import RapidFire classes for multi-config
import json
from typing import Any, Dict, List

from rapidfireai.automl import RFLangChainRagSpec, RFOpenAIAPIModelConfig
from openai import OpenAI
import tiktoken


# load input file
def load_input_questions(input_path: Path) -> List[Dict[str, Any]]:
    data = json.loads(input_path.read_text(encoding="utf-8"))

    # error handling
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list.")
    for row in data:
        if not isinstance(row, dict):
            raise ValueError("Each item must be an object.")
        if "id" in row and "question_id" not in row:
            row["question_id"] = row["id"]
        if "question_id" not in row or "question" not in row:
            raise ValueError("Each item must have question_id and question.")
    return data # return Python list




# define RAG pipeline
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_huggingface import HuggingFaceEmbeddings

BATCH_SIZE = 32
DOCS_DIR = "data/sourcedocs"
MODEL_NAME = "api-mistral-small-3.2-2506"
MAX_COMPLETION_TOKENS = 512
TOTAL_BUDGET_TOKENS = 2000 # total budget for input prompt
RESERVED_NON_CONTEXT_TOKENS = 400 # reserved for instruction prompt, user question, and scaffolding
MAX_CONTEXT_TOKENS = TOTAL_BUDGET_TOKENS - RESERVED_NON_CONTEXT_TOKENS
TOKENIZER = tiktoken.get_encoding("gpt2")

def custom_template(doc: Document) -> str:
    return f"{doc.metadata.get('source', '')}: {doc.page_content}"


def count_tokens(text: str) -> int:
    return len(TOKENIZER.encode(text))

# ensure retirved context stays within context limit (input: list of retrieved chunks)
def clip_docs_to_context_budget(docs: List[Document]) -> tuple[str, List[Document]]:
    selected_docs: List[Document] = []
    context_parts: List[str] = []
    used_tokens = 0
    sep_tokens = count_tokens("\n\n") # count tokens for separators

    for doc in docs: # iterate through chunks
        part = custom_template(doc)
        part_tokens = count_tokens(part)
        additional = part_tokens + (sep_tokens if context_parts else 0)

        if used_tokens + additional > MAX_CONTEXT_TOKENS: # if above budget, break
            break

        context_parts.append(part)
        selected_docs.append(doc)

        used_tokens += additional # accumulate tokens

    # Ensure at least one chunk is present even if first chunk exceeds budget.
    if not context_parts and docs:
        first = custom_template(docs[0])
        clipped = TOKENIZER.decode(TOKENIZER.encode(first)[:MAX_CONTEXT_TOKENS])
        context_parts = [clipped]
        selected_docs = [docs[0]]

    return "\n\n".join(context_parts), selected_docs # return prompt-ready string of kep chunks, subset of chunks that fit budget

reranker = CrossEncoderReranker(
    model=HuggingFaceCrossEncoder(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    ),
    top_n=5
)
rag = RFLangChainRagSpec( # RAG
    document_loader=DirectoryLoader( # load document
        path=DOCS_DIR,
        glob="*.rst",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        sample_seed=42,
    ),
    text_splitter=RecursiveCharacterTextSplitter.from_tiktoken_encoder( # LangChain splits by large boundaries first then smaller ones
        encoding_name="gpt2", chunk_size=256, chunk_overlap=64, add_start_index=True, # KNOBS: chunk size, chunk overlap
    ), # add_start_index=True adds a start_index for each chunk
    embedding_cfg={
        "class": HuggingFaceEmbeddings, # KNOB: embedding method?
        "model_name":"BAAI/bge-small-en-v1.5",
        "model_kwargs": {"device": "cpu"},
        "encode_kwargs": {"normalize_embeddings": True, "batch_size": BATCH_SIZE}, # how many chunks are embedded
    },
    vector_store_cfg={"type": "faiss", "batch_size": BATCH_SIZE}, # KNOBS: vector store, how many chunks inserted
    search_cfg={"type": "mmr", "k": 20}, # KNOBS: search configuration types
    reranker=reranker,
    enable_gpu_search=False,
    document_template=custom_template,
)

# instruction prompt
INSTRUCTIONS = """
You are a technical documentation assistant for Rapidfire AI.
Answer ONLY using the provided text.  Be specific and include exact values, 
parameter names, and commands when available.
Rules:
- Quote exact values from docs when answering factual questions
- If the context is insufficient, say you do not know based on the provided context.
- Keep the answer concise and factual.
- Do not invent APIs, arguments, or behaviors not supported by context.
""" # KNOB: experiment with different prompts (longer, more precise, etc)


# prepare inputs (takes raw questions and returns structured prompts and chunk objects)
def preprocess_fn(
    batch: Dict[str, List[str]], rag: RFLangChainRagSpec, prompt_manager=None
) -> Dict[str, List]:
    # retrieve top relevant .rst chunks so we can keep source metadata for output sources
    all_context_docs = rag.get_context(batch_queries=batch["question"], serialize=False)

    # convert retrieved docs to context strings with explicit per-query token budget clipping
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
        "retrieved_docs": kept_docs_per_query,  # keep for sources extraction later
        **batch,
    }



# attach RAG to LLM
openai_config = RFOpenAIAPIModelConfig(
    client_config={
        "api_key": TRITON_API_KEY,
        "base_url": "https://tritonai-api.ucsd.edu",
        "max_retries": 2,
    },
    model_config={
        "model": MODEL_NAME,
        "max_completion_tokens": MAX_COMPLETION_TOKENS, # cap on how many tokens the model can generate for output
    },
    rpm_limit=120,
    tpm_limit=1_000_000,
    rag=rag,
    prompt_manager=None,
)







# input/output
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

# extract evidence metadata for output (file name, start line, end line)
def extract_sources_from_docs(docs: List) -> List[List[Any]]:
    seen = set()
    sources = []

    for doc in docs:
        source_path = doc.metadata.get("source", "")
        if not source_path:
            continue

        text = _get_text(source_path)

        # start/end line calculation
        start_char = int(doc.metadata.get("start_index", 0)) # set starting character to start_index
        end_char = min(start_char + len(doc.page_content), len(text)) # startig character plus length of chunk
        startline = text.count("\n", 0, start_char) + 1 # counts newlines before starting character
        endline = text.count("\n", 0, end_char) + 1 # counts newlines until ending character


        filename = Path(source_path).name

        key = (filename, startline, endline)
        if key not in seen: # ensures each source entry is only added once
            seen.add(key)
            if startline > endline:
                startline, endline = endline, startline
        sources.append([filename, startline, endline])

    return sources


# one full query pass
def generate_answer(client: OpenAI, question: str) -> Dict[str, Any]:
    batch = {"question": [question]} # single item batch
    prepped = preprocess_fn(batch=batch, rag=rag, prompt_manager=None) # retrieve top chunk, serialize into text, return prompt + retrieved chunks

    messages = prepped["prompts"][0] # extract prompt
    retrieved_docs = prepped["retrieved_docs"][0] # extract doc 

    response = client.chat.completions.create( # call LLM
        model=MODEL_NAME,
        messages=messages,
        max_completion_tokens=MAX_COMPLETION_TOKENS,
    )
    answer = response.choices[0].message.content or "" # get answer text
    sources = extract_sources_from_docs(retrieved_docs) # extract evidence metadata

    return {"answer": answer.strip(), "sources": sources, "retrieved_context": prepped["prompts"][0][1]["content"]} # return final payload


def write_output(output_path: Path, rows: List[Dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True) # error handling
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False) # serialize Python list to JSON
        f.write("\n")


def main() -> None:

    # parse CLI args
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    questions = load_input_questions(input_path) # load questions, returns list of {question_id, question}
    client = make_openai_client()
    rag.build_pipeline() # initialize API client

    rows = [] # initialize empty Python list of outputs {question_id:, answer:, sources: }

    for row in questions: # iterate through questions
        if "id" in row and "question_id" not in row:
            row["question_id"] = row["id"]

        result = generate_answer(client=client, question=row["question"]) # answer and sources saved in result

        rows.append(
            {
                "question_id": row["question_id"],
                "answer": result["answer"],
                "sources": result["sources"],
        "retrieved_context": result["retrieved_context"],
            }
        )

    write_output(output_path=output_path, rows=rows) # write output


if __name__ == "__main__":
    main()
