import os
from pathlib import Path
from rapidfireai.automl import RFGridSearch, RFLangChainRagSpec, RFOpenAIAPIModelConfig
from rapidfireai.experiment import Experiment
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# API key
TRITON_API_KEY = os.getenv("TRITON_API_KEY")
if not TRITON_API_KEY:
    TRITON_API_KEY = Path("~/api-key.txt").expanduser().read_text(encoding="utf-8").splitlines()[0].strip()

DOCS_DIR = "data/sourcedocs"
MODEL_NAME = "api-mistral-small-3.2-2506"

# 8 configs varying chunk_size, chunk_overlap, k, embedding model
configs = [
    {"chunk_size": 256,  "chunk_overlap": 32, "k": 5,  "model": "sentence-transformers/all-MiniLM-L6-v2"},
    {"chunk_size": 512,  "chunk_overlap": 32, "k": 5,  "model": "sentence-transformers/all-MiniLM-L6-v2"},
    {"chunk_size": 1024, "chunk_overlap": 32, "k": 5,  "model": "sentence-transformers/all-MiniLM-L6-v2"},
    {"chunk_size": 512,  "chunk_overlap": 64, "k": 5,  "model": "sentence-transformers/all-MiniLM-L6-v2"},
    {"chunk_size": 512,  "chunk_overlap": 32, "k": 3,  "model": "sentence-transformers/all-MiniLM-L6-v2"},
    {"chunk_size": 512,  "chunk_overlap": 32, "k": 10, "model": "sentence-transformers/all-MiniLM-L6-v2"},
    {"chunk_size": 512,  "chunk_overlap": 32, "k": 5,  "model": "BAAI/bge-small-en-v1.5"},
    {"chunk_size": 256,  "chunk_overlap": 64, "k": 10, "model": "BAAI/bge-small-en-v1.5"},
]

# load eval questions
import json
eval_data = json.loads(Path("data/evaluation-set-queries.json").read_text())
# normalize id
for row in eval_data:
    if "id" in row and "question_id" not in row:
        row["question_id"] = row["id"]
questions = [row["question"] for row in eval_data]

# run each config
for i, cfg in enumerate(configs):
    print(f"\n{'='*50}")
    print(f"Config {i+1}/8: {cfg}")
    print('='*50)

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
            "model_name": cfg["model"],
            "model_kwargs": {"device": "cpu"},
            "encode_kwargs": {"normalize_embeddings": True},
        },
        vector_store_cfg={"type": "faiss"},
        search_cfg={"type": "similarity", "k": cfg["k"]},
        enable_gpu_search=False,
    )

    rag.build_pipeline()

    # retrieve chunks for all questions
    results = rag.get_context(batch_queries=questions, serialize=True)

    # save results
    output = []
    for row, context in zip(eval_data, results):
        output.append({
            "question_id": row["question_id"],
            "question": row["question"],
            "context": context,
        })

    out_path = Path(f"logs/config_{i+1}_results.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"Saved to {out_path}")

print("\nAll configs done!")