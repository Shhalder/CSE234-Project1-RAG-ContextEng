SciFact: RAG for Scientific Claim Verification
=======================

Please check out the tutorial notebook on the link below. Right click on the GitHub link to save that file locally.

RAG for scientific claim verification: `View on GitHub <https://github.com/RapidFireAI/rapidfireai/blob/main/tutorial_notebooks/rag-contexteng/rf-tutorial-scifact-full-evaluation.ipynb>`__. 

This use case notebook features an all-closed model API workflow, with Open AI calls used for both embedding for generation. So, you do not need a GPU to run this notebook.


Task, Dataset, and Prompt
-------

This tutorial shows Retrieval-Augmented Generation (RAG) for verifying scientific claims against evidence.

It uses the "SciFact" dataset from the BEIR benchmark; 
`see its details here <https://github.com/allenai/scifact>`__. 
The dataset contains scientific claims that must be labeled as SUPPORT, CONTRADICT, or NOINFO based on retrieved evidence.

The prompt format includes system instructions defining the verification task with an example, 
retrieved evidence documents with titles, and the scientific claim to verify.


Model, RAG Components, and Configuration Knobs
-------

We compare 2 generator models via OpenAI API: gpt-5-mini and gpt-4o.

There are 2 different retrieval/search strategies: similarity search and maximum marginal relevance (MMR).

The RAG pipeline uses:

- **Embeddings**: OpenAI text-embedding-3-small.
- **Vector Store**: FAISS with CPU-based exact search, i.e., no ANN approximation.
- **Chunking**: 512-token chunks with 32-token overlap using recursive character splitting with tiktoken encoding.
- **Retrieval**: Top-15 initial retrieval.
- **Reranking**: cross-encoder/ms-marco-MiniLM-L6-v2 with top-5 final documents.
- **Document Template**: Custom template including document titles with content.

All other knobs are fixed across all configs. Thus, there are a total of 4 combinations launched 
with a simple grid search: 2 generator models x 2 search strategies.
