FiQA: RAG for Financial Opinion Q&A Chatbot
=======================

Please check out the tutorial notebook on the link below. Right click on the GitHub link to save that file locally.

RAG for financial opinion Q&A chatbot: `View on GitHub <https://github.com/RapidFireAI/rapidfireai/blob/main/tutorial_notebooks/rag-contexteng/rf-tutorial-rag-fiqa.ipynb>`__. 

This use case notebook features an all-self-hosted open model workflow, with models from Hugging Face for both embedding and generation.

Or run this pre-configured Google Colab notebook on your browser; no installation required on your machine: 
`RapidFire AI RAG on Google Colab <https://tinyurl.com/rapidfireai-rag-colab>`_


Task, Dataset, and Prompt
-------

This tutorial shows Retrieval-Augmented Generation (RAG) for creating a financial opinion Q&A chatbot.

It uses the "FiQA" dataset from the BEIR benchmark; 
`see its details here <https://sites.google.com/view/fiqa/>`__. 
The dataset contains financial questions and a corpus of documents for retrieval.

The prompt format includes system instructions defining the assistant as a financial advisor 
and incorporates retrieved context along with user queries.


Model, RAG Components, and Configuration Knobs
-------

We compare 2 generator model sizes: Qwen2.5-0.5B-Instruct and Qwen2.5-3B-Instruct.

There are 2 different chunking strategies: 256-token chunks and 128-token chunks, both with 
32-token overlap using recursive character splitting with tiktoken encoding.

The RAG pipeline uses:

- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2 with GPU acceleration.
- **Vector Store**: FAISS with GPU-based exact search, i.e., no ANN approximation.
- **Retrieval**: Top-15 similarity search.
- **Reranking**: cross-encoder/ms-marco-MiniLM-L6-v2 with 2 different top-n values: 2 and 5.

All other knobs are fixed across all configs. Thus, there are a total of 8 combinations launched 
with a simple grid search: 2 generator models x 2 chunk sizes x 2 reranking top-n values.


External Vector Stores: Pinecone and PGVector
-------

RapidFire AI also supports external persistent vector stores beyond the default in-memory FAISS.
This allows you to scale to larger corpora, persist indexes across runs and experiments, and leverage managed vector DBMS services.
As of this writing, **Pinecone** (hosted serverless or pod-based) and **PostgreSQL PGVector** (self-hosted or managed) are supported.

Each external store supports three modes of operation:

- **Create mode:** Build a new index from base documents from within RapidFire AI itself and use it for RAG.
- **Read mode:** Retrieve from a pre-existing index and use it for RAG. 
- **Update mode:** Add new content to an existing index from additional base documents from within RapidFire AI itself and use it for RAG. 

See the :doc:`API: LangChain RAG Spec page</ragspecs>` for more details on how to specify these external vector stores.

The FiQA RAG tutorial notebooks have also been extended to showcase the external stores as below:

- **Pinecone**: `View on GitHub <https://github.com/RapidFireAI/rapidfireai/blob/main/tutorial_notebooks/rag-contexteng/rf-tutorial-rag-fiqa-pinecone.ipynb>`__
- **PGVector**: `View on GitHub <https://github.com/RapidFireAI/rapidfireai/blob/main/tutorial_notebooks/rag-contexteng/rf-tutorial-rag-fiqa-pgvector.ipynb>`__
