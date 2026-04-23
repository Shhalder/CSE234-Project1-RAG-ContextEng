SFT for Q&A Chatbot
=======================

Please check out the tutorial notebooks on the links below. Right click on the GitHub link to save that file locally.

SFT for customer support Q&A chatbot: `View on GitHub <https://github.com/RapidFireAI/rapidfireai/blob/main/tutorial_notebooks/fine-tuning/rf-tutorial-sft-chatqa.ipynb>`__. 
Use this version if your GPU has >= 80 GB HBM.

Lite version: `View on GitHub <https://github.com/RapidFireAI/rapidfireai/blob/main/tutorial_notebooks/fine-tuning/rf-tutorial-sft-chatqa-lite.ipynb>`__. 
Use this version if your GPU has < 80 GB HBM; it just uses smaller LLMs and finishes faster. 

Or run this pre-configured Google Colab notebook on your browser; no installation required on your machine: 
`RapidFire AI on Google Colab <https://tinyurl.com/rapidfireai-colab>`_


Task, Dataset, and Prompt
-------

This tutorial shows Supervised Fine-Tuning (SFT) for creating a customer support Q&A chatbot.

It uses the "Bitext customer support" dataset; 
`see its details on Hugging Face <https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset>`__. 
We use a sample of 5,000 training examples and 200 evaluation examples for tractable demo runtimes.

The prompt format includes a system message defining the assistant as "helpful and friendly customer 
support" with user instructions and assistant responses


Model, Adapter, and Trainer Knobs
-------

We compare 2 base model architectures: Llama-3.1-8B-Instruct and Mistral-7B-Instruct-v0.3. 
The lite version uses only one: TinyLlama-1.1B-Chat-v1.0.

There are 2 different LoRA adapter configurations: a low-capacity adapter (rank 16; 8 for lite) targeting 
only 2 modules and a high-capacity adapter (rank 128; 32 for lite) targeting 4 modules.

All other knobs are fixed across all configs. Thus, there are a total of 4 combinations, 
all launched with a simple grid search.


Multi-GPU Model Partitioning with FSDP
-------

RapidFire AI supports automated large model partitioning across GPUs (on the same machine) via PyTorch's native FSDP. 
Provide the relevant FSDP deatils in a config knob, optionally along with the number of GPUs to use for that run. 
The following notebooks showcase the use of FSDP for SFT with the corresponding LLMs:

* FSDP Lite with base model TinyLlama-1.1B-Chat-v1.0. Needs at least 2x A10 GPUs or equivalent (48 GB total HBM) to work. `View on GitHub <https://github.com/RapidFireAI/rapidfireai/blob/main/tutorial_notebooks/fine-tuning/rf-tutorial-sft-chatqa-fsdp-lite.ipynb>`__

* FSDP Regular with base model Qwen3-32B. Needs at least 4x A10 GPUs or equivalent (96 GB total HBM) to work. `View on GitHub <https://github.com/RapidFireAI/rapidfireai/blob/main/tutorial_notebooks/fine-tuning/rf-tutorial-sft-chatqa.ipynb>`__

* FSDP Large with base model Llama-3-70B-Instruct. Needs at least 8x A10 GPUs or equivalent (192 GB total HBM) to work. `View on GitHub <https://github.com/RapidFireAI/rapidfireai/blob/main/tutorial_notebooks/fine-tuning/rf-tutorial-sft-chatqa-fsdp-large.ipynb>`__

.. important::
  Although the above FSDP tutorial notebooks can work on cheap A10 GPUs, we highly recommend using at least A100s or later GPUs with NVLink support for reasonable runtimes.

