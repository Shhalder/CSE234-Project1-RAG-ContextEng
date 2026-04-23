GRPO for Math Reasoning
=======================

Please check out the tutorial notebooks on the links below. Right click on the GitHub link to save that file locally.

DPO for alignment: `View on GitHub <https://github.com/RapidFireAI/rapidfireai/blob/main/tutorial_notebooks/post-training/rf-tutorial-grpo-mathreasoning.ipynb>`__. 
Use this version if your GPU has >= 80 GB HBM.

Lite version: `View on GitHub <https://github.com/RapidFireAI/rapidfireai/blob/main/tutorial_notebooks/post-training/rf-tutorial-grpo-mathreasoning-lite.ipynb>`__. 
Use this version if your GPU has < 80 GB HBM; it just uses smaller LLMs and finishes faster. 



Task, Dataset, and Prompt
-------

This tutorial shows Group Relative Policy Optimization (GRPO) to improve mathematical reasoning capabilities. 
GRPO is an RL approach that uses multiple reward functions to provide richer training signals.

It uses the GSM8K mathematical reasoning dataset;
`see its details on Hugging Face <https://huggingface.co/datasets/openai/gsm8k>`__.
We use a sample of 500 training examples and 100 evaluation examples for tractable demo runtimes.

The prompt format includes a system message instructing the model to respond with structured reasoning
and answer tags, encouraging step-by-step mathematical problem solving with clear formatting.


Model, Adapter, and Trainer Knobs
-------

We compare 3 different base model architectures: Llama-3.1-8B-Instruct, Qwen2.5-3B-Instruct, 
and Qwen2.5-7B-Instruct, all using 4-bit quantization for efficient training.

All models use the same medium capacity LoRA configuration, targeting only 2 modules. 
We compare two different learning rates for the smaller Qwen model alone.
This results in 4 total combinations launched with a simple grid search.

There are 5 custom reward functions that collectively shape the model's behavior. 
The whole set of reward functions is used for all configs. 

* Correctness reward: Awards 2.0 points for matching the ground truth answer exactly.
* Integer reward: Awards 0.5 points for producing numeric answers (validates output format).
* Strict format reward: Awards 0.5 points for exact XML formatting compliance.
* Soft format reward: Awards 0.5 points for flexible XML formatting (more lenient matching).
* XML count reward: Fine-grained reward (up to 0.5 points) for proper XML tag usage and structure.

The lite version uses two smaller architectures: Qwen2.5-0.5B-Instruct and Llama-3.2-1B-Instruct, 
both still using 4-bit quantization. LoRA capacity is reduced with rank 16.