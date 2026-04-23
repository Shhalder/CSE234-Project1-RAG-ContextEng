DPO for Alignment
=======================

Please check out the tutorial notebooks on the links below. Right click on the GitHub link to save that file locally.

DPO for alignment: `View on GitHub <https://github.com/RapidFireAI/rapidfireai/blob/main/tutorial_notebooks/post-training/rf-tutorial-dpo-alignment.ipynb>`__. 
Use this version if your GPU has >= 80 GB HBM.

Lite version: `View on GitHub <https://github.com/RapidFireAI/rapidfireai/blob/main/tutorial_notebooks/post-training/rf-tutorial-dpo-alignment-lite.ipynb>`__. 
Use this version if your GPU has < 80 GB HBM; it just uses smaller LLMs and finishes faster. 



Task, Dataset, and Prompt
-------

This tutorial shows Direct Preference Optimization (DPO) for aligning LLMs with human preferences. 
DPO is a simpler alternative to PPO that directly optimizes the policy model using preference 
data without requiring a separate reward model.

It uses the "ultrafeedback binarized" dataset;
`see its details on Hugging Face <https://huggingface.co/datasets/trl-lib/ultrafeedback_binarized>`__.
We use a sample of 500 training examples for tractable demo runtimes. 

The dataset contains paired preference data with chosen (preferred) and rejected (dispreferred) 
responses for the same prompts.

The training starts from a pre-trained SFT model (:code:`rapidfire-ai-inc/mistral-7b-sft-bnb-4bit`) 
to ensure the model distribution is suitable for DPO alignment training.


Model, Adapter, and Trainer Knobs
-------

We use the Mistral-7B-Instruct-v0.3 base model fine-tuned with 4-bit quantization (QLoRA).

There are 4 different DPO training configurations exploring various loss functions and hyperparameters:

* Basic Bradley-Terry: Standard sigmoid loss with medium capacity LoRA (rank 64) and large beta.
* High divergence: High capacity LoRA (rank 128) with small beta to encourage divergence from reference model.
* Robust loss: Uses robust loss type with label smoothing to handle noisy preference data.
* Combined loss: Weighted combination of sigmoid, BCO pair, and SFT losses.

All configurations use QLoRA with the same target modules and are launched with a simple grid search, 
totaling 4 combinations.

The lite version simply uses a smaller LoRA rank of 16 and a subset of 3 configs from the above list.
