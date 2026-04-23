GSM8K: Context Engineering for Math Reasoning
=======================

Please check out the tutorial notebook on the link below. Right click on the GitHub link to save that file locally.

Context engineering with few-shot prompting for GSM8K math reasoning: `View on GitHub <https://github.com/RapidFireAI/rapidfireai/blob/main/tutorial_notebooks/rag-contexteng/rf-tutorial-gsm8k-fewshot.ipynb>`__. 

This use case notebook features an hybrid workflow spanning a self-hosted open LLM for embeddings and an Open AI call for generation. 


Task, Dataset, and Prompt
-------

This tutorial shows few-shot prompting as part of context engineering for solving grade school math word problems.

It uses the "GSM8K" dataset; 
`see its details here <https://huggingface.co/datasets/openai/gsm8k>`__. 
The dataset contains grade school math word problems requiring multi-step reasoning.

The prompt format includes system instructions defining the assistant as a math problem solver, 
semantically selected few-shot examples, and the target question to solve.


Model, Few-Shot Selection, and Configuration Knobs
-------

We compare 2 generator models via OpenAI API: gpt-5-mini and gpt-4o.

There are 2 different reasoning effort levels for the first model only: medium and high.

The few-shot prompting pipeline uses:

- **Example Selection**: Semantic similarity-based selection using sentence-transformers/all-MiniLM-L6-v2 embeddings.
- **Example Pool**: 10 hand-crafted examples covering diverse problem types.
- **Few-Shot k Values**: 2 different values: 3 and 5 examples per prompt.
- **Prompt Template**: Chain-of-thought style with step-by-step reasoning and final answer after "####".

All other knobs are fixed across all configs. Thus, there are a total of 6 combinations launched 
with a union of two grids across generator, reasoning effort levels, and few-shot k values: 1 x 1 x 2 + 1 x 2 x 2 = 6.