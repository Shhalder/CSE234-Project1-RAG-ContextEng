API: Multi-Config Specification
===============

The core value of RapidFire AI is in the ability to launch, compare, and dynamically 
control multiple configurations (configs) in one go. 
It is already common practice in AI to do hyperparameter sweeps via grid search, 
random search, or AutoML heuristics to generate knob values. 

RapidFire AI generalizes that notion to any type of config knobs, not just  
regular hyperparameters, but also base model architectures, prompt schemes, 
LoRA adapters, and optimizers (for training), as well as data chunking, embedding, 
retrieval, reranking, generation, and prompt schemes (for RAG) and any other 
user-defined knobs. 


Knob Set Generators
-------

To create a multi-config specification, you need two things: **knob set generators** for 
knob values and **config group generators** that take a config with set-valued knobs 
to generate groups of full configs.

We currently support two common knob set generators: :func:`List()` for a discrete 
set of values and :func:`Range()` for sampling from a continuous value interval.


.. py:function:: List(values: List[Any])

	:param values: List of discrete values for a knob; all values must be the same python data type.
	:type values: List[Any]


.. py:function:: Range(start: int | float, end: int | float, dtype: str = "int" | "float")

	:param start: Lower bound of range interval.
	:type start: int | float

	:param end: Upper bound of range interval.
	:type end: int | float

	:param dtype: Data type of value to be sampled, either :code:`"int"` or :code:`"float"`.
	:type dtype: str


**Notes:**

As of this writing, :func:`Range()` performs uniform sampling within the given interval. 
We plan to continue expanding this API and add more functionality on this front based on feedback.

Note that the return types of the knob set generators are internal to RapidFire AI and 
they are usable only within the context of the config group generators below.



Config Group Generators
-----

We currently support two common config group generators: :func:`RFGridSearch()` for grid search 
and :func:`RFRandomSearch()` for random search. 

More support for AutoML heuristics such as SHA, HyperOpt, as well as an integration with 
the popular AutoML library Optuna are coming soon. 
Likewise for RAG/context engineering, we also plan to support the AutoML heuristic syftr.


.. py:function:: RFGridSearch(configs: Dict[str, Any] | List[Dict[str, Any]], trainer_type: str = "SFT" | "DPO" | "GRPO" | None)

	:param configs: A config dictionary with :func:`List()` for at least one knob; can be a list of such config dictionaries too.
	:type configs: Dict[str, Any] | List[Dict[str, Any]]

	:param trainer_type: The fine-tuning/post-training control flow to use: "SFT", "DPO", or "GRPO". Skip this argument for :func:`run_evals()`.
	:type trainer_type: str, optional 


.. py:function:: RFRandomSearch(configs: Dict[str, Any], trainer_type: str = "SFT" | "DPO" | "GRPO" | None, num_runs: int, seed: int = 42)

	:param configs: A config dictionary with :func:`List()` or :func:`Range()` for at least one knob.
	:type configs: Dict[str, Any]

	:param trainer_type: The fine-tuning/post-training control flow to use: "SFT", "DPO", or "GRPO". Skip this argument for :func:`run_evals()`.
	:type trainer_type: str, optional

	:param num_runs: Number of runs (full combinations of knob values) to sample in total.
	:type num_runs: int

	:param seed: Seed for random sampling of knob values to construct combinations. Default is 42.
	:type seed: int, optional 


**Notes:**

For :func:`RFGridSearch()`, each knob can have either a single value or a :func:`List()` of values but no knob 
should have :func:`Range()` of values; otherwise, it will error out.

For :func:`RFRandomSearch()`, each knob can have either a single value, or a :func:`List()` of values, or a 
 :func:`Range()` of values. The semantics of sampling are independently-identically-distributed (IID), i.e.,
we uniformly randomly pick a value from each discrete set and from each continuous set to construct the 
knob combination for one run. 
Then we repeat that sampling process in an IID way to accumulate :code:`num_runs` distinct combinations.

Note that the return types of the config group generators are internal to RapidFire AI and they are usable only 
within the context of :func:`run_fit()` or :func:`run_evals()` in the :class:`Experiment` class.


**Examples:**

.. code-block:: python

	# Example 1: Based on SFT tutorial notebook
	from rapidfireai.automl import RFModelConfig, RFLoraConfig, RFSFTConfig, List

	peft_configs = List([
		RFLoraConfig(
			r=16, lora_alpha=32, lora_dropout=0.05, 
			target_modules=["q_proj", "v_proj"], bias="none"
		),
		RFLoraConfig(
			r=128, lora_alpha=256, lora_dropout=0.05,
			target_modules=["q_proj","k_proj", "v_proj","o_proj"], bias="none"
		)
	])

	config_group=RFGridSearch(
		configs=config_grid,
		trainer_type="SFT"
	)


	# Example 2: Based on GSM8K tutorial notebook
	from rapidfireai.automl import List, RFLangChainRagSpec, RFOpenAIAPIModelConfig, RFPromptManager, RFGridSearch

	openai_config1 = RFOpenAIAPIModelConfig(
		client_config={"api_key": OPENAI_API_KEY, "max_retries": 2},
		model_config={
			"model": "gpt-5-mini",
			"max_completion_tokens": 1024,
			"reasoning_effort": List(["medium", "high"]),  # 2 different reasoning levels
		},
		...
	)

	openai_config2 = RFOpenAIAPIModelConfig(
		client_config={"api_key": OPENAI_API_KEY, "max_retries": 2},
		model_config={
			"model": "gpt-4o",
			"max_completion_tokens": 1024,
			"reasoning_effort": List(["medium", "high"]),  # 2 different reasoning levels
		},
		...
	)

	config_set = {
		"openai_config": List(
			[openai_config1, openai_config2]
		),
		"batch_size": batch_size,
		...
	}

	config_group = RFGridSearch(config_set)