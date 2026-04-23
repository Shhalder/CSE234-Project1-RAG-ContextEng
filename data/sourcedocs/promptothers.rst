API: Prompt Manager and Other Eval Config Knobs
===============

RFPromptManager
------

This class wraps around some LangChain APIs to manage dynamic few-shot example selection. It provides semantic 
similarity-based example selection to construct prompts with the most relevant examples for each input query.

The individual arguments (knobs) can be :class:`List` valued or :class:`Range` valued in an :class:`RFPromptManager`. 
That is how you can specify a base set of knob combinations from which a config group can be produced. 
Also read :doc:`the Multi-Config Specification page</configs>`.

.. py:class:: RFPromptManager

  :param instructions: The main instructions for the prompt that guide the generator's behavior. This sets the overall task description and role for the assistant. Either this or :code:`instructions_file_path` must be provided.
  :type instructions: str, optional

  :param instructions_file_path: Path to a file containing the instructions. Use this as an alternative to the :code:`instructions` parameter for loading instructions from a file, say, if they are very long.
  :type instructions_file_path: str, optional

  :param examples: A list of example dictionaries for few-shot learning. Each example should be a dictionary with keys matching the expected input-output format (e.g., "question" and "answer").
  :type examples: list[dict[str, str]], optional


  :param embedding_cfg: The embedding class and its kwargs to use for computing semantic similarity between examples and queries, provided as a single dictionary. Must include a key :code:`"class"` with the class itself as value, not an instance. Options for the class include :class:`HuggingFaceEmbeddings` and :class:`OpenAIEmbeddings`. The kwargs that follow must contain all parameters needed to initialize the embedding class; required parameters vary by embedding class. For example, :class:`HuggingFaceEmbeddings` needs :code:`model_name`, :code:`model_kwargs` and :code:`device`, while :class:`OpenAIEmbeddings` needs :code:`"model"` and :code:`"api_key"`.
  :type embedding_cfg: dict[str, Any], optional


  :param example_selector_cls: The example selector class that determines how to choose relevant examples based on the input query. Must be either :code:`SemanticSimilarityExampleSelector` or :code:`MaxMarginalRelevanceExampleSelector` (for diversity) from LangChain.
  :type example_selector_cls: type[MaxMarginalRelevanceExampleSelector | SemanticSimilarityExampleSelector], optional

  :param example_prompt_template: A LangChain :code:`PromptTemplate` that defines how to format each example. Should specify :code:`input_variables` and a :code:`template` string with placeholders matching the keys in the examples dictionaries.
  :type example_prompt_template: PromptTemplate, optional

  :param k: Number of most similar or diverse examples to retrieve and include in the prompt for each query. Default is 3.
  :type k: int, optional

  .. seealso::
     - `LangChain Example Selectors <https://api.python.langchain.com/en/latest/core_api_reference.html#module-langchain_core.example_selectors/>`_
     - `LangChain Embeddings <https://reference.langchain.com/python/langchain/embeddings/>`_


**Example:**

.. code-block:: python

	# Based on GSM8K chatbot tutorial notebook; specify your INSTRUCTIONS beforehand
	fewshot_prompt_manager = RFPromptManager(
		instructions=INSTRUCTIONS,
		examples=examples,
		embedding_cfg={
			"class": HuggingFaceEmbeddings,
			"model_name": "sentence-transformers/all-MiniLM-L6-v2",
			"model_kwargs": {"device": "cuda:0"},
			"encode_kwargs": {"normalize_embeddings": True, "batch_size": batch_size},
    	},
		example_selector_cls=SemanticSimilarityExampleSelector,
		example_prompt_template=PromptTemplate(
			input_variables=["question", "answer"],
			template="Question: {question}\nAnswer: {answer}",
		),
		k=List([3, 5]),  # 2 different k values
	)




Other Eval Config Knobs
------

Finally, apart from the Generator, the following knobs can also be included in your eval config dictionary. Each of 
these can also be a knob set generator, viz., :func:`List()` for a discrete and :func:`Range()` for continuous knobs.

For more details on the four user-given functions listed below, see :doc:`the API: User-Provided Functions for Run Evals page</evalsfunctions>`.

For more details on the semantics of the online aggregation strategy arguments listed below, see :doc:`the Online Aggregation for Evals page</onlineagg>`.


**batch_size** : int
	Number of examples to process in one batch for GPU efficiency (if applicable)

**preprocess_fn** : Callable
	User-given function to preprocess a batch of examples; an eval config's RagSpec and PromptManager are input by the system

**postprocess_fn** : Callable, optional
	User-given function to postprocess a batch of examples and generations; a single cfg is passed as input by the system

**compute_metrics_fn** : Callable
	User-given evaluation function to compute eval metrics per batch

**accumulate_metrics_fn** : Callable, optional
	User-given evaluation function to aggregate algebraic eval metrics across batches. If this is not given, all metrics provided in :code:`eval_compute_metrics_fn` will be assumed to be distributive by default.

**online_strategy_kwargs** : dict[str, Any], optional
	Parameters for evals online aggregation strategy. The dictionary must include the following keys:
	
	* :code:`"strategy_name"` (str) - Must be :code:`"normal"`, :code:`"wilson"`, or :code:`"hoeffding"`.
	* :code:`"confidence_level"` (float) - Confidence level for confidence intervals on metrics. Must be in [0,1]. Default is 0.95 (95%).
	* :code:`"use_fpc"` (bool) - Whether to apply finite population correction. Default is :code:`True`.