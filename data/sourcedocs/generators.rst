API: Generator Configs
===============

RapidFire AI supports both self-hosted open LLMs and closed model LLM APIs as the generator. 
As of this writing, it wraps around the model config of vLLM for the former and the OpenAI API for the latter. 
We plan to expand support for more generator plugins, including Gemini and Claude APIs, based on feedback. 


RFvLLMModelConfig
------

This is a wrapper around vLLM's :class:`config` and :class:`SamplingParams` classes. 
The full list of their arguments are available on `this page <https://docs.vllm.ai/en/latest/api/vllm/config/index.html>`__ 
and `this page <https://docs.vllm.ai/en/v0.6.4/dev/sampling_params.html>`__, respectively.

The difference here is that the individual arguments (knobs) can be :class:`List` valued or 
:class:`Range` valued in an :class:`RFvLLMModelConfig`. 
That is how you can specify a base set of knob combinations from which a config group can 
be produced. Also read :doc:`the Multi-Config Specification page</configs>`.

.. py:class:: RFvLLMModelConfig

  :param model_config: A dictionary with key-value pairs necessary for vLLM-based generation by a self-hosted LLM. All knobs given in this dictionary are simply passed to vLLM as is. vLLM will use its defaults for unspecified knobs. We recommend listing at least the following knobs.
  
    * :code:`"model"`: Name or path of the Hugging Face model to use, e.g., "Qwen/Qwen2.5-0.5B-Instruct".
    * :code:`"dtype"`: Data type for model weights and activations, e.g., "half", "float", "bfloat16".
    * :code:`"distributed_executor_backend"`: Backend to use for distributed model workers, either "ray" or "mp" (multiprocessing). Only "mp" supported for now.
    * :code:`"max_model_len"`: Model context length (prompt and output). If unspecified, will be automatically derived from the model config.
    
  :type model_config: dict[str, Any]

  :param sampling_params: A dictionary with key-value pairs to control the sampling behavior during text generation by vLLM with a self-hosted LLM. All knobs given in this dictionary are simply passed to vLLM as is. vLLM will use its defaults for unspecified knobs. We recommend listing at least the following knobs.

    * :code:`"temperature"`: Float that controls the randomness of the sampling. Lower values make the model more deterministic, while higher values make the model more random. 0.0 means greedy sampling. Default is 1.0.
    * :code:`"top_p"`: Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. Set to 1 to consider all tokens. Default is 1.0.
    * :code:`"max_tokens"`: Maximum number of tokens to generate per output sequence. 

  :type sampling_params: dict[str, Any]

  :param rag: An instance of a RapidFire AI RAG pipeline spec.  Also read :doc:`the API: RFLangChainRagSpec page </ragspecs>`.
  :type rag: RFLangChainRagSpec

  :param prompt_manager: An instance of a RapidFire AI PromptManager. Also read :doc:`the API: Prompt Manager and Other Eval Config Knobs page </promptothers>`.
  :type prompt_manager: PromptManager

  .. seealso::
     - `vLLM Config API Reference <https://docs.vllm.ai/en/latest/api/vllm/config/index.html>`_
     - `vLLM Sampling Params API Reference <https://docs.vllm.ai/en/v0.6.4/dev/sampling_params.html>`_
     - :doc:`RapidFire AI API: RFLangChainRagSpec </ragspecs>`


**Examples:**

.. code-block:: python

	# Based on FiQA chatbot tutorial notebook
	vllm_config1 = RFvLLMModelConfig(
		model_config={
			"model": "Qwen/Qwen2.5-0.5B-Instruct",
			"dtype": "half",
			"gpu_memory_utilization": 0.7,
			"tensor_parallel_size": 1,
			"distributed_executor_backend": "mp",
			"enable_chunked_prefill": True,
			"enable_prefix_caching": True,
			"max_model_len": 2048,
			"disable_log_stats": True,  # Disable vLLM progress logging
		},
		sampling_params={
			"temperature": 0.8,
			"top_p": 0.95,
			"max_tokens": 512,
		},
		rag=rag_gpu,
		prompt_manager=None,
	)



RFOpenAIAPIModelConfig
------

This is a wrapper around OpenAI's API client config and chat completion parameters. 
The full list of their arguments are available on `this page <https://platform.openai.com/docs/api-reference/chat/create>`__.

The difference here is that the individual arguments (knobs) can be :class:`List` valued or 
:class:`Range` valued in an :class:`RFOpenAIAPIModelConfig`. 
That is how you can specify a base set of knob combinations from which a config group can 
be produced. Also read :doc:`the Multi-Config Specification page</configs>`.

.. py:class:: RFOpenAIAPIModelConfig

  :param client_config: A dictionary necessary for initializing the AsyncOpenAI client. All knobs given in this dictionary are simply passed to the AsyncOpenAI client as is. We recommend listing at least the following knobs.
  
    * :code:`"api_key"`: Your OpenAI API key for authentication. Note that we are NOT able to provide a publicly visible API key.
    * :code:`"max_retries"`: Maximum number of retry attempts for failed API calls. Default is 2.
    * :code:`"timeout"`: Request timeout in seconds. Optional.
    
  :type client_config: dict[str, Any]

  :param model_config: A dictionary to control the chat completion behavior with OpenAI's Chat Completions API. All knobs given in this dictionary are simply passed to the OpenAI API as is. The API will use its defaults for unspecified knobs. We recommend listing at least the following knobs.

    * :code:`"model"`: Name of the OpenAI model to use, e.g., "gpt-5-mini".
    * :code:`"temperature"`: What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic. OpenAI recommends altering this or :code:`"top_p"` but not both.
    * :code:`"max_completion_tokens"`: Upper bound for the number of tokens that can be generated for a completion, including visible output tokens and reasoning tokens.
    * :code:`"reasoning_effort"`: Constrains effort for reasoning models. Currently supported values are "minimal", "low", "medium", and "high". Reducing reasoning effort can result in faster responses and fewer tokens used on reasoning in a response. The gpt-5-pro model defaults to (and only supports) "high" reasoning effort.
	* :code:`"top_p"`: Alternative to temperature-based sampling called nucleus sampling. The model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered. OpenAI recommends altering this or :code:`"temperature"` but not both.

  :type model_config: dict[str, Any]

  :param rpm_limit: Rate limit for requests per minute to the OpenAI API. Used for throttling to avoid exceeding Open AI API quotas. Check the rate limit published by Open AI for details on your tier and the latest per-model limits on `this page <https://platform.openai.com/docs/guides/rate-limits>`__.
  :type rpm_limit: int

  :param tpm_limit: Rate limit for tokens per minute to the OpenAI API. Used for throttling to avoid exceeding API quotas. See the rate limit page above for details.
  :type tpm_limit: int

  :param rag: An instance of a RapidFire AI RAG pipeline spec. Also read :doc:`the API: RFLangChainRagSpec page </ragspecs>`.
  :type rag: RFLangChainRagSpec

  :param prompt_manager: An instance of a RapidFire AI PromptManager. Also read :doc:`the API: Prompt Manager and Other Eval Config Knobs page </promptothers>`.
  :type prompt_manager: PromptManager

  .. seealso::
     - `OpenAI Chat Completions API Reference <https://platform.openai.com/docs/api-reference/chat/create>`_
     - `OpenAI Python Client Documentation <https://github.com/openai/openai-python>`_
     - :doc:`API: RFLangChainRagSpec </ragspecs>`
     - :doc:`API: Prompt Manager and Other Eval Config Knobs page </promptothers>`


**Example:**

.. code-block:: python

	# Based on GSM8K chatbot tutorial notebook; specify your OPENAI_API_KEY beforehand
	openai_config1 = RFOpenAIAPIModelConfig(
		client_config={"api_key": OPENAI_API_KEY, "max_retries": 2},
		model_config={
			"model": "gpt-5-mini",
			"max_completion_tokens": 1024,
			"reasoning_effort": "medium", 
		},
		rpm_limit=500,
		tpm_limit=500_000,
		rag=None,
		prompt_manager=fewshot_prompt_manager,
	)	

