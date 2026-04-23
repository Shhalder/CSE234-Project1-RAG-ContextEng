API: User-Provided Functions for Run Fit
===============

Apart from the classes to define models, adapters, and trainers, users can create the following custom 
functions as part of their experiments.


Create Model Function
------

Mandatory user-provided function to create HuggingFace model and tokenizer objects based on the  
model type(s) and name(s) given in the :code:`RFModelConfig` and multi-config specification. 
Also read :doc:`the LoRA and Model Configs page</models>`.
A model can be imported from the Hugging Face model hub or read from a local checkpoint file. 

It is passed to :func:`run_fit()` directly. Also read :doc:`the Experiment page</experiment>`.

This function is invoked when a trainer object is created for each run. 


.. py:function:: create_model_fn(model_config: Dict[str, Any]) -> Tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizer]

  :param model_config: Dictionary injected by RapidFire AI into this user-defined function with all key-value pairs for one model config output by the config-group generator.
  :type model_config: Dict[str, Any]

  :return: Tuple containing the initialized Hugging Face model (e.g., ``AutoModelForCausalLM``, ``AutoModelForSequenceClassification``) and tokenizer (e.g., ``AutoTokenizer``, ``PreTrainedTokenizer``) objects
  :rtype: Tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizer]


**Example:**

.. code-block:: python

	# From the SFT tutorial notebook
	def sample_create_model(model_config):
    	"""Function to create model object for any given config; must return tuple of (model, tokenizer)"""
		from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForMaskedLM
		
		model_name = model_config["model_name"]
		model_type = model_config["model_type"]
		model_kwargs = model_config["model_kwargs"]
		
		if model_type == "causal_lm":
			model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
		elif model_type == "seq2seq_lm":
			model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **model_kwargs)
		elif model_type == "masked_lm":
			model = AutoModelForMaskedLM.from_pretrained(model_name, **model_kwargs)
		elif model_type == "custom":
            # Handle custom model loading logic, e.g., loading your own checkpoints
            # model = ... 
			pass
		else:
			# Default to causal LM
			model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

		tokenizer = AutoTokenizer.from_pretrained(model_name)

		return (model,tokenizer)



Compute Metrics Function
------

Optional user-provided function specifying custom evaluation metrics based on the generated 
outputs and ground truth.

It is passed to the :code:`compute_metrics` argument of :class:`RFModelConfig`. 
Also read: :doc:`the LoRA and Model Configs page</models>`.
You can create multiple variants of these functions and pass them all as a single 
:code:`List` to your :class:`RFModelConfig` to create a multi-config specification.

This function is invoked by the underlying HF trainer at a cadence controlled by the 
:code:`eval_strategy` and :code:`eval_steps` arguments.
Also read: :doc:`the Trainer Configs page</trainers>`.

.. py:function:: fit.compute_metrics_fn(eval_preds: Tuple) -> Dict[str, float]

   :param eval_preds: Tuple containing generated predictions and ground truth labels from the eval dataset.
   :type eval_preds: Tuple[List[str], List[str]]

   :return: Dictionary with user-defined metrics with names keys and numbers as values
   :rtype: Dict[str, float]


**Example:**

.. code-block:: python

	# From the SFT tutorial notebook
	def sample_compute_metrics(eval_preds):  
		"""Optional function to compute eval metrics based on predictions and labels"""
		predictions, labels = eval_preds

		# Standard text-based eval metrics: Rouge and BLEU
		import evaluate
		rouge = evaluate.load("rouge")
		bleu = evaluate.load("bleu")

		rouge_output = rouge.compute(predictions=predictions, references=labels, use_stemmer=True)
		rouge_l = rouge_output["rougeL"]
		bleu_output = bleu.compute(predictions=predictions, references=labels)
		bleu_score = bleu_output["bleu"]

		return {"rougeL": round(rouge_l, 4), "bleu": round(bleu_score, 4)}



Formatting Function
------

Optional user-provided function to format each example (row) of the dataset to construct 
the prompt and completion with relevant roles and system prompt as expected by your model. 
Apart from adding the system prompt, for conversational data it should format the user 
instruction and assistant responses as separate message dictionary entries.

It is passed to the :code:`formatting_func` argument of :class:`RFModelConfig`. 
Also read: :doc:`the LoRA and Model Configs page</models>`.
You can create multiple variants of these functions and pass them all as a single 
:code:`List` to your :class:`RFModelConfig` to create a multi-config specification.

This function is invoked by the underlying HF trainer on all examples of the train dataset 
and (if given) eval dataset on the fly.


.. py:function:: sample_formatting_fn(row: Dict[str, Any]) -> Dict[str, List[Dict[str, str]]]

   :param row: Dictionary containing a single data example with keys like "instruction" and "response"
   :type row: Dict[str, Any]

   :return: Dictionary with "prompt" and "completion" keys, each containing a list of chat messages with "role" and "content" fields
   :rtype: Dict[str, List[Dict[str, str]]]


**Example:**

.. code-block:: python

	# From the SFT tutorial notebook
	def sample_formatting_function(row):
		"""Function to preprocess each row from dataset"""
		# Special tokens for formatting
		SYSTEM_PROMPT = "You are a helpful and friendly customer support assistant. Please answer the user's query to the best of your ability."
		return {
			"prompt": [
				{"role": "system", "content": SYSTEM_PROMPT},
				{"role": "user", "content": row["instruction"]},
			],
			"completion": [
				{"role": "assistant", "content": row["response"]}
			]
		}



Reward Functions
------

User-provided reward function(s) needed for GRPO. You can create as many reward functions as you 
like with custom names.

A list of such functions is passed to the :code:`reward_funcs` argument of :class:`RFModelConfig`. 
Also read: :doc:`the LoRA and Model Configs page</models>`.
You can create multiple variants of this list with different subsets of functions and pass them 
all as a single :code:`List` to your :class:`RFModelConfig` to create a multi-config specification.

These functions are invoked by the underlying HF trainer on the generated outputs on the fly.


.. py:function:: reward_function(prompts, completions, completions_ids, trainer_state, **kwargs) -> List[float]

	:param prompts: List of input prompts that produced the completions.
	:type prompts: List[str] | List[List[Dict[str, str]]]

	:param completions: List of generated completions corresponding to above prompts.
	:type completions: List[str] | List[List[Dict[str, str]]]

	:param completions_ids: List of tokenized completions (token IDs) corresponding to each completion.
	:type completions_ids: List[List[int]]

	:param trainer_state: Current state of the trainer. Useful for implementing dynamic reward functions like curriculum learning where rewards adjust based on training progress.
	:type trainer_state: transformers.TrainerState

	:param kwargs: Additional keyword arguments containing all dataset columns (except "prompt"). For example, if the dataset contains a "ground_truth" column, it will be passed as a keyword argument.
	:type kwargs: Any

	:return: List of reward scores, one per single completion.
	:rtype: List[float] | None


**Examples:**

.. code-block:: python

    # From the GRPO tutorial notebook
    def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:

        def extract_xml_answer(text: str) -> str:
            answer = text.split("<answer>")[-1]
            answer = answer.split("</answer>")[0]
            return answer.strip()

        responses = [completion[0]['content'] for completion in completions]
        q = prompts[0][-1]['content']
        extracted_responses = [extract_xml_answer(r) for r in responses]
        # x('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
        return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

    def strict_format_reward_func(completions, **kwargs) -> list[float]:
        """Reward function that checks if the completion has a specific format."""
        import re
        pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
        responses = [completion[0]["content"] for completion in completions]
        matches = [re.match(pattern, r) for r in responses]
        return [0.5 if match else 0.0 for match in matches]

**Notes:**

Note that TRL injects into a reward function lists of prompts, completions, completion IDs, and trainer 
state as keyword arguments. You can use only a subset of these in your reward function signature as 
long as you include :code:`**kwargs`, as shown in the second example above.

Depending on the dataset format, :code:`prompts` and :code:`completions` will be either lists of 
strings (standard format) or lists of message dictionaries (conversational format). 
Standard format is usually common for text completion tasks, simple Q&A, code generation, and 
mathematical reasoning.
Conversational format is needed for multi-turn conversations, chat models with system prompts, 
role-playing scenarios, and complex dialogue systems.
Make sure your reward function can handle both cases if you dataset includes both types.

The return type of every reward function must be a list of floats, one per completion. 
It can also return :code:`None` for examples when the reward function is not applicable, 
which is useful for multi-task training. 

