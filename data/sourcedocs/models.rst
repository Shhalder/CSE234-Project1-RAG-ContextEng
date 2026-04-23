API: LoRA and Model Configs
===============

RapidFire AI's core APIs for model and adapter specifications are all thin wrappers around 
the corresponding APIs of the Hugging Face's libraries 
`transformers <https://huggingface.co/docs/transformers/en/index>`__ 
and `PEFT LoRA <https://huggingface.co/docs/peft/developer_guides/lora>`__.


RFLoraConfig 
------

This is a wrapper around :class:`LoraConfig` in HF PEFT. 
The full signature and list of arguments are available on `this page 
<https://huggingface.co/docs/peft/v0.17.0/en/package_reference/lora#peft.LoraConfig>`__.

The difference here is that the individual arguments (knobs) can be :class:`List` valued or 
:class:`Range` valued in a :class:`RFLoraConfig`. 
That is how you can specify a base set of knob combinations from which a config group can 
be produced. Also read :doc:`the Multi-Config Specification page</configs>`.


**Example:**

.. code-block:: python

    # Singleton config
    RFLoraConfig( 
        r=128, lora_alpha=256, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"], bias="none"
    )

    # 4 combinations
    RFLoraConfig(
        r=List([16, 32]), lora_alpha=List([16,32]), lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"], bias="none"
    )

    # 2 combinations
    RFLoraConfig( 
        r=64, lora_alpha=128,
        target_modules=List([["q_proj", "v_proj"], 
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]]), 
        bias="none"
    )

**Notes:**

In terms of impact on LLM behavior, the LoRA knobs that are usually experimented with are as follows:

* :code:`r` (rank): The most critical knob. Typically a power of 2 between 8 and 128. Higher rank means higher adapter learning capacity but slightly higher GPU memory footprint and compute time.

* `lora_alpha`: Controls adaptation strength. Typical values are 16, 32, and 64. Usually set to 2x :code:`r`, but this can be varied too.

* :code:`target_modules`: Which layers to apply LoRA to. Common options:

  * Query/Value only: `["q_proj", "v_proj"]`
  * Query/Key/Value: `["q_proj", "k_proj", "v_proj"]`
  * All linear layers: `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`

* :code:`lora_dropout`:  Controls regularization. Typically between 0.0 to 0.05; often set at 0.0 unless there is overfitting.

* :code:`init_lora_weights`: Initialization strategy. Most use default :code:`True` but some experiment with :code:`"gaussian"` or newer methods such as :code:`"pissa"`. 

Most other knobs such as `bias`, `use_rslora`, `modules_to_save`, etc. can be left as PEFT defaults unless you want to explore specific advanced variations.

A common combination is to start with rank 16 and 32 (resp. alpha 32 and 64) and only target query/value projection modules. Then expand to more based on observed loss and eval metrics behavior (overfitting or underfitting) and your time/compute constraints.



RFModelConfig 
------

This is a core class in the RapidFire AI API that abstracts multiple Hugging Face APIs under the 
hood to simplify and unify all model-related specifications. In particular, it unifies model 
loading, training configurations, and LoRA settings into one class. 

It gives you flexibility to try out variations of LoRA adapter structures, training arguments for 
multiple control flows (SFT, DPO, and GRPO), formatting and metrics functions, and generation specifics. 

Some of the arguments (knobs) here can also be :class:`List` valued or :class:`Range` valued depending 
on its data type, as explained below. All this helps form the base set of knob combinations from which 
a config group can be produced. Also read :doc:`the Multi-Config Specification page</configs>`.


.. py:class:: RFModelConfig

  :param model_name: Model identifier for use with Hugging Face's :code:`AutoModel.from_pretrained()`. Can be a Hugging Face model hub name (e.g., ``"Qwen/Qwen2.5-7B-Instruct"``) or local path to a checkpoint.
  :type model_name: str

  :param tokenizer: Hugging Face Tokenizer identifier, typically same as :code:`model_name` string but can be different.
  :type tokenizer: str, optional

  :param tokenizer_kwargs: Additional keyword arguments passed to tokenizer's :code:`from_pretrained()` method (e.g., :code:`padding_side`, :code:`truncation`, and :code:`model_max_length`).
  :type tokenizer_kwargs: Dict[str, Any], optional

  :param formatting_func: Custom user-given data preprocessing function for preparing a single example with system prompt, roles, etc. Can be a :class:`List` for multi-config.
  :type formatting_func: Callable | :class:`List` of Callable, optional

  :param compute_metrics: Custom user-given evaluation function passed to Hugging Face's :code:`Trainer.compute_metrics` for use during evaluation phases. Can be a :class:`List` for multi-config.
  :type compute_metrics: Callable | :class:`List` of Callable, optional

  :param peft_config: RFLoraConfig as described above; thin wrapper around :code:`peft.LoraConfig` for LoRA fine-tuning. Can be a :class:`List` for multi-config.
  :type peft_config: RFLoraConfig | :class:`List` of RFLoraConfig, optional

  :param training_args: RF trainer configuration object specifying the training flow and its parameters. Also read :doc:`the Trainer Configs page</trainers>`.
  :type training_args: :class:`RFSFTConfig`  | :class:`RFDPOConfig`  | :class:`RFGRPOConfig`

  :param model_type: Custom user-defined string to identify model type inside your :code:`create_model_fn()` given to :func:`run_fit()`; default: ``"causal_lm"``
  :type model_type: str, optional

  :param model_kwargs: Additional parameters for model initialization, passed to :code:`AutoModel.from_pretrained()` (e.g., :code:`torch_dtype`, :code:`device_map`, :code:`trust_remote_code`).
  :type model_kwargs: Dict[str, Any], optional

  :param ref_model_name: For DPO and GRPO only; akin to :code:`model_name` above but for the frozen reference model.
  :type ref_model_name: str, optional

  :param ref_model_type: For DPO and GRPO only; akin :code:`model_type` above but for the frozen reference model.
  :type ref_model_type: str, optional

  :param ref_model_kwargs: For DPO and GRPO only; akin :code:`model_kwargs` above but for the frozen reference model.
  :type ref_model_kwargs: Dict[str, Any], optional

  :param reward_funcs: Reward functions for evaluating generated outputs during training (mainly for GRPO but can be used for DPO too). Can be a :class:`List` for multi-config.
  :type reward_funcs: Callable | [Callable] | :class:`List` of Callable | :class:`List` of [Callable], optional

  :param generation_config: Arguments for text generation passed to :code:`model.generate()` (e.g., :code:`max_new_tokens`, :code:`temperature`, :code:`top_p`).
  :type generation_config: Dict[str, Any], optional

  :param num_gpus: Number of GPUs to use for each run/config produced from this :code:`RFModelConfig`. Can skip this and specify :code:`num_gpus` in :func:`run_fit()` if all runs must use same number of GPUs.
  :type num_gpus: int, optional

  .. seealso::
     - :doc:`Hugging Face Transformers documentation <https://huggingface.co/docs/transformers/>`
     - :doc:`Hugging Face PEFT library documentation <https://huggingface.co/docs/peft/>`
     - :class:`RFSFTConfig`, :class:`RFDPOConfig`, :class:`RFGRPOConfig` for training argument configurations


**Examples:**

.. code-block:: python

    # Based on the SFT tutorial notebook
    RFModelConfig(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        peft_config=rfloraconfig1,
        training_args=rfsftconfig1,
        model_type="causal_lm",
        model_kwargs={"device_map": "auto", "torch_dtype": "auto","use_cache":False},
        formatting_func=sample_formatting_function,
        compute_metrics=sample_compute_metrics, 
        generation_config = {
            "max_new_tokens": 256, "temperature": 0.6, "top_p": 0.9, 
            "top_k": 40, "repetition_penalty": 1.18,
        }
    )

    # Based on the GRPO tutorial notebook
    RFModelConfig(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        peft_config=rfloraconfig,
        training_args=rfgrpoconfig1,
        formatting_func=sample_formatting_function,
        reward_funcs=reward_funcs,
        model_kwargs={"load_in_4bit": True, "device_map": "auto", "torch_dtype": "auto", "use_cache": False},
        tokenizer_kwargs={"model_max_length": 2048, "padding_side": "left", "truncation": True}
    )

**Notes:**

Note that one :class:`RFModelConfig` object can have only one base model configuration and 
one training control flow arguments dictionary. 
But you can specify a :class:`List` of PEFT configs, formatting functions, eval metrics 
functions, and reward functions list as part of your multi-config specification.


