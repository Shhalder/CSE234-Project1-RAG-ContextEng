API: Trainer Configs
===============

RapidFire AI's core APIs for trainer specifications are all thin wrappers around the 
corresponding APIs of Hugging Face's `TRL <https://huggingface.co/docs/trl/index>`__ libraries.


RFSFTConfig
------

This is a wrapper around :class:`SFTConfig` in HF TRL. 
The full signature and list of arguments are available on `this page 
<https://huggingface.co/docs/trl/sft_trainer#trl.SFTConfig>`__.

The only difference here is that the individual arguments (knobs) can be :class:`List` valued 
or :class:`Range` valued in :class:`RFSFTConfig`. 
That is how you can specify a base set of knob combinations from which a config group can be 
produced. Also read :doc:`the Multi-Config Specification page</configs>`.
Other than the multi-config specification, this class preserves all semantics of Hugging Face's 
SFT trainer under the hood. 


**Examples:**

.. code-block:: python

	# From the SFT tutorial notebook
	RFSFTConfig(
		learning_rate=2e-4,
		lr_scheduler_type = "linear",
		per_device_train_batch_size=4,
		per_device_eval_batch_size=8,
		gradient_accumulation_steps=4,
		num_train_epochs=2,
		logging_steps=5,
		eval_strategy="steps",
		eval_steps=25,
		fp16=True,
		save_strategy="epoch",
	)

	# Two knobs have list values here
	RFSFTConfig(
		learning_rate=List([2e-4, 1e-5]),
		lr_scheduler_type=List(["linear", "cosine"]),
		per_device_train_batch_size=4,
		gradient_accumulation_steps=4,
		per_device_eval_batch_size=8,
		num_train_epochs=2,
		logging_steps=5,
		eval_strategy="steps",
		eval_steps=25,
		fp16=True,
		save_strategy="epoch",
	)


For larger LLMs that do not fit on a single GPU and need cross-GPU model partitioning (within a machine) to fit in aggregate GPU memory, specify the FSDP configuration to apply with its usual settings given a dictionary as illustrated below.


**Examples:**

.. code-block:: python

	# From the SFT FSDP Large tutorial notebook
	RFSFTConfig(
		...
		fsdp="full_shard auto_wrap",
        fsdp_config={
			"backward_prefetch": "backward_pre",
			"forward_prefetch": False,
			"use_orig_params": False,
			"cpu_ram_efficient_loading": True,
			"offload_params": False,
			"sync_module_states": True,
			"limit_all_gathers": True,
			"sharding_strategy": "FULL_SHARD",
			"auto_wrap_policy": "TRANSFORMER_BASED_WRAP"
		}
	)

	# From the SFT FSDP Lite tutorial notebook
	RFSFTConfig(
		...
		fsdp="full_shard auto_wrap",
        fsdp_config={
			"backward_prefetch": "backward_pre",
			"forward_prefetch": True,
			"use_orig_params": False,
			"cpu_ram_efficient_loading": True,
			"offload_params": True,
			"sync_module_states": True,
			"limit_all_gathers": True,
			"sharding_strategy": "FULL_SHARD",
			"auto_wrap_policy": "TRANSFORMER_BASED_WRAP"
		}
	)




RFDPOConfig
------

This is a wrapper around :class:`DPOConfig` in HF TRL. 
The full signature and list of arguments are available on `this page 
<https://huggingface.co/docs/trl/dpo_trainer#trl.DPOConfig>`__.

Again, the only difference here is that the individual arguments (knobs) can be :class:`List` 
valued or :class:`Range` valued in :class:`RFDPOConfig`. 
That is how you can specify a base set of knob combinations from which a config group can 
be produced. Also read :doc:`the Multi-Config Specification page</configs>`.
Other than the multi-config specification, this class preserves all semantics of 
Hugging Face's DPO trainer under the hood. 


**Example:**

.. code-block:: python

	# Based on the DPO tutorial notebook; one knob has list of values
	base_dpo_config = RFDPOConfig(
		model_adapter_name="default",
		ref_adapter_name="reference",
		force_use_ref_model=False, 
		loss_type="sigmoid",
		beta=List([0.1,0.001]), 
		max_prompt_length=1024,
		max_completion_length=1024,
		max_length=2048, 
		per_device_train_batch_size=2,
		gradient_accumulation_steps=4,
		learning_rate=5e-6, 
		warmup_ratio=0.1,
		weight_decay=0,
		lr_scheduler_type="linear",
		optim="adamw_8bit",
		num_train_epochs=1, 
		logging_strategy="steps",
		logging_steps=1,
		bf16=True,
		save_strategy="epoch",
	)


Just like for SFT, you can specify an FSDP configuration for DPO too for larger LLMs that need cross-GPU partitioning (within a machine).

**Example:**

.. code-block:: python

	# From the DPO FSDP Lite notebook
	base_dpo_config_lite = RFDPOConfig(
		...
		fsdp="full_shard auto_wrap",
		fsdp_config={
			"backward_prefetch": "backward_pre",
			"forward_prefetch": True,
			"use_orig_params": False,
			"cpu_ram_efficient_loading": True,
			"offload_params": False,
			"sync_module_states": True,
			"min_num_params": 1000000,
			"limit_all_gathers": True,
			"sharding_strategy": "FULL_SHARD",
			"auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
			"activation_checkpointing":False
		}
	)


RFGRPOConfig
------

This is a wrapper around :class:`GRPOConfig` in HF TRL. 
The full signature and list of arguments are available on `this page 
<https://huggingface.co/docs/trl/grpo_trainer#trl.GRPOConfig>`__.

Again, the only difference here is that the individual arguments (knobs) can be :class:`List` 
valued or :class:`Range` valued in :class:`RFGROConfig`. 
That is how you can specify a base set of knob combinations from which a config group can 
be produced. Also read :doc:`the Multi-Config Specification page</configs>`.
Other than the multi-config specification, this class preserves all semantics of 
Hugging Face's GRPO trainer under the hood. 

**Example:**

.. code-block:: python

	# Based on the GRPO tutorial notebook
	RFGRPOConfig(
		learning_rate=5e-6,
		warmup_ratio=0.1,
		weight_decay=0.1,
		max_grad_norm=0.1,
		adam_beta1=0.9,
		adam_beta2=0.99,
		lr_scheduler_type = "linear",
		per_device_train_batch_size=4,
		gradient_accumulation_steps=4,
		num_generations=8,
		optim ="adamw_8bit",
		num_train_epochs=2,
		max_prompt_length=1024,
		max_completion_length=1024,
		logging_steps=2,
		eval_steps=5,
	)

.. note::
  As of this writing, out-of-the-box support for FSDP for GRPO is still in the works. Watch this space for updates.

