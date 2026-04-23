Known Issues and Updates Coming Soon
=============

As of this writing, RapidFire AI has the following known issues. We have listed some 
recommended recourses for some. 
We are actively working on resolving these and welcome feedback on their utility for your 
use cases to help with prioritization.


Multi-GPU Model Support
------

Ordinarily, any given run's model(s) for its batch size must fit on a single GPU's memory. 
For DPO and GRPO, both the policy and reference models must fit on the GPU together. 

For SFT (and DPO), RapidFire AI can use FSDP automatically to partition a large model across GPUs on the same machine.
Provide the FSDP usage details in a config knob. See :doc:`the SFT page</sft>` FSDP section for more deatils. 
Support for GRPO coming soon. 
We also plan to add support for multi-GPU model partitioning via DeepSpeed soon.


ImportError in between Experiments
------

If you run multiple experiments back to back from the same notebook/IDE session, you might 
see the following error appear occasionally: 

.. code-block:: python

   ImportError: cannot import name 'GenerationMixin' from 'transformers.generation'

This is caused by stray Python processes from the previous experiment not ending properly. 
If you see this error, we recommend the following steps:

* Run the command :code:`ps - ef | grep python`, look for "multiprocessing.spawn"/"defunct" processes, and kill if there are any with command :code:`kill -9 [PID]`.

* Wait for about 2 minutes regardless of whether there are processes to kill as above.

* Restart the kernel and then proceed with your new experiment.


Recovering Storage Space
-------

If you run out of storage space on your machine due to experimenting with lots of LLMs, we 
recommend clearing out the ".cache" folder on your home directory that is created by 
Hugging Face to import the base models. 
One experiment's imported models are not needed for another; so, it is safe to delete them.

If you want to reclaim even more space, look at the artifacts from your experiments and 
either delete some of the files or move them to other/remote storage. 
Note that when you use LoRA adapters, RapidFire AI saves only the trained adapters in the 
checkpoints of the runs, not the base models.


Semi-Automated IC Ops
------

Triggering IC Ops manually from the dashboard is feasible only if there is a human in the loop. 
But IC Ops are useful even in offline scripted settings based on application logic, e.g., stop 
90% of runs with poor eval metrics and clone-modify the top 10% to drill down into more 
fine-grained values for their knobs.

In the near future, we plan to update the :class:`Experiment` API to let you specify such custom 
semi-automation logic for IC Ops in code using the runs' metrics and progress so far.

