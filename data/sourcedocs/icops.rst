Dashboard: Interactive Control (IC) Ops
===============

Interactive Control Operations (IC Ops) are a powerful differentiating aspect of RapidFire AI 
that enable *rapid experimentation* capability for AI customization. 


Motivation for IC Ops
----

IC Ops are control operations over the runs *in flight* in an ongoing experiment. 
They are motivated by an often under-appreciated pain point felt by many AI developers: 

* How accurate a given configuration will be is *impossible* to tell upfront in general. Experimentation is critical: one must try alternate configs based on their intuition about their specific use case, dataset, model, and eval metrics.

* Not all configs are made equal. One must be able to easily try and retry values, zoom into promising regions of values, adjust on the fly, etc. This can help reach better eval metrics/model alignment more quickly. Otherwise, one might squander their (labeled) data and/or waste resources.

* Even for a prior deployed model, one may need to adapt knobs over time as the data distribution evolves (e.g., concept drift), application schema evolves (e.g., data collection process changes), newer/better models emerge (e.g., smaller but more capable LLMs), etc.


Generic MLOps tools treat a run as a generic monolithic job and schedule them at a coarse granularity, 
leading to a disconnect between what is needed for customization and the execution layer.
IC Ops alter this status quo by giving you a whole new level of control over runs in flight. 

No need to juggle disparate tools for data-parallelism (DDP), model-parallelism (FSDP / DeepSpeed), or 
task-parallelism (W&B, Ray Tune, etc.).
RapidFire AI's execution engine handles lower level scheduling and orchestration of run adaptively 
to enable IC Ops.



Semantics of IC Ops
-----

IC Ops can be used only when a :func:`run_fit()` is actively running. 
To access the IC Ops panel, click on the "IC Ops" column buttons in the runs table
or on any run's curve on any metrics plot in the "Chart" view.
Also see :doc:`ML Metrics Dashboard</dashboard>`.

Alternatively, you can also invoke the in-notebook IC Ops control panel with the 
following code. 

As of this writing, this in-notebook panel works only on the Google
Colab deployment for :func:`run_fit()`, but we will soon support it for other environments too.

.. code-block:: python

    # Create Interactive Controller
    from rapidfireai.utils.interactive_controller import InteractiveController

    controller = InteractiveController(dispatcher_url="http://127.0.0.1:8851")
    controller.display()

The in-notebook IC Ops controller has the same operations and it looks like the following: 

.. raw:: html

    <img src="_static/notebook-icops.png" alt="In-notebook IC Ops panel" 
         style="cursor: zoom-in; max-width: 100%;" onclick="this.requestFullscreen()">


For :func:`run_evals()`, as of this writing, only jupyter is supported when its server is 
started as below. We will expand support for other IDEs soon.
Note that IC Ops panel will appear below the cell where :func:`run_evals()` is invoked.

.. code-block:: bash

   jupyter notebook --no-browser --port=8850 --ServerApp.allow_origin='*'

Open the URL provided by the above command on your browser. 
If you are running it on a remote machine, make sure to also forward 
the ports on your client :ref:`as explained here <step-3b-port-forwarding>`.


As of this writing, we support 4 IC Ops: **Stop**, **Resume**, **Clone-Modify**, and **Delete**. 
We explain each shortly below.

All IC Ops on a run are queued by the system and **executed at a chunk boundary** for that run. 
This avoids potentially non-deterministic or other inconsistent behaviors during concurrent run execution.
Note that different runs might reach their chunk boundary at different points in time. 
To control the number of chunks, set :code:`num_chunks` during :func:`run_fit()`; 
more details :doc:`on the Experiment docs page </experiments>`.

IC ops can be invoked as intermittently as you want during a long-running :func:`run_fit()`. 
So, you can launch, say, 16 configs in one go (even on a 4-GPU machine), check in after a few chunks,  
and stop bottom 80% of the runs. You can let the top performers continue for longer. Then you can 
clone and modify some to add new finer grained runs and warm start their parameters. And so on.

Under the hood, RapidFire AI automatically adjusts the apportioning of the GPUs among all ongoing 
runs to ensure maximal GPU utilization.




Stop
----

This IC Op earmarks a run to be stopped at the end of its current chunk. 
It will still be alive but it will not use any GPU resources from the next chunk. 
You will still see its minibatch-level plots advancing for the current chunk. 
You cannot stop an already stopped or deleted run. 


.. raw:: html

    <img src="_static/icop-stop2.png" alt="IC Op Stop" 
         style="cursor: zoom-in; max-width: 100%;" onclick="this.requestFullscreen()">

    <img src="_static/icop-stop.png" alt="IC Op Stop" 
         style="cursor: zoom-in; max-width: 100%;" onclick="this.requestFullscreen()">


Resume
-----

This IC Op is applicable only to a previously stopped run. 
It earmarks this run to be resumed from the next chunk onward, when it will be added to the mix of 
ongoing runs and assigned GPU(s) automatically. 
You cannot resume an already resumed or deleted run.



Clone-Modify
----

This is a powerful IC Op that is applicable to any ongoing, stopped, or resumed run. 
It allows you to add "clones" of a chosen run, called the "parent" run, during a :func:`run_fit()`. 
The IC Op panel displays an editable text box with the full knob config dictionary of the parent. 

Edit any knobs, e.g., learning rate, LoRA rank, or even base model as if you are injecting 
that new run config from code, except this is done conveniently from the metrics dashboard itself. 
As of this writing, we only support providing a single config for this IC Op.
Soon we will support providing a config-group generator such as :code:`RFGridSearch()` or
:code:`RFRandomSearch()` as well in the IC Op panel itself akin to the launching code.

You can also **warm-start** a clone using its parent's weights if you'd like. 
Warm-started clones inherit their parent's learning behavior so far and thus, they can reach better 
eval metrics faster. 
Note that warm starting is only allowed if the clones have *identical* neural architecture as the 
parent, including LoRA adapters; otherwise, it will error out.

When you are ready with your clone's config, click "Submit" to execute this IC Op.


.. raw:: html

    <img src="_static/icop-clone2.png" alt="IC Op Clone-Modify" 
         style="cursor: zoom-in; max-width: 100%;" onclick="this.requestFullscreen()">

    <img src="_static/icop-clone.png" alt="IC Op Clone-Modify" 
         style="cursor: zoom-in; max-width: 100%;" onclick="this.requestFullscreen()">


Clones will automatically appear on the plots from the next chunk onward; just refresh the page. 
RapidFire AI's adaptive scheduler automatically reapportions GPUs across all runs, including clones.
So, do not need to worry about manually splitting GPUs across models, juggling new processes, etc.
Clones are treated just like any other run; so, you can clone that clone later with IC Ops again.

You can submit multiple Clone-Modify ops on the same run or different runs whenever you want. 
They will get queued up and all clones will start together at the next chunk boundary. 

Clone-Modify combined with Stop enables you to turbocharge how you leverage your intuition about your 
AI use case, dataset, models, and eval metrics to dramatically cut down time to reaching much better 
eval metrics even within a single experiment.



Delete
----

This IC Op earmarks the run to be deleted from the next chunk onward. 
On the chart, you will see its curves vanish almost immediately. 
You cannot do any further IC Ops on a deleted run because it will not be visible. 
Note that although a deleted run vanishes from the plots, its model checkpoints are still part of 
the artifacts of that experiment so that you have post-hoc audibility.



Coming Soon: Templated Automation of IC Ops
----

IC Ops are a powerful capability to dramatically improve the effectiveness of your experiments. 
We plan to add automated template support for IC Ops based on feedback. 
This will help you apply a consistent policy for using stop, clone-modify, etc. across your projects 
and/or personnel via code. You can also create new customized semi-automated heuristics on top of 
IC Ops or schedule them for automated future execution instead of having to sit in the loop. 

Please do let us know on Discord if you have other requests regarrding how you'd like to use IC Ops!
