Glossary of Key Terms and Concepts
==========


Artifacts
-----
All files related to an experiment saved under in a local folder with its path specified by
:code:`experiments_path` in the experiment's constructor.
Includes the final (and possibly all epoch-level) model checkpoints of all runs across 
all :func:`run_fit()` in that experiment, and all associated metrics files.

Read more here: :doc:`API: Experiment</experiment>`.


Config Dictionary
-----

A dictionary of key-value pairs that specify a full model training configuration. 
A knob's value can be a singleton or set-valued (:class:`List` or :class:`Range`). 
A dictionary with set-valued knobs be fed to a config-group generator method. 
A single combination of knob values in this dictionary is called a "leaf" config. 
RapidFire AI instantiates one run per leaf config, and its values are injected via the 
:code:`model_config` argument to your :func:`create_model_fn()` function.

Read more here: :doc:`API: Multi-Config Specification</configs>`.


Config Group
-----

A set of config dictionary instances, produced in bulk by providing a config dictionary 
with set-valued knobs to a config-group generator method (see below). 
It can also be a Python list of individual config dictionaries or config-group generators recursively.

Read more here: :doc:`API: Multi-Config Specification</configs>`.


Config Group Generator
-----

A method to generate a group of config dictionaries in one go based on an input
config dictionary with set-valued knobs (:class:`List` or :class:`Range`). 
Currently supported generator methods are grid search (:class:`RFGridSearch`) and 
random search (:class:`RFRandomSearch`). Support for AutoML heuristics coming soon. 

Read more here: :doc:`API: Multi-Config Specification</configs>`.



Experiment
------

A core concept in the RapidFire AI API that defines a collection of training and evaluation 
operations performed. Each experiment is assigned a unique name that is used for both display 
of plots on the ML metrics dashboard and for artifact tracking. 
At any point in time, only one experiment can be alive. 

Read more here: :doc:`API: Experiment</experiment>`.


Experiment Ops
------

Computation methods associated with the :class:`Experiment` class of RapidFire AI: 
:func:`run_fit()`, :func:`end()`, and the constructor. 
Also includes two informational methods: :func:`get_runs_info()` and :func:`get_results()`. 
We will expand this API with more operations based on feedback, e.g., for batch testing 
and inference/generation.

Read more here: :doc:`API: Experiment</experiment>`.


Interactive Control Ops (IC Ops)
------

Operations to control runs in flight during a :func:`run_fit()`.  
RapidFire AI automatically reapportions GPU resources across runs under the hood. 
We currently support 4 IC Ops: Stop, Resume, Clone-Modify, and Delete.

Read more here: :doc:`Dashboard: Interactive Control (IC) Ops</icops>`.


Knob
-----

A single entry in the config dictionary given for experimentation. 
A knob's value can be a singleton or set-valued (:class:`List` or :class:`Range`). 

Read more here: :doc:`API: Multi-Config Specification</configs>`.


Logs
-----

Files with entries about all operations run by RapidFire AI, including IC Ops, to aid 
monitoring of debugging of experiment behaviors. 
The whole experiment log is displayed on the app under the "Experiment Log" tab. 
Likewise, all IC Ops are displayed under their own tab next to it.

Read more here: :doc:`ML Metrics Dashboard</dashboard>`.


ML Metrics Dashboard
-----

A dashboard to display plots of all ML metrics (loss and eval metrics) of all runs and 
experiments, overlay IC Ops functionality, and display informative logs.
RapidFire AI's current default dashboard is a fork of the popular OSS tool MLflow.

Read more here: :doc:`ML Metrics Dashboard</dashboard>`.



Results
------

A single DataFrame containing all loss and eval metrics values of all runs across all epochs 
across all :func:`run_fit()` invocations in this experiment so far. 
Returned by the :func:`get_results()`.

Read more here: :doc:`API: Experiment</experiment>`.



Run
-----

A central concept in RapidFire AI representing a single combination of configuration knob values
for a model trained with :func:`run_fit()`. 
It is the same concept as in ML metrics dashboards such as MLflow and Weights & Biases. 
RapidFire AI assigns each run a unique integer :code:`run_id` within an experiment.

Read more here: :doc:`API: Experiment</experiment>` and :doc:`ML Metrics Dashboard</dashboard>`.
