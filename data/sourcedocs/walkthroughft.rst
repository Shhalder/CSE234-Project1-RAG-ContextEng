Install and Get Started: Fine-Tuning and Post-Training
===============================

There are two ways to get started with RapidFire AI for SFT/RFT:

1. **Google Colab** - Zero local setup, takes under 3 minutes to get started. However, this option has limitations due to the free T4 GPU instance, viz., it works only for smaller models and small datasets.

2. **Full Installation** - Complete setup with full functionality for any environment (local machine or cloud machine), supporting models and datasets of any size.

Choose the option that best fits your needs below.


Google Colab
-----------------------

Try RapidFire AI instantly in your browser with our pre-configured Google Colab notebook. No installation required.

**Launch the notebook:** `RapidFire AI on Google Colab <https://tinyurl.com/rapidfireai-colab>`_

.. note::
  The Google Colab environment is limited to small models and datasets due to their free resource constraints.


Full Installation
-----------------------

Follow these steps to install RapidFire AI on your local machine or remote/cloud instance for complete functionality without limitations.


Step 1: Install dependencies and package
-----------------------

Obtain the RapidFire AI OSS package from pypi (includes all dependencies) and ensure it is installed correctly.

.. important::

  Requires Python 3.12+. Ensure that ``python3`` resolves to Python 3.12 before creating the venv.

.. code-block:: bash

   python3 --version  # must be 3.12.x
   python3 -m venv .venv
   source .venv/bin/activate

   pip install rapidfireai

   rapidfireai --version
   # Verify it prints the following:
   # RapidFire AI 0.14.0

Provide your Hugging Face account token to access the gated Llama and Mistral models 
showcased in the tutorial notebooks. 
If you do not have such a token, you have two options:

* Switch the :code:`model_name` in the tutorial notebook to a non-gated model from Hugging Face. Then proceed to Step 2.

* Create a Hugging Face token `as explained here <https://huggingface.co/docs/hub/en/security-tokens>`_. Then request access on the following gated models' Hugging Face pages:

  * `mistralai/Mistral-7B-Instruct-v0.3 <https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3>`_
  * `meta-llama/Llama-3.1-8B-Instruct <https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct>`_
  * `meta-llama/Llama-3.2-1B-Instruct <https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct>`_
  
  Headsup: the approval for the Llama models may take a few hours. Then provide your HF token in the same venv.

.. code-block:: bash

   source .venv/bin/activate
   pip install "huggingface-hub[cli]"

   # Replace YOUR_TOKEN with your actual HF token
   # https://huggingface.co/docs/hub/en/security-tokens
   hf auth login --token YOUR_TOKEN

   # Due to current issue: https://github.com/huggingface/xet-core/issues/527
   pip uninstall -y hf-xet


Feel free to ask us on Discord if you need any help with accessing gated Hugging Face models. Unfortunately, we are not allowed to provide a publicly visible token here for your use due to Hugging Face's policies.


Step 2: Start RapidFire AI server
------------

Run the following command to initialize rapidfireai to use the correct dependencies:

.. code-block:: bash

   rapidfireai init
   # It will install specific dependencies and initialize rapidfireai

.. note::
  You need to run init *only once* for a new venv or when switching GPU(s) on your machine. You do NOT need to run it after a reboot, start/stop of rapidfireai, or for a new terminal.


Next start RapidFire AI services: the frontend with the ML metrics dashboard and the API server. 
The frontend URL shown below can be opened on your local browser.

.. code-block:: bash

   rapidfireai start
   # It should print about 50 lines, including the following:
   # ...
   # RapidFire Frontend is ready
   # Open your browser and navigate to: http://0.0.0.0:8853
   # ...
   # Press Ctrl+C to stop all services

.. important::

  Do NOT proceed until the start is successful with "Available endpoints" printed as above. Leave this terminal running while you work through the tutorial notebooks. 
  
If you close the terminal in which you started rapidfireai or if you rebooted your machine, 
just start rapidfireai again with the above command.

If the start command fails for whatever reason, wait for half a minute and rerun it.
For diagnostics and common fixes (including Linux/macOS and Windows steps), see :doc:`Troubleshooting </troubleshooting>`.


Step 2b (optional): Forward ports if using remote machine
-----------

If you installed rapidfireai on a remote machine (e.g., on a public cloud) accessed via
ssh, you also need to forward the following port on your client machine.
Run the following ssh command with your correct username and remote machine IP.
(Or forward the port via your VSCode or other IDEs.)


.. code-block:: bash

   ssh -L 8850:localhost:8850 username@remote-machine
   ssh -L 8851:localhost:8851 username@remote-machine
   ssh -L 8853:localhost:8853 username@remote-machine


Step 2c (optional): Change the dashboard for visualization
-----------

By default RapidFire AI logs metrics plots in MLflow format and displays them automatically in 
its own MLflow-fork dashboard. 
The dashboard URI is printed when the server is started; open it in a browser. 
As of this writing, RapidFire AI also supports TensorBoard and Trackio for logging metrics.

Specify any one, two, or all three dashboards to use with the following server start argument. 
Support for other popular dashboards such as Weights & Biases and CometML is coming soon. 

.. code-block:: bash

   rapidfireai start --tracking-backends [mlflow | tensorboard | trackio]

Alternatively, set the dashboard using its environment variable as below in your python code/notebook:

.. code-block:: python

   os.environ["RF_MLFLOW_ENABLED"] = "true"
   os.environ["RF_TENSORBOARD_ENABLED"] = "true"
   os.environ["RF_TRACKIO_ENABLED"] = "true"


Step 3: Download the tutorial notebooks
------------

Only after completing Step 2, download the example tutorial notebooks (explained further here: :doc:`Example Use Case Tutorials</tutorials>`). 
You can also see them in the "tutorial_notebooks" folder under the directory where you initialized rapidfireai.

If your GPU has < 80 GB HBM, use the "lite" versions of these notebooks. The only difference is they showcase smaller LLMs and finish faster. 
Right click on the GitHub link to save that file locally.

* SFT for customer support Q&A chatbot: `View on GitHub <https://github.com/RapidFireAI/rapidfireai/blob/main/tutorial_notebooks/fine-tuning/rf-tutorial-sft-chatqa.ipynb>`__

  * Lite version: `View on GitHub <https://github.com/RapidFireAI/rapidfireai/blob/main/tutorial_notebooks/fine-tuning/rf-tutorial-sft-chatqa-lite.ipynb>`__

* DPO for alignment: `View on GitHub <https://github.com/RapidFireAI/rapidfireai/blob/main/tutorial_notebooks/post-training/rf-tutorial-dpo-alignment.ipynb>`__

  * Lite version: `View on GitHub <https://github.com/RapidFireAI/rapidfireai/blob/main/tutorial_notebooks/post-training/rf-tutorial-dpo-alignment-lite.ipynb>`__

* GRPO for math reasoning: `View on GitHub <https://github.com/RapidFireAI/rapidfireai/blob/main/tutorial_notebooks/post-training/rf-tutorial-grpo-mathreasoning.ipynb>`__

  * Lite version: `View on GitHub <https://github.com/RapidFireAI/rapidfireai/blob/main/tutorial_notebooks/post-training/rf-tutorial-grpo-mathreasoning-lite.ipynb>`__


Quickstart Video (2.5min)
^^^^^^^^^^^^^^^

.. raw:: html

    <div style="position: relative; width: 100%; height: 0; padding-bottom: 45%;">
        <iframe style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;" 
                src="https://www.youtube.com/embed/nPMBfZWqPWI?si=6h1cDj8yqhilB9ti" 
                title="RapidFire AI Quickstart Video" 
                frameborder="0" 
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" 
                referrerpolicy="strict-origin-when-cross-origin" 
                allowfullscreen>
        </iframe>
    </div>

|

Full Usage Walkthrough Video (12min)
^^^^^^^^^^^^^^^

.. raw:: html

    <div style="position: relative; width: 100%; height: 0; padding-bottom: 45%;">
        <iframe style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;" 
                src="https://www.youtube.com/embed/431LulD3Stc?si=C1nM0V5sV51cgqNz" 
                title="RapidFire AI Full Usage Walkthrough Video" 
                frameborder="0" 
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" 
                referrerpolicy="strict-origin-when-cross-origin" 
                allowfullscreen>
        </iframe>
    </div>

|


Step 4: Run the notebook cells
-------

Run the cells *one by one* as shown in the above videos. Wait for a cell to finish before running the next.

* Imports
* Load dataset and specify train and eval partitions

  If you want to run the notebook faster for demo purposes, downsample the data further as per your wish. Here are some suggested reductions. You can also reduce effective batch size by reducing either or both of :code:`per_device_train_batch_size` and :code:`gradient_accumulation_steps` in the trainer configs.

  * SFT notebook: 
     .. code-block:: python

        train_dataset=dataset["train"].select(range(128)) # 128 instead of 5000
        eval_dataset=dataset["train"].select(range(5000,5032)) # 5032 instead of 5200

  * DPO notebook: 
     .. code-block:: python

        select(range(128)) # 128 instead of 500

  * GRPO notebook: 
     .. code-block:: python

        train_dataset = get_gsm8k_questions(split="train").select(range(128)) # 128 instead of 5000
        eval_dataset = get_gsm8k_questions(split="test").select(range(32)) # 32 instead of 100

* Define example processing function

* Create named RF experiment 

* Define custom eval metrics function

* Define multi-config knobs for model, LoRA, and SFT Trainer using RapidFire AI wrapper APIs

* Define model creation function for all model types across configs

* Generate config group you want to compare in one go

* Launch multi-config training; adjust :code:`num_chunks` as per desired concurrency (see `Run Fit <experiment.html#run-fit>`__ for details)

  .. code-block:: python

     # Launch training of all configs in the config_group with swap granularity of 4 chunks
     experiment.run_fit(config_group, sample_create_model, train_dataset, eval_dataset, num_chunks=4, seed=42)

  Note that in the same experiment, you can run as many :func:`run_fit()` as you want. All their runs will be superimposed on the same plots on the dashboard.



Step 5: Monitor training behaviors on ML metrics dashboard
--------

.. raw:: html

    <img src="_static/step7.png" alt="Monitor training behaviors on ML metrics dashboard" 
         style="cursor: zoom-in; max-width: 100%;" onclick="this.requestFullscreen()">


Step 6: Interactive Control (IC) Ops: Stop, Clone-Modify; check their results 
-----

.. raw:: html

    <img src="_static/icop-stop.png" alt="IC Op: Stop" 
         style="cursor: zoom-in; max-width: 100%;" onclick="this.requestFullscreen()">


.. raw:: html

    <img src="_static/icop-clone.png" alt="IC Op: Clone-Modify" 
         style="cursor: zoom-in; max-width: 100%;" onclick="this.requestFullscreen()">


.. raw:: html

    <img src="_static/step10.png" alt="IC Op results on dashboard" 
         style="cursor: zoom-in; max-width: 100%;" onclick="this.requestFullscreen()">



Step 7: End experiment; stop server when done
------

Run the cell to end the expeirment when you are done with it.

.. code-block:: python

   experiment.end()

You can then move on to another (named) experiment in the same session. 
Run as many experiments as you like; each will have its plots apppear on the dashboard 
under its name. 
All experiment artifacts (metrics files, logs, checkpoints, etc.) are *persistent* on 
your machine in the same location as your notebook.

When you are done overall, gracefully stop the RapidFire AI session and free the ports used in one of two ways:

* Press Ctrl+C on the terminal where :code:`rapidfireai start` was performed. Wait for all services to finish cleanly. 

* In a separate terminal tab, run the stop command as follows and wait for it to finish fully. If you had run the start command as a background process, feel free to run the stop command in the same terminal tab.

.. code-block:: bash

   rapidfireai stop

.. important::
  If you kill the rapidfireai server forcibly without graceful stopping as above, you might lose some experiment artifacts and/or metadata.



Step 8: Venture Beyond!
-----

After trying out the tutorial notebooks, explore the rest of this docs website, 
especially the API and dashboard pages.
Play around more with IC Ops and/or run more experiments as you wish, including 
changing the datasets, models, config knobs, and code for the functions and rewards.


You are now up to speed! Enjoy the power of rapid AI customization with RapidFire AI!
