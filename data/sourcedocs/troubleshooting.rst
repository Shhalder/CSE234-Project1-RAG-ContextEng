Troubleshooting
===============

Use this page to diagnose and resolve common issues when installing and running RapidFire AI.

.. note::

  RapidFire AI requires Python 3.12+. Verify your shell's ``python3`` is 3.12 before creating/activating the venv.

Quick diagnostics
-----------------

If you encounter any error, run the doctor command to get a complete diagnostic report (Python env, relevant packages, GPU/CUDA, and key environment variables):

.. code-block:: bash

   rapidfireai doctor


Hugging Face permission errors (login not picked up)
---------------------------------------------------

Run the Hugging Face login from the SAME virtual environment where you installed RapidFire AI.

Activate your venv and log in:

.. code-block:: bash

   source .venv/bin/activate
   pip install huggingface-hub
   huggingface-cli login
   huggingface-cli whoami  # Prints the HF account/orgs for the credentials this venv sees


Using Jupyter notebooks:

- If you logged in while a notebook was already running, restart the notebook kernel so it picks up the new Hugging Face credentials.
- Ensure the notebook uses the same venv kernel. 


Port conflicts (services already running)
----------------------------------------

If you encounter port conflicts, you can kill existing processes.

.. code-block:: bash

   lsof -t -i:8852 | xargs kill -9  # mlflow
   lsof -t -i:8851 | xargs kill -9  # dispatcher
   lsof -t -i:8853 | xargs kill -9  # frontend server

Select specific GPU(s) to use
-----------------------------

Set the ``CUDA_VISIBLE_DEVICES`` environment variable BEFORE running ``rapidfireai start`` to control which GPU(s) RapidFire can see and use.

.. code-block:: bash

   export CUDA_VISIBLE_DEVICES=2   # use GPU index 2 only
   rapidfireai start

Multiple GPUs (example: GPUs 0 and 2):

.. code-block:: bash

   export CUDA_VISIBLE_DEVICES=0,2
   rapidfireai start

From a Python script (set before importing/starting RapidFire):

.. code-block:: python

   import os
   os.environ["CUDA_VISIBLE_DEVICES"] = "2"
   # then start your RapidFire workflow


See also
--------

- For known limitations and workarounds, see :doc:`Known Issues </issues>`.
- If you are just getting started, follow the :doc:`Walkthrough </walkthrough>`.


