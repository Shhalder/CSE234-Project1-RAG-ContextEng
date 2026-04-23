ML Metrics Dashboard
===============


RapidFire AI offers a browser-based dashboard to automatically visualize all ML metrics and lets 
you control runs on the fly from there. 
Our current default dashboard is a fork of the popular OSS tool `MLflow <https://mlflow.org/>`__, 
and it inherits much of MLflow's native features.
The dashboard URI is printed when the rapidfireai server is started; open it in a browser. 

As of this writing, apart from MLflow, RapidFire AI also supports 
`TensorBoard  <https://www.tensorflow.org/tensorboard>`__
and `Trackio <https://huggingface.co/docs/trackio/en/index>`__
for logging metrics plots. 
Specify any one, two, or all three dashboards to use with the following server start argument. 

.. code-block:: bash

   rapidfireai start --tracking-backends [mlflow | tensorboard | trackio]

Alternatively, set the dashboard using its environment variable as below in your python code/notebook:

.. code-block:: python

   os.environ["RF_MLFLOW_ENABLED"] = "true"
   os.environ["RF_TENSORBOARD_ENABLED"] = "true"
   os.environ["RF_TRACKIO_ENABLED"] = "true"

Support for other popular dashboards such as Weights & Biases and CometML is coming soon. 
The rest of this section explains the new features of our MLflow-fork dashboard.
Note that these new features are not yet available on the other dashboards.


Tabs in the Dashboard
-----

The main "Experiments" page on the dashboard has 4 main tabs: 

* Table

* Chart

* Experiment Log

* Interactive Control (IC) Log


The screenshot below shows the "Table" view of an experiment with all its runs. 
Each run represents one model with one set of config knob values, which is standard dashboard semantics.


.. raw:: html

    <img src="_static/mlflow-1-table.png" alt="Table view of runs metadata" 
         style="cursor: zoom-in; max-width: 100%;" onclick="this.requestFullscreen()">


Metrics Plots
----

The screenshot below shows the "Chart" view of an experiment with all its runs. 
Each plot corresponds to a metric, spanning :code:`loss` on the training set and evaluation set, 
as well all named metrics returned in your :func:`compute_metrics()` function in the trainer config.

We call attention to 3 key aspects of the visualizations here: 

* The x-axis "Step" for the mini batch-level plots represents absolute number of minibatches seen by that run. So, if the :code:`batch_size` is different for different runs in your experiment, they will take different numbers of steps and the curves will not line up till the end. This is not a bug but the expected correct behavior.

* The x-axis "Step" for the epoch-level plots represents absolute number of epochs seen by that run. So, if the :code:`epochs` is different for different runs in your experiment, they will take different numbers of steps again as above.

* Please refresh the browser page to get RapidFire AI's metrics reader to pull the latest data entries from the metrics files.

.. raw:: html

    <img src="_static/mlflow-2-chart.png" alt="Metrics plots on dashboard" 
         style="cursor: zoom-in; max-width: 100%;" onclick="this.requestFullscreen()">


The dashboard picks some default colors for all runs, but you can change their colors by 
clicking the "color circle" next to the run number in the "Run Name" column. 
A color palette will pop up as shown in the screenshot below.

.. raw:: html

    <img src="_static/mlflow-3-colors.png" alt="Change colors of runs" 
         style="cursor: zoom-in; max-width: 100%;" onclick="this.requestFullscreen()">


Message Logs
------

There are two continually appending message logs on the third and fourth tabs: "Experiment Log" and 
"Interactive Control Log", respectively. 
All operations you run with RapidFire AI's API will be displayed on the former. 
The latter will specifically display all the Interactive Control (IC) operations you do via the IC Ops 
panels, as shown on the screenshot below.


.. raw:: html

    <img src="_static/mlflow-4-log.png" alt="Logs on dashboard" 
         style="cursor: zoom-in; max-width: 100%;" onclick="this.requestFullscreen()">


The full experiment log will also be available as a text file saved on your local directory under the name "rapidfire.log".
