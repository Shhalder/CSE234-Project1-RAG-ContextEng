Online Aggregation for Evals
============================

When evaluating RAG/context eng. pipelines, traditional batch evals processing forces you to 
wait for the whole dataset to be processed by a config before seeing its results. 
For expensive LLM-based generators (both closed model APIs and self-hosted models) that can lead
to wastage of a lot of token spend and/or a lot of GPU resources before you realize that a config 
yields poor eval metrics. 
That can inhibit experimentation on config knobs due to the time and/or costs involved, 
ultimately hurting eval metrics and stalling AI applications.

RapidFire AI transforms the status quo by adapting the powerful idea of **online aggregation** 
from database systems research to LLM evals. 
Our adaptive execution engine, :doc:`as described on this page</difference>`, automatically 
shards the data and processes multiple configs in parallel, one shard at a time, with 
efficient swapping techniques.

This means you get **running metric estimates with confidence intervals** in real time. 
So, you can confidently stop poor configs earlier, clone better configs on the fly, and 
perform more informed exploration to reach much better eval metrics in much less time.


Example: Traditional Batch Evals vs. RapidFire AI
-------

For instance, suppose you have an evals set with 400 queries. You decide to compare, say, 
4 RAG configs in one go with RapidFire AI with number of shards set to 8. The illustration
below contrasts traditional batch evals vs. RapidFire AI's approach for a simple eval metric.

.. list-table::
   :widths: 50 50
   :class: side-by-side

   * - .. figure:: /images/rag-eval-online1.png
          :width: 100%
          :alt: Online aggregation for evals with RapidFire AI and IC Ops.

     - .. figure:: /images/rag-eval-online2.png
          :width: 100%
          :alt: Online aggregation for evals with RapidFire AI and IC Ops.


All configs are executed on the first 1/8th of the data (50 examples), with 
their **incrementally computed** eval metrics shown in real time with confidence intervals. 
In the figure, the 3 worst configs are stopped, while the best is cloned to add 2 new variants. 
The 3 running configs now continue on the second 1/8th of the data (cumulatively, 
100 examples), and so on.
One clone is then stopped halfway through the aggregation, while the other two run to completion. 
Ultimately, the other clone ends up being the best config overall.

Note that the confidence intervals shown will keep narrowing as configs see more shards, converging 
to zero when 100% of the data is seen, i.e., the metrics become exact point estimates.
Overall, compared to sequential batch evals in which the original 4 configs all run to completion, 
RapidFire AI enables you to explore more configs in less time, while reaching better eval metrics.



Types of Metrics
-----------------------

We support 2 types of metrics based on their aggregation semantics: 

* **Distributive Metrics:** These are purely additive over a given set of data points. 
  
  Examples: *count* of number of correct predictions; *sum* of output token lengths across queries.

* **Algebraic Metrics:** These are averages or proportions over a given set of data points. They can be decomposed into components that are individually distributive. 
  
  Examples: *precision*, which counts number of correct predictions and total number of data points separately and then divides them; *mean rouge-1*, which averages per-example rouge-1 values that assesses overlap of tokens between generated text and ground truth text.

When you define an eval metric via :func:`evals.compute_metrics_fn()` and :func:`evals.accumulate_metrics_fn()`, 
you must specify their type (algebraic or distributive) and value range as illustrated below. 
For metrics without a type defined, they will be displayed *as is*, i.e., without projected 
estimates or confidence intervals.

.. code-block:: python

    # Based on GSM8K tutorial use case
    metrics = {
        "Total": {"value": total},
        "Correct": {
            "value": correct,
            "is_distributive": True,
            "value_range": (0, 1),
        },
        "Accuracy": {
            "value": accuracy,
            "is_algebraic": True,
            "value_range": (0, 1),
        },
    }

Confidence Intervals
--------------------

The data points in the evals dataset are **assigned to shards uniformly randomly**, i.e., 
RapidFire AI performs sampling without replacement. 
Based on that, it supports 3 strategies to calculate confidence intervals for projected estimates of metrics. 
You can indicate the confidence level (we recommend 95%) and whether to perform "finite population correction" (FPC) or not. 
These values can be specified under the key :code:`"online_strategy_kwargs"` in your config dictionary as illustrated below.

.. code-block:: python

    # Based on FiQA RAG tutorial use case
    "online_strategy_kwargs": {
        "strategy_name": "normal",
        "confidence_level": 0.95,
        "use_fpc": True,
    },

Notation 
^^^^^^^

* :math:`N` = Total population size (total number of queries in eval set)
* :math:`n` = Sample size (number of queries processed so far)
* :math:`\hat{p}` = Observed sample proportion or average for an algebraic metric
* :math:`\bar{X}` = Sample mean for a distributive metric
* :math:`\widehat{T}` = Estimated population total for a distributive metric
* :math:`\text{Var}(\widehat{T})` = Variance of the above estimated population total
* :math:`\text{SE}` = Standard error (measure of estimate uncertainty)
* :math:`\text{CI}` = Confidence interval
* :math:`z` = Z-score for confidence level (1.96 for 95% confidence; used in Normal and Wilson)
* :math:`\alpha` = Significance level (0.05 for 95% confidence)
* :math:`n_{\text{eff}}` = Effective sample size (adjusted for FPC in Wilson)
* :math:`a, b` = Lower and upper bounds of metric value range
* :math:`R` = Range width, :math:`R = b - a`
* :math:`\varepsilon` = Margin of error (half-width of confidence interval for Hoeffding)
* :math:`\varepsilon_{\bar{X}}` = Margin of error for sample mean (Hoeffding distributive)
* :math:`\text{FPC}` = Finite population correction factor


Finite Population Correction (FPC)
^^^^^^^^^^^^^^^^^^^^^^

When sampling without replacement from finite populations, enabling FPC 
multiplies the standard error (SE) by :math:`\text{FPC} = \sqrt{(N-n)/(N-1)}` 
where :math:`N` is population size and :math:`n` is sample size.


Normal Approximation
^^^^^^^^^^^^^^^^^^^

This is the default strategy, and it uses the Central Limit Theorem. 
It is suitable for most cases with non-trivial sample sizes (n > 30). 
It provides tight intervals when the statistical assumptions hold.

* For algebraic metrics:

.. math::

   \text{SE}_{\hat{p}} = \sqrt{\frac{\hat{p}(1-\hat{p})}{n}} \times \text{FPC}

   \text{CI} = \hat{p} \pm 1.96 \cdot \text{SE}_{\hat{p}}


* For distributive metrics: 

Estimate population total :math:`\widehat{T} = N\bar{X}` with 
variance :math:`\text{Var}(\widehat{T}) = N^2 \cdot \bar{X}(1-\bar{X})/n` (FPC-adjusted).


Wilson Score
^^^^^^^^^^^

This strategy is better for small sample sizes or metrics near 0/1 boundaries. 
It is more robust than Normal Approximation for extreme proportions. 

* For algebraic metrics:

.. math::

   \text{center} = \frac{\hat{p} + z^2/(2n_{\text{eff}})}{1 + z^2/n_{\text{eff}}}

   \text{margin} = \frac{z\sqrt{\hat{p}(1-\hat{p})/n_{\text{eff}} + z^2/(4n_{\text{eff}}^2)}}{1 + z^2/n_{\text{eff}}}

where :math:`n_{\text{eff}} = n/\text{FPC}^2` when using FPC. 
The Wilson confidence interval is then :math:`[\text{center} - \text{margin}, \text{center} + \text{margin}]`,
clamped to [0, 1].

* For distributive metrics, this falls back to Normal Approximation. 



Hoeffding Bounds
^^^^^^^^^^^

This strategy is best for maximum safety (guaranteed coverage). It makes no distributional assumptions, 
but that also means its intervals are typically quite loose.

.. math::

   \varepsilon = (b-a)\sqrt{\frac{\ln(2/\alpha)}{2n}} \times \text{FPC}

   \text{CI} = [\hat{p} - \varepsilon, \hat{p} + \varepsilon]

For distributive metrics with range :math:`R=b-a`, it computes :math:`\varepsilon_{\bar{X}} = R\sqrt{\ln(2/\alpha)/(2n)}` 
and then scales to population total.


Example Impact with IC Ops
-------

Returning to our above example, suppose we have an evals set with 400 queries. We compare
16 RAG configs (say, 2 retrieval, 2 reranking, and 4 prompting schemes) with 8 shards.
Let us say it takes 10s on average per query. And you run this on one machine.

**Traditional Batch Evals**: You explore the configs sequentially, one after another.

Total time taken: 16 configs x 400 queries x 10s/query = **17.8 hours**

**RapidFire AI**: Suppose you stop the worst 15 configs after the first shard based 
on the projected estimates and confidence intervals; 
then clone the best config and add 4 new variants then stop the worst 4 configs 
after the second shard; then let only the best config continue till the last shard.

Total time taken: 16 * (400/8) * 10s + 5 * (400/8) * 10s + 1 * (6 * 400/8) * 10s = **3.8 hours**

**Result**: Runtime cut by *4.7x*, alongside likely *better eval metrics*. 

Of course, batch evals is also *embarrassingly parallel*. So, if you are able to use, say, 
10 machines for the above workload, the two runtimes will come down to 107min vs. 23min.

Note that apart from time, on closed model APIs, your token spend is also reallocated 
to more productive configs, which offers better return on investment for your use case.


Why Not Just Downsample?
------------------------

One might wonder why downsampling the eval set does not suffice here. 
:doc:`As also explained on this page</difference>`, downsampling alone has 
significant disadvantages compared to the approach offered by RapidFire AI. 

First, you have to decide a downsample size upfront, which is not trivial if your
eval metrics have high variance across examples. Point estimates without confidence 
intervals can give false confidence in a sample. You can resample manually 
over and over, but that adds manual grunt work of juggling separate samples/files. 
Finally, downsampling alone does not offer you the power of IC Ops and automated 
parallelization to try new configs on the fly--you'd have reimplement those manually.

RapidFire AI's online aggregation approach with IC Ops avoids all the above issues,
while also being **complementary** to downsampling, i.e., you can use both in 
conjunction for even lower runtimes/costs.


Takeaways
----------

RapidFire AI transforms RAG/context engineering evals from a slow, tedious 
guessing game into a fast, dynamic engineering process. 
With real-time metrics, statistically sound confidence intervals, and dynamic control, 
you can compare far more configs efficiently, make data-driven decisions, and 
optimize across the Pareto frontier of eval metrics, cost, and time.
