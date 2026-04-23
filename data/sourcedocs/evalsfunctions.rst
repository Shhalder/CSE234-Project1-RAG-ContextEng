API: User-Provided Functions for Run Evals
===============

Users can provide the following custom functions as part of their eval config to be used in :func:`run_evals()`.
Note that each leaf config can have its own set of functions for all of these.


Preprocess Function
-------------------

Mandatory user-provided function to prepare the inputs to be given to the generator model. 
It is invoked for each batch during the evaluation process before generation.
Pass it directly to the :code:`preprocess_fn` key in your eval config dictionary.

The system injects into this function the batch data, as well as the RAG spec and 
the prompt manager of an individual leaf config.


.. py:function:: preprocess_fn(batch: dict[str, list], rag: RFLangChainRagSpec, prompt_manager: RFPromptManager) -> dict[str, list]

   :param batch: Dictionary with a batch of examples with dataset field names as keys and lists as values
   :type batch: dict[str, list]

   :param rag: RAG specification object for document chunk retrieval and context serialization
   :type rag: RFLangChainRagSpec

   :param prompt_manager: Prompt manager object for handling instructions and few-shot examples
   :type prompt_manager: RFPromptManager

   :return: Dictionary with the preprocessed batch. It must have a reserved key :code:`"prompts"` for the fully formatted prompts for the generator. Other key-value pairs from the original batch can also be copied over if you want.
   :rtype: dict[str, list]


**Examples:**

.. code-block:: python

    # Example 1 from FiQA use case: RAG-based preprocessing with document chunk retrieval
    # This example demonstrates how metadata fields ingested via metadata_func in the
    # document loader (e.g., "corpus_id") are accessible on each Document object's
    # .metadata dict after retrieval, enabling retrieval evaluation.
    def sample_preprocess_fn(batch: dict[str, list], rag: RFLangChainRagSpec, prompt_manager: RFPromptManager) -> dict[str, list]:
		"""Function to prepare the final inputs given to the generator model"""
 
		INSTRUCTIONS = "Utilize your financial knowledge, give your answer or opinion to the input question or subject matter."
 
		# Perform batched retrieval over all queries; returns a list of lists of k documents per query
		all_context = rag.get_context(batch_queries=batch["query"], serialize=False)
 
		# Extract the retrieved document ids from the context.
		# The "corpus_id" metadata field was ingested via metadata_func in the document loader
		# (see RFLangChainRagSpec examples) and is now accessible on each Document object.
		retrieved_documents = [
			[doc.metadata["corpus_id"] for doc in docs] for docs in all_context
		]
 
		# Serialize the retrieved documents into a single string per query using the document_template.
		# If a custom document_template was provided in the RAG spec (e.g., to include title metadata),
		# it is applied here; otherwise the default "metadata:\ncontent" template is used.
		serialized_context = rag.serialize_documents(all_context)
		batch["query_id"] = [int(query_id) for query_id in batch["query_id"]]
 
		# Each batch to contain conversational prompt, retrieved documents, and original 'query_id', 'query', 'metadata'
		return {
			"prompts": [
				[
					{"role": "system", "content": INSTRUCTIONS},
					{
						"role": "user",
						"content": f"Here is some relevant context:\n{context}. \nNow answer the following question using the context provided earlier:\n{question}",
					},
				]
				for question, context in zip(batch["query"], serialized_context)
			],
			"retrieved_documents": retrieved_documents,
			**batch,
		}

.. code-block:: python

    # Example 2 from GSM8K use case: Few-shot learning preprocessing without RAG
    def sample_preprocess_fn(batch: dict[str, list], rag: RFLangChainRagSpec, prompt_manager: RFPromptManager) -> dict[str, list]:
		"""Function to prepare the final inputs given to the generator model"""

		return {
			"prompts": [
				[
					{"role": "system", "content": prompt_manager.get_instructions()},
					{
						"role": "user",
						"content": f"Here are some examples: \n{examples}. \nNow answer the following question:\n{question}",
					},
				]
				for question, examples in zip(
					batch["question"],
					prompt_manager.get_fewshot_examples(user_queries=batch["question"]),
				)
			],
			**batch,
		}

.. code-block:: python

    # Example 3: Multimodal preprocessing with image metadata extraction
    # For multimodal use cases, you can use the unserialized documents retrieved by the RAG
    # pipeline and extract images as URLs or base64 strings from the document metadata.
    # The image data must have been ingested via metadata_func in the document loader
    # (e.g., metadata["image_data"] = record.get("image_base64", "")).
    def sample_multimodal_preprocess_fn(
        batch: dict[str, list], rag: RFLangChainRagSpec, prompt_manager: RFPromptManager
    ) -> dict[str, list]:
        """Function to prepare multimodal inputs with text and images for the generator model"""
 
        INSTRUCTIONS = "Answer the question using the provided context and images."
 
        # Retrieve unserialized documents to access both text content and image metadata
        all_context = rag.get_context(batch_queries=batch["query"], serialize=False)
 
        # Extract base64 image strings from document metadata (one list of images per query).
        # The "image_data" field was ingested via metadata_func in the document loader.
        base64_strings = [
            [doc.metadata.get("image_data", "") for doc in docs_per_query]
            for docs_per_query in all_context
        ]
 
        # Serialize text content separately for the text portion of the prompt
        serialized_context = rag.serialize_documents(all_context)
 
        # Build multimodal conversational prompts with both text and image content blocks
        return {
            "prompts": [
                [
                    {"role": "system", "content": INSTRUCTIONS},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Here is some relevant context:\n{context}.\nNow answer the following question:\n{question}",
                            },
                            *[
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                                }
                                for b64 in images
                                if b64  # Skip empty strings (documents without images)
                            ],
                        ],
                    },
                ]
                for question, context, images in zip(
                    batch["query"], serialized_context, base64_strings
                )
            ],
            **batch,
        }



Postprocess Function
--------------------

Optional user-provided function to postprocess a batch of examples and their respective generated outputs, 
which is injected into this function by the system. 
This is useful for, say, extracting structured information from generations, adding ground truth data, 
or performing any transformations needed before computing metrics.

It is invoked for each batch during the evaluation process after generation and before metric computation.
Pass it directly to the :code:`postprocess_fn` key in your eval config dictionary.


.. py:function:: postprocess_fn(batch: dict[str, list]) -> dict[str, list]

   :param batch: Dictionary containing a batch of examples, including their respective generated outputs under the reserved key :code:`"generated_text"`.
   :type batch: dict[str, list]

   :return: Dictionary containing the postprocessed batch with any new or modified keys
   :rtype: dict[str, list]


**Examples:**

.. code-block:: python

    # Example 1 from FiQA use case: Adding ground truth documents for retrieval evaluation
    def sample_postprocess_fn(batch: dict[str, list]) -> dict[str, list]:
        """Function to postprocess outputs produced by generator model"""

        batch["ground_truth_documents"] = [
            qrels[qrels["query_id"] == query_id]["corpus_id"].tolist()
            for query_id in batch["query_id"]
        ]
        return batch

.. code-block:: python

    # Example 2 from SciFact use case: Extracting structured answers from generated text
    def extract_solution(answer):
		solution = re.search(r"####\s*(SUPPORT|CONTRADICT|NOINFO)", answer, re.IGNORECASE)
		if solution is None:
			return "INVALID"
		return solution.group(1).upper()

    def sample_postprocess_fn(batch: dict[str, list]) -> dict[str, list]:
		"""Function to postprocess outputs produced by generator model"""

		batch["ground_truth_documents"] = [
			qrels[qrels["query_id"] == query_id]["corpus_id"].tolist()
			for query_id in batch["query_id"]
		]
		batch["answer"] = [extract_solution(answer) for answer in batch["generated_text"]]
		return batch



Eval Compute Metrics Function
-------------------------

Mandatory user-provided function to compute eval metrics on a given batch of (postprocessed) examples, 
which is injected by the system. 
It should return metrics computed over the batch as a whole. 

It is invoked for each batch during the evaluation process after generation and postprocessing (if applicable). 
Pass it directly to the :code:`compute_metrics_fn` key in your eval config dictionary.


.. py:function:: eval.compute_metrics_fn(batch: dict[str, list]) -> dict[str, dict[str, Any]]

   :param batch: Dictionary containing a batch of examples, including all preprocessed fields, generated outputs, and any postprocessed fields
   :type batch: dict[str, list]

   :return: Dictionary with a metric's name as key and a dictionary as value inside which a reserved key :code:`"value"` must exist with that corresponding metric's value over this batch of examples.
   :rtype: dict[str, dict[str, Any]]


**Example:**

.. code-block:: python

    # Example 1 from FiQA use case: Metrics on retrieval accuracy
    def sample_compute_metrics_fn(batch: Dict[str, list]) -> Dict[str, Dict[str, Any]]:
		"""Function to compute all eval metrics based on retrievals and/or generations"""

		true_positives, precisions, recalls, f1_scores, ndcgs, rrs = 0, [], [], [], [], []
		total_queries = len(batch["query"])

		for pred, gt in zip(batch["retrieved_documents"], batch["ground_truth_documents"]):
			expected_set = set(gt)
			retrieved_set = set(pred)

			true_positives = len(expected_set.intersection(retrieved_set))
			precision = true_positives / len(retrieved_set) if len(retrieved_set) > 0 else 0
			recall = true_positives / len(expected_set) if len(expected_set) > 0 else 0
			f1 = (
				2 * precision * recall / (precision + recall)
				if (precision + recall) > 0
				else 0
			)

			precisions.append(precision)
			recalls.append(recall)
			f1_scores.append(f1)
			ndcgs.append(compute_ndcg_at_k(retrieved_set, expected_set, k=5))
			rrs.append(compute_rr(retrieved_set, expected_set))

		return {
			"Total": {"value": total_queries},
			"Precision": {"value": sum(precisions) / total_queries},
			"Recall": {"value": sum(recalls) / total_queries},
			"F1 Score": {"value": sum(f1_scores) / total_queries},
			"NDCG@5": {"value": sum(ndcgs) / total_queries},
			"MRR": {"value": sum(rrs) / total_queries},
		}

.. code-block:: python

    # Example 2 from GSM8K use case: Direct answer correctness check
    def sample_compute_metrics_fn(batch: dict[str, list]) -> dict[str, dict[str, Any]]:
        """Function to compute all eval metrics based on retrievals and/or generations"""

        correct = sum(
            1
            for pred, gt in zip(batch["model_answer"], batch["ground_truth"])
            if pred == gt
        )
        total = len(batch["model_answer"])
        return {
            "Correct": {"value": correct},
            "Total": {"value": total},
        }



Eval Accumulate Metrics Function
----------------------------

Optional user-provided function to aggregate algebraic eval metrics across all batches of the data. 
If this function is not provided, all metrics returned by :func:`eval.compute_metrics_fn()` 
will be assumed to be distributive (i.e., summed across batches) by default. Use this function 
when metrics require (weighted) averaging or other custom dataset-wide aggregation logic.

It is invoked once at the very end of the evaluation process after all batches have been processed.
Pass it directly to the :code:`accumulate_metrics_fn` key in your eval config dictionary.


.. py:function:: eval.accumulate_metrics_fn(aggregated_metrics: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]

   :param aggregated_metrics: Dictionary with a metric's name as key and a list of per-batch metric dictionaries as values from across all data batches. Inside each value dictionary, at least the reserved key :code:`"value"` will exist that was returned by your :code:`eval_compute_metrics_fn` function.
   :type aggregated_metrics: dict[str, list[dict[str, Any]]]

   :return: Dictionary with a metric's name as key and a dictionary as value. The value dictionary can have these keys:
   
      * :code:`"value"` (required): The metric's value over the whole dataset.
      * :code:`"is_algebraic"` (optional): Boolean indicating if the metric is algebraic.
      * :code:`"is_distributive"` (optional): Boolean indicating if the metric is distributive.
      * :code:`"value_range"` (optional): 2-tuple of floats specifying the metric's range min and max.
      
      For online aggregation support, see :doc:`the Online Aggregation for Evals page</onlineagg>`.

   :rtype: dict[str, dict[str, Any]]


**Example:**

.. code-block:: python

    # Example 1 from GSM8K use case: Weighted aggregation for accuracy computation
    def sample_accumulate_metrics_fn(aggregated_metrics: dict[str, list]) -> dict[str, dict[str, Any]]:
        """Function to accumulate eval metrics across all batches"""

        correct = sum(m.get("value", 0) for m in aggregated_metrics.get("Correct", [{}]))
        total = sum(m.get("value", 0) for m in aggregated_metrics.get("Total", [{}]))
        accuracy = correct / total if total > 0 else 0
        return {
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

.. code-block:: python

    # Example 2 from FiQA use case: Multiple algebraic metrics for retrieval accuracy
    def sample_accumulate_metrics_fn(aggregated_metrics: dict[str, list]) -> dict[str, dict[str, Any]]:
		"""Function to accumulate eval metrics across all batches"""

		num_queries_per_batch = [m["value"] for m in aggregated_metrics["Total"]]
		total_queries = sum(num_queries_per_batch)
		algebraic_metrics = ["Precision", "Recall", "F1 Score", "NDCG@5", "MRR"]

		return {
			"Total": {"value": total_queries},
			**{
				metric: {
					"value": sum(
						m["value"] * queries
						for m, queries in zip(
							aggregated_metrics[metric], num_queries_per_batch
						)
					)
					/ total_queries,
					"is_algebraic": True,
					"value_range": (0, 1),
				}
				for metric in algebraic_metrics
			},
		}
