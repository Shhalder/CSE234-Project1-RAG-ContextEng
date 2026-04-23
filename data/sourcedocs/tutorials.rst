Example Use Case Tutorials
==================

These end-to-end examples serve as tutorials for the usage of RapidFire AI's API.
They use publicly available datasets, models, and code.


For RAG and Context Engineering
------

We have one use case example each for an all-local model, all-OpenAI, and a hybrid workflow: 
FiQA RAG Q&A chatbot, SciFact RAG for scientific claim verification, and 
GSM8K few-shot/context engineering for math reasoning, respectively.
This set will expand over time to more examples based on community inputs.

You can also check out `RapidFire AI RAG on Google Colab <https://tinyurl.com/rapidfireai-rag-colab>`_ instead of your own machine; 
it showcases a simplified FiQA RAG Q&A chatbot use case.


.. toctree::

   rag_fiqa
   rag_gsm8k
   rag_scifact


For Fine-Tuning and Post-Training
------

We have one use case example for each supported control flow from HuggingFace TRL: 
SFT, DPO, and GRPO. 
This set will expand over time to more examples based on community inputs. 
The SFT use case has multiple additional variations showcased, including a "lite" version 
with a fast SLM and multiple FSDP-based versions with larger models that are auto-partitioned across GPUs.

You can also check out `RapidFire AI on Google Colab <https://tinyurl.com/rapidfireai-colab>`_ instead of your own machine; 
it showcases a simplified SFT use case.

.. toctree::

   sft
   dpo
   grpo
