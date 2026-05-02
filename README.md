Imported packages:

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import tiktoken
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from rapidfireai.automl import RFLangChainRagSpec, RFOpenAIAPIModelConfig
from sentence_transformers import CrossEncoder
