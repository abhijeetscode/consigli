"""
Configuration file for RAG Financial Document Analysis
"""

import os
from pathlib import Path
import torch


def get_device():
    """
    Auto-detect best available device for embeddings
    Priority: CUDA (NVIDIA GPU) > MPS (Apple Silicon) > CPU
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_FOLDER = str(PROJECT_ROOT / "Data")
CACHE_DIR = str(PROJECT_ROOT / "cache")
VECTOR_STORE_DIR = str(PROJECT_ROOT / "chroma_db")
WARMUP_CACHE_DIR = str(PROJECT_ROOT / "warmup_cache")

# Companies to process
COMPANIES = ["BMW", "Ford", "Tesla"]

# PDF extraction settings with chunking by title
PDF_EXTRACTION = {
    "strategy": "hi_res",
    "extract_images_in_pdf": False,
    "infer_table_structure": True,
    "chunking_strategy": "by_title",
    "max_characters": 4000,
    "new_after_n_chars": 3800,
    "combine_text_under_n_chars": 2000,
    "use_cache": True,
}

# LlamaIndex chunking settings
# Larger chunks capture more context for financial data
CHUNKING_CONFIG = {
    "chunk_size": 3000,  
    "chunk_overlap": 100,  
    "separator": " ",
}

# Vector store settings
VECTOR_STORE_CONFIG = {
    "collection_name": "financial_docs",
    "similarity_top_k": 5,  # Retrieve more chunks for comprehensive multi-company context
}

# Embedding model settings
# Using OpenAI embeddings for best retrieval quality
EMBEDDING_CONFIG = {
    "provider": "openai",
    "model_name": "text-embedding-3-small",  # High quality, cost effective
}

# OpenAI settings (for LLM generation, not embeddings)
OPENAI_CONFIG = {
    "model": "gpt-4",
    "temperature": 0,
}

# Environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Query engine settings
QUERY_ENGINE_CONFIG = {
    "use_query_refinement": False,  # Disabled for consistency
    "verbose": True,
}

# Chatbot settings
CHATBOT_CONFIG = {
    "chat_mode": "condense_question",  # or "context"
    "verbose": True,
    "use_query_refinement": False,  # Disabled for consistency
}

# Warm-up question cache settings
WARMUP_CACHE_CONFIG = {
    "cache_file": str(PROJECT_ROOT / "warmup_cache" / "warmup_answers.json"),
    "enabled": True,
}
