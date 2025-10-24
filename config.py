"""
Configuration file for RAG Financial Document Analysis
"""

import os
from pathlib import Path
import torch


def get_device():
    """
    Auto-detect best available device for embeddings
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


PROJECT_ROOT = Path(__file__).parent
DATA_FOLDER = str(PROJECT_ROOT / "Data")
CACHE_DIR = str(PROJECT_ROOT / "cache")
VECTOR_STORE_DIR = str(PROJECT_ROOT / "chroma_db")
WARMUP_CACHE_DIR = str(PROJECT_ROOT / "warmup_cache")

COMPANIES = ["BMW", "Ford", "Tesla"]

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

CHUNKING_CONFIG = {
    "chunk_size": 3000,  
    "chunk_overlap": 100,  
    "separator": " ",
}

VECTOR_STORE_CONFIG = {
    "collection_name": "financial_docs",
    "similarity_top_k": 5,
}

EMBEDDING_CONFIG = {
    "provider": "openai",
    "model_name": "text-embedding-3-small",
}

OPENAI_CONFIG = {
    "model": "gpt-4",
    "temperature": 0,
}

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

QUERY_ENGINE_CONFIG = {
    "use_query_refinement": False,
    "verbose": True,
}

CHATBOT_CONFIG = {
    "chat_mode": "condense_question",
    "verbose": True,
    "use_query_refinement": False,
}

WARMUP_CACHE_CONFIG = {
    "cache_file": str(PROJECT_ROOT / "warmup_cache" / "warmup_answers.json"),
    "enabled": True,
}
