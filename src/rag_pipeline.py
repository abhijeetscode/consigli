"""
RAG Pipeline Module
Converts extracted documents to LlamaIndex format and creates vector store
"""

from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
import chromadb
from src.models import ProcessedDocument
from config import VECTOR_STORE_DIR, CHUNKING_CONFIG, EMBEDDING_CONFIG, VECTOR_STORE_CONFIG


def convert_to_llamaindex_documents(
    processed_docs: dict[str, ProcessedDocument]
) -> list[Document]:
    """
    Convert processed documents to LlamaIndex Document format

    Args:
        processed_docs: Dictionary of ProcessedDocument objects from document_processor

    Returns:
        List of LlamaIndex Document objects with metadata
    """
    llamaindex_docs = []

    for doc_key, proc_doc in processed_docs.items():
        # Convert text elements
        for text_elem in proc_doc.texts:
            doc = Document(
                text=text_elem["text"],
                metadata={
                    "company": proc_doc.company,
                    "year": proc_doc.year,
                    "doc_key": doc_key,
                    "element_type": text_elem["type"],
                    "page_number": text_elem["metadata"].get("page_number"),
                    "source": proc_doc.pdf_name,
                    "content_type": "text"
                }
            )
            llamaindex_docs.append(doc)

        # Convert table elements
        for table_elem in proc_doc.tables:
            doc = Document(
                text=table_elem["text"],
                metadata={
                    "company": proc_doc.company,
                    "year": proc_doc.year,
                    "doc_key": doc_key,
                    "element_type": "Table",
                    "page_number": table_elem["metadata"].get("page_number"),
                    "source": proc_doc.pdf_name,
                    "content_type": "table"
                }
            )
            llamaindex_docs.append(doc)

    return llamaindex_docs


def create_vector_store_index(
    documents: list[Document],
    collection_name: str = VECTOR_STORE_CONFIG["collection_name"],
    persist_dir: str = VECTOR_STORE_DIR
) -> VectorStoreIndex:
    """
    Create vector store index with ChromaDB

    Args:
        documents: List of LlamaIndex Document objects
        collection_name: Name for ChromaDB collection
        persist_dir: Directory to persist vector store

    Returns:
        VectorStoreIndex ready for querying
    """
    # Initialize ChromaDB client
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    chroma_collection = chroma_client.get_or_create_collection(collection_name)

    # Create vector store
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Configure embeddings based on provider
    print(f"Loading embedding model: {EMBEDDING_CONFIG['model_name']}")
    if EMBEDDING_CONFIG["provider"] == "openai":
        embed_model = OpenAIEmbedding(
            model=EMBEDDING_CONFIG["model_name"]
        )
    else:
        embed_model = HuggingFaceEmbedding(
            model_name=EMBEDDING_CONFIG["model_name"],
            device=EMBEDDING_CONFIG.get("device", "cpu")
        )

    # No additional chunking needed - Unstructured handles it during PDF extraction
    # Create index without text splitter
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True
    )

    return index


def load_existing_index(
    collection_name: str = VECTOR_STORE_CONFIG["collection_name"],
    persist_dir: str = VECTOR_STORE_DIR
) -> VectorStoreIndex | None:
    """
    Load existing vector store index if available

    Args:
        collection_name: Name of ChromaDB collection
        persist_dir: Directory where vector store is persisted

    Returns:
        VectorStoreIndex if exists, None otherwise

    Raises:
        Exception: Any errors during index loading (except collection not found)
    """
    import os

    # Check if vector store directory exists
    if not os.path.exists(persist_dir):
        return None

    chroma_client = chromadb.PersistentClient(path=persist_dir)

    # Try to get collection - return None if doesn't exist
    try:
        chroma_collection = chroma_client.get_collection(collection_name)
    except Exception:
        # Collection doesn't exist yet - this is expected on first run
        return None

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # Configure embeddings based on provider
    if EMBEDDING_CONFIG["provider"] == "openai":
        embed_model = OpenAIEmbedding(
            model=EMBEDDING_CONFIG["model_name"]
        )
    else:
        embed_model = HuggingFaceEmbedding(
            model_name=EMBEDDING_CONFIG["model_name"],
            device=EMBEDDING_CONFIG.get("device", "cpu")
        )

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model
    )

    return index


def build_rag_pipeline(
    processed_docs: dict[str, ProcessedDocument],
    force_rebuild: bool = False
) -> VectorStoreIndex:
    """
    Build complete RAG pipeline from processed documents

    Args:
        processed_docs: Dictionary of ProcessedDocument objects
        force_rebuild: If True, rebuild index even if exists

    Returns:
        VectorStoreIndex ready for querying
    """
    # Try to load existing index
    if not force_rebuild:
        print("Checking for existing vector store...")
        existing_index = load_existing_index()
        if existing_index:
            print("✓ Loaded existing vector store")
            return existing_index

    print("Building new vector store...")
    print(f"Converting {len(processed_docs)} documents to LlamaIndex format...")

    # Convert documents
    llamaindex_docs = convert_to_llamaindex_documents(processed_docs)
    print(f"✓ Converted to {len(llamaindex_docs)} LlamaIndex documents")

    # Create vector store index
    print("Creating vector embeddings (this may take a few minutes)...")
    index = create_vector_store_index(llamaindex_docs)
    print("✓ Vector store created successfully")

    return index
