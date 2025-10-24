"""
Main entry point for RAG Financial Document Analysis System
"""

import os
from src.document_processor import process_all_pdfs
from src.rag_pipeline import build_rag_pipeline
from config import DATA_FOLDER, CACHE_DIR, VECTOR_STORE_DIR



def main():
    """
    Main execution pipeline demonstrating end-to-end RAG system
    """
    print("="*70)
    print("RAG FINANCIAL DOCUMENT ANALYSIS SYSTEM")
    print("="*70)
    print("\nTechnical Interview Assignment - Consigli")
    print("Automotive Sector Financial Analysis (BMW, Ford, Tesla)")
    print("\nNote: Using FinBERT embeddings for optimal financial document retrieval")
    print()

    print("\n" + "="*70)
    print("PHASE 1: DOCUMENT EXTRACTION & PROCESSING")
    print("="*70)
    print(f"\nSource: {DATA_FOLDER}")
    print(f"Cache: {CACHE_DIR}")
    print("\nProcessing Strategy:")
    print("  - High-resolution PDF parsing with layout detection")
    print("  - Structured table extraction for financial data")
    print("  - Intelligent caching to avoid redundant processing")
    print("  - Metadata preservation (company, year, page numbers)")
    print()

    all_documents = process_all_pdfs(
        data_folder=DATA_FOLDER,
        use_cache=True,
        cache_dir=CACHE_DIR
    )

    print(f"\n{'='*70}")
    print("EXTRACTION COMPLETE")
    print(f"{'='*70}")
    print(f"\nDocuments Processed: {len(all_documents)}")
    print(f"\n{'Company':<15} {'Year':<6} {'Elements':<10} {'Tables':<8} {'Text':<8}")
    print("-"*70)

    total_elements = 0
    total_tables = 0
    for doc_data in sorted(all_documents.values(), key=lambda x: (x.company, x.year or 0)):
        year_str = str(doc_data.year) if doc_data.year else "N/A"
        print(f"{doc_data.company:<15} {year_str:<6} {doc_data.metadata.total_elements:<10} "
              f"{doc_data.metadata.total_tables:<8} {doc_data.metadata.total_texts:<8}")
        total_elements += doc_data.metadata.total_elements
        total_tables += doc_data.metadata.total_tables

    print("-"*70)
    print(f"{'TOTAL':<22} {total_elements:<10} {total_tables:<8}")

    print("\n\n" + "="*70)
    print("PHASE 2: RAG PIPELINE - VECTOR STORE CREATION")
    print("="*70)
    print(f"\nVector Store: {VECTOR_STORE_DIR}")
    print("\nRAG Architecture:")
    print("  - Document chunking with semantic overlap (1024 chars, 200 overlap)")
    print("  - OpenAI embeddings (text-embedding-3-small)")
    print("  - ChromaDB for persistent vector storage")
    print("  - Metadata filtering for company/year-specific queries")
    print()

    index = build_rag_pipeline(all_documents, force_rebuild=False)

    print(f"\n{'='*70}")
    print("VECTOR STORE READY")
    print(f"{'='*70}")

    print("\n\n" + "="*70)
    print("SYSTEM INITIALIZATION COMPLETE")
    print("="*70)

    print("\nSystem Components:")
    print(f"  - Document Cache:    {CACHE_DIR}")
    print(f"  - Vector Store:      {VECTOR_STORE_DIR}")
    print(f"  - Total Documents:   {len(all_documents)}")
    print(f"  - Total Elements:    {total_elements}")
    print(f"  - Financial Tables:  {total_tables}")

    print("\nCapabilities:")
    print("  - Company-specific financial queries")
    print("  - Multi-year comparative analysis")
    print("  - Table-aware retrieval for numerical data")
    print("  - Semantic search across all documents")

    print("\nExample Queries:")
    print("  - What was BMW's total revenue in 2023?")
    print("  - Compare Tesla and Ford profits in 2022")
    print("  - What economic factors influenced Ford's 2021 performance?")
    print("  - Provide revenue summary for all companies 2021-2023")

    print("\n" + "="*70)
    print("Next step: Run query engine to test the system")
    print("="*70)
    print()

    return index, all_documents


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("\nWarning: OPENAI_API_KEY not set in environment")
        print("Please set it via: export OPENAI_API_KEY='your-key-here'")
        print()

    try:
        index, documents = main()
        print("\nSystem ready for queries!")

    except Exception as e:
        print(f"\nError during initialization: {e}")
        raise
