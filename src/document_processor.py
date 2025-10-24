"""
Document Processor Module
"""

from unstructured.partition.pdf import partition_pdf  
import json
import os
from pathlib import Path
from src.models import ProcessedDocument, DocumentMetadata


def get_cache_path(pdf_path: str, cache_dir: str = "./cache") -> str:
    """
    Generate cache file path for a given PDF

    Args:
        pdf_path: Path to PDF file
        cache_dir: Directory to store cache files

    Returns:
        Path to cache file
    """
    pdf_name = Path(pdf_path).stem
    cache_file = f"{pdf_name}_elements.json"
    return os.path.join(cache_dir, cache_file)


def save_elements_to_cache(
    elements: list[object],
    pdf_path: str,
    cache_dir: str = "./cache"
) -> str:
    """
    Save extracted elements to cache as JSON

    Args:
        elements: List of extracted elements from partition_pdf
        pdf_path: Path to PDF file
        cache_dir: Directory to store cache files

    Returns:
        Path to cache file
    """
    os.makedirs(cache_dir, exist_ok=True)

    cache_path = get_cache_path(pdf_path, cache_dir)

    cache_data = {
        "pdf_path": pdf_path,
        "pdf_name": Path(pdf_path).name,
        "total_elements": len(elements),
        "elements": []
    }

    for element in elements:
        element_data = {
            "type": str(type(element).__name__),
            "text": str(element),
            "metadata": {}
        }

        if hasattr(element, 'metadata'):
            metadata_obj = element.metadata  # type: ignore
            element_data["metadata"] = {
                "page_number": getattr(metadata_obj, 'page_number', None),
                "filename": getattr(metadata_obj, 'filename', None),
                "filetype": getattr(metadata_obj, 'filetype', None),
            }

        cache_data["elements"].append(element_data)

    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(cache_data, f, indent=2, ensure_ascii=False)

    return cache_path


def load_elements_from_cache(
    pdf_path: str,
    cache_dir: str = "./cache"
) -> list[dict[str, object]] | None:
    """
    Load extracted elements from cache if available

    Args:
        pdf_path: Path to PDF file
        cache_dir: Directory where cache files are stored

    Returns:
        List of cached elements or None if cache not found
    """
    cache_path = get_cache_path(pdf_path, cache_dir)

    if not os.path.exists(cache_path):
        return None

    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return None

    return cache_data["elements"]


def extract_pdf_elements(
    pdf_path: str,
    use_cache: bool = True,
    cache_dir: str = "./cache"
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """
    Extract text and tables from PDF with caching support
    No chunking - will be handled by LlamaIndex

    Args:
        pdf_path: Path to PDF file
        use_cache: Whether to use cached results if available
        cache_dir: Directory to store cache files

    Returns:
        Tuple of (texts, tables) - lists of extracted elements as dicts
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if use_cache:
        cached_elements = load_elements_from_cache(pdf_path, cache_dir)
        if cached_elements is not None:
            texts = [el for el in cached_elements if el["type"] not in ["Table", "Image"]]
            tables = [el for el in cached_elements if el["type"] == "Table"]
            return texts, tables

    from config import PDF_EXTRACTION

    elements = partition_pdf(
        filename=pdf_path,
        strategy=PDF_EXTRACTION["strategy"],
        extract_images_in_pdf=PDF_EXTRACTION["extract_images_in_pdf"],
        infer_table_structure=PDF_EXTRACTION["infer_table_structure"],
        chunking_strategy=PDF_EXTRACTION["chunking_strategy"],
        max_characters=PDF_EXTRACTION["max_characters"],
        new_after_n_chars=PDF_EXTRACTION["new_after_n_chars"],
        combine_text_under_n_chars=PDF_EXTRACTION["combine_text_under_n_chars"],
    )

    save_elements_to_cache(elements, pdf_path, cache_dir) # type: ignore

    texts = []
    tables = []

    for element in elements:
        element_type = str(type(element).__name__)
        
        if "Image" in element_type:
            continue
            
        element_data = {
            "type": element_type,
            "text": str(element),
            "metadata": {}
        }

        if hasattr(element, 'metadata'):
            metadata_obj = element.metadata  # type: ignore
            element_data["metadata"] = {
                "page_number": getattr(metadata_obj, 'page_number', None),
                "filename": getattr(metadata_obj, 'filename', None),
                "filetype": getattr(metadata_obj, 'filetype', None),
            }

        if "Table" in element_type:
            tables.append(element_data)
        else:
            texts.append(element_data)

    return texts, tables


def process_all_pdfs(
    data_folder: str,
    use_cache: bool = True,
    cache_dir: str = "./cache"
) -> dict[str, ProcessedDocument]:
    """
    Process all PDFs in the Data folder structure

    Expected structure:
        Data/
            BMW/BMW_Annual_Report_2021.pdf
            Ford/Ford_Annual_Report_2021.pdf
            Tesla/Tesla_Annual_Report_2022.pdf

    Note: Processes only financial annual reports (BMW, Ford, Tesla).
    news.pdf is intentionally excluded to maintain domain consistency
    with FinBERT embeddings optimized for financial documents.

    Args:
        data_folder: Path to Data folder
        use_cache: Whether to use cached results
        cache_dir: Directory to store cache files

    Returns:
        Dictionary with structure:
        {
            "BMW_2023": {"texts": [...], "tables": [...], "metadata": {...}},
            "Ford_2022": {...},
            "Tesla_2023": {...},
        }
    """
    import glob
    import re
    from config import COMPANIES

    companies = COMPANIES
    all_documents = {}

    for company in companies:
        company_folder = os.path.join(data_folder, company)

        if not os.path.exists(company_folder):
            continue

        pdf_files = glob.glob(os.path.join(company_folder, "*.pdf"))

        for pdf_path in pdf_files:
            match = re.search(r'(\d{4})', Path(pdf_path).name)
            year = int(match.group(1)) if match else None

            if not year:
                continue

            texts, tables = extract_pdf_elements(
                pdf_path,
                use_cache=use_cache,
                cache_dir=cache_dir
            )

            doc_key = f"{company}_{year}"
            all_documents[doc_key] = ProcessedDocument(
                company=company,
                year=year,
                pdf_path=pdf_path,
                pdf_name=Path(pdf_path).name,
                texts=texts,
                tables=tables,
                metadata=DocumentMetadata(
                    total_texts=len(texts),
                    total_tables=len(tables),
                    total_elements=len(texts) + len(tables)
                )
            )


    return all_documents
