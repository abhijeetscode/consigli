"""
Pydantic models for document processing
"""

from pydantic import BaseModel 


class DocumentMetadata(BaseModel):
    """Metadata for document processing statistics"""
    total_texts: int
    total_tables: int
    total_elements: int


class ProcessedDocument(BaseModel):
    """Represents a processed PDF document"""
    company: str
    year: int | None = None
    pdf_path: str
    pdf_name: str
    texts: list[dict[str, object]]
    tables: list[dict[str, object]]
    metadata: DocumentMetadata


class ElementMetadata(BaseModel):
    """Metadata for extracted PDF elements"""
    page_number: int | None = None
    filename: str | None = None
    filetype: str | None = None


class ExtractedElement(BaseModel):
    """Represents a single extracted element from PDF"""
    type: str
    text: str
    metadata: ElementMetadata
