from enum import Enum
from typing import List, Optional, Any
from pydantic import BaseModel, Field
from datetime import date, datetime

class ProcessingStatus(str, Enum):
    UPLOADED = "UPLOADED"
    PARSED = "PARSED"
    CLEANED = "CLEANED"
    SECTIONED = "SECTIONED"
    CHUNKED = "CHUNKED"
    EMBEDDED = "EMBEDDED"
    STORED = "STORED"
    READY = "READY"
    FAILED = "FAILED"

class SectionType(str, Enum):
    BUSINESS = "BUSINESS"
    RISK_FACTORS = "RISK_FACTORS"
    MDNA = "MDNA"
    FINANCIAL_STATEMENTS = "FINANCIAL_STATEMENTS"
    CYBERSECURITY = "CYBERSECURITY"
    NOTES = "NOTES"
    PRODUCTS = "PRODUCTS"
    SERVICES = "SERVICES"
    COMPETITION = "COMPETITION"
    HUMAN_CAPITAL = "HUMAN_CAPITAL"
    ESG = "ESG"
    GOVERNANCE = "GOVERNANCE"
    LEGAL_PROCEEDINGS = "LEGAL_PROCEEDINGS"
    PROPERTIES = "PROPERTIES"
    EXECUTIVE_OFFICERS = "EXECUTIVE_OFFICERS"
    MARKET_INFORMATION = "MARKET_INFORMATION"
    OTHER = "OTHER"

class DocumentType(str, Enum):
    ANNUAL_REPORT = "ANNUAL_REPORT"
    QUARTERLY_REPORT = "QUARTERLY_REPORT"
    EARNINGS_TRANSCRIPT = "EARNINGS_TRANSCRIPT"
    INVESTOR_PRESENTATION = "INVESTOR_PRESENTATION"
    OTHER = "OTHER"

class FileFormat(str, Enum):
    PDF = "PDF"
    DOCX = "DOCX"
    HTML = "HTML"
    CSV = "CSV"
    XLSX = "XLSX"

class DocumentMetadata(BaseModel):
    company_name: str
    ticker: Optional[str] = None
    company_cik: Optional[str] = None
    fiscal_year: Optional[int] = None
    filing_date: Optional[date] = None
    document_type: DocumentType
    pages: int
    language: str = "en"
    source_filename: str
    file_format: FileFormat
    document_hash: str
    created_at: datetime
    parser_version: str

class DocumentSection(BaseModel):
    section_id: str
    parent_section_id: Optional[str] = None
    level: int = Field(default=1, ge=1)
    item_number: Optional[str] = None
    section_type: SectionType
    title: str
    content: str
    page_start: int
    page_end: int

class DocumentTable(BaseModel):
    table_id: str
    title: Optional[str] = None
    columns: List[str]
    rows: List[List[Any]]
    page_start: int
    page_end: int

class DocumentFigure(BaseModel):
    figure_id: str
    title: Optional[str] = None
    description: Optional[str] = None
    page_number: int
    image_path: Optional[str] = None

class DocumentChunk(BaseModel):
    chunk_id: str
    chunk_index: int
    text: str
    token_count: int
    section_id: Optional[str] = None
    section_title: Optional[str] = None
    page_start: int
    page_end: int

class ValidationError(BaseModel):
    error_id: str
    stage: str
    message: str

class FinancialDocument(BaseModel):
    document_id: str
    metadata: DocumentMetadata
    processing_status: ProcessingStatus = ProcessingStatus.UPLOADED
    sections: List[DocumentSection] = Field(default_factory=list)
    tables: List[DocumentTable] = Field(default_factory=list)
    figures: List[DocumentFigure] = Field(default_factory=list)
    chunks: List[DocumentChunk] = Field(default_factory=list)
    validation_errors: List[ValidationError] = Field(default_factory=list)
