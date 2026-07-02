import os
import uuid
import hashlib
from datetime import datetime
from typing import Dict, Any, List

from ingestion.base_parser import BaseParser
from app.models.financial_document import (
    FinancialDocument, DocumentMetadata, DocumentType, FileFormat, ProcessingStatus
)

try:
    import pypdf
except ImportError:
    pypdf = None

class PDFParser(BaseParser):
    def parse(self, file_path: str) -> FinancialDocument:
        metadata = self.extract_metadata(file_path)
        # Store for the next pipeline stage (Cleaning -> Section Detection)
        text = self.extract_text(file_path)
        tables = self.extract_tables(file_path)
        
        # Create a valid FinancialDocument
        doc_metadata = DocumentMetadata(
            company_name=metadata.get("company_name", "Unknown"),
            document_type=DocumentType.ANNUAL_REPORT,
            pages=metadata.get("pages", 0),
            source_filename=os.path.basename(file_path),
            file_format=FileFormat.PDF,
            document_hash=metadata.get("hash", ""),
            created_at=datetime.utcnow(),
            parser_version="1.0"
        )
        
        doc = FinancialDocument(
            document_id=str(uuid.uuid4()),
            metadata=doc_metadata,
            processing_status=ProcessingStatus.PARSED
        )
        
        return doc

    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        metadata = {}
        if not os.path.exists(file_path):
            return metadata
            
        if pypdf:
            with open(file_path, "rb") as f:
                reader = pypdf.PdfReader(f)
                metadata["pages"] = len(reader.pages)
                if reader.metadata:
                    metadata["title"] = reader.metadata.title
        else:
            metadata["pages"] = 0
                
        # Extract company name generically from the first page using common SEC 10-K markers
        metadata["company_name"] = "Unknown"
        if pypdf:
            try:
                with open(file_path, "rb") as f:
                    reader = pypdf.PdfReader(f)
                    if len(reader.pages) > 0:
                        first_page = reader.pages[0].extract_text()
                        lines = [line.strip() for line in first_page.split('\n') if line.strip()]
                        for i, line in enumerate(lines):
                            line_lower = line.lower()
                            if ("(exact name of registrant" in line_lower or 
                                "(state of incorporation" in line_lower or 
                                "(state or other jurisdiction" in line_lower):
                                if i > 0:
                                    name = lines[i-1]
                                    if name.upper() in ["OR", "AND"] and i > 1:
                                        name = lines[i-2]
                                    metadata["company_name"] = name
                                break
            except Exception:
                pass
                
        # Generate hash
        with open(file_path, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        metadata["hash"] = file_hash
        
        return metadata

    def extract_text(self, file_path: str) -> str:
        text = ""
        if not os.path.exists(file_path):
            return text
            
        if pypdf:
            with open(file_path, "rb") as f:
                reader = pypdf.PdfReader(f)
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
        return text

    def extract_tables(self, file_path: str) -> List[Dict[str, Any]]:
        return []
