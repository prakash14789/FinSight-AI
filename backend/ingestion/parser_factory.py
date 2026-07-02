import os
from ingestion.base_parser import BaseParser
from ingestion.pdf_parser import PDFParser

class ParserFactory:
    @staticmethod
    def get_parser(file_path: str) -> BaseParser:
        _, ext = os.path.splitext(file_path.lower())
        
        if ext == ".pdf":
            return PDFParser()
        else:
            raise ValueError(f"Unsupported file format: {ext}")
