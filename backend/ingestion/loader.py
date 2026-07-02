import os
from ingestion.parser_factory import ParserFactory
from app.models.financial_document import FinancialDocument

class DocumentLoader:
    def load(self, file_path: str) -> FinancialDocument:
        # 1. Validate file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # 2. Ask ParserFactory for the correct parser
        parser = ParserFactory.get_parser(file_path)
        
        # 3. Call parser.parse()
        document = parser.parse(file_path)
        
        # 4. Return FinancialDocument
        return document
