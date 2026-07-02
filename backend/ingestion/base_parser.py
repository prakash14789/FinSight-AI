from abc import ABC, abstractmethod
from typing import Dict, Any, List
from app.models.financial_document import FinancialDocument

class BaseParser(ABC):
    @abstractmethod
    def parse(self, file_path: str) -> FinancialDocument:
        """Parse the document and return a FinancialDocument model."""
        pass

    @abstractmethod
    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from the document."""
        pass

    @abstractmethod
    def extract_text(self, file_path: str) -> str:
        """Extract raw text from the document."""
        pass

    @abstractmethod
    def extract_tables(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract tables from the document."""
        pass
