import sys
import os

# Add backend directory to Python path for imports
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from ingestion.loader import DocumentLoader
from ingestion.pdf_parser import PDFParser

def main():
    # Step 2: Load one PDF
    pdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../data/raw/annual_reports/microsoft_2025_annual_report.pdf'))
    
    loader = DocumentLoader()
    document = loader.load(pdf_path)

    # Step 3: Verify the output
    print("--- Document Output ---")
    print(f"Document ID: {document.document_id}")
    print(f"Company Name: {document.metadata.company_name}")
    print(f"Pages: {document.metadata.pages}")
    print(f"Document Type: {document.metadata.document_type}")
    print(f"File Format: {document.metadata.file_format}")
    print(f"Document Hash: {document.metadata.document_hash}")
    print(f"Processing Status: {document.processing_status}")
    print("-" * 25)

    # Step 4: Verify text extraction
    parser = PDFParser()
    text = parser.extract_text(pdf_path)
    
    print("\n--- Text Extraction ---")
    print(f"Total Text Length: {len(text)} characters")
    print("Sample (first 1000 characters):")
    print(text[:1000])

if __name__ == "__main__":
    main()
