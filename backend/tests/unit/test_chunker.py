import sys
import os
import unittest
import tiktoken

# Add backend directory to Python path for imports
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from app.models.financial_document import (
    FinancialDocument, DocumentSection, SectionType, DocumentMetadata, DocumentType, FileFormat, ProcessingStatus
)
from preprocessing.chunker import Chunker
from datetime import datetime

class TestChunker(unittest.TestCase):
    def setUp(self):
        # Using a small chunk size and overlap for easier, faster testing
        self.chunker = Chunker(chunk_size=10, chunk_overlap=2)
        self.encoder = tiktoken.get_encoding("cl100k_base")

    def test_invalid_init_parameters(self):
        # Negative chunk size
        with self.assertRaises(ValueError):
            Chunker(chunk_size=-5, chunk_overlap=2)
        
        # Zero chunk size
        with self.assertRaises(ValueError):
            Chunker(chunk_size=0, chunk_overlap=2)

        # Negative overlap
        with self.assertRaises(ValueError):
            Chunker(chunk_size=10, chunk_overlap=-1)

        # Overlap equal to size
        with self.assertRaises(ValueError):
            Chunker(chunk_size=10, chunk_overlap=10)

        # Overlap greater than size
        with self.assertRaises(ValueError):
            Chunker(chunk_size=10, chunk_overlap=12)

    def test_chunk_small_section(self):
        # A small section with few tokens should produce exactly one chunk
        text = "This is a small section."
        tokens = self.encoder.encode(text)
        self.assertTrue(len(tokens) < 10)
        
        section = DocumentSection(
            section_id="doc1_item_1",
            section_type=SectionType.BUSINESS,
            title="Business Section",
            content=text,
            page_start=1,
            page_end=2
        )
        
        chunks = self.chunker._chunk_section(section, "doc1", 0)
        
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].chunk_id, "doc1_chunk_0")
        self.assertEqual(chunks[0].chunk_index, 0)
        self.assertEqual(chunks[0].text, text)
        self.assertEqual(chunks[0].token_count, len(tokens))
        self.assertEqual(chunks[0].section_id, "doc1_item_1")
        self.assertEqual(chunks[0].section_title, "Business Section")
        self.assertEqual(chunks[0].page_start, 1)
        self.assertEqual(chunks[0].page_end, 2)

    def test_chunk_exact_chunk_size(self):
        # Create text that produces exactly 10 tokens (our chunk_size is 10)
        # Using 10 simple space separated words, but let's double check token count
        text = "one two three four five six seven eight nine ten"
        tokens = self.encoder.encode(text)
        # In cl100k_base, "one two three four five six seven eight nine ten" is exactly 10 tokens.
        self.assertEqual(len(tokens), 10)
        
        section = DocumentSection(
            section_id="doc1_item_exact",
            section_type=SectionType.BUSINESS,
            title="Exact Tokens",
            content=text,
            page_start=1,
            page_end=1
        )
        
        chunks = self.chunker._chunk_section(section, "doc1", 0)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].token_count, 10)
        self.assertEqual(chunks[0].text, text)

    def test_chunk_empty_section(self):
        section = DocumentSection(
            section_id="doc1_item_empty",
            section_type=SectionType.OTHER,
            title="Empty Section",
            content="   ",
            page_start=1,
            page_end=1
        )
        chunks = self.chunker._chunk_section(section, "doc1", 0)
        self.assertEqual(len(chunks), 0)

        section_none = DocumentSection(
            section_id="doc1_item_none",
            section_type=SectionType.OTHER,
            title="None Section",
            content="",
            page_start=1,
            page_end=1
        )
        chunks_none = self.chunker._chunk_section(section_none, "doc1", 0)
        self.assertEqual(len(chunks_none), 0)

    def test_chunk_large_section(self):
        # Create a text that generates more than 10 tokens
        text = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen"
        tokens = self.encoder.encode(text)
        token_count = len(tokens)
        self.assertTrue(token_count > 10)
        
        section = DocumentSection(
            section_id="doc1_item_1a",
            section_type=SectionType.RISK_FACTORS,
            title="Risk Factors",
            content=text,
            page_start=3,
            page_end=4
        )
        
        # chunk_size = 10, chunk_overlap = 2. Step is 8 tokens.
        chunks = self.chunker._chunk_section(section, "doc1", 5)
        
        # Verify multiple chunks are produced
        self.assertTrue(len(chunks) > 1)
        
        # Verify sequential indices starting at 5
        for idx, chunk in enumerate(chunks):
            self.assertEqual(chunk.chunk_index, 5 + idx)
            self.assertEqual(chunk.chunk_id, f"doc1_chunk_{5 + idx}")
            self.assertTrue(chunk.token_count <= 10)
            self.assertEqual(chunk.section_id, "doc1_item_1a")
            self.assertEqual(chunk.section_title, "Risk Factors")
            self.assertEqual(chunk.page_start, 3)
            self.assertEqual(chunk.page_end, 4)

        # Check overlapping logic
        tokens_chunk_0 = self.encoder.encode(chunks[0].text)
        tokens_chunk_1 = self.encoder.encode(chunks[1].text)
        overlap_tokens_end_c0 = tokens_chunk_0[-2:]
        overlap_tokens_start_c1 = tokens_chunk_1[:2]
        self.assertEqual(overlap_tokens_end_c0, overlap_tokens_start_c1)

    def test_chunk_document_orchestration(self):
        # Create dummy FinancialDocument
        metadata = DocumentMetadata(
            company_name="Test Company",
            ticker="TST",
            company_cik="0000000000",
            fiscal_year=2025,
            document_type=DocumentType.ANNUAL_REPORT,
            pages=10,
            language="en",
            source_filename="test.pdf",
            file_format=FileFormat.PDF,
            document_hash="hash123",
            created_at=datetime.now(),
            parser_version="1.0"
        )
        
        sections = [
            DocumentSection(
                section_id="doc1_item_1",
                section_type=SectionType.BUSINESS,
                title="Business",
                content="This is the business description section of the document.",
                page_start=1,
                page_end=2
            ),
            DocumentSection(
                section_id="doc1_item_1a",
                section_type=SectionType.RISK_FACTORS,
                title="Risk Factors",
                content="These are the primary risks associated with investing in our company.",
                page_start=3,
                page_end=4
            )
        ]
        
        doc = FinancialDocument(
            document_id="doc1",
            metadata=metadata,
            processing_status=ProcessingStatus.SECTIONED,
            sections=sections
        )
        
        processed_doc = self.chunker.chunk_document(doc)
        
        self.assertEqual(processed_doc.processing_status, ProcessingStatus.CHUNKED)
        self.assertTrue(len(processed_doc.chunks) > 0)
        
        for idx, chunk in enumerate(processed_doc.chunks):
            self.assertEqual(chunk.chunk_index, idx)
            self.assertEqual(chunk.chunk_id, f"doc1_chunk_{idx}")

    def test_chunk_document_invalid_status(self):
        # Create dummy FinancialDocument with invalid status (PARSED instead of SECTIONED)
        metadata = DocumentMetadata(
            company_name="Test Company",
            ticker="TST",
            company_cik="0000000000",
            fiscal_year=2025,
            document_type=DocumentType.ANNUAL_REPORT,
            pages=10,
            language="en",
            source_filename="test.pdf",
            file_format=FileFormat.PDF,
            document_hash="hash123",
            created_at=datetime.now(),
            parser_version="1.0"
        )
        
        doc = FinancialDocument(
            document_id="doc1",
            metadata=metadata,
            processing_status=ProcessingStatus.PARSED,
            sections=[]
        )
        
        with self.assertRaises(ValueError):
            self.chunker.chunk_document(doc)

    def test_chunk_document_no_sections(self):
        # Create dummy FinancialDocument with no sections
        metadata = DocumentMetadata(
            company_name="Test Company",
            ticker="TST",
            company_cik="0000000000",
            fiscal_year=2025,
            document_type=DocumentType.ANNUAL_REPORT,
            pages=10,
            language="en",
            source_filename="test.pdf",
            file_format=FileFormat.PDF,
            document_hash="hash123",
            created_at=datetime.now(),
            parser_version="1.0"
        )
        
        doc = FinancialDocument(
            document_id="doc1",
            metadata=metadata,
            processing_status=ProcessingStatus.SECTIONED,
            sections=[]
        )
        
        processed_doc = self.chunker.chunk_document(doc)
        self.assertEqual(processed_doc.processing_status, ProcessingStatus.CHUNKED)
        self.assertEqual(len(processed_doc.chunks), 0)

if __name__ == '__main__':
    unittest.main()
