import sys
import os
import unittest

# Add backend directory to Python path for imports
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from preprocessing.section_detector import SectionDetector
from preprocessing.cleaner import DocumentCleaner
from ingestion.pdf_parser import PDFParser
from app.models.financial_document import SectionType

class TestSectionDetector(unittest.TestCase):
    def setUp(self):
        self.detector = SectionDetector()

    def test_map_to_section_type(self):
        self.assertEqual(self.detector._map_to_section_type("1"), SectionType.BUSINESS)
        self.assertEqual(self.detector._map_to_section_type("1A"), SectionType.RISK_FACTORS)
        self.assertEqual(self.detector._map_to_section_type("1C"), SectionType.CYBERSECURITY)
        self.assertEqual(self.detector._map_to_section_type("7"), SectionType.MDNA)
        self.assertEqual(self.detector._map_to_section_type("8"), SectionType.FINANCIAL_STATEMENTS)
        self.assertEqual(self.detector._map_to_section_type("10"), SectionType.GOVERNANCE)
        self.assertEqual(self.detector._map_to_section_type("5"), SectionType.MARKET_INFORMATION)
        
        # Test default/unknown
        self.assertEqual(self.detector._map_to_section_type("4"), SectionType.OTHER)
        self.assertEqual(self.detector._map_to_section_type("1B"), SectionType.OTHER)
        self.assertEqual(self.detector._map_to_section_type(""), SectionType.OTHER)
        self.assertEqual(self.detector._map_to_section_type(None), SectionType.OTHER)

    def test_identify_headings(self):
        raw_text = """
        Item 1. Business
        Item 1A. Risk Factors
        Item 7. Management's Discussion and Analysis
        
        Some filler text...
        
        ITEM 1. BUSINESS
        This is the business content.
        
        ITEM 1A. RISK FACTORS
        Here are the risks.
        
        ITEM 7. MANAGEMENT’S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS
        MD&A content.
        """
        
        headings = self.detector._identify_headings(raw_text)
        
        self.assertEqual(len(headings), 6)
        
        self.assertEqual(headings[0]["item_number"], "1")
        self.assertEqual(headings[0]["title"], "Business")
        
        self.assertEqual(headings[1]["item_number"], "1A")
        self.assertEqual(headings[1]["title"], "Risk Factors")
        
        self.assertEqual(headings[3]["item_number"], "1")
        self.assertEqual(headings[3]["title"], "BUSINESS")
        
        self.assertEqual(headings[4]["item_number"], "1A")
        self.assertEqual(headings[4]["title"], "RISK FACTORS")
        
        self.assertEqual(headings[5]["item_number"], "7")
        self.assertEqual(headings[5]["title"], "MANAGEMENT’S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS")

    def test_detect_sections_slicing(self):
        text = """
        ITEM 1. BUSINESS
        Business content goes here.
        
        ITEM 1A. RISK FACTORS
        Risk factors go here.
        
        ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS
        MD&A discussion.
        """
        
        sections = self.detector.detect_sections(text, document_id="doc_123")
        
        self.assertEqual(len(sections), 3)
        
        # Section 1: Business
        self.assertEqual(sections[0].section_id, "doc_123_item_1")
        self.assertEqual(sections[0].title, "BUSINESS")
        self.assertEqual(sections[0].section_type, SectionType.BUSINESS)
        self.assertEqual(sections[0].content, "Business content goes here.")
        self.assertEqual(sections[0].page_start, 0)
        self.assertEqual(sections[0].page_end, 0)
        
        # Section 2: Risk Factors
        self.assertEqual(sections[1].section_id, "doc_123_item_1a")
        self.assertEqual(sections[1].title, "RISK FACTORS")
        self.assertEqual(sections[1].section_type, SectionType.RISK_FACTORS)
        self.assertEqual(sections[1].content, "Risk factors go here.")
        
        # Section 3: MD&A
        self.assertEqual(sections[2].section_id, "doc_123_item_7")
        self.assertEqual(sections[2].title, "MANAGEMENT'S DISCUSSION AND ANALYSIS")
        self.assertEqual(sections[2].section_type, SectionType.MDNA)
        self.assertEqual(sections[2].content, "MD&A discussion.")

    def test_detect_sections_toc_filtering(self):
        text = """
        Item 1. Business 3
        Item 1A. Risk Factors 16
        
        ITEM 1. BUSINESS
        This is a substantial business section content. It describes the company's business model, operations, segments, and other important aspects of the enterprise. We need this content to be longer than 500 characters so that the SectionDetector identifies it as the first substantial Business section rather than a Table of Contents entry. Let's add more filler text to exceed 500 characters easily: Microsoft is a technology company committed to making digital technology and artificial intelligence available broadly. Our mission is to empower every person.
        
        ITEM 1A. RISK FACTORS
        Risk factors go here.
        """
        sections = self.detector.detect_sections(text, document_id="doc_123")
        
        # The TOC entries "Business 3" and "Risk Factors 16" should be discarded
        # Only the actual body "BUSINESS" and "RISK FACTORS" sections should be returned
        self.assertEqual(len(sections), 2)
        self.assertEqual(sections[0].title, "BUSINESS")
        self.assertTrue(len(sections[0].content) > 500)
        self.assertEqual(sections[1].title, "RISK FACTORS")

    def test_integration_microsoft_report(self):
        pdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../data/raw/annual_reports/microsoft_2025_annual_report.pdf'))
        if not os.path.exists(pdf_path):
            self.skipTest("Microsoft 2025 report PDF not found in expected path.")

        parser = PDFParser()
        cleaner = DocumentCleaner()
        
        raw_text = parser.extract_text(pdf_path)
        cleaned_text = cleaner.clean_text(raw_text)
        
        sections = self.detector.detect_sections(cleaned_text, document_id="msft_2025")
        
        print("\n--- INTEGRATION TEST SECTIONS ---")
        print(f"Total Sections Detected: {len(sections)}")
        for idx, sec in enumerate(sections):
            print(f"Section {idx+1}:")
            print(f"  ID: {sec.section_id}")
            print(f"  Title: {sec.title}")
            print(f"  Type: {sec.section_type}")
            print(f"  Content Length: {len(sec.content)} chars")
            print(f"  Preview: {sec.content[:150]}...")
            print("-" * 25)
            
        # Assertions
        # 1. We expect to find the body sections (around 22 major sections)
        self.assertTrue(len(sections) >= 15)
        self.assertTrue(len(sections) <= 25)
        
        # 2. Verify specific expected items are found
        section_types = [sec.section_type for sec in sections]
        self.assertIn(SectionType.BUSINESS, section_types)
        self.assertIn(SectionType.RISK_FACTORS, section_types)
        self.assertIn(SectionType.CYBERSECURITY, section_types)
        self.assertIn(SectionType.MDNA, section_types)
        self.assertIn(SectionType.FINANCIAL_STATEMENTS, section_types)
        
        # 3. Check specific content exists and is non-empty
        business_sec = next(sec for sec in sections if sec.section_type == SectionType.BUSINESS)
        self.assertTrue(len(business_sec.content) > 500)
        
        risk_sec = next(sec for sec in sections if sec.section_type == SectionType.RISK_FACTORS)
        self.assertTrue(len(risk_sec.content) > 500)

if __name__ == '__main__':
    unittest.main()
