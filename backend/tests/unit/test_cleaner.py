import sys
import os
import unittest

# Add backend directory to Python path for imports
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from preprocessing.cleaner import DocumentCleaner
from ingestion.pdf_parser import PDFParser

class TestDocumentCleaner(unittest.TestCase):
    def setUp(self):
        self.cleaner = DocumentCleaner()

    def test_normalize_newlines(self):
        raw = "Line 1   \r\nLine 2\t  \r\n\r\nLine 3"
        expected = "Line 1\nLine 2\n\nLine 3"
        self.assertEqual(self.cleaner.normalize_newlines(raw), expected)

    def test_remove_extra_spaces(self):
        raw = "This   is   a\t\ttest  with   multiple spaces."
        expected = "This is a test with multiple spaces."
        self.assertEqual(self.cleaner.remove_extra_spaces(raw), expected)
        
        # Test cleaning spaces inside hyphenated words
        self.assertEqual(self.cleaner.remove_extra_spaces("multi-   space"), "multi-space")
        # Ensure bullet points or normal dashes are not broken
        self.assertEqual(self.cleaner.remove_extra_spaces(" - Item one"), " - Item one")

    def test_remove_blank_lines(self):
        raw = "\n\nLine 1\n\n\n\nLine 2\n\nLine 3\n\n"
        # strip outer, and collapse 3+ newlines to 2 newlines (one empty line)
        expected = "Line 1\n\nLine 2\n\nLine 3"
        self.assertEqual(self.cleaner.remove_blank_lines(raw), expected)

    def test_fix_broken_words(self):
        raw = "This is a devel-\nopment in technology.\nThis is U.S.-\nbased."
        expected = "This is a development in technology.\nThis is U.S.-based."
        self.assertEqual(self.cleaner.fix_broken_words(raw), expected)

    def test_remove_page_numbers(self):
        raw = "Line before\n 21 \nLine after\n45\n"
        expected = "Line before\nLine after\n"
        self.assertEqual(self.cleaner.remove_page_numbers(raw), expected)

    def test_remove_generic_headers_footers(self):
        raw = "PART IItem 1\nSome Content\nPART II Item 1A\nMore Content"
        expected = "Some Content\nMore Content"
        self.assertEqual(self.cleaner.remove_generic_headers_footers(raw), expected)

    def test_join_split_item_headers(self):
        raw = "Item 1.\nBusiness\nItem 1A.\nRisk Factors\nItem 7.  \n  Management's Discussion and Analysis of Financial Condition and Results of Operations"
        expected = "Item 1. Business\nItem 1A. Risk Factors\nItem 7. Management's Discussion and Analysis of Financial Condition and Results of Operations"
        self.assertEqual(self.cleaner.join_split_item_headers(raw), expected)

    def test_clean_text_pipeline(self):
        raw = """
        PART IItem 1
        
        Item 1.
        Business
        
        This is a multi-     space test.
        The system is U.S.-
        based and has seen huge devel-
        opment this year.
        
        3
        """
        # clean_text should process all these
        cleaned = self.cleaner.clean_text(raw)
        
        self.assertNotIn("PART IItem 1", cleaned)
        self.assertIn("Item 1. Business", cleaned)
        self.assertIn("multi-space test.", cleaned)
        self.assertIn("U.S.-based", cleaned)
        self.assertIn("development", cleaned)
        # Page number 3 should be removed
        self.assertNotIn("\n3\n", cleaned)
        self.assertFalse(cleaned.startswith("\n"))
        self.assertFalse(cleaned.endswith("\n"))

    def test_integration_microsoft_report(self):
        pdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../data/raw/annual_reports/microsoft_2025_annual_report.pdf'))
        if not os.path.exists(pdf_path):
            self.skipTest("Microsoft 2025 report PDF not found in expected path.")

        parser = PDFParser()
        raw_text = parser.extract_text(pdf_path)
        
        self.assertTrue(len(raw_text) > 0)
        
        cleaned_text = self.cleaner.clean_text(raw_text)
        
        print("\n--- INTEGRATION TEST STATS ---")
        print(f"Raw Text Length: {len(raw_text)} chars")
        print(f"Cleaned Text Length: {len(cleaned_text)} chars")
        print(f"Reduction: {len(raw_text) - len(cleaned_text)} chars ({(1 - len(cleaned_text)/len(raw_text))*100:.2f}%)")
        print("Sample (first 800 chars of cleaned text):")
        print(cleaned_text[:800])
        print("-------------------------------")
        
        # Verify no lone digits exist (page numbers removed)
        # We search line by line for any line consisting only of digits
        for line in cleaned_text.split('\n'):
            stripped = line.strip()
            if stripped.isdigit():
                self.fail(f"Found remaining page number line: '{line}'")
                
        # Verify generic page headers are stripped
        self.assertNotIn("PART IItem 1", cleaned_text)
        self.assertNotIn("PART IItem 1A", cleaned_text)
        self.assertNotIn("PART IItem 7", cleaned_text)

if __name__ == '__main__':
    unittest.main()
