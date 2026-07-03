import re

class DocumentCleaner:
    def __init__(self):
        """Initialize the DocumentCleaner."""
        pass

    def normalize_newlines(self, text: str) -> str:
        """
        Standardizes newline characters to \n and strips trailing whitespace from each line.
        """
        if not text:
            return ""
        # Replace non-breaking spaces with normal spaces
        text = text.replace('\xa0', ' ')
        # Convert carriage returns
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        # Strip trailing whitespace from each line
        lines = [line.rstrip() for line in text.split('\n')]
        return '\n'.join(lines)

    def remove_extra_spaces(self, text: str) -> str:
        """
        Replaces multiple horizontal spaces and tabs with a single space.
        Does not affect newlines.
        Also cleans up spaces inside hyphenated words (e.g., 'multi- space' -> 'multi-space').
        """
        if not text:
            return ""
        text = re.sub(r'[ \t]+', ' ', text)
        # Remove spaces around hyphens when they are part of a word (e.g. 'multi- space' -> 'multi-space')
        text = re.sub(r'(\w+)-\s+(\w+)', r'\1-\2', text)
        return text

    def remove_blank_lines(self, text: str) -> str:
        """
        Collapses consecutive empty lines to at most one blank line (two newlines).
        Strips leading and trailing empty lines from the entire text.
        """
        if not text:
            return ""
        # Strip outer whitespace
        text = text.strip()
        # Collapse 3 or more consecutive newlines to 2 newlines (one blank line)
        return re.sub(r'\n{3,}', '\n\n', text)

    def fix_broken_words(self, text: str) -> str:
        """
        Resolves words hyphenated and broken across line breaks.
        For example, 'devel-\\nopment' becomes 'development'.
        """
        if not text:
            return ""
        
        # 1. Join hyphenated words split across lines where the continuation is lowercase (remove hyphen)
        # e.g., 'devel-\nopment' -> 'development'
        text = re.sub(r'(\w+)-\n\s*([a-z]\w*)', r'\1\2', text)
        
        # 2. Join other line-break hyphens while preserving the hyphen (e.g. 'U.S.-\nbased' -> 'U.S.-based')
        text = re.sub(r'(\S+)-\n\s*(\w+)', r'\1-\2', text)
        
        return text

    def remove_page_numbers(self, text: str) -> str:
        """
        Removes lines that consist solely of digits (page numbers).
        """
        if not text:
            return ""
        # Match lines with only digits and optional whitespace, removing the line and its trailing newline
        return re.sub(r'(?m)^\s*\d+\s*$\n?', '', text)

    def remove_generic_headers_footers(self, text: str) -> str:
        """
        Removes generic repeated page headers and footers typical in SEC filings.
        For example, lines matching 'PART IItem 1' or 'PART I Item 1A' or 'PART IItem 1B, 1C'.
        Uses [ \\t]* instead of \\s* to prevent matching across newlines.
        """
        if not text:
            return ""
        # Match lines starting with PART, followed by Roman numerals, followed by optional horizontal spaces, then Item and anything else
        return re.sub(r'(?mi)^[ \t]*PART[ \t]+([IVXLC]+?)[ \t]*(Item[ \t]+.*)$\n?', '', text)

    def join_split_item_headers(self, text: str) -> str:
        """
        Resolves cases where section item headers are split from their titles.
        For example:
        Item 1.
        Business
        becomes:
        Item 1. Business
        """
        if not text:
            return ""
        # Join lines starting with 'Item' followed by number and period, then newline, then a capital letter heading line
        return re.sub(r'(?m)^[ \t]*([Ii][Tt][Ee][Mm][ \t]+\d+[A-Z]?\.?)[ \t]*\n[ \t]*([A-Z].*)$', r'\1 \2', text)

    def clean_text(self, text: str) -> str:
        """
        Runs the full text-cleaning pipeline in logical order.
        """
        if not text:
            return ""
            
        # 1. Normalize newlines first
        text = self.normalize_newlines(text)
        
        # 2. Fix broken words across lines
        text = self.fix_broken_words(text)
        
        # 3. Remove page numbers
        text = self.remove_page_numbers(text)
        
        # 4. Remove generic headers/footers
        text = self.remove_generic_headers_footers(text)
        
        # 5. Join split item headers
        text = self.join_split_item_headers(text)
        
        # 6. Remove extra spaces (horizontal)
        text = self.remove_extra_spaces(text)
        
        # 7. Collapse consecutive blank lines last
        text = self.remove_blank_lines(text)
        
        return text
