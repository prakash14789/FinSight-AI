import re
from typing import List, Dict, Any
from app.models.financial_document import DocumentSection, SectionType

# Constant for tuning the Table of Contents filtering heuristic
MIN_BODY_SECTION_LENGTH = 500

class SectionDetector:
    def __init__(self):
        """Initialize the SectionDetector."""
        self.item_map = {
            "1": SectionType.BUSINESS,
            "1A": SectionType.RISK_FACTORS,
            "1C": SectionType.CYBERSECURITY,
            "2": SectionType.PROPERTIES,
            "3": SectionType.LEGAL_PROCEEDINGS,
            "5": SectionType.MARKET_INFORMATION,
            "7": SectionType.MDNA,
            "8": SectionType.FINANCIAL_STATEMENTS,
            "10": SectionType.GOVERNANCE,
        }

    def _map_to_section_type(self, item_number: str) -> SectionType:
        """
        Maps an SEC item number (e.g. '1', '1A', '7') to a SectionType enum.
        Defaults to SectionType.OTHER if not in the known dictionary map.
        """
        if not item_number:
            return SectionType.OTHER
        # Standardize key (strip spacing and make uppercase)
        key = item_number.strip().upper()
        return self.item_map.get(key, SectionType.OTHER)

    def _identify_headings(self, text: str) -> List[Dict[str, Any]]:
        """
        Scans the text using regex to identify section headings.
        """
        headings = []
        if not text:
            return headings

        # Regex matches lines starting with "Item/ITEM" followed by spaces, item number, optional period, spaces, and the title.
        # Group 1 captures the item number (e.g. '1A').
        # Group 2 captures the human-readable title (e.g. 'Risk Factors').
        pattern = re.compile(r'(?m)^[ \t]*[Ii][Tt][Ee][Mm][ \t]+(\d+[A-Z]?)\.?[ \t]+(\w.*)$')

        for match in pattern.finditer(text):
            item_number = match.group(1)
            title = match.group(2).strip()
            
            headings.append({
                "item_number": item_number,
                "title": title,
                "start_idx": match.start(),
                "end_idx": match.end()
            })

        # Sort headings by their start index to ensure logical order
        headings.sort(key=lambda x: x["start_idx"])
        return headings

    def detect_sections(self, cleaned_text: str, document_id: str = "") -> List[DocumentSection]:
        """
        Orchestrates section boundary detection, slices the content between headings,
        filters out TOC entries based on the 'first substantial Business section' heuristic,
        and returns a list of DocumentSection objects.
        """
        sections = []
        if not cleaned_text:
            return sections

        headings = self._identify_headings(cleaned_text)
        if not headings:
            # If no sections are detected, return a single catch-all section
            sections.append(DocumentSection(
                section_id=f"{document_id}_item_all",
                title="Full Document",
                section_type=SectionType.OTHER,
                content=cleaned_text.strip(),
                page_start=0,
                page_end=0
            ))
            return sections

        num_headings = len(headings)
        for i in range(num_headings):
            curr_h = headings[i]
            
            # Start of the content is after the current heading ends
            content_start = curr_h["end_idx"]
            
            # End of the content is the start of the next heading (or end of document)
            if i + 1 < num_headings:
                content_end = headings[i+1]["start_idx"]
            else:
                content_end = len(cleaned_text)
                
            content = cleaned_text[content_start:content_end].strip()
            
            item_number = curr_h["item_number"]
            sec_type = self._map_to_section_type(item_number)
            
            sections.append(DocumentSection(
                section_id=f"{document_id}_item_{item_number.lower()}",
                title=curr_h["title"],
                section_type=sec_type,
                content=content,
                page_start=0,
                page_end=0
            ))

        # SEC 10-K reports usually contain a Table of Contents before the actual
        # document body. We consider the first BUSINESS section with substantial
        # content as the beginning of the real report and discard earlier duplicate
        # headings originating from the Table of Contents.
        body_start_idx = 0
        for idx, sec in enumerate(sections):
            if (
                sec.section_type == SectionType.BUSINESS
                and len(sec.content) > MIN_BODY_SECTION_LENGTH
            ):
                body_start_idx = idx
                break
                
        # Keep only sections starting from that body start index
        sections = sections[body_start_idx:]

        return sections
