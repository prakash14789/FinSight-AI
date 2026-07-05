import tiktoken
from typing import List
from app.models.financial_document import FinancialDocument, DocumentSection, DocumentChunk, ProcessingStatus

class Chunker:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 100, encoding_name: str = "cl100k_base"):
        """
        Initialize the Chunker with target chunk size, overlap, and tokenizer encoding.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoder = tiktoken.get_encoding(encoding_name)

    def chunk_document(self, document: FinancialDocument) -> FinancialDocument:
        """
        Iterates through sections of the FinancialDocument, chunks each section
        independently by calling _chunk_section, populates document.chunks, and updates 
        the processing status to CHUNKED.
        
        Enforces that the document has been successfully processed by the Section Detector.
        """
        if document.processing_status != ProcessingStatus.SECTIONED:
            raise ValueError("Document must be SECTIONED before chunking.")

        if not document.sections:
            document.chunks = []
            document.processing_status = ProcessingStatus.CHUNKED
            return document

        all_chunks = []
        for section in document.sections:
            section_chunks = self._chunk_section(
                section=section,
                document_id=document.document_id,
                start_chunk_index=len(all_chunks)
            )
            all_chunks.extend(section_chunks)
            
        document.chunks = all_chunks
        document.processing_status = ProcessingStatus.CHUNKED
        return document

    def _chunk_section(
        self, 
        section: DocumentSection, 
        document_id: str, 
        start_chunk_index: int
    ) -> List[DocumentChunk]:
        """
        Chunks a single section independently. Handles token-based windowing using
        tiktoken, metadata propagation, and unique ID generation.
        """
        content = section.content
        if not content or not content.strip():
            return []

        # Encode content to tokens
        tokens = self.encoder.encode(content)
        total_tokens = len(tokens)

        # If section is small (<= chunk_size), keep it as a single chunk
        if total_tokens <= self.chunk_size:
            return [DocumentChunk(
                chunk_id=f"{document_id}_chunk_{start_chunk_index}",
                chunk_index=start_chunk_index,
                text=content,
                token_count=total_tokens,
                section_id=section.section_id,
                section_title=section.title,
                page_start=section.page_start,
                page_end=section.page_end
            )]

        # Otherwise, create overlapping token windows
        step = self.chunk_size - self.chunk_overlap
        chunks = []
        idx = 0
        chunk_seq_idx = start_chunk_index
        while idx < total_tokens:
            end_idx = min(idx + self.chunk_size, total_tokens)
            chunk_tokens = tokens[idx:end_idx]
            
            chunk_token_count = len(chunk_tokens)
            chunk_text = self.encoder.decode(chunk_tokens)
            
            chunks.append(DocumentChunk(
                chunk_id=f"{document_id}_chunk_{chunk_seq_idx}",
                chunk_index=chunk_seq_idx,
                text=chunk_text,
                token_count=chunk_token_count,
                section_id=section.section_id,
                section_title=section.title,
                page_start=section.page_start,
                page_end=section.page_end
            ))
            
            chunk_seq_idx += 1
            idx += step
            
            if end_idx == total_tokens:
                break

        return chunks
