# Universal Financial Document Model

## Purpose
The `FinancialDocument` is the central, foundational object that every subsystem in FinSight AI uses to exchange data. It provides a universal contract so that regardless of the original source format (PDF, DOCX, HTML, CSV), the downstream systems (RAG Engine, Analytics Engine, FastAPI, React Dashboard) receive standardized, predictable structured data. 

This model abstracts away the complexities of parsing and file formats, ensuring that the rest of the application deals purely with business logic and intelligence.

## Responsibilities
The model acts as the unified data payload for a document as it moves through the pipeline. It contains:
- **Metadata**: Core identifying information, source details, hashing (for deduplication), and parser versions (for backward compatibility).
- **Sections**: Extracted logical blocks (e.g., Risk Factors, MD&A) mapped to strict semantic types (`SectionType`), ensuring consistent analytics and targeted retrieval regardless of original company-specific headings.
- **Tables**: Structured tabular data (rows and columns) extracted from the document to enable direct mathematical computations (e.g., Revenue Growth) rather than just presentation.
- **Figures**: References to extracted charts, graphs, and images along with their descriptions.
- **Chunks**: The sequential text segments derived from sections, enriched with `chunk_index` and `token_count` to allow the retriever to sort and manage context windows intelligently.
- **Processing Status**: An enumerated tracking state (`ProcessingStatus`) that represents exactly where the document is within the asynchronous data pipeline.
- **Validation Errors**: An optional log of issues encountered during parsing, keeping the pipeline robust without crashing silently.

## What it deliberately does not contain
- **Embeddings**: The model does not store high-dimensional float arrays (embeddings). Embeddings belong strictly in the Vector Database (e.g., ChromaDB). The `FinancialDocument` remains lightweight and represents the parsed document itself, not the mathematical vector index.

## Lifecycle
Every document moves sequentially through the following states, tracked by the `ProcessingStatus` field:
1. `UPLOADED`: Document received and metadata initialized.
2. `PARSED`: Raw text, tables, and figures extracted.
3. `CLEANED`: Noise removed, text normalized.
4. `SECTIONED`: Logical blocks identified and typed via `SectionType`.
5. `CHUNKED`: Sections broken down into bite-sized `DocumentChunk`s.
6. `EMBEDDED`: Vectors generated and mapped.
7. `STORED`: Persisted in the Vector Database and relational storage.
8. `READY`: Fully available for RAG and analytics queries.
*(Optional state: `FAILED` for pipeline errors).*

## Relationships (Subsystem Interactions)
- **Parsers (PDF/DOCX)**: Ingest raw files and output a populated `FinancialDocument` (up to the `SECTIONED` state).
- **Chunker/Embedder**: Consumes the `FinancialDocument`, populates the `Chunks`, and interfaces with the embedding model.
- **Storage/VectorDB**: Uses the `FinancialDocument` chunks for indexing, while structured tables and metadata can be sent to relational stores like Postgres.
- **RAG Engine**: Queries the vector store and retrieves contextual chunks mapped back to the structured `FinancialDocument` sections.
- **Analytics Engine**: Consumes structured `Tables` directly from the `FinancialDocument` (via pandas) to calculate financial metrics like Gross Margin or Debt Ratio without relying on text parsing.
- **API & Dashboard**: Exposes the `FinancialDocument` structure cleanly to the React frontend for visual rendering.
