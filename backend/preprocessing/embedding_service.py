import os
import time
import random
import logging
from abc import ABC, abstractmethod
import openai
from pydantic import BaseModel
from app.models.financial_document import DocumentChunk

logger = logging.getLogger(__name__)

# Type aliases
EmbeddingVector = list[float]
EmbeddingBatch = list[EmbeddingVector]

# Dimension mapping for OpenAI embedding models
MODEL_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,  # Standard legacy model
}

class EmbeddingResult(BaseModel):
    chunk_id: str
    embedding: EmbeddingVector

class EmbeddingService(ABC):
    @abstractmethod
    def embed_texts(self, texts: list[str]) -> EmbeddingBatch:
        """
        Embeds a list of plain text strings into a list of vector embeddings.
        """
        pass

    def embed_chunks(self, chunks: list[DocumentChunk]) -> list[EmbeddingResult]:
        """
        Extracts text from DocumentChunks, embeds them using embed_texts,
        and constructs structured EmbeddingResult outputs.
        """
        if not chunks:
            return []
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embed_texts(texts)
        if len(embeddings) != len(chunks):
            raise ValueError(f"Mismatch: embedded {len(texts)} texts but got {len(embeddings)} vectors.")
        
        return [
            EmbeddingResult(chunk_id=chunk.chunk_id, embedding=emb)
            for chunk, emb in zip(chunks, embeddings)
        ]

class OpenAIEmbeddingService(EmbeddingService):
    def __init__(
        self, 
        api_key: str | None = None, 
        model: str = "text-embedding-3-small", 
        batch_size: int = 100,
        max_retries: int = 3,
        base_delay: float = 1.0,
        timeout: float = 10.0
    ):
        """
        Initialize the OpenAI Embedding Service.
        """
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OpenAI API key must be provided or set in the OPENAI_API_KEY environment variable.")
        
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")

        if model not in MODEL_DIMENSIONS:
            raise ValueError(f"Unsupported model name: {model}. Supported: {list(MODEL_DIMENSIONS.keys())}")

        self.client = openai.OpenAI(api_key=key, timeout=timeout)
        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.expected_dim = MODEL_DIMENSIONS[model]

    def embed_texts(self, texts: list[str]) -> EmbeddingBatch:
        """
        Embeds a list of texts by batching them and making requests to OpenAI API with error handling and retries.
        """
        if not texts:
            return []

        # Validate that no empty or whitespace-only strings are present
        for idx, text in enumerate(texts):
            if not text or not text.strip():
                raise ValueError(f"Input text at index {idx} cannot be empty or whitespace-only.")

        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_embeddings = self._embed_batch_with_retry(batch)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def _embed_batch_with_retry(self, batch: list[str]) -> EmbeddingBatch:
        """
        Sends a single batch of texts to the OpenAI embeddings API with retries and exponential backoff + jitter.
        """
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.model
                )
                embeddings = [data.embedding for data in response.data]
                
                # Validate dimensions of returned vectors
                for i, vector in enumerate(embeddings):
                    if len(vector) != self.expected_dim:
                        raise ValueError(
                            f"Embedding {i} has dimension {len(vector)} instead of {self.expected_dim}"
                        )
                
                # Check that order matches input size
                if len(embeddings) != len(batch):
                    raise ValueError(f"Returned embeddings count mismatch: expected {len(batch)}, got {len(embeddings)}")
                
                return embeddings
            
            except openai.APITimeoutError as e:
                if attempt >= self.max_retries:
                    raise e
                
                logger.warning(
                    "OpenAI API timeout on attempt %d/%d. Retrying...",
                    attempt + 1,
                    self.max_retries + 1
                )
                time.sleep(self._compute_backoff_delay(attempt + 1))

            except openai.APIStatusError as e:
                # Immediate fail on non-retryable status errors (400, 401, 403, 404)
                if e.status_code in (400, 401, 403, 404):
                    raise e
                
                if attempt >= self.max_retries:
                    raise e
                
                logger.warning(
                    "OpenAI API status error %s on attempt %d/%d. Retrying...",
                    e.status_code,
                    attempt + 1,
                    self.max_retries + 1
                )
                time.sleep(self._compute_backoff_delay(attempt + 1))

            except openai.APIConnectionError as e:
                if attempt >= self.max_retries:
                    raise e
                
                logger.warning(
                    "OpenAI API connection error on attempt %d/%d. Retrying...",
                    attempt + 1,
                    self.max_retries + 1
                )
                time.sleep(self._compute_backoff_delay(attempt + 1))

    def _compute_backoff_delay(self, retry_count: int) -> float:
        """
        Compute delay with exponential backoff and jitter.
        """
        delay = self.base_delay * (2 ** (retry_count - 1))
        jitter = random.uniform(0, 0.5 * delay)
        return delay + jitter
