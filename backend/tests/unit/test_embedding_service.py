import sys
import os
import unittest
from unittest.mock import MagicMock, patch
import openai

# Add backend directory to Python path for imports
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from app.models.financial_document import DocumentChunk
from preprocessing.embedding_service import OpenAIEmbeddingService, EmbeddingResult

class TestOpenAIEmbeddingService(unittest.TestCase):
    def setUp(self):
        # Initialize with dummy API key for testing
        self.service = OpenAIEmbeddingService(
            api_key="sk-test-key",
            model="text-embedding-3-small",
            batch_size=2,
            max_retries=2,
            base_delay=1.0  # Higher base delay is safe because time.sleep is mocked
        )

    def test_invalid_init_parameters(self):
        # Negative batch size
        with self.assertRaises(ValueError):
            OpenAIEmbeddingService(api_key="sk-test-key", batch_size=-1)
        
        # Zero batch size
        with self.assertRaises(ValueError):
            OpenAIEmbeddingService(api_key="sk-test-key", batch_size=0)

        # Negative retries
        with self.assertRaises(ValueError):
            OpenAIEmbeddingService(api_key="sk-test-key", max_retries=-1)

        # Invalid model name
        with self.assertRaises(ValueError):
            OpenAIEmbeddingService(api_key="sk-test-key", model="unsupported-model")

    def test_invalid_text_inputs(self):
        # Texts with empty strings
        with self.assertRaises(ValueError) as ctx:
            self.service.embed_texts(["Valid", ""])
        self.assertIn("cannot be empty or whitespace-only", str(ctx.exception))

        # Texts with whitespace only
        with self.assertRaises(ValueError) as ctx:
            self.service.embed_texts(["   ", "Valid"])
        self.assertIn("cannot be empty or whitespace-only", str(ctx.exception))

    def test_empty_chunks_or_texts(self):
        # Empty text list should return immediately with empty list
        res_texts = self.service.embed_texts([])
        self.assertEqual(res_texts, [])

        # Empty chunk list should return immediately with empty list
        res_chunks = self.service.embed_chunks([])
        self.assertEqual(res_chunks, [])

    def test_successful_embedding_texts(self):
        mock_data = [
            MagicMock(embedding=[0.1] * 1536),
            MagicMock(embedding=[0.2] * 1536)
        ]
        mock_response = MagicMock()
        mock_response.data = mock_data

        with patch.object(self.service.client.embeddings, "create", return_value=mock_response) as mock_create:
            texts = ["Text one", "Text two"]
            embeddings = self.service.embed_texts(texts)

            self.assertEqual(len(embeddings), 2)
            self.assertEqual(len(embeddings[0]), 1536)
            self.assertEqual(embeddings[0], [0.1] * 1536)
            self.assertEqual(embeddings[1], [0.2] * 1536)
            mock_create.assert_called_once_with(input=["Text one", "Text two"], model="text-embedding-3-small")

    def test_embed_chunks_success_and_order_preservation(self):
        # Batch size is 2. We'll send 3 chunks to force batching.
        mock_data_batch_1 = [
            MagicMock(embedding=[0.1] * 1536),
            MagicMock(embedding=[0.2] * 1536)
        ]
        mock_data_batch_2 = [
            MagicMock(embedding=[0.3] * 1536)
        ]

        mock_response_1 = MagicMock()
        mock_response_1.data = mock_data_batch_1
        mock_response_2 = MagicMock()
        mock_response_2.data = mock_data_batch_2

        chunks = [
            DocumentChunk(chunk_id="c_0", chunk_index=0, text="First", token_count=1, page_start=1, page_end=1),
            DocumentChunk(chunk_id="c_1", chunk_index=1, text="Second", token_count=1, page_start=1, page_end=1),
            DocumentChunk(chunk_id="c_2", chunk_index=2, text="Third", token_count=1, page_start=1, page_end=1)
        ]

        with patch.object(self.service.client.embeddings, "create") as mock_create:
            mock_create.side_effect = [mock_response_1, mock_response_2]
            results = self.service.embed_chunks(chunks)

            # Verify batching call count (exactly 2 calls for 3 chunks with batch size 2)
            self.assertEqual(mock_create.call_count, 2)
            
            # Verify structured output results
            self.assertEqual(len(results), 3)
            self.assertIsInstance(results[0], EmbeddingResult)
            
            # Verify order and mapping
            self.assertEqual(results[0].chunk_id, "c_0")
            self.assertEqual(results[0].embedding, [0.1] * 1536)
            self.assertEqual(results[1].chunk_id, "c_1")
            self.assertEqual(results[1].embedding, [0.2] * 1536)
            self.assertEqual(results[2].chunk_id, "c_2")
            self.assertEqual(results[2].embedding, [0.3] * 1536)

    def test_batching_call_count_exact(self):
        # We will send 5 texts. Batch size is 2, so it must call the API exactly 3 times.
        mock_data_1 = [MagicMock(embedding=[0.1] * 1536), MagicMock(embedding=[0.2] * 1536)]
        mock_data_2 = [MagicMock(embedding=[0.3] * 1536), MagicMock(embedding=[0.4] * 1536)]
        mock_data_3 = [MagicMock(embedding=[0.5] * 1536)]

        mock_res_1 = MagicMock(data=mock_data_1)
        mock_res_2 = MagicMock(data=mock_data_2)
        mock_res_3 = MagicMock(data=mock_data_3)

        with patch.object(self.service.client.embeddings, "create") as mock_create:
            mock_create.side_effect = [mock_res_1, mock_res_2, mock_res_3]
            res = self.service.embed_texts(["T1", "T2", "T3", "T4", "T5"])
            
            self.assertEqual(len(res), 5)
            self.assertEqual(mock_create.call_count, 3)

    def test_dimension_validation(self):
        # Mock returns incorrect dimension size (e.g. 100 instead of 1536)
        mock_data = [MagicMock(embedding=[0.1] * 100)]
        mock_response = MagicMock()
        mock_response.data = mock_data

        with patch.object(self.service.client.embeddings, "create", return_value=mock_response):
            with self.assertRaises(ValueError) as ctx:
                self.service.embed_texts(["Valid Text"])
            self.assertIn("dimension 100 instead of 1536", str(ctx.exception))

    def test_embedding_count_mismatch(self):
        # Mock returns fewer embeddings than input texts
        mock_data = [MagicMock(embedding=[0.1] * 1536)]
        mock_response = MagicMock()
        mock_response.data = mock_data

        with patch.object(self.service.client.embeddings, "create", return_value=mock_response):
            # Input has 2 texts, but mock only returns 1 embedding
            with self.assertRaises(ValueError) as ctx:
                self.service.embed_texts(["Text A", "Text B"])
            self.assertIn("count mismatch", str(ctx.exception))

    def test_non_retryable_errors(self):
        # Setup 401 Authentication Error
        mock_error = openai.AuthenticationError(
            message="Invalid API Key",
            response=MagicMock(status_code=401),
            body=None
        )

        with patch.object(self.service.client.embeddings, "create", side_effect=mock_error) as mock_create:
            with self.assertRaises(openai.AuthenticationError):
                self.service.embed_texts(["Some text"])
            
            # Should not retry; API call should be made exactly once
            self.assertEqual(mock_create.call_count, 1)

    @patch("time.sleep")
    def test_retryable_errors_eventual_success(self, mock_sleep):
        # Setup mock to fail with 429 RateLimit once, then succeed
        mock_rate_limit = openai.RateLimitError(
            message="Rate limit exceeded",
            response=MagicMock(status_code=429),
            body=None
        )
        mock_success_response = MagicMock()
        mock_success_response.data = [MagicMock(embedding=[0.5] * 1536)]

        with patch.object(self.service.client.embeddings, "create") as mock_create:
            mock_create.side_effect = [mock_rate_limit, mock_success_response]

            res = self.service.embed_texts(["Retry text"])
            self.assertEqual(len(res), 1)
            self.assertEqual(res[0], [0.5] * 1536)
            self.assertEqual(mock_create.call_count, 2)
            # Check mock sleep was called to avoid real sleeping delay
            mock_sleep.assert_called_once()

    @patch("time.sleep")
    def test_api_timeout_error_retry(self, mock_sleep):
        # Setup mock to fail with TimeoutError once, then succeed
        mock_timeout = openai.APITimeoutError(
            request=MagicMock()
        )
        mock_success = MagicMock(data=[MagicMock(embedding=[0.6] * 1536)])

        with patch.object(self.service.client.embeddings, "create") as mock_create:
            mock_create.side_effect = [mock_timeout, mock_success]

            res = self.service.embed_texts(["Timeout retry text"])
            self.assertEqual(len(res), 1)
            self.assertEqual(res[0], [0.6] * 1536)
            self.assertEqual(mock_create.call_count, 2)
            mock_sleep.assert_called_once()

    @patch("time.sleep")
    def test_retryable_errors_exhausted(self, mock_sleep):
        # Setup mock to fail with connection error repeatedly
        mock_error = openai.APIConnectionError(
            message="Connection timeout",
            request=MagicMock()
        )

        with patch.object(self.service.client.embeddings, "create", side_effect=mock_error) as mock_create:
            # max_retries = 2, so it will attempt 1 original + 2 retries = 3 calls total
            with self.assertRaises(openai.APIConnectionError):
                self.service.embed_texts(["Connection text"])

            self.assertEqual(mock_create.call_count, 3)
            self.assertEqual(mock_sleep.call_count, 2)

if __name__ == '__main__':
    unittest.main()
