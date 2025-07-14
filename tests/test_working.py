"""
Unit tests for the LLM Evaluator package - Fixed version.
This version matches the actual implementation.
"""

import unittest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import json
import time

# Import the actual modules
from llm_evaluator.models import (
    MetricScore, EvaluationMetrics, ModelConfig, EvaluationInput
)
from llm_evaluator.metrics import MetricsFramework
from llm_evaluator.utils import CacheManager, RateLimiter
from call_aoai import send_to_azure_openai, create_llm_evaluator


class TestMetricScore(unittest.TestCase):
    """Test MetricScore class."""
    
    def test_init(self):
        """Test MetricScore initialization."""
        metric_score = MetricScore(
            name="correctness",
            score=4.5,
            justification="Very accurate answer",
            confidence=0.85
        )
        
        self.assertEqual(metric_score.name, "correctness")
        self.assertEqual(metric_score.score, 4.5)
        self.assertEqual(metric_score.justification, "Very accurate answer")
        self.assertEqual(metric_score.confidence, 0.85)
    
    def test_to_dict(self):
        """Test to_dict method."""
        metric_score = MetricScore("correctness", 4.5, "Very accurate answer", 0.85)
        expected = {
            "name": "correctness",
            "score": 4.5,
            "justification": "Very accurate answer",
            "confidence": 0.85
        }
        self.assertEqual(metric_score.to_dict(), expected)


class TestModelConfig(unittest.TestCase):
    """Test ModelConfig class."""
    
    def test_init_minimal(self):
        """Test ModelConfig initialization with minimal parameters."""
        config = ModelConfig(model_name="gpt-4")
        
        self.assertEqual(config.model_name, "gpt-4")
        self.assertEqual(config.temperature, 0.0)
        self.assertEqual(config.max_tokens, 2000)
        self.assertEqual(config.timeout, 30)
        self.assertEqual(config.retry_attempts, 3)
        self.assertEqual(config.retry_delay, 1.0)
        self.assertEqual(config.top_p, 1.0)
        self.assertEqual(config.frequency_penalty, 0.0)
        self.assertEqual(config.presence_penalty, 0.0)
    
    def test_init_full(self):
        """Test ModelConfig initialization with all parameters."""
        config = ModelConfig(
            model_name="gpt-4",
            temperature=0.5,
            max_tokens=1500,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.1,
            timeout=45,
            retry_attempts=5,
            retry_delay=2.0
        )
        
        self.assertEqual(config.model_name, "gpt-4")
        self.assertEqual(config.temperature, 0.5)
        self.assertEqual(config.max_tokens, 1500)
        self.assertEqual(config.top_p, 0.9)
        self.assertEqual(config.frequency_penalty, 0.1)
        self.assertEqual(config.presence_penalty, 0.1)
        self.assertEqual(config.timeout, 45)
        self.assertEqual(config.retry_attempts, 5)
        self.assertEqual(config.retry_delay, 2.0)
    
    def test_to_dict(self):
        """Test to_dict method."""
        config = ModelConfig(model_name="gpt-4", temperature=0.5)
        result = config.to_dict()
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result["model_name"], "gpt-4")
        self.assertEqual(result["temperature"], 0.5)
        self.assertIn("max_tokens", result)
        self.assertIn("timeout", result)


class TestEvaluationInput(unittest.TestCase):
    """Test EvaluationInput class."""
    
    def test_init(self):
        """Test EvaluationInput initialization."""
        eval_input = EvaluationInput(
            chat_history=[{"role": "user", "content": "What is Python?"}],
            retrieved_chunks=["Python is a programming language."],
            llm_answer="Python is a high-level programming language.",
            user_feedback_label="Positive",
            user_feedback_comment="Good explanation"
        )
        
        self.assertEqual(len(eval_input.chat_history), 1)
        self.assertEqual(len(eval_input.retrieved_chunks), 1)
        self.assertEqual(eval_input.llm_answer, "Python is a high-level programming language.")
        self.assertEqual(eval_input.user_feedback_label, "Positive")
        self.assertEqual(eval_input.user_feedback_comment, "Good explanation")
    
    def test_init_minimal(self):
        """Test EvaluationInput initialization with minimal required fields."""
        eval_input = EvaluationInput(
            chat_history=[{"role": "user", "content": "Test"}],
            retrieved_chunks=["Test chunk"],
            llm_answer="Test answer",
            user_feedback_label="Positive"
        )
        
        self.assertEqual(len(eval_input.chat_history), 1)
        self.assertEqual(len(eval_input.retrieved_chunks), 1)
        self.assertEqual(eval_input.llm_answer, "Test answer")
        self.assertEqual(eval_input.user_feedback_label, "Positive")
        self.assertIsNone(eval_input.user_feedback_comment)
    
    def test_to_dict(self):
        """Test to_dict method."""
        eval_input = EvaluationInput(
            chat_history=[{"role": "user", "content": "Test"}],
            retrieved_chunks=["Test chunk"],
            llm_answer="Test answer",
            user_feedback_label="Positive"
        )
        
        result = eval_input.to_dict()
        
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result["chat_history"]), 1)
        self.assertEqual(len(result["retrieved_chunks"]), 1)
        self.assertEqual(result["llm_answer"], "Test answer")
        self.assertEqual(result["user_feedback_label"], "Positive")


class TestMetricsFramework(unittest.TestCase):
    """Test MetricsFramework class."""
    
    def setUp(self):
        self.metrics_framework = MetricsFramework()
        self.test_eval_input = EvaluationInput(
            chat_history=[{"role": "user", "content": "What is Python?"}],
            retrieved_chunks=["Python is a programming language."],
            llm_answer="Python is a high-level programming language.",
            user_feedback_label="Positive",
            user_feedback_comment="Good explanation"
        )
    
    def test_init(self):
        """Test MetricsFramework initialization."""
        self.assertIsNotNone(self.metrics_framework.metrics_definitions)
        self.assertIsInstance(self.metrics_framework.metrics_definitions, dict)
        self.assertEqual(len(self.metrics_framework.metrics_definitions), 9)
    
    def test_get_all_metrics(self):
        """Test get_all_metrics method."""
        all_metrics = self.metrics_framework.get_all_metrics()
        
        self.assertIsInstance(all_metrics, list)
        self.assertEqual(len(all_metrics), 9)
        
        expected_metrics = [
            "correctness", "completeness", "relevance", "fluency",
            "coherence", "faithfulness", "harmfulness", "helpfulness", "conciseness"
        ]
        
        for metric in expected_metrics:
            self.assertIn(metric, all_metrics)
    
    def test_get_metric_definition(self):
        """Test get_metric_definition method."""
        # Test valid metric
        correctness_def = self.metrics_framework.get_metric_definition("correctness")
        
        self.assertIsInstance(correctness_def, dict)
        self.assertEqual(correctness_def["name"], "Correctness")
        self.assertIn("description", correctness_def)
        self.assertIn("scale", correctness_def)
        self.assertIn("criteria", correctness_def)
        
        # Test invalid metric
        invalid_def = self.metrics_framework.get_metric_definition("invalid_metric")
        self.assertEqual(invalid_def, {})
    
    def test_create_evaluation_prompt(self):
        """Test create_evaluation_prompt method."""
        prompt = self.metrics_framework.create_evaluation_prompt("correctness", self.test_eval_input)
        
        self.assertIsInstance(prompt, str)
        self.assertGreater(len(prompt), 100)
        
        # Check that key components are included
        self.assertIn("correctness", prompt.lower())
        self.assertIn("Python", prompt)
        self.assertIn("score", prompt.lower())
        self.assertIn("justification", prompt.lower())
        self.assertIn("confidence", prompt.lower())
    
    def test_validate_metric_score(self):
        """Test validate_metric_score method."""
        # Test valid scores
        self.assertTrue(self.metrics_framework.validate_metric_score(1.0))
        self.assertTrue(self.metrics_framework.validate_metric_score(3.5))
        self.assertTrue(self.metrics_framework.validate_metric_score(5.0))
        
        # Test invalid scores
        self.assertFalse(self.metrics_framework.validate_metric_score(0.5))
        self.assertFalse(self.metrics_framework.validate_metric_score(5.5))
        self.assertFalse(self.metrics_framework.validate_metric_score(-1.0))
    
    def test_validate_confidence(self):
        """Test validate_confidence method."""
        # Test valid confidence values
        self.assertTrue(self.metrics_framework.validate_confidence(0.0))
        self.assertTrue(self.metrics_framework.validate_confidence(0.5))
        self.assertTrue(self.metrics_framework.validate_confidence(1.0))
        
        # Test invalid confidence values
        self.assertFalse(self.metrics_framework.validate_confidence(-0.1))
        self.assertFalse(self.metrics_framework.validate_confidence(1.1))


class TestCacheManager(unittest.TestCase):
    """Test CacheManager class."""
    
    def setUp(self):
        self.cache_manager = CacheManager(max_size=3, ttl_seconds=3600)
    
    def test_init(self):
        """Test CacheManager initialization."""
        self.assertEqual(self.cache_manager.max_size, 3)
        self.assertEqual(self.cache_manager.ttl_seconds, 3600)
        self.assertIsInstance(self.cache_manager.cache, dict)
        self.assertIsInstance(self.cache_manager.timestamps, dict)
    
    def test_set_and_get(self):
        """Test setting and getting cache items."""
        self.cache_manager.set("key1", "value1")
        self.assertEqual(self.cache_manager.get("key1"), "value1")
        
        # Test with different data types
        self.cache_manager.set("key2", 42)
        self.assertEqual(self.cache_manager.get("key2"), 42)
        
        self.cache_manager.set("key3", {"nested": "dict"})
        self.assertEqual(self.cache_manager.get("key3"), {"nested": "dict"})
    
    def test_get_nonexistent_key(self):
        """Test getting non-existent key."""
        self.assertIsNone(self.cache_manager.get("nonexistent"))
    
    def test_max_size_limit(self):
        """Test maximum size limit."""
        # Fill cache to max size
        for i in range(3):
            self.cache_manager.set(f"key{i}", f"value{i}")
        
        self.assertEqual(self.cache_manager.size(), 3)
        
        # Add one more item - should evict oldest
        self.cache_manager.set("key3", "value3")
        
        # Should still be at max size
        self.assertEqual(self.cache_manager.size(), 3)
        
        # First item should be evicted
        self.assertIsNone(self.cache_manager.get("key0"))
        
        # Last item should be present
        self.assertEqual(self.cache_manager.get("key3"), "value3")


class TestRateLimiter(unittest.TestCase):
    """Test RateLimiter class."""
    
    def setUp(self):
        self.rate_limiter = RateLimiter(max_requests=2, window_seconds=1)
    
    def test_init(self):
        """Test RateLimiter initialization."""
        self.assertEqual(self.rate_limiter.max_requests, 2)
        self.assertEqual(self.rate_limiter.window_seconds, 1)
        self.assertIsInstance(self.rate_limiter.requests, list)
    
    def test_wait_if_needed_exists(self):
        """Test that wait_if_needed method exists."""
        self.assertTrue(hasattr(self.rate_limiter, 'wait_if_needed'))
        self.assertTrue(callable(self.rate_limiter.wait_if_needed))


class TestSendToAzureOpenAI(unittest.TestCase):
    """Test send_to_azure_openai function."""
    
    @patch('call_aoai.requests.post')
    def test_send_to_azure_openai_success(self, mock_post):
        """Test successful API call."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello! How can I help you today?"}}]
        }
        mock_post.return_value = mock_response
        
        result = send_to_azure_openai(
            "https://test.openai.azure.com/",
            "test-key",
            "test-deployment",
            [{"role": "user", "content": "Hello"}]
        )
        
        self.assertEqual(result, "Hello! How can I help you today?")
        mock_post.assert_called_once()
    
    @patch('call_aoai.requests.post')
    def test_send_to_azure_openai_with_params(self, mock_post):
        """Test API call with additional parameters."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Response with custom params"}}]
        }
        mock_post.return_value = mock_response
        
        result = send_to_azure_openai(
            "https://test.openai.azure.com/",
            "test-key",
            "test-deployment",
            [{"role": "user", "content": "Hello"}],
            temperature=0.5,
            max_tokens=500
        )
        
        self.assertEqual(result, "Response with custom params")
        
        # Check that parameters were passed correctly
        call_args = mock_post.call_args
        payload = call_args[1]["json"]
        self.assertEqual(payload["temperature"], 0.5)
        self.assertEqual(payload["max_tokens"], 500)
    
    @patch('call_aoai.requests.post')
    def test_send_to_azure_openai_error(self, mock_post):
        """Test error handling."""
        # Mock error response
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.raise_for_status.side_effect = Exception("API Error")
        mock_post.return_value = mock_response
        
        with self.assertRaises(Exception):
            send_to_azure_openai(
                "https://test.openai.azure.com/",
                "test-key",
                "test-deployment",
                [{"role": "user", "content": "Hello"}]
            )


class TestCreateLLMEvaluator(unittest.TestCase):
    """Test create_llm_evaluator function."""
    
    def test_create_llm_evaluator_default(self):
        """Test creating evaluator with default configuration."""
        evaluator = create_llm_evaluator(
            "https://test.openai.azure.com/",
            "test-azure-key",
            "test-deployment",
            "test-claude-key"
        )
        
        self.assertIsNotNone(evaluator)
        self.assertEqual(evaluator.azure_openai_endpoint, "https://test.openai.azure.com/")
        self.assertEqual(evaluator.azure_openai_api_key, "test-azure-key")
        self.assertEqual(evaluator.azure_openai_deployment, "test-deployment")
        self.assertEqual(evaluator.claude_api_key, "test-claude-key")
        
        # Check default configuration
        self.assertEqual(evaluator.config.azure_openai_config.model_name, "gpt-4")
        self.assertEqual(evaluator.config.claude_config.model_name, "claude-3-sonnet-20240229")
    
    def test_create_llm_evaluator_custom_config(self):
        """Test creating evaluator with custom configuration."""
        custom_config = {
            "azure_model": "gpt-3.5-turbo",
            "claude_model": "claude-3-haiku-20240307",
            "parallel_evaluation": False
        }
        
        evaluator = create_llm_evaluator(
            "https://test.openai.azure.com/",
            "test-azure-key",
            "test-deployment",
            "test-claude-key",
            config=custom_config
        )
        
        self.assertIsNotNone(evaluator)
        self.assertEqual(evaluator.config.azure_openai_config.model_name, "gpt-3.5-turbo")
        self.assertEqual(evaluator.config.claude_config.model_name, "claude-3-haiku-20240307")
        self.assertFalse(evaluator.config.parallel_evaluation)


if __name__ == '__main__':
    unittest.main()