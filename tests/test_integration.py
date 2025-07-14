"""
Integration tests for the LLM Evaluator package.
Tests the interaction between different components.
"""

import unittest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from llm_evaluator import LLMEvaluator, EvaluationInput, ModelConfig, EvaluationConfig
from llm_evaluator.models import MetricScore, EvaluationMetrics
from llm_evaluator.metrics import MetricsFramework
from llm_evaluator.utils import CacheManager, RateLimiter
from call_aoai import create_llm_evaluator


class TestLLMEvaluatorIntegration(unittest.TestCase):
    """Integration tests for LLMEvaluator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = LLMEvaluator(
            azure_openai_endpoint="https://test.openai.azure.com/",
            azure_openai_api_key="test-key",
            azure_openai_deployment="test-deployment",
            claude_api_key="test-key"
        )
        
        self.eval_input = EvaluationInput(
            chat_history=[{"role": "user", "content": "What is Python?"}],
            retrieved_chunks=["Python is a programming language."],
            llm_answer="Python is a high-level programming language known for its readability.",
            user_feedback_label="Positive",
            user_feedback_comment="Good explanation"
        )
    
    def test_evaluator_initialization(self):
        """Test that evaluator initializes all components correctly."""
        # Check that all required components are initialized
        self.assertIsNotNone(self.evaluator.metrics_framework)
        self.assertIsNotNone(self.evaluator.azure_integration)
        self.assertIsNotNone(self.evaluator.claude_integration)
        self.assertIsNotNone(self.evaluator.cache_manager)
        self.assertIsNotNone(self.evaluator.rate_limiter)
        
        # Check configuration
        self.assertIsNotNone(self.evaluator.config)
        self.assertIsInstance(self.evaluator.config, EvaluationConfig)
    
    def test_get_metrics_info(self):
        """Test get_metrics_info method."""
        metrics_info = self.evaluator.get_metrics_info()
        
        self.assertIsInstance(metrics_info, dict)
        self.assertEqual(len(metrics_info), 9)
        
        # Check that metrics info contains expected structure
        for metric_name, metric_info in metrics_info.items():
            self.assertIn("name", metric_info)
            self.assertIn("description", metric_info)
            self.assertIn("scale", metric_info)
            self.assertIn("criteria", metric_info)
    
    def test_cache_key_generation(self):
        """Test cache key generation is consistent."""
        # Test that same input generates same cache key
        key1 = self.evaluator._generate_cache_key(self.eval_input)
        key2 = self.evaluator._generate_cache_key(self.eval_input)
        
        self.assertEqual(key1, key2)
        self.assertIsInstance(key1, str)
        self.assertGreater(len(key1), 0)
        
        # Test that different inputs generate different keys
        different_input = EvaluationInput(
            chat_history=[{"role": "user", "content": "What is Java?"}],
            retrieved_chunks=["Java is a programming language."],
            llm_answer="Java is a programming language.",
            user_feedback_label="Positive"
        )
        
        key3 = self.evaluator._generate_cache_key(different_input)
        self.assertNotEqual(key1, key3)
    
    def test_evaluator_with_custom_config(self):
        """Test evaluator with custom configuration."""
        azure_config = ModelConfig(
            model_name="gpt-3.5-turbo",
            temperature=0.5,
            max_tokens=1500
        )
        
        claude_config = ModelConfig(
            model_name="claude-3-haiku-20240307",
            temperature=0.7,
            max_tokens=1800
        )
        
        eval_config = EvaluationConfig(
            azure_openai_config=azure_config,
            claude_config=claude_config,
            enabled_metrics=["correctness", "completeness", "relevance"],
            parallel_evaluation=False,
            cache_enabled=False,
            rate_limit_enabled=False
        )
        
        evaluator = LLMEvaluator(
            azure_openai_endpoint="https://test.openai.azure.com/",
            azure_openai_api_key="test-key",
            azure_openai_deployment="test-deployment",
            claude_api_key="test-key",
            config=eval_config
        )
        
        # Check that custom config is applied
        self.assertEqual(evaluator.config.azure_openai_config.model_name, "gpt-3.5-turbo")
        self.assertEqual(evaluator.config.claude_config.model_name, "claude-3-haiku-20240307")
        self.assertEqual(len(evaluator.config.enabled_metrics), 3)
        self.assertFalse(evaluator.config.parallel_evaluation)
        self.assertFalse(evaluator.config.cache_enabled)
        self.assertFalse(evaluator.config.rate_limit_enabled)
        
        # Check that optional components are None when disabled
        self.assertIsNone(evaluator.cache_manager)
        self.assertIsNone(evaluator.rate_limiter)


class TestCreateLLMEvaluatorIntegration(unittest.TestCase):
    """Integration tests for create_llm_evaluator function."""
    
    def test_create_evaluator_with_default_config(self):
        """Test creating evaluator with default configuration."""
        evaluator = create_llm_evaluator(
            "https://test.openai.azure.com/",
            "test-azure-key",
            "test-deployment",
            "test-claude-key"
        )
        
        # Check that evaluator is properly initialized
        self.assertIsInstance(evaluator, LLMEvaluator)
        self.assertIsNotNone(evaluator.metrics_framework)
        self.assertIsNotNone(evaluator.azure_integration)
        self.assertIsNotNone(evaluator.claude_integration)
        
        # Check default configuration values
        self.assertEqual(evaluator.config.azure_openai_config.model_name, "gpt-4")
        self.assertEqual(evaluator.config.claude_config.model_name, "claude-3-sonnet-20240229")
        self.assertEqual(evaluator.config.azure_openai_config.temperature, 0.0)
        self.assertEqual(evaluator.config.azure_openai_config.max_tokens, 2000)
        self.assertTrue(evaluator.config.parallel_evaluation)
        self.assertTrue(evaluator.config.cache_enabled)
        self.assertTrue(evaluator.config.rate_limit_enabled)
    
    def test_create_evaluator_with_custom_config(self):
        """Test creating evaluator with custom configuration."""
        custom_config = {
            "azure_model": "gpt-3.5-turbo",
            "claude_model": "claude-3-haiku-20240307",
            "azure_temperature": 0.3,
            "claude_temperature": 0.4,
            "azure_max_tokens": 1200,
            "claude_max_tokens": 1400,
            "parallel_evaluation": False,
            "cache_enabled": False,
            "rate_limit_enabled": False
        }
        
        evaluator = create_llm_evaluator(
            "https://test.openai.azure.com/",
            "test-azure-key",
            "test-deployment",
            "test-claude-key",
            config=custom_config
        )
        
        # Check that custom configuration is applied
        self.assertEqual(evaluator.config.azure_openai_config.model_name, "gpt-3.5-turbo")
        self.assertEqual(evaluator.config.claude_config.model_name, "claude-3-haiku-20240307")
        self.assertEqual(evaluator.config.azure_openai_config.temperature, 0.3)
        self.assertEqual(evaluator.config.claude_config.temperature, 0.4)
        self.assertEqual(evaluator.config.azure_openai_config.max_tokens, 1200)
        self.assertEqual(evaluator.config.claude_config.max_tokens, 1400)
        self.assertFalse(evaluator.config.parallel_evaluation)
        self.assertFalse(evaluator.config.cache_enabled)
        self.assertFalse(evaluator.config.rate_limit_enabled)


class TestComponentIntegration(unittest.TestCase):
    """Test integration between different components."""
    
    def test_metrics_framework_with_evaluation_input(self):
        """Test metrics framework with evaluation input."""
        metrics_framework = MetricsFramework()
        
        eval_input = EvaluationInput(
            chat_history=[{"role": "user", "content": "What is machine learning?"}],
            retrieved_chunks=["Machine learning is a subset of AI."],
            llm_answer="Machine learning is a subset of artificial intelligence.",
            user_feedback_label="Positive",
            user_feedback_comment="Accurate and concise"
        )
        
        # Test that prompts can be generated for all metrics
        metrics = metrics_framework.get_all_metrics()
        
        for metric in metrics:
            prompt = metrics_framework.create_evaluation_prompt(metric, eval_input)
            
            self.assertIsInstance(prompt, str)
            self.assertGreater(len(prompt), 50)
            self.assertIn(metric, prompt.lower())
            self.assertIn("machine learning", prompt.lower())
            self.assertIn("subset of artificial intelligence", prompt)
    
    def test_cache_manager_with_evaluation_data(self):
        """Test cache manager with evaluation data."""
        cache_manager = CacheManager(max_size=3, ttl_seconds=3600)
        
        # Create test evaluation data
        eval_input = EvaluationInput(
            chat_history=[{"role": "user", "content": "Test question"}],
            retrieved_chunks=["Test context"],
            llm_answer="Test answer",
            user_feedback_label="Positive"
        )
        
        # Test caching evaluation input
        cache_key = "test_evaluation_1"
        cache_manager.set(cache_key, eval_input)
        
        retrieved_input = cache_manager.get(cache_key)
        self.assertIsNotNone(retrieved_input)
        self.assertEqual(retrieved_input.llm_answer, "Test answer")
        self.assertEqual(retrieved_input.user_feedback_label, "Positive")
    
    def test_model_config_integration(self):
        """Test model config integration with different components."""
        azure_config = ModelConfig(
            model_name="gpt-4",
            temperature=0.2,
            max_tokens=1500,
            timeout=60
        )
        
        claude_config = ModelConfig(
            model_name="claude-3-sonnet-20240229",
            temperature=0.1,
            max_tokens=2000,
            timeout=45
        )
        
        eval_config = EvaluationConfig(
            azure_openai_config=azure_config,
            claude_config=claude_config,
            enabled_metrics=["correctness", "helpfulness", "relevance"],
            parallel_evaluation=True,
            cache_enabled=True,
            rate_limit_enabled=True
        )
        
        # Test that configuration can be serialized
        config_dict = eval_config.to_dict()
        
        self.assertIsInstance(config_dict, dict)
        self.assertIn("azure_openai_config", config_dict)
        self.assertIn("claude_config", config_dict)
        self.assertIn("enabled_metrics", config_dict)
        
        # Check that nested configurations are properly serialized
        self.assertEqual(config_dict["azure_openai_config"]["model_name"], "gpt-4")
        self.assertEqual(config_dict["claude_config"]["model_name"], "claude-3-sonnet-20240229")
        self.assertEqual(len(config_dict["enabled_metrics"]), 3)
    
    def test_evaluation_data_flow(self):
        """Test data flow through evaluation components."""
        # Test the complete data flow from input to configuration
        eval_input = EvaluationInput(
            chat_history=[
                {"role": "user", "content": "Explain quantum computing"},
                {"role": "assistant", "content": "I'll explain quantum computing for you."}
            ],
            retrieved_chunks=[
                "Quantum computing uses quantum mechanics principles.",
                "Quantum computers use qubits instead of classical bits."
            ],
            llm_answer="Quantum computing leverages quantum mechanical phenomena like superposition and entanglement to process information in fundamentally different ways than classical computers.",
            user_feedback_label="Positive",
            user_feedback_comment="Comprehensive and accurate explanation"
        )
        
        # Convert to dict and back to test serialization
        input_dict = eval_input.to_dict()
        
        self.assertIsInstance(input_dict, dict)
        self.assertEqual(len(input_dict["chat_history"]), 2)
        self.assertEqual(len(input_dict["retrieved_chunks"]), 2)
        self.assertIn("quantum computing", input_dict["llm_answer"].lower())
        self.assertEqual(input_dict["user_feedback_label"], "Positive")
        self.assertEqual(input_dict["user_feedback_comment"], "Comprehensive and accurate explanation")
        
        # Test that metrics framework can handle this input
        metrics_framework = MetricsFramework()
        prompt = metrics_framework.create_evaluation_prompt("correctness", eval_input)
        
        self.assertIn("quantum computing", prompt.lower())
        self.assertIn("superposition", prompt.lower())
        self.assertIn("entanglement", prompt.lower())
        self.assertIn("Positive", prompt)


if __name__ == '__main__':
    unittest.main()