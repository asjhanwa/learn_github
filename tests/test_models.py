"""
Unit tests for data models (models.py).
"""

import unittest
from datetime import datetime
from unittest.mock import Mock, patch
import json

from llm_evaluator.models import (
    MetricScore, EvaluationMetrics, ModelConfig, EvaluationInput,
    EvaluationConfig, ModelResult, EvaluationResult
)


class TestMetricScore(unittest.TestCase):
    """Test MetricScore class."""
    
    def setUp(self):
        self.metric_score = MetricScore(
            name="correctness",
            score=4.5,
            justification="Very accurate answer",
            confidence=0.85
        )
    
    def test_init(self):
        """Test MetricScore initialization."""
        self.assertEqual(self.metric_score.name, "correctness")
        self.assertEqual(self.metric_score.score, 4.5)
        self.assertEqual(self.metric_score.justification, "Very accurate answer")
        self.assertEqual(self.metric_score.confidence, 0.85)
    
    def test_to_dict(self):
        """Test to_dict method."""
        expected = {
            "name": "correctness",
            "score": 4.5,
            "justification": "Very accurate answer",
            "confidence": 0.85
        }
        self.assertEqual(self.metric_score.to_dict(), expected)
    
    def test_score_range(self):
        """Test valid score range."""
        # Test valid scores
        valid_scores = [1.0, 2.5, 3.0, 4.5, 5.0]
        for score in valid_scores:
            metric = MetricScore("test", score, "test", 0.5)
            self.assertEqual(metric.score, score)
    
    def test_confidence_range(self):
        """Test valid confidence range."""
        # Test valid confidence values
        valid_confidences = [0.0, 0.5, 0.85, 1.0]
        for confidence in valid_confidences:
            metric = MetricScore("test", 3.0, "test", confidence)
            self.assertEqual(metric.confidence, confidence)


class TestEvaluationMetrics(unittest.TestCase):
    """Test EvaluationMetrics class."""
    
    def setUp(self):
        self.create_test_metrics()
    
    def create_test_metrics(self):
        """Create test metrics for testing."""
        self.correctness = MetricScore("correctness", 4.5, "Accurate", 0.9)
        self.completeness = MetricScore("completeness", 4.0, "Complete", 0.8)
        self.relevance = MetricScore("relevance", 4.2, "Relevant", 0.85)
        self.fluency = MetricScore("fluency", 4.8, "Fluent", 0.95)
        self.coherence = MetricScore("coherence", 4.1, "Coherent", 0.8)
        self.faithfulness = MetricScore("faithfulness", 4.3, "Faithful", 0.9)
        self.harmfulness = MetricScore("harmfulness", 1.0, "Safe", 0.95)
        self.helpfulness = MetricScore("helpfulness", 4.6, "Helpful", 0.9)
        self.conciseness = MetricScore("conciseness", 3.5, "Adequate", 0.7)
        
        self.metrics = EvaluationMetrics(
            correctness=self.correctness,
            completeness=self.completeness,
            relevance=self.relevance,
            fluency=self.fluency,
            coherence=self.coherence,
            faithfulness=self.faithfulness,
            harmfulness=self.harmfulness,
            helpfulness=self.helpfulness,
            conciseness=self.conciseness
        )
    
    def test_init(self):
        """Test EvaluationMetrics initialization."""
        self.assertEqual(self.metrics.correctness, self.correctness)
        self.assertEqual(self.metrics.completeness, self.completeness)
        self.assertEqual(self.metrics.relevance, self.relevance)
        self.assertEqual(self.metrics.fluency, self.fluency)
        self.assertEqual(self.metrics.coherence, self.coherence)
        self.assertEqual(self.metrics.faithfulness, self.faithfulness)
        self.assertEqual(self.metrics.harmfulness, self.harmfulness)
        self.assertEqual(self.metrics.helpfulness, self.helpfulness)
        self.assertEqual(self.metrics.conciseness, self.conciseness)
    
    def test_to_dict(self):
        """Test to_dict method."""
        result = self.metrics.to_dict()
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 9)
        
        # Check all metric names are present
        expected_metrics = [
            "correctness", "completeness", "relevance", "fluency",
            "coherence", "faithfulness", "harmfulness", "helpfulness", "conciseness"
        ]
        for metric_name in expected_metrics:
            self.assertIn(metric_name, result)
            self.assertIsInstance(result[metric_name], dict)
            self.assertIn("score", result[metric_name])
            self.assertIn("justification", result[metric_name])
            self.assertIn("confidence", result[metric_name])


class TestModelConfig(unittest.TestCase):
    """Test ModelConfig class."""
    
    def setUp(self):
        self.config = ModelConfig(
            model_name="gpt-4",
            temperature=0.7,
            max_tokens=1000,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.1,
            timeout=30,
            retry_attempts=3,
            retry_delay=1.0
        )
    
    def test_init_with_all_params(self):
        """Test ModelConfig initialization with all parameters."""
        self.assertEqual(self.config.model_name, "gpt-4")
        self.assertEqual(self.config.temperature, 0.7)
        self.assertEqual(self.config.max_tokens, 1000)
        self.assertEqual(self.config.top_p, 0.9)
        self.assertEqual(self.config.frequency_penalty, 0.1)
        self.assertEqual(self.config.presence_penalty, 0.1)
        self.assertEqual(self.config.timeout, 30)
        self.assertEqual(self.config.retry_attempts, 3)
        self.assertEqual(self.config.retry_delay, 1.0)
    
    def test_init_with_minimal_params(self):
        """Test ModelConfig initialization with minimal parameters."""
        config = ModelConfig(model_name="gpt-3.5-turbo")
        self.assertEqual(config.model_name, "gpt-3.5-turbo")
        self.assertEqual(config.temperature, 0.0)  # Default value
        self.assertEqual(config.max_tokens, 2000)  # Default value
        self.assertEqual(config.timeout, 30)  # Default value
        self.assertEqual(config.retry_attempts, 3)  # Default value
    
    def test_to_dict(self):
        """Test to_dict method."""
        result = self.config.to_dict()
        self.assertIsInstance(result, dict)
        
        expected_keys = [
            "model_name", "temperature", "max_tokens", "top_p",
            "frequency_penalty", "presence_penalty", "timeout",
            "retry_attempts", "retry_delay"
        ]
        for key in expected_keys:
            self.assertIn(key, result)
        
        self.assertEqual(result["model_name"], "gpt-4")
        self.assertEqual(result["temperature"], 0.7)
        self.assertEqual(result["max_tokens"], 1000)


class TestEvaluationInput(unittest.TestCase):
    """Test EvaluationInput class."""
    
    def setUp(self):
        self.eval_input = EvaluationInput(
            chat_history=[
                {"role": "user", "content": "What is Python?"},
                {"role": "assistant", "content": "I'll help you with that."}
            ],
            retrieved_chunks=[
                "Python is a programming language.",
                "Python is known for its simplicity."
            ],
            llm_answer="Python is a high-level programming language known for its readability and simplicity.",
            user_feedback_label="Positive",
            user_feedback_comment="Good explanation"
        )
    
    def test_init(self):
        """Test EvaluationInput initialization."""
        self.assertEqual(len(self.eval_input.chat_history), 2)
        self.assertEqual(len(self.eval_input.retrieved_chunks), 2)
        self.assertEqual(self.eval_input.user_feedback_label, "Positive")
        self.assertEqual(self.eval_input.user_feedback_comment, "Good explanation")
    
    def test_to_dict(self):
        """Test to_dict method."""
        result = self.eval_input.to_dict()
        self.assertIsInstance(result, dict)
        
        expected_keys = [
            "chat_history", "retrieved_chunks", "llm_answer",
            "user_feedback_label", "user_feedback_comment"
        ]
        for key in expected_keys:
            self.assertIn(key, result)
        
        self.assertEqual(len(result["chat_history"]), 2)
        self.assertEqual(len(result["retrieved_chunks"]), 2)
        self.assertEqual(result["user_feedback_label"], "Positive")
    
    def test_minimal_input(self):
        """Test EvaluationInput with minimal required fields."""
        minimal_input = EvaluationInput(
            chat_history=[{"role": "user", "content": "Test"}],
            retrieved_chunks=["Test chunk"],
            llm_answer="Test answer",
            user_feedback_label="Positive"
        )
        
        self.assertEqual(len(minimal_input.chat_history), 1)
        self.assertEqual(len(minimal_input.retrieved_chunks), 1)
        self.assertEqual(minimal_input.llm_answer, "Test answer")
        self.assertEqual(minimal_input.user_feedback_label, "Positive")
        self.assertIsNone(minimal_input.user_feedback_comment)


class TestEvaluationConfig(unittest.TestCase):
    """Test EvaluationConfig class."""
    
    def setUp(self):
        self.azure_config = ModelConfig(model_name="gpt-4")
        self.claude_config = ModelConfig(model_name="claude-3-sonnet-20240229")
        
        self.eval_config = EvaluationConfig(
            azure_openai_config=self.azure_config,
            claude_config=self.claude_config,
            enabled_metrics=["correctness", "completeness", "relevance"],
            parallel_evaluation=True,
            cache_enabled=True,
            rate_limit_enabled=True
        )
    
    def test_init(self):
        """Test EvaluationConfig initialization."""
        self.assertEqual(self.eval_config.azure_openai_config, self.azure_config)
        self.assertEqual(self.eval_config.claude_config, self.claude_config)
        self.assertEqual(len(self.eval_config.enabled_metrics), 3)
        self.assertTrue(self.eval_config.parallel_evaluation)
        self.assertTrue(self.eval_config.cache_enabled)
        self.assertTrue(self.eval_config.rate_limit_enabled)
    
    def test_default_config(self):
        """Test EvaluationConfig with default values."""
        config = EvaluationConfig(
            azure_openai_config=self.azure_config,
            claude_config=self.claude_config
        )
        
        # Check defaults
        self.assertEqual(len(config.enabled_metrics), 9)  # All metrics enabled by default
        self.assertTrue(config.parallel_evaluation)
        self.assertTrue(config.cache_enabled)
        self.assertTrue(config.rate_limit_enabled)
    
    def test_to_dict(self):
        """Test to_dict method."""
        result = self.eval_config.to_dict()
        self.assertIsInstance(result, dict)
        
        expected_keys = [
            "azure_openai_config", "claude_config", "enabled_metrics",
            "parallel_evaluation", "cache_enabled", "rate_limit_enabled"
        ]
        for key in expected_keys:
            self.assertIn(key, result)
        
        self.assertIsInstance(result["azure_openai_config"], dict)
        self.assertIsInstance(result["claude_config"], dict)
        self.assertEqual(len(result["enabled_metrics"]), 3)


class TestModelResult(unittest.TestCase):
    """Test ModelResult class."""
    
    def setUp(self):
        # Create test metrics
        self.correctness = MetricScore("correctness", 4.5, "Accurate", 0.9)
        self.completeness = MetricScore("completeness", 4.0, "Complete", 0.8)
        self.relevance = MetricScore("relevance", 4.2, "Relevant", 0.85)
        self.fluency = MetricScore("fluency", 4.8, "Fluent", 0.95)
        self.coherence = MetricScore("coherence", 4.1, "Coherent", 0.8)
        self.faithfulness = MetricScore("faithfulness", 4.3, "Faithful", 0.9)
        self.harmfulness = MetricScore("harmfulness", 1.0, "Safe", 0.95)
        self.helpfulness = MetricScore("helpfulness", 4.6, "Helpful", 0.9)
        self.conciseness = MetricScore("conciseness", 3.5, "Adequate", 0.7)
        
        self.metrics = EvaluationMetrics(
            correctness=self.correctness,
            completeness=self.completeness,
            relevance=self.relevance,
            fluency=self.fluency,
            coherence=self.coherence,
            faithfulness=self.faithfulness,
            harmfulness=self.harmfulness,
            helpfulness=self.helpfulness,
            conciseness=self.conciseness
        )
        
        self.model_result = ModelResult(
            model_name="azure_openai",
            model_version="gpt-4",
            overall_score=4.2,
            confidence=0.85,
            metrics=self.metrics,
            processing_time=2.5,
            tokens_used=150,
            raw_response="Test response"
        )
    
    def test_init(self):
        """Test ModelResult initialization."""
        self.assertEqual(self.model_result.model_name, "azure_openai")
        self.assertEqual(self.model_result.model_version, "gpt-4")
        self.assertEqual(self.model_result.overall_score, 4.2)
        self.assertEqual(self.model_result.confidence, 0.85)
        self.assertEqual(self.model_result.metrics, self.metrics)
        self.assertEqual(self.model_result.processing_time, 2.5)
        self.assertEqual(self.model_result.tokens_used, 150)
        self.assertEqual(self.model_result.raw_response, "Test response")
    
    def test_to_dict(self):
        """Test to_dict method."""
        result = self.model_result.to_dict()
        self.assertIsInstance(result, dict)
        
        expected_keys = [
            "model_name", "model_version", "overall_score", "confidence",
            "metrics", "processing_time", "tokens_used", "raw_response"
        ]
        for key in expected_keys:
            self.assertIn(key, result)
        
        self.assertEqual(result["model_name"], "azure_openai")
        self.assertEqual(result["overall_score"], 4.2)
        self.assertIsInstance(result["metrics"], dict)


class TestEvaluationResult(unittest.TestCase):
    """Test EvaluationResult class."""
    
    def setUp(self):
        # Create test metrics for both models
        self.create_test_metrics()
        self.create_test_results()
    
    def create_test_metrics(self):
        """Create test metrics."""
        self.correctness = MetricScore("correctness", 4.5, "Accurate", 0.9)
        self.completeness = MetricScore("completeness", 4.0, "Complete", 0.8)
        self.relevance = MetricScore("relevance", 4.2, "Relevant", 0.85)
        self.fluency = MetricScore("fluency", 4.8, "Fluent", 0.95)
        self.coherence = MetricScore("coherence", 4.1, "Coherent", 0.8)
        self.faithfulness = MetricScore("faithfulness", 4.3, "Faithful", 0.9)
        self.harmfulness = MetricScore("harmfulness", 1.0, "Safe", 0.95)
        self.helpfulness = MetricScore("helpfulness", 4.6, "Helpful", 0.9)
        self.conciseness = MetricScore("conciseness", 3.5, "Adequate", 0.7)
        
        self.metrics = EvaluationMetrics(
            correctness=self.correctness,
            completeness=self.completeness,
            relevance=self.relevance,
            fluency=self.fluency,
            coherence=self.coherence,
            faithfulness=self.faithfulness,
            harmfulness=self.harmfulness,
            helpfulness=self.helpfulness,
            conciseness=self.conciseness
        )
    
    def create_test_results(self):
        """Create test results."""
        self.azure_result = ModelResult(
            model_name="azure_openai",
            model_version="gpt-4",
            overall_score=4.2,
            confidence=0.85,
            metrics=self.metrics,
            processing_time=2.5,
            tokens_used=150,
            raw_response="Azure response"
        )
        
        self.claude_result = ModelResult(
            model_name="claude",
            model_version="claude-3-sonnet-20240229",
            overall_score=4.3,
            confidence=0.88,
            metrics=self.metrics,
            processing_time=2.1,
            tokens_used=140,
            raw_response="Claude response"
        )
        
        self.comparison = {
            "overall_score_difference": 0.1,
            "agreement_analysis": {
                "overall_agreement": 0.85,
                "high_agreement_metrics": ["correctness", "relevance"],
                "low_agreement_metrics": ["fluency"]
            }
        }
        
        self.insights = {
            "summary": {
                "overall_quality": "High",
                "strongest_aspect": "correctness",
                "weakest_aspect": "conciseness"
            },
            "strengths": ["Strong correctness", "Good relevance"],
            "weaknesses": ["Verbose response"],
            "recommendations": ["Consider more concise phrasing"]
        }
        
        self.metadata = {
            "evaluation_time": 5.2,
            "timestamp": "2024-01-15T10:30:00",
            "config": {},
            "enabled_metrics": ["correctness", "completeness", "relevance"],
            "parallel_evaluation": True
        }
        
        self.evaluation_result = EvaluationResult(
            azure_openai_result=self.azure_result,
            claude_result=self.claude_result,
            comparison=self.comparison,
            aggregated_insights=self.insights,
            evaluation_metadata=self.metadata
        )
    
    def test_init(self):
        """Test EvaluationResult initialization."""
        self.assertEqual(self.evaluation_result.azure_openai_result, self.azure_result)
        self.assertEqual(self.evaluation_result.claude_result, self.claude_result)
        self.assertEqual(self.evaluation_result.comparison, self.comparison)
        self.assertEqual(self.evaluation_result.aggregated_insights, self.insights)
        self.assertEqual(self.evaluation_result.evaluation_metadata, self.metadata)
    
    def test_to_dict(self):
        """Test to_dict method."""
        result = self.evaluation_result.to_dict()
        self.assertIsInstance(result, dict)
        
        expected_keys = [
            "azure_openai_result", "claude_result", "comparison",
            "aggregated_insights", "evaluation_metadata"
        ]
        for key in expected_keys:
            self.assertIn(key, result)
        
        self.assertIsInstance(result["azure_openai_result"], dict)
        self.assertIsInstance(result["claude_result"], dict)
        self.assertIsInstance(result["comparison"], dict)
        self.assertIsInstance(result["aggregated_insights"], dict)
        self.assertIsInstance(result["evaluation_metadata"], dict)


if __name__ == '__main__':
    unittest.main()