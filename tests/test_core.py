"""
Unit tests for core evaluator functionality (core.py).
"""

import unittest
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime

from llm_evaluator.core import LLMEvaluator
from llm_evaluator.models import (
    EvaluationInput, EvaluationConfig, ModelConfig, 
    ModelResult, EvaluationResult, EvaluationMetrics, MetricScore
)


class TestLLMEvaluator(unittest.TestCase):
    """Test LLMEvaluator class."""
    
    def setUp(self):
        # Create test configurations
        self.azure_config = ModelConfig(model_name="gpt-4")
        self.claude_config = ModelConfig(model_name="claude-3-sonnet-20240229")
        
        self.eval_config = EvaluationConfig(
            azure_openai_config=self.azure_config,
            claude_config=self.claude_config,
            parallel_evaluation=True,
            cache_enabled=True,
            rate_limit_enabled=True
        )
        
        # Create test evaluation input
        self.eval_input = EvaluationInput(
            chat_history=[{"role": "user", "content": "What is Python?"}],
            retrieved_chunks=["Python is a programming language."],
            llm_answer="Python is a high-level programming language.",
            user_feedback_label="Positive",
            user_feedback_comment="Good explanation"
        )
        
        # Create evaluator instance
        self.evaluator = LLMEvaluator(
            azure_openai_endpoint="https://test.openai.azure.com/",
            azure_openai_api_key="test-key",
            azure_openai_deployment="test-deployment",
            claude_api_key="test-key",
            config=self.eval_config
        )
    
    def test_init(self):
        """Test LLMEvaluator initialization."""
        self.assertEqual(self.evaluator.azure_openai_endpoint, "https://test.openai.azure.com/")
        self.assertEqual(self.evaluator.azure_openai_api_key, "test-key")
        self.assertEqual(self.evaluator.azure_openai_deployment, "test-deployment")
        self.assertEqual(self.evaluator.claude_api_key, "test-key")
        self.assertEqual(self.evaluator.config, self.eval_config)
        
        # Check that components are initialized
        self.assertIsNotNone(self.evaluator.metrics_framework)
        self.assertIsNotNone(self.evaluator.azure_integration)
        self.assertIsNotNone(self.evaluator.claude_integration)
        self.assertIsNotNone(self.evaluator.cache_manager)
        self.assertIsNotNone(self.evaluator.rate_limiter)
    
    def test_init_with_default_config(self):
        """Test LLMEvaluator initialization with default config."""
        evaluator = LLMEvaluator(
            azure_openai_endpoint="https://test.openai.azure.com/",
            azure_openai_api_key="test-key",
            azure_openai_deployment="test-deployment",
            claude_api_key="test-key"
        )
        
        # Should have default config
        self.assertIsNotNone(evaluator.config)
        self.assertEqual(evaluator.config.azure_openai_config.model_name, "gpt-4")
        self.assertEqual(evaluator.config.claude_config.model_name, "claude-3-sonnet-20240229")
    
    def test_init_with_cache_disabled(self):
        """Test LLMEvaluator initialization with cache disabled."""
        config = EvaluationConfig(
            azure_openai_config=self.azure_config,
            claude_config=self.claude_config,
            cache_enabled=False
        )
        
        evaluator = LLMEvaluator(
            azure_openai_endpoint="https://test.openai.azure.com/",
            azure_openai_api_key="test-key",
            azure_openai_deployment="test-deployment",
            claude_api_key="test-key",
            config=config
        )
        
        self.assertIsNone(evaluator.cache_manager)
    
    def test_init_with_rate_limit_disabled(self):
        """Test LLMEvaluator initialization with rate limiting disabled."""
        config = EvaluationConfig(
            azure_openai_config=self.azure_config,
            claude_config=self.claude_config,
            rate_limit_enabled=False
        )
        
        evaluator = LLMEvaluator(
            azure_openai_endpoint="https://test.openai.azure.com/",
            azure_openai_api_key="test-key",
            azure_openai_deployment="test-deployment",
            claude_api_key="test-key",
            config=config
        )
        
        self.assertIsNone(evaluator.rate_limiter)
    
    def test_generate_cache_key(self):
        """Test _generate_cache_key method."""
        cache_key = self.evaluator._generate_cache_key(self.eval_input)
        
        self.assertIsInstance(cache_key, str)
        self.assertGreater(len(cache_key), 0)
        
        # Same input should generate same key
        cache_key2 = self.evaluator._generate_cache_key(self.eval_input)
        self.assertEqual(cache_key, cache_key2)
        
        # Different input should generate different key
        different_input = EvaluationInput(
            chat_history=[{"role": "user", "content": "What is Java?"}],
            retrieved_chunks=["Java is a programming language."],
            llm_answer="Java is a programming language.",
        )
        different_key = self.evaluator._generate_cache_key(different_input)
        self.assertNotEqual(cache_key, different_key)
    
    def test_get_metrics_info(self):
        """Test get_metrics_info method."""
        metrics_info = self.evaluator.get_metrics_info()
        
        self.assertIsInstance(metrics_info, dict)
        self.assertEqual(len(metrics_info), 9)
        
        # Check that all expected metrics are present
        expected_metrics = [
            "correctness", "completeness", "relevance", "fluency",
            "coherence", "faithfulness", "harmfulness", "helpfulness", "conciseness"
        ]
        
        for metric in expected_metrics:
            self.assertIn(metric, metrics_info)
            self.assertIn("name", metrics_info[metric])
            self.assertIn("description", metrics_info[metric])
    
    def create_mock_model_result(self, model_name="test_model"):
        """Create a mock ModelResult for testing."""
        # Create mock metrics
        correctness = MetricScore("correctness", 4.5, "Accurate", 0.9)
        completeness = MetricScore("completeness", 4.0, "Complete", 0.8)
        relevance = MetricScore("relevance", 4.2, "Relevant", 0.85)
        fluency = MetricScore("fluency", 4.8, "Fluent", 0.95)
        coherence = MetricScore("coherence", 4.1, "Coherent", 0.8)
        faithfulness = MetricScore("faithfulness", 4.3, "Faithful", 0.9)
        harmfulness = MetricScore("harmfulness", 1.0, "Safe", 0.95)
        helpfulness = MetricScore("helpfulness", 4.6, "Helpful", 0.9)
        conciseness = MetricScore("conciseness", 3.5, "Adequate", 0.7)
        
        metrics = EvaluationMetrics(
            correctness=correctness,
            completeness=completeness,
            relevance=relevance,
            fluency=fluency,
            coherence=coherence,
            faithfulness=faithfulness,
            harmfulness=harmfulness,
            helpfulness=helpfulness,
            conciseness=conciseness
        )
        
        return ModelResult(
            model_name=model_name,
            model_version="test-version",
            overall_score=4.2,
            confidence=0.85,
            metrics=metrics,
            processing_time=2.5,
            tokens_used=150,
            raw_response="Test response"
        )
    
    @patch('llm_evaluator.core.LLMEvaluator._evaluate_with_model')
    async def test_evaluate_parallel(self, mock_evaluate):
        """Test evaluate method with parallel evaluation."""
        # Create mock results
        azure_result = self.create_mock_model_result("azure_openai")
        claude_result = self.create_mock_model_result("claude")
        
        mock_evaluate.side_effect = [azure_result, claude_result]
        
        # Test parallel evaluation
        self.evaluator.config.parallel_evaluation = True
        result = await self.evaluator.evaluate(self.eval_input)
        
        self.assertIsInstance(result, EvaluationResult)
        self.assertEqual(result.azure_openai_result, azure_result)
        self.assertEqual(result.claude_result, claude_result)
        
        # Should have called evaluate_with_model twice
        self.assertEqual(mock_evaluate.call_count, 2)
    
    @patch('llm_evaluator.core.LLMEvaluator._evaluate_with_model')
    async def test_evaluate_sequential(self, mock_evaluate):
        """Test evaluate method with sequential evaluation."""
        # Create mock results
        azure_result = self.create_mock_model_result("azure_openai")
        claude_result = self.create_mock_model_result("claude")
        
        mock_evaluate.side_effect = [azure_result, claude_result]
        
        # Test sequential evaluation
        self.evaluator.config.parallel_evaluation = False
        result = await self.evaluator.evaluate(self.eval_input)
        
        self.assertIsInstance(result, EvaluationResult)
        self.assertEqual(result.azure_openai_result, azure_result)
        self.assertEqual(result.claude_result, claude_result)
        
        # Should have called evaluate_with_model twice
        self.assertEqual(mock_evaluate.call_count, 2)
    
    @patch('llm_evaluator.core.LLMEvaluator._evaluate_with_model')
    async def test_evaluate_with_cache(self, mock_evaluate):
        """Test evaluate method with caching."""
        # Create mock result
        azure_result = self.create_mock_model_result("azure_openai")
        claude_result = self.create_mock_model_result("claude")
        
        mock_evaluate.side_effect = [azure_result, claude_result]
        
        # First evaluation - should call models
        result1 = await self.evaluator.evaluate(self.eval_input)
        
        # Second evaluation with same input - should use cache
        result2 = await self.evaluator.evaluate(self.eval_input)
        
        # Should have only called evaluate_with_model twice (first time only)
        self.assertEqual(mock_evaluate.call_count, 2)
        
        # Results should be the same
        self.assertEqual(result1.azure_openai_result.overall_score, result2.azure_openai_result.overall_score)
    
    @patch('llm_evaluator.core.LLMEvaluator._evaluate_with_model')
    async def test_evaluate_without_cache(self, mock_evaluate):
        """Test evaluate method without caching."""
        # Disable cache
        self.evaluator.cache_manager = None
        
        # Create mock result
        azure_result = self.create_mock_model_result("azure_openai")
        claude_result = self.create_mock_model_result("claude")
        
        mock_evaluate.side_effect = [azure_result, claude_result, azure_result, claude_result]
        
        # First evaluation
        result1 = await self.evaluator.evaluate(self.eval_input)
        
        # Second evaluation with same input - should call models again
        result2 = await self.evaluator.evaluate(self.eval_input)
        
        # Should have called evaluate_with_model four times (twice each evaluation)
        self.assertEqual(mock_evaluate.call_count, 4)
    
    @patch('llm_evaluator.core.LLMEvaluator._evaluate_with_model')
    async def test_batch_evaluate(self, mock_evaluate):
        """Test batch_evaluate method."""
        # Create mock results
        azure_result = self.create_mock_model_result("azure_openai")
        claude_result = self.create_mock_model_result("claude")
        
        mock_evaluate.side_effect = [azure_result, claude_result, azure_result, claude_result]
        
        # Create batch inputs
        batch_inputs = [self.eval_input, self.eval_input]
        
        results = await self.evaluator.batch_evaluate(batch_inputs)
        
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 2)
        
        for result in results:
            self.assertIsInstance(result, EvaluationResult)
            self.assertEqual(result.azure_openai_result.model_name, "azure_openai")
            self.assertEqual(result.claude_result.model_name, "claude")
    
    @patch('llm_evaluator.integrations.AzureOpenAIIntegration.make_request')
    @patch('llm_evaluator.metrics.MetricsFramework.parse_model_response')
    async def test_evaluate_with_model(self, mock_parse, mock_request):
        """Test _evaluate_with_model method."""
        # Mock API response
        mock_request.return_value = '{"score": 4.5, "justification": "Good", "confidence": 0.9}'
        
        # Mock parsed response
        mock_metric_score = MetricScore("correctness", 4.5, "Good", 0.9)
        mock_parse.return_value = mock_metric_score
        
        # Test with Azure integration
        result = await self.evaluator._evaluate_with_model(
            self.eval_input, 
            self.evaluator.azure_integration, 
            "azure_openai"
        )
        
        self.assertIsInstance(result, ModelResult)
        self.assertEqual(result.model_name, "azure_openai")
        self.assertIsInstance(result.overall_score, float)
        self.assertIsInstance(result.confidence, float)
        self.assertIsInstance(result.metrics, EvaluationMetrics)
        
        # Should have called the model for each enabled metric
        expected_calls = len(self.evaluator.config.enabled_metrics)
        self.assertEqual(mock_request.call_count, expected_calls)
    
    def test_create_comparison(self):
        """Test _create_comparison method."""
        azure_result = self.create_mock_model_result("azure_openai")
        claude_result = self.create_mock_model_result("claude")
        
        # Modify scores slightly for comparison
        claude_result.overall_score = 4.3
        
        comparison = self.evaluator._create_comparison(azure_result, claude_result)
        
        self.assertIsInstance(comparison, dict)
        self.assertIn("overall_score_difference", comparison)
        self.assertIn("agreement_analysis", comparison)
        
        # Check score difference
        expected_diff = abs(azure_result.overall_score - claude_result.overall_score)
        self.assertAlmostEqual(comparison["overall_score_difference"], expected_diff, places=2)
        
        # Check agreement analysis
        self.assertIn("overall_agreement", comparison["agreement_analysis"])
        self.assertIn("high_agreement_metrics", comparison["agreement_analysis"])
        self.assertIn("low_agreement_metrics", comparison["agreement_analysis"])
    
    def test_generate_insights(self):
        """Test _generate_insights method."""
        azure_result = self.create_mock_model_result("azure_openai")
        claude_result = self.create_mock_model_result("claude")
        
        insights = self.evaluator._generate_insights(azure_result, claude_result, self.eval_input)
        
        self.assertIsInstance(insights, dict)
        self.assertIn("summary", insights)
        self.assertIn("strengths", insights)
        self.assertIn("weaknesses", insights)
        self.assertIn("recommendations", insights)
        
        # Check summary
        summary = insights["summary"]
        self.assertIn("overall_quality", summary)
        self.assertIn("strongest_aspect", summary)
        self.assertIn("weakest_aspect", summary)
        
        # Check that strengths and weaknesses are lists
        self.assertIsInstance(insights["strengths"], list)
        self.assertIsInstance(insights["weaknesses"], list)
        self.assertIsInstance(insights["recommendations"], list)
    
    def test_determine_quality_level(self):
        """Test _determine_quality_level method."""
        # Test high quality
        high_score = 4.5
        self.assertEqual(self.evaluator._determine_quality_level(high_score), "High")
        
        # Test medium quality
        medium_score = 3.5
        self.assertEqual(self.evaluator._determine_quality_level(medium_score), "Medium")
        
        # Test low quality
        low_score = 2.0
        self.assertEqual(self.evaluator._determine_quality_level(low_score), "Low")
    
    def test_identify_strongest_aspect(self):
        """Test _identify_strongest_aspect method."""
        azure_result = self.create_mock_model_result("azure_openai")
        claude_result = self.create_mock_model_result("claude")
        
        # Modify one metric to be clearly strongest
        azure_result.metrics.fluency.score = 5.0
        claude_result.metrics.fluency.score = 5.0
        
        strongest = self.evaluator._identify_strongest_aspect(azure_result, claude_result)
        
        self.assertIsInstance(strongest, str)
        self.assertIn(strongest, [
            "correctness", "completeness", "relevance", "fluency",
            "coherence", "faithfulness", "harmfulness", "helpfulness", "conciseness"
        ])
    
    def test_identify_weakest_aspect(self):
        """Test _identify_weakest_aspect method."""
        azure_result = self.create_mock_model_result("azure_openai")
        claude_result = self.create_mock_model_result("claude")
        
        # Modify one metric to be clearly weakest
        azure_result.metrics.conciseness.score = 1.0
        claude_result.metrics.conciseness.score = 1.0
        
        weakest = self.evaluator._identify_weakest_aspect(azure_result, claude_result)
        
        self.assertIsInstance(weakest, str)
        self.assertIn(weakest, [
            "correctness", "completeness", "relevance", "fluency",
            "coherence", "faithfulness", "harmfulness", "helpfulness", "conciseness"
        ])
    
    def test_generate_strengths(self):
        """Test _generate_strengths method."""
        azure_result = self.create_mock_model_result("azure_openai")
        claude_result = self.create_mock_model_result("claude")
        
        strengths = self.evaluator._generate_strengths(azure_result, claude_result)
        
        self.assertIsInstance(strengths, list)
        self.assertGreater(len(strengths), 0)
        
        # Each strength should be a string
        for strength in strengths:
            self.assertIsInstance(strength, str)
    
    def test_generate_weaknesses(self):
        """Test _generate_weaknesses method."""
        azure_result = self.create_mock_model_result("azure_openai")
        claude_result = self.create_mock_model_result("claude")
        
        weaknesses = self.evaluator._generate_weaknesses(azure_result, claude_result)
        
        self.assertIsInstance(weaknesses, list)
        
        # Each weakness should be a string
        for weakness in weaknesses:
            self.assertIsInstance(weakness, str)
    
    def test_generate_recommendations(self):
        """Test _generate_recommendations method."""
        azure_result = self.create_mock_model_result("azure_openai")
        claude_result = self.create_mock_model_result("claude")
        
        recommendations = self.evaluator._generate_recommendations(azure_result, claude_result)
        
        self.assertIsInstance(recommendations, list)
        
        # Each recommendation should be a string
        for recommendation in recommendations:
            self.assertIsInstance(recommendation, str)
    
    def test_calculate_model_agreement(self):
        """Test _calculate_model_agreement method."""
        azure_result = self.create_mock_model_result("azure_openai")
        claude_result = self.create_mock_model_result("claude")
        
        # Modify some scores for testing
        claude_result.metrics.correctness.score = 4.0
        claude_result.metrics.completeness.score = 3.5
        
        agreement = self.evaluator._calculate_model_agreement(azure_result, claude_result)
        
        self.assertIsInstance(agreement, float)
        self.assertGreaterEqual(agreement, 0.0)
        self.assertLessEqual(agreement, 1.0)
    
    def test_identify_agreement_metrics(self):
        """Test _identify_agreement_metrics method."""
        azure_result = self.create_mock_model_result("azure_openai")
        claude_result = self.create_mock_model_result("claude")
        
        # Modify some scores for testing
        claude_result.metrics.correctness.score = 4.5  # High agreement
        claude_result.metrics.completeness.score = 2.0  # Low agreement
        
        high_agreement, low_agreement = self.evaluator._identify_agreement_metrics(
            azure_result, claude_result
        )
        
        self.assertIsInstance(high_agreement, list)
        self.assertIsInstance(low_agreement, list)
        
        # Should contain metric names
        for metric in high_agreement:
            self.assertIn(metric, [
                "correctness", "completeness", "relevance", "fluency",
                "coherence", "faithfulness", "harmfulness", "helpfulness", "conciseness"
            ])
        
        for metric in low_agreement:
            self.assertIn(metric, [
                "correctness", "completeness", "relevance", "fluency",
                "coherence", "faithfulness", "harmfulness", "helpfulness", "conciseness"
            ])


if __name__ == '__main__':
    unittest.main()