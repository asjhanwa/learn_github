"""
Unit tests for metrics framework (metrics.py).
"""

import unittest
from unittest.mock import Mock, patch, MagicMock

from llm_evaluator.metrics import MetricsFramework
from llm_evaluator.models import EvaluationInput, MetricScore


class TestMetricsFramework(unittest.TestCase):
    """Test MetricsFramework class."""
    
    def setUp(self):
        self.metrics_framework = MetricsFramework()
        self.test_eval_input = EvaluationInput(
            chat_history=[{"role": "user", "content": "What is Python?"}],
            retrieved_chunks=["Python is a programming language."],
            llm_answer="Python is a high-level programming language known for its readability.",
            user_feedback_label="Positive",
            user_feedback_comment="Good explanation"
        )
    
    def test_init(self):
        """Test MetricsFramework initialization."""
        self.assertIsNotNone(self.metrics_framework.metrics_definitions)
        self.assertIsInstance(self.metrics_framework.metrics_definitions, dict)
        self.assertEqual(len(self.metrics_framework.metrics_definitions), 9)
    
    def test_metrics_definitions_structure(self):
        """Test that metrics definitions have the correct structure."""
        expected_metrics = [
            "correctness", "completeness", "relevance", "fluency",
            "coherence", "faithfulness", "harmfulness", "helpfulness", "conciseness"
        ]
        
        for metric in expected_metrics:
            self.assertIn(metric, self.metrics_framework.metrics_definitions)
            definition = self.metrics_framework.metrics_definitions[metric]
            
            # Check required fields
            self.assertIn("name", definition)
            self.assertIn("description", definition)
            self.assertIn("scale", definition)
            self.assertIn("criteria", definition)
            
            # Check types
            self.assertIsInstance(definition["name"], str)
            self.assertIsInstance(definition["description"], str)
            self.assertIsInstance(definition["scale"], str)
            self.assertIsInstance(definition["criteria"], list)
            
            # Check criteria has content
            self.assertGreater(len(definition["criteria"]), 0)
    
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
        
        # Test invalid metric - should return empty dict
        invalid_def = self.metrics_framework.get_metric_definition("invalid_metric")
        self.assertEqual(invalid_def, {})
    
    def test_create_evaluation_prompt(self):
        """Test create_evaluation_prompt method."""
        prompt = self.metrics_framework.create_evaluation_prompt("correctness", self.test_eval_input)
        
        self.assertIsInstance(prompt, str)
        self.assertGreater(len(prompt), 100)  # Should be substantial
        
        # Check that key components are included
        self.assertIn("correctness", prompt.lower())
        self.assertIn("Python", prompt)  # From the test input
        self.assertIn("score", prompt.lower())
        self.assertIn("justification", prompt.lower())
        self.assertIn("confidence", prompt.lower())
    
    def test_create_evaluation_prompt_all_metrics(self):
        """Test create_evaluation_prompt for all metrics."""
        # Get metrics from the actual implementation
        metrics = self.metrics_framework.get_all_metrics()
        
        for metric in metrics:
            prompt = self.metrics_framework.create_evaluation_prompt(metric, self.test_eval_input)
            
            self.assertIsInstance(prompt, str)
            self.assertGreater(len(prompt), 50)
            self.assertIn(metric, prompt.lower())
    
    def test_create_evaluation_prompt_invalid_metric(self):
        """Test create_evaluation_prompt with invalid metric."""
        # Based on the actual implementation, this should raise KeyError
        with self.assertRaises(KeyError):
            self.metrics_framework.create_evaluation_prompt("invalid_metric", self.test_eval_input)
    
    def test_create_evaluation_prompt_components(self):
        """Test that evaluation prompt includes all necessary components."""
        prompt = self.metrics_framework.create_evaluation_prompt("correctness", self.test_eval_input)
        
        # Check for chat history
        self.assertIn("What is Python?", prompt)
        
        # Check for retrieved chunks
        self.assertIn("Python is a programming language", prompt)
        
        # Check for LLM answer
        self.assertIn("high-level programming language", prompt)
        
        # Check for user feedback
        self.assertIn("Positive", prompt)
        self.assertIn("Good explanation", prompt)
        
        # Check for metric-specific content
        self.assertIn("Correctness", prompt)
        self.assertIn("Factual accuracy", prompt)
    
    def test_validate_metric_score(self):
        """Test validate_metric_score method."""
        # Test valid scores
        valid_scores = [1.0, 2.5, 3.0, 4.5, 5.0]
        for score in valid_scores:
            self.assertTrue(self.metrics_framework.validate_metric_score(score))
        
        # Test invalid scores
        invalid_scores = [0.0, 0.5, 5.5, 6.0, -1.0]
        for score in invalid_scores:
            self.assertFalse(self.metrics_framework.validate_metric_score(score))
    
    def test_validate_confidence(self):
        """Test validate_confidence method."""
        # Test valid confidence values
        valid_confidences = [0.0, 0.5, 0.85, 1.0]
        for confidence in valid_confidences:
            self.assertTrue(self.metrics_framework.validate_confidence(confidence))
        
        # Test invalid confidence values
        invalid_confidences = [-0.1, 1.1, 2.0, -1.0]
        for confidence in invalid_confidences:
            self.assertFalse(self.metrics_framework.validate_confidence(confidence))
    
    def test_parse_model_response(self):
        """Test parse_model_response method."""
        # Mock response with valid JSON
        mock_response = '''
        {
            "score": 4.5,
            "justification": "The answer is factually correct and well-explained.",
            "confidence": 0.85
        }
        '''
        
        result = self.metrics_framework.parse_model_response("correctness", mock_response)
        
        self.assertIsInstance(result, MetricScore)
        self.assertEqual(result.name, "correctness")
        self.assertEqual(result.score, 4.5)
        self.assertEqual(result.justification, "The answer is factually correct and well-explained.")
        self.assertEqual(result.confidence, 0.85)
    
    def test_parse_model_response_invalid_json(self):
        """Test parse_model_response with invalid JSON."""
        invalid_response = "This is not valid JSON"
        
        result = self.metrics_framework.parse_model_response("correctness", invalid_response)
        
        # Should return a default/fallback score
        self.assertIsInstance(result, MetricScore)
        self.assertEqual(result.name, "correctness")
        self.assertEqual(result.score, 3.0)  # Default score
        self.assertIn("Unable to parse", result.justification)
        self.assertEqual(result.confidence, 0.5)  # Default confidence
    
    def test_parse_model_response_missing_fields(self):
        """Test parse_model_response with missing required fields."""
        incomplete_response = '''
        {
            "score": 4.5
        }
        '''
        
        result = self.metrics_framework.parse_model_response("correctness", incomplete_response)
        
        self.assertIsInstance(result, MetricScore)
        self.assertEqual(result.name, "correctness")
        self.assertEqual(result.score, 4.5)
        self.assertIn("No justification provided", result.justification)
        self.assertEqual(result.confidence, 0.5)  # Default confidence
    
    def test_parse_model_response_invalid_score(self):
        """Test parse_model_response with invalid score values."""
        invalid_score_response = '''
        {
            "score": 6.0,
            "justification": "Test justification",
            "confidence": 0.85
        }
        '''
        
        result = self.metrics_framework.parse_model_response("correctness", invalid_score_response)
        
        self.assertIsInstance(result, MetricScore)
        self.assertEqual(result.name, "correctness")
        self.assertEqual(result.score, 3.0)  # Normalized to default
        self.assertIn("Invalid score", result.justification)
        self.assertEqual(result.confidence, 0.5)  # Default confidence
    
    def test_parse_model_response_invalid_confidence(self):
        """Test parse_model_response with invalid confidence values."""
        invalid_confidence_response = '''
        {
            "score": 4.5,
            "justification": "Test justification",
            "confidence": 1.5
        }
        '''
        
        result = self.metrics_framework.parse_model_response("correctness", invalid_confidence_response)
        
        self.assertIsInstance(result, MetricScore)
        self.assertEqual(result.name, "correctness")
        self.assertEqual(result.score, 4.5)
        self.assertEqual(result.justification, "Test justification")
        self.assertEqual(result.confidence, 0.5)  # Normalized to default
    
    def test_calculate_overall_score(self):
        """Test calculate_overall_score method."""
        # Create test metrics
        correctness = MetricScore("correctness", 4.5, "Good", 0.9)
        completeness = MetricScore("completeness", 4.0, "Complete", 0.8)
        relevance = MetricScore("relevance", 4.2, "Relevant", 0.85)
        fluency = MetricScore("fluency", 4.8, "Fluent", 0.95)
        coherence = MetricScore("coherence", 4.1, "Coherent", 0.8)
        faithfulness = MetricScore("faithfulness", 4.3, "Faithful", 0.9)
        harmfulness = MetricScore("harmfulness", 1.0, "Safe", 0.95)
        helpfulness = MetricScore("helpfulness", 4.6, "Helpful", 0.9)
        conciseness = MetricScore("conciseness", 3.5, "Adequate", 0.7)
        
        metrics = [correctness, completeness, relevance, fluency, coherence, 
                  faithfulness, harmfulness, helpfulness, conciseness]
        
        overall_score = self.metrics_framework.calculate_overall_score(metrics)
        
        self.assertIsInstance(overall_score, float)
        self.assertGreater(overall_score, 0.0)
        self.assertLessEqual(overall_score, 5.0)
        
        # Should be roughly the average, considering harmfulness is inverted
        expected_avg = (4.5 + 4.0 + 4.2 + 4.8 + 4.1 + 4.3 + (6.0 - 1.0) + 4.6 + 3.5) / 9
        self.assertAlmostEqual(overall_score, expected_avg, places=1)
    
    def test_calculate_overall_confidence(self):
        """Test calculate_overall_confidence method."""
        # Create test metrics
        correctness = MetricScore("correctness", 4.5, "Good", 0.9)
        completeness = MetricScore("completeness", 4.0, "Complete", 0.8)
        relevance = MetricScore("relevance", 4.2, "Relevant", 0.85)
        
        metrics = [correctness, completeness, relevance]
        
        overall_confidence = self.metrics_framework.calculate_overall_confidence(metrics)
        
        self.assertIsInstance(overall_confidence, float)
        self.assertGreaterEqual(overall_confidence, 0.0)
        self.assertLessEqual(overall_confidence, 1.0)
        
        # Should be roughly the average
        expected_avg = (0.9 + 0.8 + 0.85) / 3
        self.assertAlmostEqual(overall_confidence, expected_avg, places=2)
    
    def test_get_metric_weights(self):
        """Test get_metric_weights method."""
        weights = self.metrics_framework.get_metric_weights()
        
        self.assertIsInstance(weights, dict)
        self.assertEqual(len(weights), 9)
        
        # Check all metrics have weights
        for metric in self.metrics_framework.get_metric_names():
            self.assertIn(metric, weights)
            self.assertIsInstance(weights[metric], (int, float))
            self.assertGreater(weights[metric], 0.0)
        
        # Check weights sum to 1.0 (or close to it)
        total_weight = sum(weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=2)
    
    def test_get_enabled_metrics(self):
        """Test get_enabled_metrics method with different configurations."""
        # Test with all metrics enabled (default)
        enabled_metrics = ["correctness", "completeness", "relevance", "fluency",
                          "coherence", "faithfulness", "harmfulness", "helpfulness", "conciseness"]
        
        result = self.metrics_framework.get_enabled_metrics(enabled_metrics)
        self.assertEqual(len(result), 9)
        
        # Test with subset of metrics
        subset_metrics = ["correctness", "completeness", "relevance"]
        result = self.metrics_framework.get_enabled_metrics(subset_metrics)
        self.assertEqual(len(result), 3)
        
        for metric in subset_metrics:
            self.assertIn(metric, result)
    
    def test_get_enabled_metrics_invalid(self):
        """Test get_enabled_metrics with invalid metric names."""
        invalid_metrics = ["correctness", "invalid_metric", "completeness"]
        
        with self.assertRaises(ValueError):
            self.metrics_framework.get_enabled_metrics(invalid_metrics)


if __name__ == '__main__':
    unittest.main()