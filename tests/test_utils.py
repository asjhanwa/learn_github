"""
Unit tests for utility classes (utils.py).
"""

import unittest
import asyncio
import time
import json
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock, mock_open
from pathlib import Path

from llm_evaluator.utils import (
    CacheManager, RateLimiter, AnalysisUtils, ExportUtils
)
from llm_evaluator.models import EvaluationResult, ModelResult, EvaluationMetrics, MetricScore


class TestCacheManager(unittest.TestCase):
    """Test CacheManager class."""
    
    def setUp(self):
        self.cache_manager = CacheManager(max_size=5, ttl_seconds=3600)
    
    def test_init(self):
        """Test CacheManager initialization."""
        self.assertEqual(self.cache_manager.max_size, 5)
        self.assertEqual(self.cache_manager.ttl_seconds, 3600)
        self.assertIsInstance(self.cache_manager.cache, dict)
        self.assertIsInstance(self.cache_manager.timestamps, dict)
        self.assertEqual(len(self.cache_manager.cache), 0)
    
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
    
    def test_ttl_expiration(self):
        """Test TTL expiration."""
        # Create cache with very short TTL
        short_cache = CacheManager(max_size=5, ttl_seconds=0.1)
        short_cache.set("key1", "value1")
        
        # Should still be available immediately
        self.assertEqual(short_cache.get("key1"), "value1")
        
        # Wait for TTL to expire
        time.sleep(0.2)
        
        # Should now be expired
        self.assertIsNone(short_cache.get("key1"))
    
    def test_max_size_limit(self):
        """Test maximum size limit."""
        # Fill cache to max size
        for i in range(5):
            self.cache_manager.set(f"key{i}", f"value{i}")
        
        self.assertEqual(self.cache_manager.size(), 5)
        
        # Add one more item - should evict oldest
        self.cache_manager.set("key5", "value5")
        
        # Should still be at max size
        self.assertEqual(self.cache_manager.size(), 5)
        
        # First item should be evicted
        self.assertIsNone(self.cache_manager.get("key0"))
        
        # Last item should be present
        self.assertEqual(self.cache_manager.get("key5"), "value5")
    
    def test_size_method(self):
        """Test size method."""
        self.assertEqual(self.cache_manager.size(), 0)
        
        self.cache_manager.set("key1", "value1")
        self.assertEqual(self.cache_manager.size(), 1)
        
        self.cache_manager.set("key2", "value2")
        self.assertEqual(self.cache_manager.size(), 2)
    
    def test_clear_method(self):
        """Test clear method."""
        self.cache_manager.set("key1", "value1")
        self.cache_manager.set("key2", "value2")
        self.assertEqual(self.cache_manager.size(), 2)
        
        self.cache_manager.clear()
        self.assertEqual(self.cache_manager.size(), 0)
        self.assertIsNone(self.cache_manager.get("key1"))
        self.assertIsNone(self.cache_manager.get("key2"))
    
    def test_contains_method(self):
        """Test contains method."""
        self.assertFalse(self.cache_manager.contains("key1"))
        
        self.cache_manager.set("key1", "value1")
        self.assertTrue(self.cache_manager.contains("key1"))
        
        # Test with expired key
        short_cache = CacheManager(max_size=5, ttl_seconds=0.1)
        short_cache.set("key1", "value1")
        time.sleep(0.2)
        self.assertFalse(short_cache.contains("key1"))


class TestRateLimiter(unittest.TestCase):
    """Test RateLimiter class."""
    
    def setUp(self):
        self.rate_limiter = RateLimiter(max_requests=3, window_seconds=1)
    
    def test_init(self):
        """Test RateLimiter initialization."""
        self.assertEqual(self.rate_limiter.max_requests, 3)
        self.assertEqual(self.rate_limiter.window_seconds, 1)
        self.assertIsInstance(self.rate_limiter.request_times, list)
        self.assertEqual(len(self.rate_limiter.request_times), 0)
    
    def test_can_make_request(self):
        """Test can_make_request method."""
        # Should be able to make initial requests
        self.assertTrue(self.rate_limiter.can_make_request())
        
        # Add requests manually to test limit
        current_time = time.time()
        for i in range(3):
            self.rate_limiter.request_times.append(current_time)
        
        # Should not be able to make more requests
        self.assertFalse(self.rate_limiter.can_make_request())
    
    def test_record_request(self):
        """Test record_request method."""
        initial_count = len(self.rate_limiter.request_times)
        
        self.rate_limiter.record_request()
        
        self.assertEqual(len(self.rate_limiter.request_times), initial_count + 1)
        
        # Check that timestamp is recent
        latest_time = self.rate_limiter.request_times[-1]
        self.assertLess(abs(time.time() - latest_time), 1.0)  # Within 1 second
    
    @patch('asyncio.sleep')
    async def test_wait_if_needed_no_wait(self, mock_sleep):
        """Test wait_if_needed when no waiting is required."""
        # Should not wait when under limit
        await self.rate_limiter.wait_if_needed()
        mock_sleep.assert_not_called()
    
    @patch('asyncio.sleep')
    async def test_wait_if_needed_with_wait(self, mock_sleep):
        """Test wait_if_needed when waiting is required."""
        # Fill up the request limit
        current_time = time.time()
        for i in range(3):
            self.rate_limiter.request_times.append(current_time)
        
        await self.rate_limiter.wait_if_needed()
        
        # Should have called sleep
        mock_sleep.assert_called_once()
    
    def test_cleanup_old_requests(self):
        """Test cleanup of old requests."""
        # Add old requests
        old_time = time.time() - 2  # 2 seconds ago
        for i in range(5):
            self.rate_limiter.request_times.append(old_time)
        
        # Add recent request
        self.rate_limiter.request_times.append(time.time())
        
        # Should have 6 requests total
        self.assertEqual(len(self.rate_limiter.request_times), 6)
        
        # Clean up old requests
        self.rate_limiter._cleanup_old_requests()
        
        # Should only have 1 recent request
        self.assertEqual(len(self.rate_limiter.request_times), 1)


class TestAnalysisUtils(unittest.TestCase):
    """Test AnalysisUtils class."""
    
    def setUp(self):
        self.create_test_results()
    
    def create_test_results(self):
        """Create test evaluation results."""
        # Create test metrics
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
        
        # Create test results
        azure_result = ModelResult(
            model_name="azure_openai",
            model_version="gpt-4",
            overall_score=4.2,
            confidence=0.85,
            metrics=metrics,
            processing_time=2.5,
            tokens_used=150,
            raw_response="Azure response"
        )
        
        claude_result = ModelResult(
            model_name="claude",
            model_version="claude-3-sonnet-20240229",
            overall_score=4.3,
            confidence=0.88,
            metrics=metrics,
            processing_time=2.1,
            tokens_used=140,
            raw_response="Claude response"
        )
        
        self.test_results = [
            EvaluationResult(
                azure_openai_result=azure_result,
                claude_result=claude_result,
                comparison={"overall_score_difference": 0.1},
                aggregated_insights={"summary": {"overall_quality": "High"}},
                evaluation_metadata={"evaluation_time": 5.2}
            ),
            EvaluationResult(
                azure_openai_result=azure_result,
                claude_result=claude_result,
                comparison={"overall_score_difference": 0.2},
                aggregated_insights={"summary": {"overall_quality": "Medium"}},
                evaluation_metadata={"evaluation_time": 4.8}
            )
        ]
    
    def test_calculate_score_statistics(self):
        """Test calculate_score_statistics method."""
        stats = AnalysisUtils.calculate_score_statistics(self.test_results)
        
        self.assertIsInstance(stats, dict)
        self.assertIn("azure_openai", stats)
        self.assertIn("claude", stats)
        self.assertIn("comparison", stats)
        
        # Check Azure stats
        azure_stats = stats["azure_openai"]
        self.assertIn("mean", azure_stats)
        self.assertIn("std_dev", azure_stats)
        self.assertIn("min", azure_stats)
        self.assertIn("max", azure_stats)
        self.assertIn("count", azure_stats)
        
        # Check Claude stats
        claude_stats = stats["claude"]
        self.assertIn("mean", claude_stats)
        self.assertIn("std_dev", claude_stats)
        self.assertIn("min", claude_stats)
        self.assertIn("max", claude_stats)
        self.assertIn("count", claude_stats)
        
        # Check comparison stats
        comparison_stats = stats["comparison"]
        self.assertIn("agreement_rate", comparison_stats)
        self.assertIn("mean_difference", comparison_stats)
        self.assertIn("correlation", comparison_stats)
    
    def test_analyze_metric_trends(self):
        """Test analyze_metric_trends method."""
        trends = AnalysisUtils.analyze_metric_trends(self.test_results)
        
        self.assertIsInstance(trends, dict)
        
        # Should have trends for each metric
        expected_metrics = [
            "correctness", "completeness", "relevance", "fluency",
            "coherence", "faithfulness", "harmfulness", "helpfulness", "conciseness"
        ]
        
        for metric in expected_metrics:
            self.assertIn(metric, trends)
            metric_trend = trends[metric]
            
            self.assertIn("azure_openai", metric_trend)
            self.assertIn("claude", metric_trend)
            self.assertIn("agreement", metric_trend)
    
    def test_identify_outliers(self):
        """Test identify_outliers method."""
        outliers = AnalysisUtils.identify_outliers(self.test_results)
        
        self.assertIsInstance(outliers, list)
        
        # Each outlier should have required fields
        for outlier in outliers:
            self.assertIn("index", outlier)
            self.assertIn("reason", outlier)
            self.assertIn("score", outlier)
            self.assertIn("threshold", outlier)
    
    def test_calculate_model_agreement(self):
        """Test calculate_model_agreement method."""
        agreement = AnalysisUtils.calculate_model_agreement(self.test_results)
        
        self.assertIsInstance(agreement, dict)
        self.assertIn("overall_agreement", agreement)
        self.assertIn("metric_agreements", agreement)
        self.assertIn("high_agreement_metrics", agreement)
        self.assertIn("low_agreement_metrics", agreement)
        
        # Check agreement score is between 0 and 1
        self.assertGreaterEqual(agreement["overall_agreement"], 0.0)
        self.assertLessEqual(agreement["overall_agreement"], 1.0)
    
    def test_generate_insights_summary(self):
        """Test generate_insights_summary method."""
        summary = AnalysisUtils.generate_insights_summary(self.test_results)
        
        self.assertIsInstance(summary, dict)
        self.assertIn("total_evaluations", summary)
        self.assertIn("average_scores", summary)
        self.assertIn("model_performance", summary)
        self.assertIn("quality_distribution", summary)
        self.assertIn("processing_stats", summary)
        
        # Check data types
        self.assertIsInstance(summary["total_evaluations"], int)
        self.assertIsInstance(summary["average_scores"], dict)
        self.assertIsInstance(summary["model_performance"], dict)
        self.assertIsInstance(summary["quality_distribution"], dict)
        self.assertIsInstance(summary["processing_stats"], dict)
    
    def test_calculate_correlation(self):
        """Test calculate_correlation method."""
        azure_scores = [4.2, 4.3, 4.1]
        claude_scores = [4.3, 4.4, 4.2]
        
        correlation = AnalysisUtils.calculate_correlation(azure_scores, claude_scores)
        
        self.assertIsInstance(correlation, float)
        self.assertGreaterEqual(correlation, -1.0)
        self.assertLessEqual(correlation, 1.0)
    
    def test_calculate_correlation_empty_lists(self):
        """Test calculate_correlation with empty lists."""
        correlation = AnalysisUtils.calculate_correlation([], [])
        self.assertEqual(correlation, 0.0)
    
    def test_calculate_correlation_single_value(self):
        """Test calculate_correlation with single values."""
        correlation = AnalysisUtils.calculate_correlation([4.2], [4.3])
        self.assertEqual(correlation, 0.0)  # No correlation possible with single values


class TestExportUtils(unittest.TestCase):
    """Test ExportUtils class."""
    
    def setUp(self):
        self.create_test_results()
    
    def create_test_results(self):
        """Create test evaluation results."""
        # Create test metrics
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
        
        # Create test results
        azure_result = ModelResult(
            model_name="azure_openai",
            model_version="gpt-4",
            overall_score=4.2,
            confidence=0.85,
            metrics=metrics,
            processing_time=2.5,
            tokens_used=150,
            raw_response="Azure response"
        )
        
        claude_result = ModelResult(
            model_name="claude",
            model_version="claude-3-sonnet-20240229",
            overall_score=4.3,
            confidence=0.88,
            metrics=metrics,
            processing_time=2.1,
            tokens_used=140,
            raw_response="Claude response"
        )
        
        self.test_results = [
            EvaluationResult(
                azure_openai_result=azure_result,
                claude_result=claude_result,
                comparison={"overall_score_difference": 0.1},
                aggregated_insights={"summary": {"overall_quality": "High"}},
                evaluation_metadata={"evaluation_time": 5.2}
            )
        ]
    
    @patch('builtins.open', new_callable=mock_open)
    def test_to_json(self, mock_file):
        """Test to_json method."""
        filename = "test_results.json"
        
        ExportUtils.to_json(self.test_results, filename)
        
        # Check that file was opened for writing
        mock_file.assert_called_once_with(filename, 'w', encoding='utf-8')
        
        # Check that JSON was written
        handle = mock_file()
        handle.write.assert_called()
        
        # Check that valid JSON was written
        written_content = ''.join(call[0][0] for call in handle.write.call_args_list)
        self.assertIsInstance(json.loads(written_content), list)
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('csv.writer')
    def test_to_csv(self, mock_csv_writer, mock_file):
        """Test to_csv method."""
        filename = "test_results.csv"
        
        ExportUtils.to_csv(self.test_results, filename)
        
        # Check that file was opened for writing
        mock_file.assert_called_once_with(filename, 'w', newline='', encoding='utf-8')
        
        # Check that CSV writer was created
        mock_csv_writer.assert_called_once()
    
    @patch('builtins.open', new_callable=mock_open)
    def test_create_report(self, mock_file):
        """Test create_report method."""
        filename = "test_report.txt"
        
        ExportUtils.create_report(self.test_results, filename)
        
        # Check that file was opened for writing
        mock_file.assert_called_once_with(filename, 'w', encoding='utf-8')
        
        # Check that content was written
        handle = mock_file()
        handle.write.assert_called()
        
        # Check that report contains expected sections
        written_content = ''.join(call[0][0] for call in handle.write.call_args_list)
        self.assertIn("LLM Evaluation Report", written_content)
        self.assertIn("Summary", written_content)
        self.assertIn("Detailed Results", written_content)
    
    def test_prepare_visualization_data(self):
        """Test prepare_visualization_data method."""
        viz_data = ExportUtils.prepare_visualization_data(self.test_results)
        
        self.assertIsInstance(viz_data, dict)
        
        # Check expected data components
        expected_components = [
            "score_distribution", "metric_comparison", "model_performance",
            "quality_trends", "processing_stats"
        ]
        
        for component in expected_components:
            self.assertIn(component, viz_data)
    
    def test_format_score_for_display(self):
        """Test format_score_for_display method."""
        # Test with float
        self.assertEqual(ExportUtils.format_score_for_display(4.567), "4.57")
        
        # Test with int
        self.assertEqual(ExportUtils.format_score_for_display(4), "4.00")
        
        # Test with None
        self.assertEqual(ExportUtils.format_score_for_display(None), "N/A")
        
        # Test with string
        self.assertEqual(ExportUtils.format_score_for_display("4.567"), "4.567")
    
    def test_generate_csv_headers(self):
        """Test generate_csv_headers method."""
        headers = ExportUtils.generate_csv_headers()
        
        self.assertIsInstance(headers, list)
        self.assertGreater(len(headers), 0)
        
        # Check that essential headers are present
        expected_headers = [
            "evaluation_id", "azure_overall_score", "claude_overall_score",
            "score_difference", "azure_correctness", "claude_correctness"
        ]
        
        for header in expected_headers:
            self.assertIn(header, headers)
    
    def test_convert_result_to_csv_row(self):
        """Test convert_result_to_csv_row method."""
        row = ExportUtils.convert_result_to_csv_row(self.test_results[0], 0)
        
        self.assertIsInstance(row, list)
        self.assertGreater(len(row), 0)
        
        # Check that row contains expected data types
        self.assertIsInstance(row[0], int)  # evaluation_id
        self.assertIsInstance(row[1], str)  # azure_overall_score (formatted)
        self.assertIsInstance(row[2], str)  # claude_overall_score (formatted)


if __name__ == '__main__':
    unittest.main()