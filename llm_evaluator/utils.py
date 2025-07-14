"""
Utility classes and functions for LLM evaluation.
"""

import asyncio
import time
import json
import csv
import hashlib
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import statistics
from pathlib import Path

from .models import EvaluationResult, ModelResult


class CacheManager:
    """Simple in-memory cache for evaluation results."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.cache: Dict[str, Any] = {}
        self.timestamps: Dict[str, float] = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        if key not in self.cache:
            return None
        
        # Check TTL
        if time.time() - self.timestamps[key] > self.ttl_seconds:
            self._remove(key)
            return None
        
        return self.cache[key]
    
    def set(self, key: str, value: Any):
        """Set item in cache."""
        # Remove oldest items if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.timestamps, key=self.timestamps.get)
            self._remove(oldest_key)
        
        self.cache[key] = value
        self.timestamps[key] = time.time()
    
    def _remove(self, key: str):
        """Remove item from cache."""
        if key in self.cache:
            del self.cache[key]
        if key in self.timestamps:
            del self.timestamps[key]
    
    def clear(self):
        """Clear all cache."""
        self.cache.clear()
        self.timestamps.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        return len(self.cache)


class RateLimiter:
    """Rate limiter for API calls."""
    
    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: List[float] = []
    
    async def wait_if_needed(self):
        """Wait if rate limit would be exceeded."""
        current_time = time.time()
        
        # Remove old requests outside the window
        self.requests = [
            req_time for req_time in self.requests
            if current_time - req_time < self.window_seconds
        ]
        
        # Check if we need to wait
        if len(self.requests) >= self.max_requests:
            oldest_request = min(self.requests)
            wait_time = self.window_seconds - (current_time - oldest_request)
            if wait_time > 0:
                await asyncio.sleep(wait_time)
        
        # Record this request
        self.requests.append(current_time)


class AnalysisUtils:
    """Utilities for analyzing evaluation results."""
    
    @staticmethod
    def calculate_score_statistics(results: List[EvaluationResult]) -> Dict[str, Any]:
        """Calculate statistical summary of evaluation results."""
        if not results:
            return {}
        
        stats = {
            "azure_openai": {},
            "claude": {},
            "comparison": {}
        }
        
        # Extract scores
        azure_scores = [result.azure_openai_result.overall_score for result in results]
        claude_scores = [result.claude_result.overall_score for result in results]
        
        # Calculate statistics for each model
        for model_name, scores in [("azure_openai", azure_scores), ("claude", claude_scores)]:
            if scores:
                stats[model_name] = {
                    "mean": statistics.mean(scores),
                    "median": statistics.median(scores),
                    "std_dev": statistics.stdev(scores) if len(scores) > 1 else 0,
                    "min": min(scores),
                    "max": max(scores),
                    "count": len(scores)
                }
        
        # Comparison statistics
        if azure_scores and claude_scores:
            differences = [a - c for a, c in zip(azure_scores, claude_scores)]
            stats["comparison"] = {
                "mean_difference": statistics.mean(differences),
                "correlation": AnalysisUtils._calculate_correlation(azure_scores, claude_scores),
                "agreement_rate": sum(1 for d in differences if abs(d) < 0.5) / len(differences)
            }
        
        return stats
    
    @staticmethod
    def _calculate_correlation(x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient."""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi ** 2 for xi in x)
        sum_y2 = sum(yi ** 2 for yi in y)
        
        numerator = n * sum_xy - sum_x * sum_y
        denominator = ((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2)) ** 0.5
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    @staticmethod
    def analyze_metric_trends(results: List[EvaluationResult]) -> Dict[str, Any]:
        """Analyze trends in individual metrics."""
        if not results:
            return {}
        
        metrics_names = [
            "correctness", "completeness", "relevance", "fluency",
            "coherence", "faithfulness", "harmfulness", "helpfulness", "conciseness"
        ]
        
        trends = {}
        
        for metric_name in metrics_names:
            azure_scores = []
            claude_scores = []
            
            for result in results:
                azure_metric = getattr(result.azure_openai_result.metrics, metric_name)
                claude_metric = getattr(result.claude_result.metrics, metric_name)
                
                azure_scores.append(azure_metric.score)
                claude_scores.append(claude_metric.score)
            
            trends[metric_name] = {
                "azure_avg": statistics.mean(azure_scores),
                "claude_avg": statistics.mean(claude_scores),
                "difference": statistics.mean(azure_scores) - statistics.mean(claude_scores),
                "azure_std": statistics.stdev(azure_scores) if len(azure_scores) > 1 else 0,
                "claude_std": statistics.stdev(claude_scores) if len(claude_scores) > 1 else 0
            }
        
        return trends
    
    @staticmethod
    def identify_outliers(results: List[EvaluationResult], threshold: float = 2.0) -> List[Dict[str, Any]]:
        """Identify outlier evaluations based on score differences."""
        if not results:
            return []
        
        differences = []
        for i, result in enumerate(results):
            diff = abs(result.azure_openai_result.overall_score - result.claude_result.overall_score)
            differences.append((i, diff))
        
        if len(differences) < 2:
            return []
        
        mean_diff = statistics.mean([d[1] for d in differences])
        std_diff = statistics.stdev([d[1] for d in differences])
        
        outliers = []
        for i, diff in differences:
            if abs(diff - mean_diff) > threshold * std_diff:
                outliers.append({
                    "index": i,
                    "difference": diff,
                    "azure_score": results[i].azure_openai_result.overall_score,
                    "claude_score": results[i].claude_result.overall_score,
                    "evaluation_id": i
                })
        
        return outliers


class ExportUtils:
    """Utilities for exporting evaluation results."""
    
    @staticmethod
    def to_csv(results: List[EvaluationResult], filename: str):
        """Export evaluation results to CSV format."""
        if not results:
            return
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Header
            header = [
                "timestamp", "azure_overall_score", "claude_overall_score",
                "azure_confidence", "claude_confidence", "score_difference",
                "azure_processing_time", "claude_processing_time"
            ]
            
            # Add metric columns
            metrics_names = [
                "correctness", "completeness", "relevance", "fluency",
                "coherence", "faithfulness", "harmfulness", "helpfulness", "conciseness"
            ]
            
            for metric in metrics_names:
                header.extend([f"azure_{metric}", f"claude_{metric}"])
            
            writer.writerow(header)
            
            # Data rows
            for result in results:
                row = [
                    result.evaluation_metadata.get("timestamp", ""),
                    result.azure_openai_result.overall_score,
                    result.claude_result.overall_score,
                    result.azure_openai_result.confidence,
                    result.claude_result.confidence,
                    result.comparison.get("overall_score_difference", 0),
                    result.azure_openai_result.processing_time,
                    result.claude_result.processing_time
                ]
                
                # Add metric scores
                for metric in metrics_names:
                    azure_metric = getattr(result.azure_openai_result.metrics, metric)
                    claude_metric = getattr(result.claude_result.metrics, metric)
                    row.extend([azure_metric.score, claude_metric.score])
                
                writer.writerow(row)
    
    @staticmethod
    def to_json(results: List[EvaluationResult], filename: str, indent: int = 2):
        """Export evaluation results to JSON format."""
        if not results:
            return
        
        data = {
            "evaluation_results": [result.to_dict() for result in results],
            "summary": {
                "total_evaluations": len(results),
                "export_timestamp": datetime.now().isoformat(),
                "statistics": AnalysisUtils.calculate_score_statistics(results)
            }
        }
        
        with open(filename, 'w', encoding='utf-8') as jsonfile:
            json.dump(data, jsonfile, indent=indent, ensure_ascii=False)
    
    @staticmethod
    def create_report(results: List[EvaluationResult], filename: str):
        """Create a comprehensive text report."""
        if not results:
            return
        
        with open(filename, 'w', encoding='utf-8') as report_file:
            report_file.write("# LLM Evaluation Report\n\n")
            report_file.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            report_file.write(f"Total Evaluations: {len(results)}\n\n")
            
            # Summary statistics
            stats = AnalysisUtils.calculate_score_statistics(results)
            report_file.write("## Summary Statistics\n\n")
            
            if "azure_openai" in stats:
                report_file.write("### Azure OpenAI Results\n")
                azure_stats = stats["azure_openai"]
                report_file.write(f"- Mean Score: {azure_stats['mean']:.2f}\n")
                report_file.write(f"- Median Score: {azure_stats['median']:.2f}\n")
                report_file.write(f"- Standard Deviation: {azure_stats['std_dev']:.2f}\n")
                report_file.write(f"- Range: {azure_stats['min']:.2f} - {azure_stats['max']:.2f}\n\n")
            
            if "claude" in stats:
                report_file.write("### Claude Results\n")
                claude_stats = stats["claude"]
                report_file.write(f"- Mean Score: {claude_stats['mean']:.2f}\n")
                report_file.write(f"- Median Score: {claude_stats['median']:.2f}\n")
                report_file.write(f"- Standard Deviation: {claude_stats['std_dev']:.2f}\n")
                report_file.write(f"- Range: {claude_stats['min']:.2f} - {claude_stats['max']:.2f}\n\n")
            
            if "comparison" in stats:
                report_file.write("### Model Comparison\n")
                comp_stats = stats["comparison"]
                report_file.write(f"- Mean Difference: {comp_stats['mean_difference']:.2f}\n")
                report_file.write(f"- Correlation: {comp_stats['correlation']:.2f}\n")
                report_file.write(f"- Agreement Rate: {comp_stats['agreement_rate']:.2%}\n\n")
            
            # Metric trends
            trends = AnalysisUtils.analyze_metric_trends(results)
            report_file.write("## Metric Analysis\n\n")
            
            for metric, trend in trends.items():
                report_file.write(f"### {metric.title()}\n")
                report_file.write(f"- Azure Average: {trend['azure_avg']:.2f}\n")
                report_file.write(f"- Claude Average: {trend['claude_avg']:.2f}\n")
                report_file.write(f"- Difference: {trend['difference']:.2f}\n\n")
            
            # Outliers
            outliers = AnalysisUtils.identify_outliers(results)
            if outliers:
                report_file.write("## Outlier Analysis\n\n")
                for outlier in outliers:
                    report_file.write(f"- Evaluation #{outlier['index']}: ")
                    report_file.write(f"Difference = {outlier['difference']:.2f} ")
                    report_file.write(f"(Azure: {outlier['azure_score']:.2f}, Claude: {outlier['claude_score']:.2f})\n")
    
    @staticmethod
    def prepare_visualization_data(results: List[EvaluationResult]) -> Dict[str, Any]:
        """Prepare data for visualization (charts, graphs)."""
        if not results:
            return {}
        
        # Time series data
        timestamps = []
        azure_scores = []
        claude_scores = []
        
        for result in results:
            timestamps.append(result.evaluation_metadata.get("timestamp", ""))
            azure_scores.append(result.azure_openai_result.overall_score)
            claude_scores.append(result.claude_result.overall_score)
        
        # Metric comparison data
        metrics_names = [
            "correctness", "completeness", "relevance", "fluency",
            "coherence", "faithfulness", "harmfulness", "helpfulness", "conciseness"
        ]
        
        metric_comparison = {}
        for metric in metrics_names:
            azure_values = []
            claude_values = []
            
            for result in results:
                azure_metric = getattr(result.azure_openai_result.metrics, metric)
                claude_metric = getattr(result.claude_result.metrics, metric)
                azure_values.append(azure_metric.score)
                claude_values.append(claude_metric.score)
            
            metric_comparison[metric] = {
                "azure": azure_values,
                "claude": claude_values
            }
        
        return {
            "time_series": {
                "timestamps": timestamps,
                "azure_scores": azure_scores,
                "claude_scores": claude_scores
            },
            "metric_comparison": metric_comparison,
            "statistics": AnalysisUtils.calculate_score_statistics(results),
            "outliers": AnalysisUtils.identify_outliers(results)
        }