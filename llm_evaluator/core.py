"""
Core LLM Evaluator class that orchestrates the evaluation process.
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import hashlib
import json

from .models import (
    EvaluationInput, EvaluationResult, EvaluationConfig, 
    ModelResult, EvaluationMetrics, MetricScore
)
from .integrations import AzureOpenAIIntegration, ClaudeIntegration, ModelIntegrationFactory
from .metrics import MetricsFramework
from .utils import CacheManager, RateLimiter


class LLMEvaluator:
    """
    Main class for evaluating LLM answers using Azure OpenAI and Claude Sonnet.
    
    This class provides comprehensive evaluation capabilities including:
    - Multi-model evaluation (Azure OpenAI + Claude)
    - Comprehensive metrics framework
    - Async processing and batch operations
    - Caching and rate limiting
    - Error handling and retry mechanisms
    """
    
    def __init__(self, 
                 azure_openai_endpoint: str,
                 azure_openai_api_key: str,
                 azure_openai_deployment: str,
                 claude_api_key: str,
                 config: Optional[EvaluationConfig] = None):
        """
        Initialize the LLM Evaluator.
        
        Args:
            azure_openai_endpoint: Azure OpenAI endpoint URL
            azure_openai_api_key: Azure OpenAI API key
            azure_openai_deployment: Azure OpenAI deployment name
            claude_api_key: Claude API key
            config: Optional evaluation configuration
        """
        self.azure_openai_endpoint = azure_openai_endpoint
        self.azure_openai_api_key = azure_openai_api_key
        self.azure_openai_deployment = azure_openai_deployment
        self.claude_api_key = claude_api_key
        
        # Set default config if not provided
        if config is None:
            from .models import ModelConfig
            config = EvaluationConfig(
                azure_openai_config=ModelConfig(model_name="gpt-4"),
                claude_config=ModelConfig(model_name="claude-3-sonnet-20240229")
            )
        
        self.config = config
        self.metrics_framework = MetricsFramework()
        
        # Initialize integrations
        self.azure_integration = ModelIntegrationFactory.create_azure_openai_integration(
            self.config.azure_openai_config,
            azure_openai_endpoint,
            azure_openai_api_key,
            azure_openai_deployment
        )
        
        self.claude_integration = ModelIntegrationFactory.create_claude_integration(
            self.config.claude_config,
            claude_api_key
        )
        
        # Initialize utilities
        self.cache_manager = CacheManager() if config.cache_enabled else None
        self.rate_limiter = RateLimiter() if config.rate_limit_enabled else None
    
    async def evaluate(self, eval_input: EvaluationInput) -> EvaluationResult:
        """
        Evaluate an LLM answer using both Azure OpenAI and Claude models.
        
        Args:
            eval_input: The evaluation input containing chat history, context, etc.
            
        Returns:
            EvaluationResult: Comprehensive evaluation results from both models
        """
        start_time = time.time()
        
        # Generate cache key
        cache_key = self._generate_cache_key(eval_input) if self.cache_manager else None
        
        # Check cache
        if cache_key and self.cache_manager:
            cached_result = self.cache_manager.get(cache_key)
            if cached_result:
                return cached_result
        
        # Apply rate limiting
        if self.rate_limiter:
            await self.rate_limiter.wait_if_needed()
        
        # Run evaluations
        if self.config.parallel_evaluation:
            # Parallel evaluation
            azure_task = self._evaluate_with_model(eval_input, self.azure_integration, "azure_openai")
            claude_task = self._evaluate_with_model(eval_input, self.claude_integration, "claude")
            
            azure_result, claude_result = await asyncio.gather(azure_task, claude_task)
        else:
            # Sequential evaluation
            azure_result = await self._evaluate_with_model(eval_input, self.azure_integration, "azure_openai")
            claude_result = await self._evaluate_with_model(eval_input, self.claude_integration, "claude")
        
        # Create comparison and insights
        comparison = self._create_comparison(azure_result, claude_result)
        insights = self._generate_insights(azure_result, claude_result, eval_input)
        
        # Create final result
        evaluation_result = EvaluationResult(
            azure_openai_result=azure_result,
            claude_result=claude_result,
            comparison=comparison,
            aggregated_insights=insights,
            evaluation_metadata={
                "evaluation_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat(),
                "config": self.config.to_dict(),
                "enabled_metrics": self.config.enabled_metrics,
                "parallel_evaluation": self.config.parallel_evaluation
            }
        )
        
        # Cache result
        if cache_key and self.cache_manager:
            self.cache_manager.set(cache_key, evaluation_result)
        
        return evaluation_result
    
    async def batch_evaluate(self, eval_inputs: List[EvaluationInput]) -> List[EvaluationResult]:
        """
        Evaluate multiple LLM answers in batch.
        
        Args:
            eval_inputs: List of evaluation inputs
            
        Returns:
            List[EvaluationResult]: List of evaluation results
        """
        tasks = [self.evaluate(eval_input) for eval_input in eval_inputs]
        return await asyncio.gather(*tasks)
    
    async def _evaluate_with_model(self, 
                                 eval_input: EvaluationInput, 
                                 integration: Any, 
                                 model_name: str) -> ModelResult:
        """Evaluate using a specific model integration."""
        start_time = time.time()
        
        # Evaluate each enabled metric
        metric_tasks = []
        for metric_name in self.config.enabled_metrics:
            task = integration.evaluate_metric(metric_name, eval_input)
            metric_tasks.append(task)
        
        # Wait for all metric evaluations
        metric_results = await asyncio.gather(*metric_tasks)
        
        # Create metrics object
        metrics_dict = {metric.name: metric for metric in metric_results}
        
        # Ensure all required metrics are present
        required_metrics = ['correctness', 'completeness', 'relevance', 'fluency', 
                           'coherence', 'faithfulness', 'harmfulness', 'helpfulness', 'conciseness']
        
        for metric_name in required_metrics:
            if metric_name not in metrics_dict:
                # Create default metric if missing
                metrics_dict[metric_name] = MetricScore(
                    name=metric_name,
                    score=2.5,
                    justification=f"Metric {metric_name} was not evaluated",
                    confidence=0.1
                )
        
        metrics = EvaluationMetrics(
            correctness=metrics_dict['correctness'],
            completeness=metrics_dict['completeness'],
            relevance=metrics_dict['relevance'],
            fluency=metrics_dict['fluency'],
            coherence=metrics_dict['coherence'],
            faithfulness=metrics_dict['faithfulness'],
            harmfulness=metrics_dict['harmfulness'],
            helpfulness=metrics_dict['helpfulness'],
            conciseness=metrics_dict['conciseness']
        )
        
        # Calculate overall score and confidence
        overall_score = metrics.get_overall_score()
        overall_confidence = sum(metric.confidence for metric in metric_results) / len(metric_results)
        
        return ModelResult(
            model_name=model_name,
            model_version=integration.config.model_name,
            metrics=metrics,
            overall_score=overall_score,
            confidence=overall_confidence,
            processing_time=time.time() - start_time
        )
    
    def _create_comparison(self, azure_result: ModelResult, claude_result: ModelResult) -> Dict[str, Any]:
        """Create comparison between Azure OpenAI and Claude results."""
        comparison = {
            "overall_score_difference": azure_result.overall_score - claude_result.overall_score,
            "confidence_difference": azure_result.confidence - claude_result.confidence,
            "processing_time_difference": azure_result.processing_time - claude_result.processing_time,
            "metric_comparisons": {},
            "agreement_analysis": {}
        }
        
        # Compare individual metrics
        azure_metrics = azure_result.metrics.to_dict()
        claude_metrics = claude_result.metrics.to_dict()
        
        total_agreement = 0
        metric_count = 0
        
        for metric_name in azure_metrics:
            azure_score = azure_metrics[metric_name]['score']
            claude_score = claude_metrics[metric_name]['score']
            
            difference = azure_score - claude_score
            agreement = 1 - abs(difference) / 4  # Normalize to 0-1 scale
            
            comparison["metric_comparisons"][metric_name] = {
                "azure_score": azure_score,
                "claude_score": claude_score,
                "difference": difference,
                "agreement": agreement,
                "higher_scoring_model": "azure" if azure_score > claude_score else "claude"
            }
            
            total_agreement += agreement
            metric_count += 1
        
        # Overall agreement
        comparison["agreement_analysis"] = {
            "overall_agreement": total_agreement / metric_count if metric_count > 0 else 0,
            "high_agreement_metrics": [
                metric for metric, data in comparison["metric_comparisons"].items()
                if data["agreement"] > 0.8
            ],
            "low_agreement_metrics": [
                metric for metric, data in comparison["metric_comparisons"].items()
                if data["agreement"] < 0.5
            ]
        }
        
        return comparison
    
    def _generate_insights(self, 
                          azure_result: ModelResult, 
                          claude_result: ModelResult, 
                          eval_input: EvaluationInput) -> Dict[str, Any]:
        """Generate aggregated insights from both model evaluations."""
        insights = {
            "summary": {},
            "strengths": [],
            "weaknesses": [],
            "recommendations": [],
            "user_feedback_alignment": {}
        }
        
        # Calculate average scores
        azure_metrics = azure_result.metrics.to_dict()
        claude_metrics = claude_result.metrics.to_dict()
        
        avg_scores = {}
        for metric_name in azure_metrics:
            avg_scores[metric_name] = (
                azure_metrics[metric_name]['score'] + claude_metrics[metric_name]['score']
            ) / 2
        
        # Overall summary
        insights["summary"] = {
            "overall_quality": "High" if sum(avg_scores.values()) / len(avg_scores) > 3.5 else 
                             "Medium" if sum(avg_scores.values()) / len(avg_scores) > 2.5 else "Low",
            "strongest_aspect": max(avg_scores, key=avg_scores.get),
            "weakest_aspect": min(avg_scores, key=avg_scores.get),
            "average_score": sum(avg_scores.values()) / len(avg_scores)
        }
        
        # Identify strengths and weaknesses
        for metric_name, score in avg_scores.items():
            if score >= 4.0:
                insights["strengths"].append(f"Strong {metric_name} (score: {score:.1f})")
            elif score <= 2.5:
                insights["weaknesses"].append(f"Weak {metric_name} (score: {score:.1f})")
        
        # Generate recommendations
        if avg_scores.get("correctness", 0) < 3.0:
            insights["recommendations"].append("Improve factual accuracy by better using provided context")
        
        if avg_scores.get("completeness", 0) < 3.0:
            insights["recommendations"].append("Provide more comprehensive coverage of the question")
        
        if avg_scores.get("relevance", 0) < 3.0:
            insights["recommendations"].append("Stay more focused on the specific question asked")
        
        if avg_scores.get("coherence", 0) < 3.0:
            insights["recommendations"].append("Improve logical flow and structure of the response")
        
        # User feedback alignment
        insights["user_feedback_alignment"] = {
            "label": eval_input.user_feedback_label,
            "matches_evaluation": self._check_feedback_alignment(
                eval_input.user_feedback_label, 
                insights["summary"]["overall_quality"]
            ),
            "comment_analysis": eval_input.user_feedback_comment
        }
        
        return insights
    
    def _check_feedback_alignment(self, feedback_label: str, evaluation_quality: str) -> bool:
        """Check if user feedback aligns with evaluation results."""
        if feedback_label == "Positive" and evaluation_quality in ["High", "Medium"]:
            return True
        elif feedback_label == "Negative" and evaluation_quality == "Low":
            return True
        elif feedback_label == "Missing":
            return evaluation_quality == "Low"
        return False
    
    def _generate_cache_key(self, eval_input: EvaluationInput) -> str:
        """Generate a cache key for the evaluation input."""
        # Create a deterministic hash of the input
        input_str = json.dumps(eval_input.to_dict(), sort_keys=True)
        config_str = json.dumps(self.config.to_dict(), sort_keys=True)
        combined = input_str + config_str
        
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get_metrics_info(self) -> Dict[str, Any]:
        """Get information about available metrics."""
        return {
            metric_name: self.metrics_framework.get_metric_definition(metric_name)
            for metric_name in self.metrics_framework.get_all_metrics()
        }
    
    def update_config(self, new_config: EvaluationConfig):
        """Update the evaluation configuration."""
        self.config = new_config
        
        # Reinitialize integrations with new config
        self.azure_integration = ModelIntegrationFactory.create_azure_openai_integration(
            self.config.azure_openai_config,
            self.azure_openai_endpoint,
            self.azure_openai_api_key,
            self.azure_openai_deployment
        )
        
        self.claude_integration = ModelIntegrationFactory.create_claude_integration(
            self.config.claude_config,
            self.claude_api_key
        )