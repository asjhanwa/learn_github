"""
Data models for LLM evaluation results and configurations.
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class MetricScore:
    """Individual metric score with justification."""
    name: str
    score: float  # 1-5 scale
    justification: str
    confidence: float  # 0-1 scale
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "score": self.score,
            "justification": self.justification,
            "confidence": self.confidence
        }


@dataclass
class EvaluationMetrics:
    """Collection of all evaluation metrics."""
    correctness: MetricScore
    completeness: MetricScore
    relevance: MetricScore
    fluency: MetricScore
    coherence: MetricScore
    faithfulness: MetricScore
    harmfulness: MetricScore
    helpfulness: MetricScore
    conciseness: MetricScore
    
    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        return {
            "correctness": self.correctness.to_dict(),
            "completeness": self.completeness.to_dict(),
            "relevance": self.relevance.to_dict(),
            "fluency": self.fluency.to_dict(),
            "coherence": self.coherence.to_dict(),
            "faithfulness": self.faithfulness.to_dict(),
            "harmfulness": self.harmfulness.to_dict(),
            "helpfulness": self.helpfulness.to_dict(),
            "conciseness": self.conciseness.to_dict()
        }
    
    def get_overall_score(self) -> float:
        """Calculate overall weighted score."""
        # Define weights for different metrics
        weights = {
            "correctness": 0.20,
            "completeness": 0.15,
            "relevance": 0.15,
            "fluency": 0.10,
            "coherence": 0.10,
            "faithfulness": 0.15,
            "harmfulness": -0.10,  # Negative weight for harmfulness
            "helpfulness": 0.10,
            "conciseness": 0.05
        }
        
        total_score = 0
        for metric_name, weight in weights.items():
            metric_score = getattr(self, metric_name).score
            if metric_name == "harmfulness":
                # For harmfulness, lower score is better, so invert it
                metric_score = 6 - metric_score
            total_score += weight * metric_score
        
        return round(total_score, 2)


@dataclass
class ModelResult:
    """Result from a single model evaluation."""
    model_name: str
    model_version: str
    metrics: EvaluationMetrics
    overall_score: float
    confidence: float
    processing_time: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "metrics": self.metrics.to_dict(),
            "overall_score": self.overall_score,
            "confidence": self.confidence,
            "processing_time": self.processing_time,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class EvaluationResult:
    """Complete evaluation result with comparison between models."""
    azure_openai_result: ModelResult
    claude_result: ModelResult
    comparison: Dict[str, Any]
    aggregated_insights: Dict[str, Any]
    evaluation_metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "azure_openai_result": self.azure_openai_result.to_dict(),
            "claude_result": self.claude_result.to_dict(),
            "comparison": self.comparison,
            "aggregated_insights": self.aggregated_insights,
            "evaluation_metadata": self.evaluation_metadata
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


@dataclass
class ModelConfig:
    """Configuration for model parameters."""
    model_name: str
    temperature: float = 0.0
    max_tokens: int = 2000
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "timeout": self.timeout,
            "retry_attempts": self.retry_attempts,
            "retry_delay": self.retry_delay
        }


@dataclass
class EvaluationInput:
    """Input data for LLM evaluation."""
    chat_history: List[Dict[str, str]]
    retrieved_chunks: List[str]
    llm_answer: str
    user_feedback_label: str  # "Positive", "Negative", or "Missing"
    user_feedback_comment: Optional[str] = None
    additional_context: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "chat_history": self.chat_history,
            "retrieved_chunks": self.retrieved_chunks,
            "llm_answer": self.llm_answer,
            "user_feedback_label": self.user_feedback_label,
            "user_feedback_comment": self.user_feedback_comment,
            "additional_context": self.additional_context
        }


@dataclass
class EvaluationConfig:
    """Configuration for evaluation process."""
    azure_openai_config: ModelConfig
    claude_config: ModelConfig
    enabled_metrics: List[str] = field(default_factory=lambda: [
        "correctness", "completeness", "relevance", "fluency", 
        "coherence", "faithfulness", "harmfulness", "helpfulness", "conciseness"
    ])
    custom_scoring_rubric: Optional[Dict[str, Any]] = None
    parallel_evaluation: bool = True
    cache_enabled: bool = True
    rate_limit_enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "azure_openai_config": self.azure_openai_config.to_dict(),
            "claude_config": self.claude_config.to_dict(),
            "enabled_metrics": self.enabled_metrics,
            "custom_scoring_rubric": self.custom_scoring_rubric,
            "parallel_evaluation": self.parallel_evaluation,
            "cache_enabled": self.cache_enabled,
            "rate_limit_enabled": self.rate_limit_enabled
        }