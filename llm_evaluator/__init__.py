"""
LLM Evaluator Package

A comprehensive package for evaluating LLM answers using both Azure OpenAI and Claude Sonnet models.
Provides detailed metrics, structured responses, and production-ready features.
"""

from .core import LLMEvaluator
from .models import EvaluationResult, EvaluationMetrics, ModelConfig, EvaluationInput, EvaluationConfig
from .metrics import MetricsFramework
from .utils import AnalysisUtils, ExportUtils

__version__ = "1.0.0"
__author__ = "LLM Evaluator Team"

__all__ = [
    "LLMEvaluator",
    "EvaluationResult", 
    "EvaluationMetrics",
    "ModelConfig",
    "EvaluationInput",
    "EvaluationConfig",
    "MetricsFramework",
    "AnalysisUtils",
    "ExportUtils",
]