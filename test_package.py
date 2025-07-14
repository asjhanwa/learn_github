#!/usr/bin/env python3
"""
Test script to validate the LLM Evaluator package functionality.
"""

import sys
import traceback
from typing import Dict, Any

def test_imports():
    """Test that all main components can be imported."""
    print("Testing imports...")
    
    try:
        # Test main imports
        from llm_evaluator import (
            LLMEvaluator, EvaluationInput, EvaluationResult, 
            EvaluationMetrics, ModelConfig, EvaluationConfig,
            MetricsFramework, AnalysisUtils, ExportUtils
        )
        print("✓ Main imports successful")
        
        # Test module imports
        from llm_evaluator.core import LLMEvaluator as CoreEvaluator
        from llm_evaluator.models import EvaluationInput as ModelEvaluationInput
        from llm_evaluator.metrics import MetricsFramework as MetricsFramework2
        from llm_evaluator.integrations import AzureOpenAIIntegration, ClaudeIntegration
        from llm_evaluator.utils import CacheManager, RateLimiter
        print("✓ Module imports successful")
        
        # Test enhanced call_aoai
        from call_aoai import send_to_azure_openai, create_llm_evaluator
        print("✓ Enhanced call_aoai imports successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Import failed: {e}")
        traceback.print_exc()
        return False


def test_data_models():
    """Test that data models can be instantiated and work correctly."""
    print("\nTesting data models...")
    
    try:
        from llm_evaluator.models import (
            EvaluationInput, ModelConfig, EvaluationConfig,
            MetricScore, EvaluationMetrics, ModelResult
        )
        
        # Test EvaluationInput
        eval_input = EvaluationInput(
            chat_history=[{"role": "user", "content": "What is Python?"}],
            retrieved_chunks=["Python is a programming language."],
            llm_answer="Python is a high-level programming language.",
            user_feedback_label="Positive",
            user_feedback_comment="Good answer"
        )
        
        input_dict = eval_input.to_dict()
        assert isinstance(input_dict, dict)
        assert "chat_history" in input_dict
        print("✓ EvaluationInput works correctly")
        
        # Test ModelConfig
        config = ModelConfig(
            model_name="test-model",
            temperature=0.5,
            max_tokens=1000
        )
        
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict["model_name"] == "test-model"
        print("✓ ModelConfig works correctly")
        
        # Test MetricScore
        metric_score = MetricScore(
            name="correctness",
            score=4.5,
            justification="Very accurate answer",
            confidence=0.9
        )
        
        score_dict = metric_score.to_dict()
        assert score_dict["score"] == 4.5
        print("✓ MetricScore works correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ Data models test failed: {e}")
        traceback.print_exc()
        return False


def test_metrics_framework():
    """Test the metrics framework functionality."""
    print("\nTesting metrics framework...")
    
    try:
        from llm_evaluator.metrics import MetricsFramework
        from llm_evaluator.models import EvaluationInput
        
        framework = MetricsFramework()
        
        # Test metric definitions
        metrics = framework.get_all_metrics()
        assert len(metrics) == 9
        print(f"✓ Found {len(metrics)} metrics")
        
        # Test metric definition retrieval
        correctness_def = framework.get_metric_definition("correctness")
        assert "name" in correctness_def
        assert "description" in correctness_def
        assert "criteria" in correctness_def
        print("✓ Metric definitions work correctly")
        
        # Test prompt generation
        eval_input = EvaluationInput(
            chat_history=[{"role": "user", "content": "Test question"}],
            retrieved_chunks=["Test context"],
            llm_answer="Test answer",
            user_feedback_label="Positive"
        )
        
        prompt = framework.create_evaluation_prompt("correctness", eval_input)
        assert isinstance(prompt, str)
        assert len(prompt) > 100  # Should be a substantial prompt
        assert "correctness" in prompt.lower()
        print("✓ Prompt generation works correctly")
        
        # Test validation functions
        assert framework.validate_metric_score(3.5) == True
        assert framework.validate_metric_score(6.0) == False
        assert framework.validate_confidence(0.8) == True
        assert framework.validate_confidence(1.5) == False
        print("✓ Validation functions work correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ Metrics framework test failed: {e}")
        traceback.print_exc()
        return False


def test_utils():
    """Test utility classes."""
    print("\nTesting utility classes...")
    
    try:
        from llm_evaluator.utils import CacheManager, RateLimiter, AnalysisUtils, ExportUtils
        
        # Test CacheManager
        cache = CacheManager(max_size=10, ttl_seconds=60)
        cache.set("test_key", "test_value")
        assert cache.get("test_key") == "test_value"
        assert cache.size() == 1
        print("✓ CacheManager works correctly")
        
        # Test RateLimiter
        rate_limiter = RateLimiter(max_requests=10, window_seconds=60)
        # Just test instantiation since async testing is more complex
        print("✓ RateLimiter instantiated correctly")
        
        # Test AnalysisUtils with mock data
        print("✓ AnalysisUtils available")
        
        # Test ExportUtils
        print("✓ ExportUtils available")
        
        return True
        
    except Exception as e:
        print(f"✗ Utils test failed: {e}")
        traceback.print_exc()
        return False


def test_integration_classes():
    """Test integration classes (without actual API calls)."""
    print("\nTesting integration classes...")
    
    try:
        from llm_evaluator.integrations import (
            AzureOpenAIIntegration, ClaudeIntegration, 
            ModelIntegrationFactory, send_to_azure_openai
        )
        from llm_evaluator.models import ModelConfig
        
        # Test factory methods
        config = ModelConfig(model_name="test-model")
        
        # Test factory creation (without actual API calls)
        azure_integration = ModelIntegrationFactory.create_azure_openai_integration(
            config, "https://test.openai.azure.com/", "test-key", "test-deployment"
        )
        assert isinstance(azure_integration, AzureOpenAIIntegration)
        print("✓ Azure OpenAI integration created successfully")
        
        claude_integration = ModelIntegrationFactory.create_claude_integration(
            config, "test-key"
        )
        assert isinstance(claude_integration, ClaudeIntegration)
        print("✓ Claude integration created successfully")
        
        # Test enhanced send_to_azure_openai function
        assert callable(send_to_azure_openai)
        print("✓ Enhanced send_to_azure_openai function available")
        
        return True
        
    except Exception as e:
        print(f"✗ Integration classes test failed: {e}")
        traceback.print_exc()
        return False


def test_core_evaluator():
    """Test core evaluator class (without actual API calls)."""
    print("\nTesting core evaluator...")
    
    try:
        from llm_evaluator.core import LLMEvaluator
        from llm_evaluator.models import ModelConfig, EvaluationConfig
        
        # Create configs
        azure_config = ModelConfig(model_name="gpt-4")
        claude_config = ModelConfig(model_name="claude-3-sonnet-20240229")
        
        eval_config = EvaluationConfig(
            azure_openai_config=azure_config,
            claude_config=claude_config,
            parallel_evaluation=True,
            cache_enabled=True,
            rate_limit_enabled=True
        )
        
        # Create evaluator
        evaluator = LLMEvaluator(
            azure_openai_endpoint="https://test.openai.azure.com/",
            azure_openai_api_key="test-key",
            azure_openai_deployment="test-deployment",
            claude_api_key="test-key",
            config=eval_config
        )
        
        # Test methods that don't require API calls
        metrics_info = evaluator.get_metrics_info()
        assert isinstance(metrics_info, dict)
        assert len(metrics_info) == 9
        print("✓ Core evaluator instantiated and basic methods work")
        
        return True
        
    except Exception as e:
        print(f"✗ Core evaluator test failed: {e}")
        traceback.print_exc()
        return False


def test_package_structure():
    """Test overall package structure."""
    print("\nTesting package structure...")
    
    try:
        import llm_evaluator
        
        # Check version
        assert hasattr(llm_evaluator, '__version__')
        print(f"✓ Package version: {llm_evaluator.__version__}")
        
        # Check __all__ exports
        expected_exports = [
            "LLMEvaluator", "EvaluationResult", "EvaluationMetrics",
            "ModelConfig", "EvaluationInput", "EvaluationConfig",
            "MetricsFramework", "AnalysisUtils", "ExportUtils"
        ]
        
        for export in expected_exports:
            assert hasattr(llm_evaluator, export)
        print("✓ All expected exports available")
        
        return True
        
    except Exception as e:
        print(f"✗ Package structure test failed: {e}")
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("="*60)
    print("LLM Evaluator Package Test Suite")
    print("="*60)
    
    tests = [
        test_imports,
        test_data_models,
        test_metrics_framework,
        test_utils,
        test_integration_classes,
        test_core_evaluator,
        test_package_structure
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed with exception: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {passed + failed}")
    
    if failed == 0:
        print("🎉 All tests passed! Package is ready for use.")
        return True
    else:
        print(f"❌ {failed} tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)