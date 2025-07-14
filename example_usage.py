#!/usr/bin/env python3
"""
Example usage of the LLM Evaluator package.

This script demonstrates how to use the comprehensive LLM evaluation package
to evaluate LLM answers using both Azure OpenAI and Claude Sonnet models.
"""

import asyncio
from typing import Dict, List, Any
import json

# Import the LLM evaluator components
from llm_evaluator import (
    LLMEvaluator, 
    EvaluationInput, 
    ModelConfig, 
    EvaluationConfig,
    AnalysisUtils,
    ExportUtils
)

# You can also use the convenience function from call_aoai
from call_aoai import create_llm_evaluator


async def basic_evaluation_example():
    """Basic example of evaluating a single LLM answer."""
    print("=" * 60)
    print("Basic LLM Evaluation Example")
    print("=" * 60)
    
    # Configuration (replace with your actual credentials)
    azure_endpoint = "https://your-resource.openai.azure.com/"
    azure_api_key = "your-azure-api-key"
    azure_deployment = "gpt-4"
    claude_api_key = "your-claude-api-key"
    
    # Create evaluator
    evaluator = create_llm_evaluator(
        azure_endpoint=azure_endpoint,
        azure_api_key=azure_api_key,
        azure_deployment=azure_deployment,
        claude_api_key=claude_api_key,
        config={
            "parallel_evaluation": True,
            "cache_enabled": True,
            "rate_limit_enabled": True
        }
    )
    
    # Create evaluation input
    eval_input = EvaluationInput(
        chat_history=[
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "I can help you with that question."}
        ],
        retrieved_chunks=[
            "France is a country in Western Europe. Its capital and largest city is Paris.",
            "Paris is located in the north-central part of France, on the Seine River."
        ],
        llm_answer="The capital of France is Paris. It is located in the north-central part of the country and is also the largest city in France.",
        user_feedback_label="Positive",
        user_feedback_comment="Very accurate and helpful answer!"
    )
    
    try:
        # Evaluate the answer
        print("Evaluating LLM answer...")
        result = await evaluator.evaluate(eval_input)
        
        # Display results
        print("\n" + "=" * 40)
        print("EVALUATION RESULTS")
        print("=" * 40)
        
        print(f"\nAzure OpenAI Overall Score: {result.azure_openai_result.overall_score:.2f}")
        print(f"Claude Overall Score: {result.claude_result.overall_score:.2f}")
        print(f"Score Difference: {result.comparison['overall_score_difference']:.2f}")
        
        print("\n" + "-" * 30)
        print("DETAILED METRICS")
        print("-" * 30)
        
        # Display metric scores
        metrics = ["correctness", "completeness", "relevance", "fluency", 
                  "coherence", "faithfulness", "harmfulness", "helpfulness", "conciseness"]
        
        for metric in metrics:
            azure_metric = getattr(result.azure_openai_result.metrics, metric)
            claude_metric = getattr(result.claude_result.metrics, metric)
            
            print(f"\n{metric.title()}:")
            print(f"  Azure: {azure_metric.score:.1f} (confidence: {azure_metric.confidence:.2f})")
            print(f"  Claude: {claude_metric.score:.1f} (confidence: {claude_metric.confidence:.2f})")
            print(f"  Difference: {azure_metric.score - claude_metric.score:.1f}")
        
        print("\n" + "-" * 30)
        print("AGGREGATED INSIGHTS")
        print("-" * 30)
        
        insights = result.aggregated_insights
        print(f"Overall Quality: {insights['summary']['overall_quality']}")
        print(f"Strongest Aspect: {insights['summary']['strongest_aspect']}")
        print(f"Weakest Aspect: {insights['summary']['weakest_aspect']}")
        
        if insights['strengths']:
            print(f"\nStrengths:")
            for strength in insights['strengths']:
                print(f"  - {strength}")
        
        if insights['weaknesses']:
            print(f"\nWeaknesses:")
            for weakness in insights['weaknesses']:
                print(f"  - {weakness}")
        
        if insights['recommendations']:
            print(f"\nRecommendations:")
            for rec in insights['recommendations']:
                print(f"  - {rec}")
        
        print("\n" + "-" * 30)
        print("PROCESSING DETAILS")
        print("-" * 30)
        
        print(f"Azure Processing Time: {result.azure_openai_result.processing_time:.2f}s")
        print(f"Claude Processing Time: {result.claude_result.processing_time:.2f}s")
        print(f"Total Evaluation Time: {result.evaluation_metadata['evaluation_time']:.2f}s")
        
        return result
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        return None


async def batch_evaluation_example():
    """Example of batch evaluation with multiple inputs."""
    print("\n" + "=" * 60)
    print("Batch LLM Evaluation Example")
    print("=" * 60)
    
    # Configuration
    azure_endpoint = "https://your-resource.openai.azure.com/"
    azure_api_key = "your-azure-api-key"
    azure_deployment = "gpt-4"
    claude_api_key = "your-claude-api-key"
    
    # Create evaluator
    evaluator = create_llm_evaluator(
        azure_endpoint=azure_endpoint,
        azure_api_key=azure_api_key,
        azure_deployment=azure_deployment,
        claude_api_key=claude_api_key
    )
    
    # Create multiple evaluation inputs
    eval_inputs = [
        EvaluationInput(
            chat_history=[{"role": "user", "content": "What is Python?"}],
            retrieved_chunks=["Python is a programming language known for its simplicity."],
            llm_answer="Python is a high-level programming language known for its readability and simplicity.",
            user_feedback_label="Positive",
            user_feedback_comment="Good explanation"
        ),
        EvaluationInput(
            chat_history=[{"role": "user", "content": "How do you cook pasta?"}],
            retrieved_chunks=["Pasta should be cooked in boiling salted water."],
            llm_answer="To cook pasta, bring a large pot of salted water to boil, add pasta, and cook until al dente.",
            user_feedback_label="Positive",
            user_feedback_comment="Complete instructions"
        ),
        EvaluationInput(
            chat_history=[{"role": "user", "content": "What is machine learning?"}],
            retrieved_chunks=["Machine learning is a subset of AI that enables computers to learn from data."],
            llm_answer="Machine learning is when computers learn stuff automatically.",
            user_feedback_label="Negative",
            user_feedback_comment="Too vague and informal"
        )
    ]
    
    try:
        # Batch evaluate
        print("Evaluating multiple LLM answers...")
        results = await evaluator.batch_evaluate(eval_inputs)
        
        print(f"\nProcessed {len(results)} evaluations")
        
        # Analyze results
        stats = AnalysisUtils.calculate_score_statistics(results)
        
        print("\n" + "-" * 30)
        print("BATCH ANALYSIS")
        print("-" * 30)
        
        if "azure_openai" in stats:
            print(f"Azure OpenAI - Mean: {stats['azure_openai']['mean']:.2f}, Std: {stats['azure_openai']['std_dev']:.2f}")
        
        if "claude" in stats:
            print(f"Claude - Mean: {stats['claude']['mean']:.2f}, Std: {stats['claude']['std_dev']:.2f}")
        
        if "comparison" in stats:
            print(f"Model Agreement Rate: {stats['comparison']['agreement_rate']:.2%}")
            print(f"Correlation: {stats['comparison']['correlation']:.2f}")
        
        # Export results
        print("\n" + "-" * 30)
        print("EXPORTING RESULTS")
        print("-" * 30)
        
        # Export to different formats
        ExportUtils.to_json(results, "evaluation_results.json")
        ExportUtils.to_csv(results, "evaluation_results.csv")
        ExportUtils.create_report(results, "evaluation_report.txt")
        
        print("Results exported to:")
        print("  - evaluation_results.json")
        print("  - evaluation_results.csv")
        print("  - evaluation_report.txt")
        
        # Prepare visualization data
        viz_data = ExportUtils.prepare_visualization_data(results)
        print(f"\nVisualization data prepared with {len(viz_data)} components")
        
        return results
        
    except Exception as e:
        print(f"Error during batch evaluation: {str(e)}")
        return None


async def advanced_configuration_example():
    """Example with advanced configuration options."""
    print("\n" + "=" * 60)
    print("Advanced Configuration Example")
    print("=" * 60)
    
    # Custom model configurations
    azure_config = ModelConfig(
        model_name="gpt-4",
        temperature=0.0,  # Deterministic responses
        max_tokens=3000,
        top_p=0.95,
        frequency_penalty=0.1,
        presence_penalty=0.1,
        timeout=45,
        retry_attempts=5,
        retry_delay=2.0
    )
    
    claude_config = ModelConfig(
        model_name="claude-3-sonnet-20240229",
        temperature=0.0,
        max_tokens=3000,
        timeout=45,
        retry_attempts=5,
        retry_delay=2.0
    )
    
    # Custom evaluation configuration
    eval_config = EvaluationConfig(
        azure_openai_config=azure_config,
        claude_config=claude_config,
        enabled_metrics=["correctness", "completeness", "relevance", "helpfulness"],  # Subset of metrics
        parallel_evaluation=False,  # Sequential evaluation
        cache_enabled=True,
        rate_limit_enabled=True
    )
    
    # Create evaluator with custom config
    evaluator = LLMEvaluator(
        azure_openai_endpoint="https://your-resource.openai.azure.com/",
        azure_openai_api_key="your-azure-api-key",
        azure_openai_deployment="gpt-4",
        claude_api_key="your-claude-api-key",
        config=eval_config
    )
    
    # Get metrics information
    metrics_info = evaluator.get_metrics_info()
    print(f"Available metrics: {list(metrics_info.keys())}")
    
    # Example evaluation with custom config
    eval_input = EvaluationInput(
        chat_history=[{"role": "user", "content": "Explain quantum computing"}],
        retrieved_chunks=[
            "Quantum computing uses quantum mechanics principles for computation.",
            "Quantum computers use qubits instead of classical bits."
        ],
        llm_answer="Quantum computing leverages quantum mechanical phenomena like superposition and entanglement to process information in ways that classical computers cannot.",
        user_feedback_label="Positive",
        user_feedback_comment="Good technical explanation"
    )
    
    try:
        result = await evaluator.evaluate(eval_input)
        print(f"\nEvaluation completed with custom configuration")
        print(f"Enabled metrics: {eval_config.enabled_metrics}")
        print(f"Parallel evaluation: {eval_config.parallel_evaluation}")
        print(f"Overall score difference: {result.comparison['overall_score_difference']:.2f}")
        
        return result
        
    except Exception as e:
        print(f"Error with advanced configuration: {str(e)}")
        return None


def display_package_info():
    """Display information about the LLM evaluator package."""
    print("=" * 60)
    print("LLM Evaluator Package Information")
    print("=" * 60)
    
    print("""
This comprehensive package provides:

1. **Detailed Metrics Framework**: 9 comprehensive evaluation metrics
   - Correctness (factual accuracy)
   - Completeness (coverage of the question)
   - Relevance (pertinence to the query)
   - Fluency (language quality and readability)
   - Coherence (logical flow and consistency)
   - Faithfulness (adherence to provided context)
   - Harmfulness (safety and appropriateness)
   - Helpfulness (utility to the user)
   - Conciseness (brevity without losing meaning)

2. **Structured Response Format**: Detailed JSON with scores, justifications, and insights

3. **Advanced Prompt Engineering**: Context-aware prompts with few-shot examples

4. **Error Handling & Reliability**: Retry mechanisms, timeout handling, graceful degradation

5. **Configuration & Flexibility**: Configurable metrics, custom scoring rubrics

6. **Analysis & Reporting**: Statistical analysis, visualization data, export capabilities

7. **Integration Features**: Async support, batch processing, caching, rate limiting

Key Classes:
- LLMEvaluator: Main evaluator class
- EvaluationInput: Input data structure
- EvaluationResult: Comprehensive results
- ModelConfig: Model configuration
- EvaluationConfig: Evaluation configuration
- AnalysisUtils: Statistical analysis utilities
- ExportUtils: Export and reporting utilities
    """)


async def main():
    """Main function demonstrating all examples."""
    display_package_info()
    
    print("\n" + "!" * 60)
    print("NOTE: This is a demonstration script.")
    print("Replace the placeholder credentials with your actual API keys.")
    print("!" * 60)
    
    # Comment out the actual evaluations since we don't have real API keys
    # Uncomment these lines when you have actual credentials
    
    # await basic_evaluation_example()
    # await batch_evaluation_example()
    # await advanced_configuration_example()
    
    print("\nTo use this package with real API keys:")
    print("1. Replace placeholder credentials in the examples")
    print("2. Uncomment the evaluation function calls")
    print("3. Run the script")
    
    print("\nFor more examples, see the package documentation.")


if __name__ == "__main__":
    asyncio.run(main())