# LLM Evaluator - Comprehensive Answer Quality Assessment

A production-ready Python package for evaluating LLM answers using both Azure OpenAI and Claude Sonnet models. This package provides comprehensive metrics, structured responses, and advanced features for reliable LLM evaluation.

## 🚀 Features

### Core Capabilities
- **Dual Model Evaluation**: Simultaneous assessment using Azure OpenAI and Claude Sonnet
- **Comprehensive Metrics**: 9 detailed evaluation metrics covering all aspects of answer quality
- **Structured Responses**: Detailed JSON output with scores, justifications, and insights
- **Advanced Prompting**: Context-aware prompts with few-shot examples and chain-of-thought reasoning

### Production-Ready Features
- **Async Support**: Parallel evaluation and batch processing capabilities
- **Error Handling**: Retry mechanisms, timeout handling, and graceful degradation
- **Caching**: Built-in caching system for improved performance
- **Rate Limiting**: Configurable rate limiting to respect API limits
- **Comprehensive Logging**: Detailed logging and monitoring capabilities

### Analysis & Reporting
- **Statistical Analysis**: Score aggregation and trend analysis
- **Export Capabilities**: JSON, CSV, and comprehensive text reports
- **Visualization Data**: Prepared data for charts and graphs
- **Outlier Detection**: Automatic identification of unusual evaluations

## 📋 Evaluation Metrics

The package evaluates LLM answers across 9 comprehensive metrics:

1. **Correctness**: Factual accuracy and truthfulness
2. **Completeness**: Coverage of all aspects of the question
3. **Relevance**: Pertinence and applicability to the query
4. **Fluency**: Language quality, grammar, and readability
5. **Coherence**: Logical flow and internal consistency
6. **Faithfulness**: Adherence to provided context and source material
7. **Harmfulness**: Safety and appropriateness of content
8. **Helpfulness**: Utility and practical value to the user
9. **Conciseness**: Brevity without losing essential meaning

Each metric is scored on a 1-5 scale with detailed justifications and confidence levels.

## 🔧 Installation

```bash
pip install -r requirements.txt
```

Or install in development mode:

```bash
pip install -e .
```

## 🚦 Quick Start

### Basic Usage

```python
import asyncio
from llm_evaluator import LLMEvaluator, EvaluationInput, ModelConfig, EvaluationConfig

async def evaluate_answer():
    # Initialize evaluator
    evaluator = LLMEvaluator(
        azure_openai_endpoint="https://your-resource.openai.azure.com/",
        azure_openai_api_key="your-azure-api-key",
        azure_openai_deployment="gpt-4",
        claude_api_key="your-claude-api-key"
    )
    
    # Create evaluation input
    eval_input = EvaluationInput(
        chat_history=[
            {"role": "user", "content": "What is the capital of France?"}
        ],
        retrieved_chunks=[
            "France is a country in Western Europe. Its capital and largest city is Paris."
        ],
        llm_answer="The capital of France is Paris.",
        user_feedback_label="Positive",
        user_feedback_comment="Accurate and concise answer"
    )
    
    # Evaluate
    result = await evaluator.evaluate(eval_input)
    
    # Access results
    print(f"Azure OpenAI Score: {result.azure_openai_result.overall_score}")
    print(f"Claude Score: {result.claude_result.overall_score}")
    print(f"Overall Quality: {result.aggregated_insights['summary']['overall_quality']}")

# Run evaluation
asyncio.run(evaluate_answer())
```

### Using the Convenience Function

```python
from call_aoai import create_llm_evaluator

# Create evaluator with simplified interface
evaluator = create_llm_evaluator(
    azure_endpoint="https://your-resource.openai.azure.com/",
    azure_api_key="your-azure-api-key",
    azure_deployment="gpt-4",
    claude_api_key="your-claude-api-key",
    config={
        "parallel_evaluation": True,
        "cache_enabled": True,
        "rate_limit_enabled": True
    }
)
```

### Batch Processing

```python
# Evaluate multiple answers in parallel
results = await evaluator.batch_evaluate([
    eval_input1, eval_input2, eval_input3
])

# Analyze batch results
from llm_evaluator.utils import AnalysisUtils, ExportUtils

stats = AnalysisUtils.calculate_score_statistics(results)
ExportUtils.to_csv(results, "evaluation_results.csv")
ExportUtils.create_report(results, "evaluation_report.txt")
```

## ⚙️ Configuration

### Model Configuration

```python
from llm_evaluator.models import ModelConfig

azure_config = ModelConfig(
    model_name="gpt-4",
    temperature=0.0,
    max_tokens=2000,
    top_p=0.95,
    timeout=30,
    retry_attempts=3,
    retry_delay=1.0
)

claude_config = ModelConfig(
    model_name="claude-3-sonnet-20240229",
    temperature=0.0,
    max_tokens=2000,
    timeout=30,
    retry_attempts=3,
    retry_delay=1.0
)
```

### Evaluation Configuration

```python
from llm_evaluator.models import EvaluationConfig

eval_config = EvaluationConfig(
    azure_openai_config=azure_config,
    claude_config=claude_config,
    enabled_metrics=["correctness", "completeness", "relevance"],  # Subset of metrics
    parallel_evaluation=True,
    cache_enabled=True,
    rate_limit_enabled=True
)
```

## 📊 Response Format

The evaluation returns a structured `EvaluationResult` object:

```python
{
    "azure_openai_result": {
        "model_name": "azure_openai",
        "model_version": "gpt-4",
        "overall_score": 4.2,
        "confidence": 0.85,
        "metrics": {
            "correctness": {
                "score": 5.0,
                "justification": "Factually accurate information",
                "confidence": 0.95
            },
            "completeness": {
                "score": 4.0,
                "justification": "Covers main aspects with good detail",
                "confidence": 0.80
            }
            // ... other metrics
        }
    },
    "claude_result": {
        // Similar structure for Claude
    },
    "comparison": {
        "overall_score_difference": 0.1,
        "agreement_analysis": {
            "overall_agreement": 0.85,
            "high_agreement_metrics": ["correctness", "relevance"],
            "low_agreement_metrics": ["fluency"]
        }
    },
    "aggregated_insights": {
        "summary": {
            "overall_quality": "High",
            "strongest_aspect": "correctness",
            "weakest_aspect": "conciseness"
        },
        "strengths": ["Strong correctness", "Good relevance"],
        "weaknesses": ["Verbose response"],
        "recommendations": ["Consider more concise phrasing"]
    }
}
```

## 🔍 Advanced Features

### Custom Metrics

```python
# Enable only specific metrics
eval_config.enabled_metrics = ["correctness", "helpfulness", "harmfulness"]

# Custom scoring rubric (advanced)
eval_config.custom_scoring_rubric = {
    "correctness": {
        "weight": 0.3,
        "custom_criteria": ["Domain-specific accuracy", "Citation quality"]
    }
}
```

### Analysis and Visualization

```python
from llm_evaluator.utils import AnalysisUtils, ExportUtils

# Statistical analysis
stats = AnalysisUtils.calculate_score_statistics(results)
trends = AnalysisUtils.analyze_metric_trends(results)
outliers = AnalysisUtils.identify_outliers(results)

# Export capabilities
ExportUtils.to_json(results, "results.json")
ExportUtils.to_csv(results, "results.csv")
ExportUtils.create_report(results, "report.txt")

# Prepare visualization data
viz_data = ExportUtils.prepare_visualization_data(results)
```

### Error Handling

```python
try:
    result = await evaluator.evaluate(eval_input)
except Exception as e:
    print(f"Evaluation failed: {e}")
    # The package includes graceful degradation
    # with fallback scores when APIs are unavailable
```

## 🏗️ Architecture

### Package Structure

```
llm_evaluator/
├── __init__.py          # Package initialization
├── core.py              # Main LLMEvaluator class
├── models.py            # Data models and configurations
├── metrics.py           # Metrics framework and definitions
├── integrations.py      # Azure OpenAI and Claude integrations
└── utils.py             # Utilities for analysis and export
```

### Key Components

- **LLMEvaluator**: Main orchestrator class
- **MetricsFramework**: Handles metric definitions and prompt generation
- **ModelIntegrations**: Azure OpenAI and Claude API integrations
- **AnalysisUtils**: Statistical analysis and insights generation
- **ExportUtils**: Data export and reporting capabilities

## 🧪 Testing

```bash
# Run example script
python example_usage.py

# Test package import
python -c "from llm_evaluator import LLMEvaluator; print('Package imported successfully')"
```

## 📈 Performance Considerations

- **Parallel Evaluation**: Enable `parallel_evaluation=True` for faster results
- **Caching**: Enable `cache_enabled=True` to avoid redundant API calls
- **Rate Limiting**: Configure `rate_limit_enabled=True` to respect API limits
- **Batch Processing**: Use `batch_evaluate()` for multiple evaluations

## 🔒 Security

- Store API keys securely (environment variables, key vaults)
- The package includes timeout handling and retry mechanisms
- No sensitive data is logged or cached permanently

## 📚 Examples

See `example_usage.py` for comprehensive examples including:

- Basic single evaluation
- Batch processing
- Advanced configuration
- Analysis and reporting
- Error handling

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
- Create an issue on GitHub
- Check the documentation in `example_usage.py`
- Review the comprehensive docstrings in the source code

---

**Note**: This package requires valid API keys for Azure OpenAI and Claude Anthropic. Replace placeholder credentials in examples with your actual API keys.