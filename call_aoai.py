import requests
from typing import Dict, List, Any, Optional

def send_to_azure_openai(endpoint: str, api_key: str, deployment_name: str, messages: list, **kwargs):
    """
    Enhanced Azure OpenAI function with additional parameters.
    Maintains backward compatibility while adding new features.

    Args:
        endpoint (str): Your Azure OpenAI endpoint (e.g., https://<resource-name>.openai.azure.com/)
        api_key (str): Your Azure OpenAI API key
        deployment_name (str): The deployment name of your model (e.g., "gpt-35-turbo")
        messages (list): A list of message dicts as per OpenAI chat format
        **kwargs: Additional parameters (temperature, max_tokens, etc.)

    Returns:
        str: The content of the assistant's reply
    """
    url = f"{endpoint.rstrip('/')}/openai/deployments/{deployment_name}/chat/completions?api-version=2024-02-15-preview"

    headers = {
        "Content-Type": "application/json",
        "api-key": api_key
    }

    payload = {
        "messages": messages,
        "temperature": kwargs.get("temperature", 0.7),
        "max_tokens": kwargs.get("max_tokens", 1000),
        "top_p": kwargs.get("top_p", 1.0),
        "frequency_penalty": kwargs.get("frequency_penalty", 0.0),
        "presence_penalty": kwargs.get("presence_penalty", 0.0)
    }

    timeout = kwargs.get("timeout", 10)
    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()  # Raises exception for 4xx/5xx errors

    return response.json()["choices"][0]["message"]["content"]


# Import the new LLM evaluator for easy access
try:
    from llm_evaluator import LLMEvaluator, EvaluationInput, ModelConfig, EvaluationConfig
    
    def create_llm_evaluator(azure_endpoint: str, 
                           azure_api_key: str, 
                           azure_deployment: str,
                           claude_api_key: str,
                           config: Optional[Dict[str, Any]] = None) -> LLMEvaluator:
        """
        Convenience function to create an LLM evaluator.
        
        Args:
            azure_endpoint: Azure OpenAI endpoint
            azure_api_key: Azure OpenAI API key
            azure_deployment: Azure OpenAI deployment name
            claude_api_key: Claude API key
            config: Optional configuration dictionary
        
        Returns:
            LLMEvaluator: Configured evaluator instance
        """
        if config is None:
            config = {}
        
        # Create model configs
        azure_config = ModelConfig(
            model_name=config.get("azure_model", "gpt-4"),
            temperature=config.get("azure_temperature", 0.0),
            max_tokens=config.get("azure_max_tokens", 2000)
        )
        
        claude_config = ModelConfig(
            model_name=config.get("claude_model", "claude-3-sonnet-20240229"),
            temperature=config.get("claude_temperature", 0.0),
            max_tokens=config.get("claude_max_tokens", 2000)
        )
        
        # Create evaluation config
        eval_config = EvaluationConfig(
            azure_openai_config=azure_config,
            claude_config=claude_config,
            parallel_evaluation=config.get("parallel_evaluation", True),
            cache_enabled=config.get("cache_enabled", True),
            rate_limit_enabled=config.get("rate_limit_enabled", True)
        )
        
        return LLMEvaluator(
            azure_endpoint, azure_api_key, azure_deployment, claude_api_key, eval_config
        )
    
except ImportError:
    # Fallback if the llm_evaluator package is not available
    def create_llm_evaluator(*args, **kwargs):
        raise ImportError("LLM evaluator package not available. Please ensure it's properly installed.")
