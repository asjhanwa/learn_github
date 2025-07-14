"""
Model integrations for Azure OpenAI and Claude Sonnet.
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Tuple
import requests
from datetime import datetime

from .models import ModelConfig, MetricScore, EvaluationInput
from .metrics import MetricsFramework


class BaseModelIntegration:
    """Base class for model integrations."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.metrics_framework = MetricsFramework()
    
    async def evaluate_metric(self, 
                            metric_name: str, 
                            eval_input: EvaluationInput) -> MetricScore:
        """Evaluate a specific metric. To be implemented by subclasses."""
        raise NotImplementedError
    
    def _parse_evaluation_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response from model evaluation."""
        try:
            # Try to extract JSON from response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON found in response")
            
            json_str = response[start_idx:end_idx]
            parsed = json.loads(json_str)
            
            # Validate required fields
            required_fields = ['reasoning', 'score', 'confidence', 'justification']
            for field in required_fields:
                if field not in parsed:
                    raise ValueError(f"Missing required field: {field}")
            
            # Validate score and confidence ranges
            if not self.metrics_framework.validate_metric_score(parsed['score']):
                raise ValueError(f"Invalid score: {parsed['score']}")
            
            if not self.metrics_framework.validate_confidence(parsed['confidence']):
                raise ValueError(f"Invalid confidence: {parsed['confidence']}")
            
            return parsed
            
        except (json.JSONDecodeError, ValueError) as e:
            # Fallback parsing if JSON is malformed
            return self._fallback_parse(response, str(e))
    
    def _fallback_parse(self, response: str, error_msg: str) -> Dict[str, Any]:
        """Fallback parsing when JSON parsing fails."""
        # Simple heuristic parsing
        lines = response.strip().split('\n')
        
        # Try to extract score from text
        score = 3.0  # Default moderate score
        confidence = 0.5  # Default moderate confidence
        
        # Look for score indicators
        for line in lines:
            if 'score' in line.lower():
                try:
                    # Extract number from line
                    import re
                    numbers = re.findall(r'\d+\.?\d*', line)
                    if numbers:
                        potential_score = float(numbers[0])
                        if 1 <= potential_score <= 5:
                            score = potential_score
                except:
                    pass
        
        return {
            'reasoning': f"Fallback parsing due to malformed response: {error_msg}",
            'score': score,
            'confidence': confidence,
            'justification': response[:500] + "..." if len(response) > 500 else response
        }


class AzureOpenAIIntegration(BaseModelIntegration):
    """Integration with Azure OpenAI models."""
    
    def __init__(self, config: ModelConfig, endpoint: str, api_key: str, deployment_name: str):
        super().__init__(config)
        self.endpoint = endpoint.rstrip('/')
        self.api_key = api_key
        self.deployment_name = deployment_name
        self.url = f"{self.endpoint}/openai/deployments/{deployment_name}/chat/completions?api-version=2024-02-15-preview"
    
    async def evaluate_metric(self, 
                            metric_name: str, 
                            eval_input: EvaluationInput) -> MetricScore:
        """Evaluate a specific metric using Azure OpenAI."""
        
        prompt = self.metrics_framework.create_evaluation_prompt(
            metric_name, eval_input, "azure_openai"
        )
        
        messages = [
            {
                "role": "system",
                "content": "You are an expert AI evaluator. Provide precise, well-reasoned evaluations in the requested JSON format."
            },
            {
                "role": "user", 
                "content": prompt
            }
        ]
        
        start_time = time.time()
        
        # Retry mechanism
        last_error = None
        for attempt in range(self.config.retry_attempts):
            try:
                response = await self._make_request(messages)
                parsed_response = self._parse_evaluation_response(response)
                
                return MetricScore(
                    name=metric_name,
                    score=parsed_response['score'],
                    justification=parsed_response['justification'],
                    confidence=parsed_response['confidence']
                )
                
            except Exception as e:
                last_error = e
                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
        
        # If all retries failed, return a default low-confidence score
        return MetricScore(
            name=metric_name,
            score=2.5,
            justification=f"Error in evaluation: {str(last_error)}",
            confidence=0.1
        )
    
    async def _make_request(self, messages: List[Dict[str, str]]) -> str:
        """Make request to Azure OpenAI API."""
        
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key
        }
        
        payload = {
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
            "frequency_penalty": self.config.frequency_penalty,
            "presence_penalty": self.config.presence_penalty
        }
        
        # Use asyncio to make the request (simulating async behavior)
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: requests.post(
                self.url,
                headers=headers,
                json=payload,
                timeout=self.config.timeout
            )
        )
        
        response.raise_for_status()
        response_data = response.json()
        
        return response_data["choices"][0]["message"]["content"]


class ClaudeIntegration(BaseModelIntegration):
    """Integration with Claude Sonnet models."""
    
    def __init__(self, config: ModelConfig, api_key: str):
        super().__init__(config)
        self.api_key = api_key
        self.url = "https://api.anthropic.com/v1/messages"
        self.model_name = config.model_name or "claude-3-sonnet-20240229"
    
    async def evaluate_metric(self, 
                            metric_name: str, 
                            eval_input: EvaluationInput) -> MetricScore:
        """Evaluate a specific metric using Claude."""
        
        prompt = self.metrics_framework.create_evaluation_prompt(
            metric_name, eval_input, "claude"
        )
        
        start_time = time.time()
        
        # Retry mechanism
        last_error = None
        for attempt in range(self.config.retry_attempts):
            try:
                response = await self._make_request(prompt)
                parsed_response = self._parse_evaluation_response(response)
                
                return MetricScore(
                    name=metric_name,
                    score=parsed_response['score'],
                    justification=parsed_response['justification'],
                    confidence=parsed_response['confidence']
                )
                
            except Exception as e:
                last_error = e
                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
        
        # If all retries failed, return a default low-confidence score
        return MetricScore(
            name=metric_name,
            score=2.5,
            justification=f"Error in evaluation: {str(last_error)}",
            confidence=0.1
        )
    
    async def _make_request(self, prompt: str) -> str:
        """Make request to Claude API."""
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
        
        payload = {
            "model": self.model_name,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        
        # Use asyncio to make the request (simulating async behavior)
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: requests.post(
                self.url,
                headers=headers,
                json=payload,
                timeout=self.config.timeout
            )
        )
        
        response.raise_for_status()
        response_data = response.json()
        
        return response_data["content"][0]["text"]


class ModelIntegrationFactory:
    """Factory for creating model integrations."""
    
    @staticmethod
    def create_azure_openai_integration(config: ModelConfig, 
                                      endpoint: str, 
                                      api_key: str, 
                                      deployment_name: str) -> AzureOpenAIIntegration:
        """Create Azure OpenAI integration."""
        return AzureOpenAIIntegration(config, endpoint, api_key, deployment_name)
    
    @staticmethod
    def create_claude_integration(config: ModelConfig, 
                                api_key: str) -> ClaudeIntegration:
        """Create Claude integration."""
        return ClaudeIntegration(config, api_key)


# Enhanced version of the original call_aoai function
def send_to_azure_openai(endpoint: str, api_key: str, deployment_name: str, 
                        messages: list, **kwargs) -> str:
    """
    Enhanced version of the original Azure OpenAI function with additional parameters.
    Maintains backward compatibility while adding new features.
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
    response.raise_for_status()
    
    return response.json()["choices"][0]["message"]["content"]