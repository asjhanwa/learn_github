"""
Unit tests for call_aoai.py helper functions.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import json

from call_aoai import send_to_azure_openai, create_llm_evaluator
from llm_evaluator import LLMEvaluator, ModelConfig, EvaluationConfig


class TestSendToAzureOpenAI(unittest.TestCase):
    """Test send_to_azure_openai function."""
    
    def setUp(self):
        self.endpoint = "https://test.openai.azure.com/"
        self.api_key = "test-key"
        self.deployment_name = "test-deployment"
        self.messages = [{"role": "user", "content": "Hello"}]
    
    @patch('call_aoai.requests.post')
    def test_send_to_azure_openai_success(self, mock_post):
        """Test successful API call."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello! How can I help you today?"}}]
        }
        mock_post.return_value = mock_response
        
        result = send_to_azure_openai(
            self.endpoint, 
            self.api_key, 
            self.deployment_name, 
            self.messages
        )
        
        self.assertEqual(result, "Hello! How can I help you today?")
        mock_post.assert_called_once()
        
        # Check that the correct URL was called
        call_args = mock_post.call_args
        self.assertIn(self.deployment_name, call_args[1]["url"])
        self.assertIn("chat/completions", call_args[1]["url"])
        self.assertIn("api-version", call_args[1]["url"])
    
    @patch('call_aoai.requests.post')
    def test_send_to_azure_openai_with_kwargs(self, mock_post):
        """Test API call with additional parameters."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Response with custom params"}}]
        }
        mock_post.return_value = mock_response
        
        result = send_to_azure_openai(
            self.endpoint, 
            self.api_key, 
            self.deployment_name, 
            self.messages,
            temperature=0.5,
            max_tokens=500,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.1
        )
        
        self.assertEqual(result, "Response with custom params")
        
        # Check that kwargs were passed in the payload
        call_args = mock_post.call_args
        payload = call_args[1]["json"]
        self.assertEqual(payload["temperature"], 0.5)
        self.assertEqual(payload["max_tokens"], 500)
        self.assertEqual(payload["top_p"], 0.9)
        self.assertEqual(payload["frequency_penalty"], 0.1)
        self.assertEqual(payload["presence_penalty"], 0.1)
    
    @patch('call_aoai.requests.post')
    def test_send_to_azure_openai_default_params(self, mock_post):
        """Test API call with default parameters."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Response with defaults"}}]
        }
        mock_post.return_value = mock_response
        
        result = send_to_azure_openai(
            self.endpoint, 
            self.api_key, 
            self.deployment_name, 
            self.messages
        )
        
        self.assertEqual(result, "Response with defaults")
        
        # Check that default parameters were used
        call_args = mock_post.call_args
        payload = call_args[1]["json"]
        self.assertEqual(payload["temperature"], 0.7)
        self.assertEqual(payload["max_tokens"], 1000)
        self.assertEqual(payload["top_p"], 1.0)
        self.assertEqual(payload["frequency_penalty"], 0.0)
        self.assertEqual(payload["presence_penalty"], 0.0)
    
    @patch('call_aoai.requests.post')
    def test_send_to_azure_openai_headers(self, mock_post):
        """Test that correct headers are sent."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Test response"}}]
        }
        mock_post.return_value = mock_response
        
        send_to_azure_openai(
            self.endpoint, 
            self.api_key, 
            self.deployment_name, 
            self.messages
        )
        
        # Check headers
        call_args = mock_post.call_args
        headers = call_args[1]["headers"]
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertEqual(headers["api-key"], self.api_key)
    
    @patch('call_aoai.requests.post')
    def test_send_to_azure_openai_timeout(self, mock_post):
        """Test timeout parameter."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Test response"}}]
        }
        mock_post.return_value = mock_response
        
        send_to_azure_openai(
            self.endpoint, 
            self.api_key, 
            self.deployment_name, 
            self.messages,
            timeout=30
        )
        
        # Check timeout
        call_args = mock_post.call_args
        self.assertEqual(call_args[1]["timeout"], 30)
    
    @patch('call_aoai.requests.post')
    def test_send_to_azure_openai_error(self, mock_post):
        """Test error handling."""
        # Mock error response
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.raise_for_status.side_effect = Exception("API Error")
        mock_post.return_value = mock_response
        
        with self.assertRaises(Exception):
            send_to_azure_openai(
                self.endpoint, 
                self.api_key, 
                self.deployment_name, 
                self.messages
            )
    
    @patch('call_aoai.requests.post')
    def test_send_to_azure_openai_url_construction(self, mock_post):
        """Test URL construction."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Test response"}}]
        }
        mock_post.return_value = mock_response
        
        # Test with endpoint that has trailing slash
        send_to_azure_openai(
            "https://test.openai.azure.com/", 
            self.api_key, 
            self.deployment_name, 
            self.messages
        )
        
        call_args = mock_post.call_args
        url = call_args[1]["url"]
        self.assertIn("https://test.openai.azure.com/openai/deployments", url)
        self.assertNotIn("//openai", url)  # No double slash
        
        # Test with endpoint that doesn't have trailing slash
        mock_post.reset_mock()
        send_to_azure_openai(
            "https://test.openai.azure.com", 
            self.api_key, 
            self.deployment_name, 
            self.messages
        )
        
        call_args = mock_post.call_args
        url = call_args[1]["url"]
        self.assertIn("https://test.openai.azure.com/openai/deployments", url)
    
    @patch('call_aoai.requests.post')
    def test_send_to_azure_openai_messages_format(self, mock_post):
        """Test that messages are passed correctly."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Test response"}}]
        }
        mock_post.return_value = mock_response
        
        complex_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]
        
        send_to_azure_openai(
            self.endpoint, 
            self.api_key, 
            self.deployment_name, 
            complex_messages
        )
        
        # Check that messages were passed correctly
        call_args = mock_post.call_args
        payload = call_args[1]["json"]
        self.assertEqual(payload["messages"], complex_messages)


class TestCreateLLMEvaluator(unittest.TestCase):
    """Test create_llm_evaluator function."""
    
    def setUp(self):
        self.azure_endpoint = "https://test.openai.azure.com/"
        self.azure_api_key = "test-azure-key"
        self.azure_deployment = "test-deployment"
        self.claude_api_key = "test-claude-key"
    
    def test_create_llm_evaluator_default_config(self):
        """Test creating evaluator with default configuration."""
        evaluator = create_llm_evaluator(
            self.azure_endpoint,
            self.azure_api_key,
            self.azure_deployment,
            self.claude_api_key
        )
        
        self.assertIsInstance(evaluator, LLMEvaluator)
        self.assertEqual(evaluator.azure_openai_endpoint, self.azure_endpoint)
        self.assertEqual(evaluator.azure_openai_api_key, self.azure_api_key)
        self.assertEqual(evaluator.azure_openai_deployment, self.azure_deployment)
        self.assertEqual(evaluator.claude_api_key, self.claude_api_key)
        
        # Check default configuration
        self.assertEqual(evaluator.config.azure_openai_config.model_name, "gpt-4")
        self.assertEqual(evaluator.config.claude_config.model_name, "claude-3-sonnet-20240229")
        self.assertEqual(evaluator.config.azure_openai_config.temperature, 0.0)
        self.assertEqual(evaluator.config.claude_config.temperature, 0.0)
        self.assertEqual(evaluator.config.azure_openai_config.max_tokens, 2000)
        self.assertEqual(evaluator.config.claude_config.max_tokens, 2000)
        self.assertTrue(evaluator.config.parallel_evaluation)
        self.assertTrue(evaluator.config.cache_enabled)
        self.assertTrue(evaluator.config.rate_limit_enabled)
    
    def test_create_llm_evaluator_custom_config(self):
        """Test creating evaluator with custom configuration."""
        custom_config = {
            "azure_model": "gpt-3.5-turbo",
            "claude_model": "claude-3-haiku-20240307",
            "azure_temperature": 0.5,
            "claude_temperature": 0.7,
            "azure_max_tokens": 1500,
            "claude_max_tokens": 1800,
            "parallel_evaluation": False,
            "cache_enabled": False,
            "rate_limit_enabled": False
        }
        
        evaluator = create_llm_evaluator(
            self.azure_endpoint,
            self.azure_api_key,
            self.azure_deployment,
            self.claude_api_key,
            config=custom_config
        )
        
        self.assertIsInstance(evaluator, LLMEvaluator)
        
        # Check custom configuration
        self.assertEqual(evaluator.config.azure_openai_config.model_name, "gpt-3.5-turbo")
        self.assertEqual(evaluator.config.claude_config.model_name, "claude-3-haiku-20240307")
        self.assertEqual(evaluator.config.azure_openai_config.temperature, 0.5)
        self.assertEqual(evaluator.config.claude_config.temperature, 0.7)
        self.assertEqual(evaluator.config.azure_openai_config.max_tokens, 1500)
        self.assertEqual(evaluator.config.claude_config.max_tokens, 1800)
        self.assertFalse(evaluator.config.parallel_evaluation)
        self.assertFalse(evaluator.config.cache_enabled)
        self.assertFalse(evaluator.config.rate_limit_enabled)
    
    def test_create_llm_evaluator_partial_config(self):
        """Test creating evaluator with partial custom configuration."""
        partial_config = {
            "azure_model": "gpt-3.5-turbo",
            "parallel_evaluation": False
        }
        
        evaluator = create_llm_evaluator(
            self.azure_endpoint,
            self.azure_api_key,
            self.azure_deployment,
            self.claude_api_key,
            config=partial_config
        )
        
        self.assertIsInstance(evaluator, LLMEvaluator)
        
        # Check that specified configs are applied
        self.assertEqual(evaluator.config.azure_openai_config.model_name, "gpt-3.5-turbo")
        self.assertFalse(evaluator.config.parallel_evaluation)
        
        # Check that defaults are used for unspecified configs
        self.assertEqual(evaluator.config.claude_config.model_name, "claude-3-sonnet-20240229")
        self.assertEqual(evaluator.config.azure_openai_config.temperature, 0.0)
        self.assertTrue(evaluator.config.cache_enabled)
        self.assertTrue(evaluator.config.rate_limit_enabled)
    
    def test_create_llm_evaluator_model_config_creation(self):
        """Test that ModelConfig objects are created correctly."""
        evaluator = create_llm_evaluator(
            self.azure_endpoint,
            self.azure_api_key,
            self.azure_deployment,
            self.claude_api_key
        )
        
        # Check that config objects are ModelConfig instances
        self.assertIsInstance(evaluator.config.azure_openai_config, ModelConfig)
        self.assertIsInstance(evaluator.config.claude_config, ModelConfig)
        
        # Check that EvaluationConfig is created
        self.assertIsInstance(evaluator.config, EvaluationConfig)
    
    def test_create_llm_evaluator_config_validation(self):
        """Test that configuration values are validated."""
        # Test with valid custom configuration
        valid_config = {
            "azure_temperature": 0.8,
            "claude_temperature": 0.9,
            "azure_max_tokens": 500,
            "claude_max_tokens": 600
        }
        
        evaluator = create_llm_evaluator(
            self.azure_endpoint,
            self.azure_api_key,
            self.azure_deployment,
            self.claude_api_key,
            config=valid_config
        )
        
        self.assertEqual(evaluator.config.azure_openai_config.temperature, 0.8)
        self.assertEqual(evaluator.config.claude_config.temperature, 0.9)
        self.assertEqual(evaluator.config.azure_openai_config.max_tokens, 500)
        self.assertEqual(evaluator.config.claude_config.max_tokens, 600)
    
    def test_create_llm_evaluator_empty_config(self):
        """Test creating evaluator with empty config dictionary."""
        evaluator = create_llm_evaluator(
            self.azure_endpoint,
            self.azure_api_key,
            self.azure_deployment,
            self.claude_api_key,
            config={}
        )
        
        self.assertIsInstance(evaluator, LLMEvaluator)
        
        # Should use all defaults
        self.assertEqual(evaluator.config.azure_openai_config.model_name, "gpt-4")
        self.assertEqual(evaluator.config.claude_config.model_name, "claude-3-sonnet-20240229")
        self.assertEqual(evaluator.config.azure_openai_config.temperature, 0.0)
        self.assertEqual(evaluator.config.claude_config.temperature, 0.0)
        self.assertTrue(evaluator.config.parallel_evaluation)
        self.assertTrue(evaluator.config.cache_enabled)
        self.assertTrue(evaluator.config.rate_limit_enabled)
    
    def test_create_llm_evaluator_credentials_passed(self):
        """Test that credentials are passed correctly to the evaluator."""
        evaluator = create_llm_evaluator(
            self.azure_endpoint,
            self.azure_api_key,
            self.azure_deployment,
            self.claude_api_key
        )
        
        # Check that credentials are stored correctly
        self.assertEqual(evaluator.azure_openai_endpoint, self.azure_endpoint)
        self.assertEqual(evaluator.azure_openai_api_key, self.azure_api_key)
        self.assertEqual(evaluator.azure_openai_deployment, self.azure_deployment)
        self.assertEqual(evaluator.claude_api_key, self.claude_api_key)
    
    def test_create_llm_evaluator_config_types(self):
        """Test that configuration creates proper types."""
        config = {
            "parallel_evaluation": "true",  # String instead of boolean
            "cache_enabled": "false",       # String instead of boolean
            "rate_limit_enabled": "true"    # String instead of boolean
        }
        
        evaluator = create_llm_evaluator(
            self.azure_endpoint,
            self.azure_api_key,
            self.azure_deployment,
            self.claude_api_key,
            config=config
        )
        
        # Function should handle string values properly
        self.assertIsInstance(evaluator.config.parallel_evaluation, bool)
        self.assertIsInstance(evaluator.config.cache_enabled, bool)
        self.assertIsInstance(evaluator.config.rate_limit_enabled, bool)


class TestCallAOAIImportFallback(unittest.TestCase):
    """Test import fallback functionality."""
    
    @patch('call_aoai.LLMEvaluator', side_effect=ImportError("Package not available"))
    def test_create_llm_evaluator_import_error(self, mock_evaluator):
        """Test that import error is handled gracefully."""
        with self.assertRaises(ImportError) as context:
            create_llm_evaluator(
                "https://test.openai.azure.com/",
                "test-key",
                "test-deployment",
                "test-key"
            )
        
        self.assertIn("LLM evaluator package not available", str(context.exception))


if __name__ == '__main__':
    unittest.main()