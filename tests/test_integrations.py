"""
Unit tests for integration classes (integrations.py).
"""

import unittest
import asyncio
import json
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from unittest.mock import call as mock_call

from llm_evaluator.integrations import (
    AzureOpenAIIntegration, ClaudeIntegration, ModelIntegrationFactory
)
from llm_evaluator.models import ModelConfig, MetricScore


class TestAzureOpenAIIntegration(unittest.TestCase):
    """Test AzureOpenAIIntegration class."""
    
    def setUp(self):
        self.config = ModelConfig(
            model_name="gpt-4",
            temperature=0.7,
            max_tokens=1000,
            timeout=30
        )
        
        self.integration = AzureOpenAIIntegration(
            config=self.config,
            endpoint="https://test.openai.azure.com/",
            api_key="test-key",
            deployment_name="test-deployment"
        )
    
    def test_init(self):
        """Test AzureOpenAIIntegration initialization."""
        self.assertEqual(self.integration.config, self.config)
        self.assertEqual(self.integration.endpoint, "https://test.openai.azure.com/")
        self.assertEqual(self.integration.api_key, "test-key")
        self.assertEqual(self.integration.deployment_name, "test-deployment")
    
    def test_prepare_headers(self):
        """Test prepare_headers method."""
        headers = self.integration.prepare_headers()
        
        self.assertIsInstance(headers, dict)
        self.assertIn("Content-Type", headers)
        self.assertIn("api-key", headers)
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertEqual(headers["api-key"], "test-key")
    
    def test_prepare_payload(self):
        """Test prepare_payload method."""
        messages = [{"role": "user", "content": "Test message"}]
        payload = self.integration.prepare_payload(messages)
        
        self.assertIsInstance(payload, dict)
        self.assertIn("messages", payload)
        self.assertIn("temperature", payload)
        self.assertIn("max_tokens", payload)
        
        self.assertEqual(payload["messages"], messages)
        self.assertEqual(payload["temperature"], 0.7)
        self.assertEqual(payload["max_tokens"], 1000)
    
    def test_build_url(self):
        """Test build_url method."""
        url = self.integration.build_url()
        
        self.assertIsInstance(url, str)
        self.assertIn("https://test.openai.azure.com/", url)
        self.assertIn("test-deployment", url)
        self.assertIn("chat/completions", url)
        self.assertIn("api-version", url)
    
    @patch('requests.post')
    async def test_make_request_success(self, mock_post):
        """Test make_request with successful response."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Test response"}}],
            "usage": {"total_tokens": 150}
        }
        mock_post.return_value = mock_response
        
        messages = [{"role": "user", "content": "Test message"}]
        result = await self.integration.make_request(messages)
        
        self.assertEqual(result, "Test response")
        mock_post.assert_called_once()
    
    @patch('requests.post')
    async def test_make_request_failure(self, mock_post):
        """Test make_request with failed response."""
        # Mock failed response
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.raise_for_status.side_effect = Exception("API Error")
        mock_post.return_value = mock_response
        
        messages = [{"role": "user", "content": "Test message"}]
        
        with self.assertRaises(Exception):
            await self.integration.make_request(messages)
    
    @patch('requests.post')
    async def test_make_request_with_retry(self, mock_post):
        """Test make_request with retry mechanism."""
        # Mock first call to fail, second to succeed
        mock_response_fail = Mock()
        mock_response_fail.status_code = 500
        mock_response_fail.raise_for_status.side_effect = Exception("Server Error")
        
        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {
            "choices": [{"message": {"content": "Success response"}}],
            "usage": {"total_tokens": 150}
        }
        
        mock_post.side_effect = [mock_response_fail, mock_response_success]
        
        messages = [{"role": "user", "content": "Test message"}]
        
        # Configure retry
        self.integration.config.retry_attempts = 2
        self.integration.config.retry_delay = 0.1
        
        result = await self.integration.make_request(messages)
        
        self.assertEqual(result, "Success response")
        self.assertEqual(mock_post.call_count, 2)
    
    def test_validate_response(self):
        """Test validate_response method."""
        # Test valid response
        valid_response = {
            "choices": [{"message": {"content": "Test response"}}],
            "usage": {"total_tokens": 150}
        }
        
        self.assertTrue(self.integration.validate_response(valid_response))
        
        # Test invalid response - no choices
        invalid_response = {"usage": {"total_tokens": 150}}
        self.assertFalse(self.integration.validate_response(invalid_response))
        
        # Test invalid response - empty choices
        empty_choices_response = {"choices": []}
        self.assertFalse(self.integration.validate_response(empty_choices_response))
        
        # Test invalid response - no message content
        no_content_response = {"choices": [{"message": {}}]}
        self.assertFalse(self.integration.validate_response(no_content_response))
    
    def test_extract_content(self):
        """Test extract_content method."""
        response = {
            "choices": [{"message": {"content": "Test content"}}],
            "usage": {"total_tokens": 150}
        }
        
        content = self.integration.extract_content(response)
        self.assertEqual(content, "Test content")
    
    def test_extract_usage_info(self):
        """Test extract_usage_info method."""
        response = {
            "choices": [{"message": {"content": "Test content"}}],
            "usage": {"total_tokens": 150, "prompt_tokens": 50, "completion_tokens": 100}
        }
        
        usage = self.integration.extract_usage_info(response)
        self.assertEqual(usage["total_tokens"], 150)
        self.assertEqual(usage["prompt_tokens"], 50)
        self.assertEqual(usage["completion_tokens"], 100)
        
        # Test response without usage info
        response_no_usage = {"choices": [{"message": {"content": "Test content"}}]}
        usage_no_info = self.integration.extract_usage_info(response_no_usage)
        self.assertEqual(usage_no_info["total_tokens"], 0)


class TestClaudeIntegration(unittest.TestCase):
    """Test ClaudeIntegration class."""
    
    def setUp(self):
        self.config = ModelConfig(
            model_name="claude-3-sonnet-20240229",
            temperature=0.7,
            max_tokens=1000,
            timeout=30
        )
        
        self.integration = ClaudeIntegration(
            config=self.config,
            api_key="test-key"
        )
    
    def test_init(self):
        """Test ClaudeIntegration initialization."""
        self.assertEqual(self.integration.config, self.config)
        self.assertEqual(self.integration.api_key, "test-key")
        self.assertEqual(self.integration.api_url, "https://api.anthropic.com/v1/messages")
    
    def test_prepare_headers(self):
        """Test prepare_headers method."""
        headers = self.integration.prepare_headers()
        
        self.assertIsInstance(headers, dict)
        self.assertIn("Content-Type", headers)
        self.assertIn("x-api-key", headers)
        self.assertIn("anthropic-version", headers)
        
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertEqual(headers["x-api-key"], "test-key")
    
    def test_prepare_payload(self):
        """Test prepare_payload method."""
        messages = [{"role": "user", "content": "Test message"}]
        payload = self.integration.prepare_payload(messages)
        
        self.assertIsInstance(payload, dict)
        self.assertIn("model", payload)
        self.assertIn("messages", payload)
        self.assertIn("temperature", payload)
        self.assertIn("max_tokens", payload)
        
        self.assertEqual(payload["model"], "claude-3-sonnet-20240229")
        self.assertEqual(payload["messages"], messages)
        self.assertEqual(payload["temperature"], 0.7)
        self.assertEqual(payload["max_tokens"], 1000)
    
    def test_convert_messages_format(self):
        """Test convert_messages_format method."""
        # Test with system message
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"}
        ]
        
        converted = self.integration.convert_messages_format(messages)
        
        self.assertIsInstance(converted, list)
        self.assertEqual(len(converted), 1)  # System message removed, user message kept
        self.assertEqual(converted[0]["role"], "user")
        self.assertEqual(converted[0]["content"], "Hello")
    
    def test_convert_messages_format_no_system(self):
        """Test convert_messages_format with no system message."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"}
        ]
        
        converted = self.integration.convert_messages_format(messages)
        
        self.assertEqual(len(converted), 2)
        self.assertEqual(converted[0]["role"], "user")
        self.assertEqual(converted[1]["role"], "assistant")
    
    @patch('requests.post')
    async def test_make_request_success(self, mock_post):
        """Test make_request with successful response."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": [{"text": "Test response"}],
            "usage": {"input_tokens": 50, "output_tokens": 100}
        }
        mock_post.return_value = mock_response
        
        messages = [{"role": "user", "content": "Test message"}]
        result = await self.integration.make_request(messages)
        
        self.assertEqual(result, "Test response")
        mock_post.assert_called_once()
    
    @patch('requests.post')
    async def test_make_request_failure(self, mock_post):
        """Test make_request with failed response."""
        # Mock failed response
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.raise_for_status.side_effect = Exception("API Error")
        mock_post.return_value = mock_response
        
        messages = [{"role": "user", "content": "Test message"}]
        
        with self.assertRaises(Exception):
            await self.integration.make_request(messages)
    
    def test_validate_response(self):
        """Test validate_response method."""
        # Test valid response
        valid_response = {
            "content": [{"text": "Test response"}],
            "usage": {"input_tokens": 50, "output_tokens": 100}
        }
        
        self.assertTrue(self.integration.validate_response(valid_response))
        
        # Test invalid response - no content
        invalid_response = {"usage": {"input_tokens": 50, "output_tokens": 100}}
        self.assertFalse(self.integration.validate_response(invalid_response))
        
        # Test invalid response - empty content
        empty_content_response = {"content": []}
        self.assertFalse(self.integration.validate_response(empty_content_response))
        
        # Test invalid response - no text in content
        no_text_response = {"content": [{"type": "image"}]}
        self.assertFalse(self.integration.validate_response(no_text_response))
    
    def test_extract_content(self):
        """Test extract_content method."""
        response = {
            "content": [{"text": "Test content"}],
            "usage": {"input_tokens": 50, "output_tokens": 100}
        }
        
        content = self.integration.extract_content(response)
        self.assertEqual(content, "Test content")
    
    def test_extract_usage_info(self):
        """Test extract_usage_info method."""
        response = {
            "content": [{"text": "Test content"}],
            "usage": {"input_tokens": 50, "output_tokens": 100}
        }
        
        usage = self.integration.extract_usage_info(response)
        self.assertEqual(usage["input_tokens"], 50)
        self.assertEqual(usage["output_tokens"], 100)
        self.assertEqual(usage["total_tokens"], 150)
        
        # Test response without usage info
        response_no_usage = {"content": [{"text": "Test content"}]}
        usage_no_info = self.integration.extract_usage_info(response_no_usage)
        self.assertEqual(usage_no_info["total_tokens"], 0)


class TestModelIntegrationFactory(unittest.TestCase):
    """Test ModelIntegrationFactory class."""
    
    def setUp(self):
        self.config = ModelConfig(model_name="test-model")
    
    def test_create_azure_openai_integration(self):
        """Test create_azure_openai_integration method."""
        integration = ModelIntegrationFactory.create_azure_openai_integration(
            config=self.config,
            endpoint="https://test.openai.azure.com/",
            api_key="test-key",
            deployment_name="test-deployment"
        )
        
        self.assertIsInstance(integration, AzureOpenAIIntegration)
        self.assertEqual(integration.config, self.config)
        self.assertEqual(integration.endpoint, "https://test.openai.azure.com/")
        self.assertEqual(integration.api_key, "test-key")
        self.assertEqual(integration.deployment_name, "test-deployment")
    
    def test_create_claude_integration(self):
        """Test create_claude_integration method."""
        integration = ModelIntegrationFactory.create_claude_integration(
            config=self.config,
            api_key="test-key"
        )
        
        self.assertIsInstance(integration, ClaudeIntegration)
        self.assertEqual(integration.config, self.config)
        self.assertEqual(integration.api_key, "test-key")
    
    def test_create_integration_invalid_type(self):
        """Test create_integration with invalid type."""
        with self.assertRaises(ValueError):
            ModelIntegrationFactory.create_integration(
                integration_type="invalid_type",
                config=self.config,
                api_key="test-key"
            )
    
    def test_create_integration_azure_openai(self):
        """Test create_integration with azure_openai type."""
        integration = ModelIntegrationFactory.create_integration(
            integration_type="azure_openai",
            config=self.config,
            endpoint="https://test.openai.azure.com/",
            api_key="test-key",
            deployment_name="test-deployment"
        )
        
        self.assertIsInstance(integration, AzureOpenAIIntegration)
    
    def test_create_integration_claude(self):
        """Test create_integration with claude type."""
        integration = ModelIntegrationFactory.create_integration(
            integration_type="claude",
            config=self.config,
            api_key="test-key"
        )
        
        self.assertIsInstance(integration, ClaudeIntegration)
    
    def test_get_available_integrations(self):
        """Test get_available_integrations method."""
        integrations = ModelIntegrationFactory.get_available_integrations()
        
        self.assertIsInstance(integrations, list)
        self.assertIn("azure_openai", integrations)
        self.assertIn("claude", integrations)
    
    def test_validate_integration_config(self):
        """Test validate_integration_config method."""
        # Test valid Azure OpenAI config
        valid_azure_config = {
            "endpoint": "https://test.openai.azure.com/",
            "api_key": "test-key",
            "deployment_name": "test-deployment"
        }
        
        self.assertTrue(
            ModelIntegrationFactory.validate_integration_config("azure_openai", valid_azure_config)
        )
        
        # Test invalid Azure OpenAI config - missing endpoint
        invalid_azure_config = {
            "api_key": "test-key",
            "deployment_name": "test-deployment"
        }
        
        self.assertFalse(
            ModelIntegrationFactory.validate_integration_config("azure_openai", invalid_azure_config)
        )
        
        # Test valid Claude config
        valid_claude_config = {
            "api_key": "test-key"
        }
        
        self.assertTrue(
            ModelIntegrationFactory.validate_integration_config("claude", valid_claude_config)
        )
        
        # Test invalid Claude config - missing api_key
        invalid_claude_config = {}
        
        self.assertFalse(
            ModelIntegrationFactory.validate_integration_config("claude", invalid_claude_config)
        )


class TestIntegrationHelpers(unittest.TestCase):
    """Test integration helper functions."""
    
    @patch('requests.post')
    def test_send_to_azure_openai(self, mock_post):
        """Test send_to_azure_openai helper function."""
        from llm_evaluator.integrations import send_to_azure_openai
        
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Test response"}}]
        }
        mock_post.return_value = mock_response
        
        result = send_to_azure_openai(
            endpoint="https://test.openai.azure.com/",
            api_key="test-key",
            deployment_name="test-deployment",
            messages=[{"role": "user", "content": "Test message"}]
        )
        
        self.assertEqual(result, "Test response")
        mock_post.assert_called_once()
        
        # Check that the correct URL was called
        call_args = mock_post.call_args
        self.assertIn("test-deployment", call_args[1]["url"])
        self.assertIn("chat/completions", call_args[1]["url"])
    
    @patch('requests.post')
    def test_send_to_azure_openai_with_kwargs(self, mock_post):
        """Test send_to_azure_openai with additional kwargs."""
        from llm_evaluator.integrations import send_to_azure_openai
        
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Test response"}}]
        }
        mock_post.return_value = mock_response
        
        result = send_to_azure_openai(
            endpoint="https://test.openai.azure.com/",
            api_key="test-key",
            deployment_name="test-deployment",
            messages=[{"role": "user", "content": "Test message"}],
            temperature=0.5,
            max_tokens=500
        )
        
        self.assertEqual(result, "Test response")
        
        # Check that kwargs were passed in the payload
        call_args = mock_post.call_args
        payload = call_args[1]["json"]
        self.assertEqual(payload["temperature"], 0.5)
        self.assertEqual(payload["max_tokens"], 500)
    
    @patch('requests.post')
    def test_send_to_azure_openai_failure(self, mock_post):
        """Test send_to_azure_openai with failed response."""
        from llm_evaluator.integrations import send_to_azure_openai
        
        # Mock failed response
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.raise_for_status.side_effect = Exception("API Error")
        mock_post.return_value = mock_response
        
        with self.assertRaises(Exception):
            send_to_azure_openai(
                endpoint="https://test.openai.azure.com/",
                api_key="test-key",
                deployment_name="test-deployment",
                messages=[{"role": "user", "content": "Test message"}]
            )


if __name__ == '__main__':
    unittest.main()