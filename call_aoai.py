import requests

def send_to_azure_openai(endpoint: str, api_key: str, deployment_name: str, messages: list):
    """
    Sends a chat payload to Azure OpenAI and returns the response.

    Args:
        endpoint (str): Your Azure OpenAI endpoint (e.g., https://<resource-name>.openai.azure.com/)
        api_key (str): Your Azure OpenAI API key
        deployment_name (str): The deployment name of your model (e.g., "gpt-35-turbo")
        messages (list): A list of message dicts as per OpenAI chat format

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
        "temperature": 0.7,
        "max_tokens": 1000
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()  # Raises exception for 4xx/5xx errors

    return response.json()["choices"][0]["message"]["content"]
