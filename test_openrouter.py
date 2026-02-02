#!/usr/bin/env python3
"""
Test OpenRouter API integration
"""

import os
from openai import OpenAI

# Test with OpenRouter
client = OpenAI(
    api_key=os.environ.get("OPENROUTER_API_KEY", "test-key"),
    base_url="https://openrouter.ai/api/v1",
)

try:
    response = client.chat.completions.create(
        model="openai/gpt-4-turbo",
        messages=[
            {"role": "user", "content": "Hello! Can you tell me about yourself in one sentence?"}
        ],
        max_tokens=100
    )
    print("✅ OpenRouter API test successful!")
    print(f"Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"❌ OpenRouter API test failed: {e}")
    print("Make sure OPENROUTER_API_KEY is set to a valid key from https://openrouter.ai/keys")