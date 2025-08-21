"""
Sample file demonstrating various AI model usages.
This file is used for testing the HL-SCAI scanner.
"""

import cohere
import openai
import transformers
from anthropic import Anthropic
from transformers import pipeline

# OpenAI usage
openai.api_key = "test-key"
client = openai.Client()


def test_openai():
    response = client.chat.completions.create(model="gpt-4", messages=[{"role": "user", "content": "Hello!"}])

    # Using a model variable
    model_name = "gpt-3.5-turbo"
    response2 = client.chat.completions.create(
        model=model_name, messages=[{"role": "system", "content": "You are helpful."}]
    )


# Anthropic usage
def test_anthropic():
    client = Anthropic(api_key="test-key")
    response = client.messages.create(
        model="claude-3-opus-20240229", messages=[{"role": "user", "content": "Hello Claude!"}]
    )


# HuggingFace usage
def test_huggingface():
    # Direct model loading
    model = transformers.AutoModel.from_pretrained("bert-base-uncased")

    # Pipeline usage
    classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    # With model variable
    model_id = "facebook/bart-large-mnli"
    zero_shot = pipeline("zero-shot-classification", model=model_id)


# Cohere usage
def test_cohere():
    co = cohere.Client("test-key")
    response = co.generate(model="command", prompt="Once upon a time")


if __name__ == "__main__":
    print("This is a sample file for testing HL-SCAI scanner")
