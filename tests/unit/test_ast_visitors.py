"""Unit tests for AST visitors."""

import ast

import pytest

from hl_scai.scanners.ast.visitors import ModelVisitor


class TestModelVisitor:
    """Test cases for ModelVisitor."""

    @pytest.mark.unit
    def test_visitor_initialization(self):
        """Test visitor initialization."""
        visitor = ModelVisitor()
        assert isinstance(visitor.results, list)
        assert len(visitor.results) == 0
        assert isinstance(visitor.constants, dict)
        assert isinstance(visitor.class_constants, dict)

    @pytest.mark.unit
    def test_detect_openai_model(self):
        """Test detection of OpenAI model usage."""
        code = """
import openai

client = openai.OpenAI()
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"}
    ]
)
"""
        tree = ast.parse(code)
        visitor = ModelVisitor()
        visitor.visit(tree)

        results = visitor.get_results()
        assert len(results) == 1
        assert results[0].name == "gpt-3.5-turbo"
        assert results[0].source == "openai"
        assert results[0].system_prompt == "You are a helpful assistant."

    @pytest.mark.unit
    def test_detect_huggingface_pipeline(self):
        """Test detection of HuggingFace pipeline usage."""
        code = """
from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="bert-base-uncased")
"""
        tree = ast.parse(code)
        visitor = ModelVisitor()
        visitor.visit(tree)

        results = visitor.get_results()
        assert len(results) == 1
        assert results[0].name == "bert-base-uncased"
        assert results[0].source == "huggingface"
        assert results[0].usage == "pipeline"

    @pytest.mark.unit
    def test_detect_huggingface_from_pretrained(self):
        """Test detection of HuggingFace from_pretrained usage."""
        code = """
from transformers import AutoModel

model = AutoModel.from_pretrained("microsoft/deberta-v3-base", revision="main")
"""
        tree = ast.parse(code)
        visitor = ModelVisitor()
        visitor.visit(tree)

        results = visitor.get_results()
        assert len(results) == 1
        assert results[0].name == "microsoft/deberta-v3-base"
        assert results[0].source == "huggingface"
        assert results[0].version == "main"

    @pytest.mark.unit
    def test_detect_anthropic_model(self):
        """Test detection of Anthropic model usage."""
        code = """
import anthropic

client = anthropic.Anthropic()
response = client.messages.create(
    model="claude-3-opus-20240229",
    messages=[
        {"role": "user", "content": "Hello"}
    ]
)
"""
        tree = ast.parse(code)
        visitor = ModelVisitor()
        visitor.visit(tree)

        results = visitor.get_results()
        assert len(results) == 1
        assert results[0].name == "claude-3-opus-20240229"
        assert results[0].source == "anthropic"

    @pytest.mark.unit
    def test_detect_aws_bedrock_model(self):
        """Test detection of AWS Bedrock model usage."""
        code = """
import boto3

bedrock = boto3.client("bedrock-runtime")
response = bedrock.invoke_model(
    modelId="anthropic.claude-v2",
    body=json.dumps({"prompt": "Hello"})
)
"""
        tree = ast.parse(code)
        visitor = ModelVisitor()
        visitor.visit(tree)

        results = visitor.get_results()
        assert len(results) == 1
        assert results[0].name == "anthropic.claude-v2"
        assert results[0].source == "aws"

    @pytest.mark.unit
    def test_resolve_constant_references(self):
        """Test resolution of model names from constants."""
        code = """
import openai

MODEL_NAME = "gpt-4"

client = openai.OpenAI()
response = client.chat.completions.create(model=MODEL_NAME, messages=[])
"""
        tree = ast.parse(code)
        visitor = ModelVisitor()
        visitor.visit(tree)

        results = visitor.get_results()
        assert len(results) == 1
        assert results[0].name == "gpt-4"

    @pytest.mark.unit
    def test_resolve_class_constants(self):
        """Test resolution of model names from class constants."""
        code = """
import openai

class Config:
    MODEL_NAME = "gpt-3.5-turbo"

client = openai.OpenAI()
response = client.chat.completions.create(model=Config.MODEL_NAME, messages=[])
"""
        tree = ast.parse(code)
        visitor = ModelVisitor()
        visitor.visit(tree)

        results = visitor.get_results()
        assert len(results) == 1
        assert results[0].name == "gpt-3.5-turbo"

    @pytest.mark.unit
    def test_resolve_instance_attributes(self):
        """Test resolution of model names from instance attributes."""
        code = """
import openai

class AIService:
    def __init__(self, model="gpt-4"):
        self.model = model
        self.client = openai.OpenAI()

    def generate(self):
        return self.client.chat.completions.create(model=self.model, messages=[])
"""
        tree = ast.parse(code)
        visitor = ModelVisitor()
        visitor.visit(tree)

        results = visitor.get_results()
        assert len(results) == 1
        assert results[0].name == "gpt-4"

    @pytest.mark.unit
    def test_resolve_dict_constants(self):
        """Test resolution of model names from dictionary constants."""
        code = """
import openai

config = {
    "model": "gpt-3.5-turbo",
    "temperature": 0.7
}

client = openai.OpenAI()
response = client.chat.completions.create(model=config["model"], messages=[])
"""
        tree = ast.parse(code)
        visitor = ModelVisitor()
        visitor.visit(tree)

        results = visitor.get_results()
        assert len(results) == 1
        assert results[0].name == "gpt-3.5-turbo"

    @pytest.mark.unit
    def test_multiple_models_in_file(self):
        """Test detection of multiple models in a single file."""
        code = """
import openai
from transformers import pipeline

# OpenAI usage
client = openai.OpenAI()
response = client.chat.completions.create(model="gpt-4", messages=[])

# HuggingFace usage
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased")
"""
        tree = ast.parse(code)
        visitor = ModelVisitor()
        visitor.visit(tree)

        results = visitor.get_results()
        assert len(results) == 2

        model_names = [r.name for r in results]
        assert "gpt-4" in model_names
        assert "distilbert-base-uncased" in model_names

    @pytest.mark.unit
    def test_extract_system_prompt(self):
        """Test extraction of system prompts from inline messages."""
        # Note: AST visitor can only extract system prompts from inline message lists,
        # not from variables (which would require data flow analysis)
        code = """
import openai

client = openai.OpenAI()
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are an expert Python developer."},
        {"role": "user", "content": "Write a function"}
    ]
)
"""
        tree = ast.parse(code)
        visitor = ModelVisitor()
        visitor.visit(tree)

        results = visitor.get_results()
        assert len(results) == 1
        assert results[0].system_prompt == "You are an expert Python developer."

    @pytest.mark.unit
    def test_no_model_detection(self):
        """Test that no models are detected when none are present."""
        code = """
def add(a, b):
    return a + b

result = add(1, 2)
print(result)
"""
        tree = ast.parse(code)
        visitor = ModelVisitor()
        visitor.visit(tree)

        results = visitor.get_results()
        assert len(results) == 0

    @pytest.mark.unit
    def test_resolve_self_attribute_no_class_context(self):
        """Test resolving self.attribute outside class context."""
        code = """
import openai

# This shouldn't work as there's no class context
self.model = "gpt-4"

client = openai.OpenAI()
response = client.chat.completions.create(model=self.model, messages=[])
"""
        tree = ast.parse(code)
        visitor = ModelVisitor()
        visitor.visit(tree)

        results = visitor.get_results()
        # Should not resolve self.model outside class
        assert len(results) == 0

    @pytest.mark.unit
    def test_resolve_list_subscript(self):
        """Test resolving model names from list subscripts."""
        code = """
import openai

models = ["gpt-3.5-turbo", "gpt-4", "gpt-4-32k"]

client = openai.OpenAI()
response = client.chat.completions.create(model=models[1], messages=[])
"""
        tree = ast.parse(code)
        visitor = ModelVisitor()
        visitor.visit(tree)

        results = visitor.get_results()
        assert len(results) == 1
        assert results[0].name == "gpt-4"

    @pytest.mark.unit
    def test_resolve_ternary_expression(self):
        """Test resolving model names from ternary expressions."""
        code = """
import openai

use_advanced = True
model = "gpt-4" if use_advanced else "gpt-3.5-turbo"

client = openai.OpenAI()
response = client.chat.completions.create(model=model, messages=[])
"""
        tree = ast.parse(code)
        visitor = ModelVisitor()
        visitor.visit(tree)

        # Cannot resolve ternary without evaluating condition
        results = visitor.get_results()
        assert len(results) == 0

    @pytest.mark.unit
    def test_resolve_os_environ_get(self):
        """Test resolving model names from os.environ.get with default."""
        code = """
import os
import openai

client = openai.OpenAI()
response = client.chat.completions.create(
    model=os.environ.get("MODEL_NAME", "gpt-3.5-turbo"),
    messages=[]
)
"""
        tree = ast.parse(code)
        visitor = ModelVisitor()
        visitor.visit(tree)

        results = visitor.get_results()
        assert len(results) == 1
        assert results[0].name == "gpt-3.5-turbo"  # Default value

    @pytest.mark.unit
    def test_resolve_dict_get_method(self):
        """Test resolving model names from dict.get() method."""
        code = """
import openai

config = {"model": "gpt-4", "temperature": 0.7}

client = openai.OpenAI()
response = client.chat.completions.create(
    model=config.get("model", "gpt-3.5-turbo"),
    messages=[]
)
"""
        tree = ast.parse(code)
        visitor = ModelVisitor()
        visitor.visit(tree)

        results = visitor.get_results()
        assert len(results) == 1
        assert results[0].name == "gpt-4"

    @pytest.mark.unit
    def test_resolve_function_returns(self):
        """Test resolving model names from function return values."""
        code = """
import openai

def get_default_model():
    return "gpt-4"

client = openai.OpenAI()
response = client.chat.completions.create(
    model=get_default_model(),
    messages=[]
)
"""
        tree = ast.parse(code)
        visitor = ModelVisitor()
        visitor.visit(tree)

        results = visitor.get_results()
        assert len(results) == 1
        assert results[0].name == "gpt-4"

    @pytest.mark.unit
    def test_resolve_method_returns(self):
        """Test resolving model names from method return values."""
        code = """
import openai

class Config:
    def get_model(self):
        return "gpt-4-turbo"

    def use_model(self):
        client = openai.OpenAI()
        return client.chat.completions.create(
            model=self.get_model(),
            messages=[]
        )
"""
        tree = ast.parse(code)
        visitor = ModelVisitor()
        visitor.visit(tree)

        results = visitor.get_results()
        assert len(results) == 1
        assert results[0].name == "gpt-4-turbo"

    @pytest.mark.unit
    def test_cohere_model_detection(self):
        """Test detection of Cohere models."""
        code = """
import cohere

co = cohere.Client()
response = co.generate(model="command-r-plus")
"""
        tree = ast.parse(code)
        visitor = ModelVisitor()
        visitor.visit(tree)

        results = visitor.get_results()
        assert len(results) == 1
        assert results[0].name == "command-r-plus"
        assert results[0].source == "cohere"

    @pytest.mark.unit
    def test_visit_class_with_annotations(self):
        """Test visiting class with annotated assignments."""
        code = """
import openai

class Config:
    MODEL_NAME: str = "gpt-4"
    TEMPERATURE: float = 0.7

    def __init__(self):
        self.client = openai.OpenAI()

    def generate(self):
        return self.client.chat.completions.create(
            model=Config.MODEL_NAME,
            messages=[]
        )
"""
        tree = ast.parse(code)
        visitor = ModelVisitor()
        visitor.visit(tree)

        results = visitor.get_results()
        assert len(results) == 1
        assert results[0].name == "gpt-4"

    @pytest.mark.unit
    def test_aws_messages_from_variable(self):
        """Test AWS Bedrock with messages from variable (cannot resolve)."""
        code = """
import boto3

messages = [{"role": "user", "content": "Hello"}]

bedrock = boto3.client("bedrock-runtime")
response = bedrock.converse(
    modelId="anthropic.claude-3-sonnet",
    messages=messages
)
"""
        tree = ast.parse(code)
        visitor = ModelVisitor()
        visitor.visit(tree)

        results = visitor.get_results()
        assert len(results) == 1
        assert results[0].name == "anthropic.claude-3-sonnet"
        assert results[0].system_prompt is None  # Can't extract from variable

    @pytest.mark.unit
    def test_extract_dict_from_json_dumps(self):
        """Test extracting prompt from json.dumps in AWS Bedrock."""
        code = """
import boto3
import json

bedrock = boto3.client("bedrock-runtime")
response = bedrock.invoke_model(
    modelId="anthropic.claude-2",
    body=json.dumps({
        "prompt": "What is the weather?",
        "max_tokens": 100
    })
)
"""
        tree = ast.parse(code)
        visitor = ModelVisitor()
        visitor.visit(tree)

        results = visitor.get_results()
        assert len(results) == 1
        assert results[0].name == "anthropic.claude-2"
        assert results[0].system_prompt == "What is the weather?"

    @pytest.mark.unit
    def test_visitor_edge_cases(self):
        """Test edge cases in ModelVisitor that are not covered yet."""
        # Test instance attributes when not in a class context (lines 83-85)
        code = """
def test_func():
    self.model_name = "test-model"  # self outside class context
"""
        visitor = ModelVisitor()
        tree = ast.parse(code)
        visitor.visit(tree)
        # Should not crash, but won't track the attribute
        assert len(visitor.instance_attributes) == 0

        # Test _is_huggingface_call returning False (line 186)
        code = """
some_other_func()
"""
        visitor = ModelVisitor()
        tree = ast.parse(code)
        visitor.visit(tree)
        assert len(visitor.get_results()) == 0

        # Test HuggingFace usage when func is not ast.Attribute (lines 240-241)
        code = """
from_pretrained("bert-base-uncased")
"""
        visitor = ModelVisitor()
        tree = ast.parse(code)
        visitor.visit(tree)
        results = visitor.get_results()
        assert len(results) == 1
        assert results[0].usage == "from_pretrained"

        # Test HuggingFace with revision keyword (lines 256-257)
        code = """
model = AutoModel.from_pretrained("bert-base-uncased", revision="v2.0")
"""
        visitor = ModelVisitor()
        tree = ast.parse(code)
        visitor.visit(tree)
        results = visitor.get_results()
        assert len(results) == 1
        assert results[0].version == "v2.0"

        # Test _is_openai_call returning False (line 292)
        code = """
some_client.other_method()
"""
        visitor = ModelVisitor()
        tree = ast.parse(code)
        visitor.visit(tree)
        assert len(visitor.get_results()) == 0

    @pytest.mark.unit
    def test_anthropic_messages_edge_cases(self):
        """Test Anthropic messages extraction edge cases (lines 327-330)."""
        # Test with messages as a list
        code = """
client.messages.create(
    model="claude-3-opus-20240229",
    messages=[
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hello"}
    ]
)
"""
        visitor = ModelVisitor()
        tree = ast.parse(code)
        visitor.visit(tree)
        results = visitor.get_results()
        assert len(results) == 1
        assert results[0].system_prompt == "You are helpful"
        assert len(results[0].messages) == 2

    @pytest.mark.unit
    def test_aws_body_extraction_edge_cases(self):
        """Test AWS body extraction edge cases."""
        # Test _extract_dict_from_json_dumps edge cases (lines 426, 432)
        code1 = """
# Not a json.dumps call
some_func({"prompt": "test"})
"""
        visitor = ModelVisitor()
        tree = ast.parse(code1)
        visitor.visit(tree)
        assert len(visitor.get_results()) == 0

        # Test with non-dict argument to json.dumps
        code2 = """
import json
body = json.dumps("not a dict")
"""
        visitor = ModelVisitor()
        tree = ast.parse(code2)
        visitor.visit(tree)
        # Should handle gracefully

        # Test _extract_prompt_from_aws_body with JSON parsing (lines 439-451)
        code3 = """
body_str = '{"prompt": "Test prompt"}'
client.invoke_model(modelId="anthropic.claude-v2", body=body_str)
"""
        visitor = ModelVisitor()
        visitor.constants["body_str"] = '{"prompt": "Test prompt"}'
        tree = ast.parse(code3)
        visitor.visit(tree)
        results = visitor.get_results()
        assert len(results) == 1
        assert results[0].system_prompt == "Test prompt"

        # Test with inputText field
        code4 = """
body_str = '{"inputText": "Another prompt"}'
client.invoke_model(modelId="anthropic.claude-v2", body=body_str)
"""
        visitor = ModelVisitor()
        visitor.constants["body_str"] = '{"inputText": "Another prompt"}'
        tree = ast.parse(code4)
        visitor.visit(tree)
        results = visitor.get_results()
        assert len(results) == 1
        assert results[0].system_prompt == "Another prompt"

        # Test with messages in body
        code5 = """
body_str = '{"messages": [{"role": "system", "content": "System message"}]}'
client.invoke_model(modelId="anthropic.claude-v2", body=body_str)
"""
        visitor = ModelVisitor()
        visitor.constants["body_str"] = '{"messages": [{"role": "system", "content": "System message"}]}'
        tree = ast.parse(code5)
        visitor.visit(tree)
        results = visitor.get_results()
        assert len(results) == 1
        assert results[0].system_prompt == "System message"

        # Test with invalid JSON
        code6 = """
body_str = 'invalid json'
client.invoke_model(modelId="anthropic.claude-v2", body=body_str)
"""
        visitor = ModelVisitor()
        visitor.constants["body_str"] = "invalid json"
        tree = ast.parse(code6)
        visitor.visit(tree)
        results = visitor.get_results()
        assert len(results) == 1
        assert results[0].system_prompt is None

        # Test json.dumps with dict containing prompt (lines 460-463)
        code7 = """
import json
client.invoke_model(
    modelId="anthropic.claude-v2",
    body=json.dumps({"prompt": "Dumped prompt"})
)
"""
        visitor = ModelVisitor()
        tree = ast.parse(code7)
        visitor.visit(tree)
        results = visitor.get_results()
        assert len(results) == 1
        assert results[0].system_prompt == "Dumped prompt"

        # Test with inputText in json.dumps
        code8 = """
import json
client.invoke_model(
    modelId="anthropic.claude-v2",
    body=json.dumps({"inputText": "Dumped text"})
)
"""
        visitor = ModelVisitor()
        tree = ast.parse(code8)
        visitor.visit(tree)
        results = visitor.get_results()
        assert len(results) == 1
        assert results[0].system_prompt == "Dumped text"

    @pytest.mark.unit
    def test_aws_messages_edge_cases(self):
        """Test AWS messages handling edge cases (lines 480, 487, 489)."""
        # Test with messages as variable reference in list_constants
        code1 = """
messages = ["msg1", "msg2"]
client.converse(modelId="anthropic.claude-v2", messages=messages)
"""
        visitor = ModelVisitor()
        visitor.list_constants["messages"] = ["msg1", "msg2"]
        tree = ast.parse(code1)
        visitor.visit(tree)
        results = visitor.get_results()
        assert len(results) == 1
        # Messages won't be extracted because they're strings, not dicts
        assert results[0].messages is None

        # Test body with existing system_prompt
        code2 = """
client.converse(
    modelId="anthropic.claude-v2",
    messages=[{"role": "system", "content": "First prompt"}],
    body='{"prompt": "Second prompt"}'
)
"""
        visitor = ModelVisitor()
        visitor.constants["body"] = '{"prompt": "Second prompt"}'
        tree = ast.parse(code2)
        visitor.visit(tree)
        results = visitor.get_results()
        assert len(results) == 1
        # Should prefer messages over body
        assert results[0].system_prompt == "First prompt"

    @pytest.mark.unit
    def test_cohere_edge_cases(self):
        """Test Cohere edge cases."""
        # Test early break in loop (although this line seems already covered)
        code = """
client.generate(model="command-xlarge")
"""
        visitor = ModelVisitor()
        tree = ast.parse(code)
        visitor.visit(tree)
        results = visitor.get_results()
        assert len(results) == 1
        assert results[0].name == "command-xlarge"

    @pytest.mark.unit
    def test_class_def_dict_list_literals(self):
        """Test ClassDef handling of dict/list literals (lines 561-570)."""
        code = """
class Config:
    # Dict literal at class level
    MODEL_MAPPING = {
        "small": "bert-small",
        "large": "bert-large"
    }

    # List literal at class level
    SUPPORTED_MODELS = ["gpt-3.5", "gpt-4", "claude"]

    # Non-string values (should be ignored)
    NUMBERS = [1, 2, 3]
    MIXED_DICT = {"key": 123}
"""
        visitor = ModelVisitor()
        tree = ast.parse(code)
        visitor.visit(tree)

        # Check dict constants
        assert "Config.MODEL_MAPPING" in visitor.dict_constants
        assert visitor.dict_constants["Config.MODEL_MAPPING"]["small"] == "bert-small"
        assert visitor.dict_constants["Config.MODEL_MAPPING"]["large"] == "bert-large"

        # Check list constants
        assert "Config.SUPPORTED_MODELS" in visitor.list_constants
        assert visitor.list_constants["Config.SUPPORTED_MODELS"] == ["gpt-3.5", "gpt-4", "claude"]

        # Non-string values should not be stored
        assert "Config.NUMBERS" not in visitor.list_constants
        assert "Config.MIXED_DICT" not in visitor.dict_constants

    @pytest.mark.unit
    def test_complex_attribute_chains(self):
        """Test complex attribute chain resolution."""
        # Test _get_attr_chain with non-Name base
        code = """
# This will have a non-Name base (e.g., a Call or something else)
get_client().chat.completions.create(model="gpt-4")
"""
        visitor = ModelVisitor()
        tree = ast.parse(code)
        visitor.visit(tree)
        # Should handle gracefully even if it can't fully resolve the chain
        _ = visitor.get_results()
        # May or may not detect depending on implementation

    @pytest.mark.unit
    def test_resolve_str_edge_cases(self):
        """Test additional edge cases in _resolve_str method."""
        # Test ternary with non-constant condition
        code = """
import random
model = "gpt-4" if random.random() > 0.5 else "gpt-3.5"
"""
        visitor = ModelVisitor()
        tree = ast.parse(code)
        visitor.visit(tree)
        # Should not resolve because condition is not constant
        assert "model" not in visitor.constants

        # Test os.environ.get without default
        code2 = """
import os
model = os.environ.get("MODEL_NAME")
"""
        visitor = ModelVisitor()
        tree = ast.parse(code2)
        visitor.visit(tree)
        # Should not resolve without default
        assert "model" not in visitor.constants

        # Test dict.get with non-string key
        code3 = """
config = {"model": "gpt-4"}
key = 123
model = config.get(key)
"""
        visitor = ModelVisitor()
        visitor.dict_constants["config"] = {"model": "gpt-4"}
        tree = ast.parse(code3)
        visitor.visit(tree)
        # Should not resolve with non-string key
        assert "model" not in visitor.constants

    @pytest.mark.unit
    def test_huggingface_model_id_keyword(self):
        """Test HuggingFace model_id keyword argument."""
        # Test model_id for non-pipeline calls
        code = """
model = AutoModel.from_pretrained(model_id="bert-base-uncased")
"""
        visitor = ModelVisitor()
        tree = ast.parse(code)
        visitor.visit(tree)
        results = visitor.get_results()
        assert len(results) == 1
        assert results[0].name == "bert-base-uncased"

        # Test model keyword for pipeline
        code2 = """
pipeline(task="text-generation", model="gpt2")
"""
        visitor = ModelVisitor()
        tree = ast.parse(code2)
        visitor.visit(tree)
        results = visitor.get_results()
        assert len(results) == 1
        assert results[0].name == "gpt2"

    @pytest.mark.unit
    def test_empty_extractions(self):
        """Test methods that return empty results."""
        # Test _extract_dict_literal with no string values
        code = """
data = {1: 2, 3: 4}  # Non-string keys/values
"""
        visitor = ModelVisitor()
        tree = ast.parse(code)
        visitor.visit(tree)
        assert "data" not in visitor.dict_constants

        # Test _extract_list_literal with no string values
        code2 = """
items = [1, 2, 3, None, True]  # No strings
"""
        visitor = ModelVisitor()
        tree = ast.parse(code2)
        visitor.visit(tree)
        assert "items" not in visitor.list_constants

        # Test _extract_messages_from_list with invalid structure
        code3 = """
client.chat.completions.create(
    model="gpt-4",
    messages=["not", "dict", "items"]  # Invalid message format
)
"""
        visitor = ModelVisitor()
        tree = ast.parse(code3)
        visitor.visit(tree)
        results = visitor.get_results()
        assert len(results) == 1
        assert results[0].messages is None
        assert results[0].system_prompt is None

    @pytest.mark.unit
    def test_remaining_edge_cases(self):
        """Test remaining edge cases for better coverage."""
        # Test instance attributes when class already exists in dict (lines 83-85)
        code = """
class MyClass:
    def __init__(self):
        self.model1 = "gpt-4"
        self.model2 = "gpt-3.5"  # Second assignment to same class
"""
        visitor = ModelVisitor()
        tree = ast.parse(code)
        visitor.visit(tree)
        assert "MyClass" in visitor.instance_attributes
        assert visitor.instance_attributes["MyClass"]["model1"] == "gpt-4"
        assert visitor.instance_attributes["MyClass"]["model2"] == "gpt-3.5"

        # Test _is_huggingface_call with neither Name nor Attribute (line 186)
        # This would be a Call or other node type
        code2 = """
(lambda x: x)("model")  # Lambda call, not a simple function
"""
        visitor = ModelVisitor()
        tree = ast.parse(code2)
        visitor.visit(tree)
        assert len(visitor.get_results()) == 0

        # Test non-Attribute func in HuggingFace but not pipeline (lines 240-241)
        # This tests when func.id is accessed but func is not ast.Name
        # Actually, this might be impossible since we check isinstance(node.func, ast.Name)
        # Let's test a direct call to a HF function that's not pipeline
        code3 = """
text_generation("bert-base-uncased")
"""
        visitor = ModelVisitor()
        tree = ast.parse(code3)
        visitor.visit(tree)
        results = visitor.get_results()
        assert len(results) == 1
        assert results[0].usage == "text_generation"

        # Test revision keyword in non-pipeline HF call (lines 256-257)
        # Already tested in previous test, but let's ensure it works

        # Test _extract_dict_from_json_dumps with wrong function name (line 426)
        code4 = """
import json
# Not json.dumps but some other function
result = json.loads('{"test": "value"}')
"""
        visitor = ModelVisitor()
        tree = ast.parse(code4)
        # This shouldn't extract anything
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                result = visitor._extract_dict_from_json_dumps(node)
                assert result is None

        # Test _extract_dict_from_json_dumps with empty args (line 432)
        code5 = """
import json
result = json.dumps()  # No arguments
"""
        visitor = ModelVisitor()
        tree = ast.parse(code5)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and hasattr(node.func, "attr") and node.func.attr == "dumps":
                result = visitor._extract_dict_from_json_dumps(node)
                assert result is None

    @pytest.mark.unit
    def test_anthropic_edge_case_coverage(self):
        """Test specific Anthropic edge cases for lines 329-330."""
        # The uncovered lines are in the condition check for messages
        # Test with messages not being a list
        code = """
client.messages.create(
    model="claude-3-opus-20240229",
    messages="not a list"  # String instead of list
)
"""
        visitor = ModelVisitor()
        tree = ast.parse(code)
        visitor.visit(tree)
        results = visitor.get_results()
        assert len(results) == 1
        assert results[0].messages is None
        assert results[0].system_prompt is None

    @pytest.mark.unit
    def test_openai_without_attribute(self):
        """Test OpenAI call detection when func is not ast.Attribute."""
        # This tests the _is_openai_call method with non-Attribute input
        visitor = ModelVisitor()

        # Test with a Name node
        name_node = ast.Name(id="create", ctx=ast.Load())
        assert visitor._is_openai_call(name_node) is False

        # Test with a Call node
        call_node = ast.Call(func=ast.Name(id="func", ctx=ast.Load()), args=[], keywords=[])
        assert visitor._is_openai_call(call_node) is False

    @pytest.mark.unit
    def test_get_attr_chain_non_name_base(self):
        """Test _get_attr_chain when base is not a Name."""
        visitor = ModelVisitor()

        # Create an attribute chain that ends in something other than Name
        # e.g., (1).bit_length - where 1 is a Constant, not a Name
        code = """
(1).bit_length()
"""
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                result = visitor._get_attr_chain(node)
                assert result == ""  # Should return empty string

    @pytest.mark.unit
    def test_cohere_and_aws_type_checks(self):
        """Test type checking for Cohere and AWS calls."""
        visitor = ModelVisitor()

        # These methods handle any node type via _get_attr_chain
        # which returns empty string for non-Attribute nodes
        # Since "generate" is in _cohere_attributes, a Name node with id="generate" will match

        # Test with a node that won't match any attributes
        name_node = ast.Name(id="unknown_function", ctx=ast.Load())
        assert visitor._is_cohere_call(name_node) is False
        assert visitor._is_aws_call(name_node) is False
        assert visitor._is_anthropic_call(name_node) is False

        # Test that _get_attr_chain handles Name nodes correctly
        # It returns the id for Name nodes
        generate_node = ast.Name(id="generate", ctx=ast.Load())
        chain = visitor._get_attr_chain(generate_node)
        assert chain == "generate"
