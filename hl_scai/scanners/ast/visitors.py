import ast
from typing import Any

from ...models.ast import ASTModelResult


class ModelVisitor(ast.NodeVisitor):
    _huggingface_attributes = [
        "from_pretrained",
        "pipeline",
        "text_generation",
        "image_classification",
        "chat_completion",
        "text_to_image",
        "image_to_text",
        "question_answering",
        "translation",
        "zero_shot_classification",
        "feature_extraction",
        "summarization",
    ]
    _openai_attributes = ["completions.create", "chat.completions.create", "embeddings.create", "images.generate"]
    _anthropic_attributes = ["messages.create"]
    _aws_attributes = ["invoke_model", "invoke_model_with_response_stream", "converse", "converse_stream"]
    _cohere_attributes = ["generate"]

    def __init__(self) -> None:
        self.results: list[ASTModelResult] = []
        self.constants: dict[str, str] = {}
        self.class_constants: dict[str, dict[str, str]] = {}  # Track class-level constants
        self.current_class: str | None = None  # Track current class context
        self.current_function: str | None = None  # Track current function context
        self.instance_attributes: dict[str, dict[str, str]] = {}  # Track instance attributes per class
        self.dict_constants: dict[str, dict[str, str]] = {}  # Track dictionary literals
        self.list_constants: dict[str, list[str]] = {}  # Track list literals
        self.function_returns: dict[str, str] = {}  # Track function return values
        self.method_returns: dict[str, dict[str, str]] = {}  # Track method return values per class

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Track function context, especially for __init__ methods."""
        old_function = self.current_function
        self.current_function = node.name

        # If this is an __init__ method, track instance attribute assignments
        if node.name == "__init__" and self.current_class:
            # Check for default parameter values
            # defaults are aligned to the right side of args
            num_args = len(node.args.args) - 1  # Exclude 'self'
            num_defaults = len(node.args.defaults)
            if num_defaults > 0:
                # Map parameter names to their defaults
                for i, default in enumerate(node.args.defaults):
                    # Calculate which argument this default belongs to
                    arg_idx = num_args - num_defaults + i + 1  # +1 to skip 'self'
                    if arg_idx < len(node.args.args):
                        param_name = node.args.args[arg_idx].arg
                        if isinstance(default, ast.Constant) and isinstance(default.value, str):
                            if self.current_class not in self.instance_attributes:
                                self.instance_attributes[self.current_class] = {}
                            self.instance_attributes[self.current_class][param_name] = default.value

        # Check if this function has a simple return statement returning a string
        for stmt in node.body:
            if isinstance(stmt, ast.Return) and stmt.value:
                return_val = self._resolve_str(stmt.value)
                if return_val:
                    if self.current_class:
                        # It's a method
                        if self.current_class not in self.method_returns:
                            self.method_returns[self.current_class] = {}
                        self.method_returns[self.current_class][node.name] = return_val
                    else:
                        # It's a module-level function
                        self.function_returns[node.name] = return_val
                    # Only track the first return for simplicity
                    break

        self.generic_visit(node)
        self.current_function = old_function

    def visit_Assign(self, node: ast.Assign) -> None:
        # Handle instance attribute assignments in __init__
        if self.current_function == "__init__" and self.current_class:
            if len(node.targets) == 1:
                target = node.targets[0]
                # Check if it's self.attribute = value
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                ):
                    attr_name = target.attr
                    # Try to resolve the value
                    resolved_value = self._resolve_str(node.value)
                    if resolved_value and not resolved_value.startswith("<"):
                        if self.current_class not in self.instance_attributes:
                            self.instance_attributes[self.current_class] = {}
                        self.instance_attributes[self.current_class][attr_name] = resolved_value

        # Handle all assignment targets (including multiple assignment)
        for target in node.targets:
            if isinstance(target, ast.Name):
                target_name = target.id
                val = node.value

                # Try to resolve any value
                resolved_value = self._resolve_str(val)
                if resolved_value:
                    self.constants[target_name] = resolved_value
                # Also check for specific literal types
                elif isinstance(val, ast.Dict):
                    dict_val = self._extract_dict_literal(val)
                    if dict_val:
                        self.dict_constants[target_name] = dict_val
                elif isinstance(val, ast.List):
                    list_val = self._extract_list_literal(val)
                    if list_val:
                        self.list_constants[target_name] = list_val

        # Keep recursing
        self.generic_visit(node)

    def _extract_dict_literal(self, node: ast.Dict) -> dict[str, str] | None:
        """Extract string key-value pairs from a dictionary literal."""
        result = {}
        for key, value in zip(node.keys, node.values):
            # Only handle string keys and string values
            if key and isinstance(key, ast.Constant) and isinstance(key.value, str):
                key_str = key.value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    result[key_str] = value.value

        return result if result else None

    def _extract_list_literal(self, node: ast.List) -> list[str] | None:
        """Extract string values from a list literal."""
        result = []
        for item in node.elts:
            if isinstance(item, ast.Constant) and isinstance(item.value, str):
                result.append(item.value)

        return result if result else None

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func

        # huggingface
        if self._is_huggingface_call(func):
            self.results.extend(self._extract_from_huggingface_call(node))
        # openai
        elif self._is_openai_call(func):
            self.results.extend(self._extract_from_openai_call(node))
        # anthropic
        elif self._is_anthropic_call(func):
            self.results.extend(self._extract_from_anthropic_call(node))
        # aws
        elif self._is_aws_call(func):
            self.results.extend(self._extract_from_aws_call(node))
        # cohere
        elif self._is_cohere_call(func):
            self.results.extend(self._extract_from_cohere_call(node))

        self.generic_visit(node)

    def get_results(self) -> list[ASTModelResult]:
        return self.results

    def _resolve_str(self, node: ast.AST) -> str | None:
        """
        Resolve various AST patterns to string values:
         - String literals
         - Name references to tracked constants
         - Attribute access (self.X, ClassName.X)
         - Subscript access (dict["key"], list[0])
         - Ternary expressions
         - Environment variable calls with defaults
         - Otherwise → None
        """
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name):
            return self.constants.get(node.id)

        # Attribute access
        if isinstance(node, ast.Attribute):
            # Try to resolve attribute access
            if isinstance(node.value, ast.Name):
                base_name = node.value.id
                attr_name = node.attr

                if base_name == "self" and self.current_class:
                    # First check instance attributes
                    instance_attrs = self.instance_attributes.get(self.current_class, {})
                    if attr_name in instance_attrs:
                        return instance_attrs[attr_name]
                    # Then check class attributes
                    class_consts = self.class_constants.get(self.current_class, {})
                    if attr_name in class_consts:
                        return class_consts[attr_name]
                else:
                    # Check if it's a class name reference (e.g., Config.MODEL_NAME)
                    class_consts = self.class_constants.get(base_name, {})
                    if attr_name in class_consts:
                        return class_consts[attr_name]

        # Subscript access (dict["key"] or list[0])
        if isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Name):
                var_name = node.value.id

                # Dictionary subscript
                if isinstance(node.slice, ast.Constant):
                    key = node.slice.value
                    if isinstance(key, str) and var_name in self.dict_constants:
                        return self.dict_constants[var_name].get(key)
                    elif isinstance(key, int) and var_name in self.list_constants:
                        # List subscript with integer index
                        lst = self.list_constants[var_name]
                        if 0 <= key < len(lst):
                            return lst[key]

        # Ternary/conditional expression
        if isinstance(node, ast.IfExp):
            # Try to resolve both branches
            true_val = self._resolve_str(node.body)
            false_val = self._resolve_str(node.orelse)
            # If both branches resolve to strings, we could return one
            # For now, let's be conservative and return None
            # unless we can evaluate the condition
            if isinstance(node.test, ast.Constant):
                return true_val if node.test.value else false_val

        # Method calls - handle common patterns
        if isinstance(node, ast.Call):
            # os.environ.get("KEY", "default") or os.getenv("KEY", "default")
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Attribute)
                and isinstance(node.func.value.value, ast.Name)
            ):

                if node.func.value.value.id == "os" and node.func.value.attr == "environ" and node.func.attr == "get":
                    # Check if there's a default value
                    if len(node.args) >= 2:
                        return self._resolve_str(node.args[1])

            # os.getenv("KEY", "default")
            elif (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "os"
                and node.func.attr == "getenv"
            ):
                # Check if there's a default value
                if len(node.args) >= 2:
                    return self._resolve_str(node.args[1])

            # dict.get("key", "default")
            elif (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and isinstance(node.func.value, ast.Name)
            ):
                var_name = node.func.value.id
                if var_name in self.dict_constants and len(node.args) >= 1:
                    key_node = node.args[0]
                    if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
                        key = key_node.value
                        result = self.dict_constants[var_name].get(key)
                        if result:
                            return result
                        # Check for default value
                        elif len(node.args) >= 2:
                            return self._resolve_str(node.args[1])

            # Simple function calls (e.g., get_default_model())
            elif isinstance(node.func, ast.Name) and node.func.id in self.function_returns:
                return self.function_returns[node.func.id]

            # Method calls (e.g., self.get_model_name())
            elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                if node.func.value.id == "self" and self.current_class:
                    method_name = node.func.attr
                    class_methods = self.method_returns.get(self.current_class, {})
                    if method_name in class_methods:
                        return class_methods[method_name]

        return None

    def _get_attr_chain(self, node: ast.AST) -> str:
        """
        If node is something like
           Attribute(Attribute(Attribute(Name('self'),'openai_client'),
                               'chat'),
                     'completions'),
                     'create')
        this returns ['self','openai_client','chat','completions','create'].
        Otherwise returns None.
        """
        chain: list[str] = []
        # peel off Attribute layers
        while isinstance(node, ast.Attribute):
            chain.append(node.attr)
            node = node.value
        # finally should be a Name (e.g. "self")
        if isinstance(node, ast.Name):
            chain.append(node.id)
            return ".".join(list(reversed(chain)))
        return ""

    def _is_huggingface_call(self, func: ast.AST) -> bool:
        # Handle direct name references (e.g., pipeline)
        if isinstance(func, ast.Name) and func.id in self._huggingface_attributes:
            return True
        # Handle attribute chains
        if isinstance(func, ast.Attribute):
            chain = self._get_attr_chain(func)
            return any(attr in chain for attr in self._huggingface_attributes)
        return False

    def _extract_from_huggingface_call(self, node: ast.Call) -> list[ASTModelResult]:
        models = []
        model_name = None
        version = None
        usage = None
        source = "huggingface"

        # Handle pipeline calls specially
        if isinstance(node.func, ast.Name) and node.func.id == "pipeline":
            # For pipeline, model is typically a keyword argument
            for kw in node.keywords:
                if kw.arg == "model":
                    model_name = self._resolve_str(kw.value)
                    break
            usage = "pipeline"
        else:
            # For other calls (from_pretrained, etc.)
            if node.args:
                model_name = self._resolve_str(node.args[0])

            for kw in node.keywords:
                if kw.arg == "revision":
                    version = self._resolve_str(kw.value)
                elif kw.arg == "model_id":
                    model_name = self._resolve_str(kw.value)
                elif kw.arg == "model":
                    model_name = self._resolve_str(kw.value)

            if isinstance(node.func, ast.Attribute):
                usage = self._get_attr_chain(node.func)
            elif isinstance(node.func, ast.Name):
                usage = node.func.id
            else:
                usage = "unknown"

        if version is None:
            version = "latest"

        # Only record if we could resolve the model name
        if model_name is not None:
            models.append(ASTModelResult(name=model_name, version=version, source=source, usage=usage))

        return models

    def _is_openai_call(self, func: ast.AST) -> bool:
        chain = self._get_attr_chain(func)
        return any(attr in chain for attr in self._openai_attributes)

    def _extract_from_openai_call(self, node: ast.Call) -> list[ASTModelResult]:
        models = []
        model_name = None
        version = "latest"
        source = "openai"
        usage = None
        messages = None
        system_prompt = None

        for kw in node.keywords:
            if kw.arg == "model":
                model_name = self._resolve_str(kw.value)
            elif kw.arg == "messages" and isinstance(kw.value, ast.List):
                # Extract messages
                messages = self._extract_messages_from_list(kw.value)
                if messages:
                    system_prompt = self._extract_system_prompt_from_messages(messages)

        usage = self._get_attr_chain(node.func)

        if model_name is not None:
            models.append(
                ASTModelResult(
                    name=model_name,
                    version=version,
                    source=source,
                    usage=usage,
                    system_prompt=system_prompt,
                    messages=messages,
                )
            )

        return models

    def _is_anthropic_call(self, func: ast.AST) -> bool:
        chain = self._get_attr_chain(func)
        return any(attr in chain for attr in self._anthropic_attributes)

    def _extract_from_anthropic_call(self, node: ast.Call) -> list[ASTModelResult]:
        models = []
        model_name = None
        version = "latest"
        source = "anthropic"
        usage = None
        messages = None
        system_prompt = None

        for kw in node.keywords:
            if kw.arg == "model":
                model_name = self._resolve_str(kw.value)
            elif kw.arg == "messages" and isinstance(kw.value, ast.List):
                # Extract messages
                messages = self._extract_messages_from_list(kw.value)
                if messages:
                    system_prompt = self._extract_system_prompt_from_messages(messages)

        usage = self._get_attr_chain(node.func)

        if model_name is not None:
            models.append(
                ASTModelResult(
                    name=model_name,
                    version=version,
                    source=source,
                    usage=usage,
                    system_prompt=system_prompt,
                    messages=messages,
                )
            )

        return models

    def _is_aws_call(self, func: ast.AST) -> bool:
        chain = self._get_attr_chain(func)
        return any(attr in chain for attr in self._aws_attributes)

    def _extract_dict_from_json_dumps(self, node: ast.Call) -> dict[str, Any] | None:
        """Extract dictionary data from a json.dumps() call."""
        # Check if this is a json.dumps call
        if not (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "dumps"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "json"
        ):
            return None

        # Get the first argument (the object being dumped)
        if node.args and isinstance(node.args[0], ast.Dict):
            return self._extract_dict_literal(node.args[0])

        return None

    def _extract_prompt_from_aws_body(self, body_node: ast.AST) -> str | None:
        """Extract prompt from AWS Bedrock body parameter."""
        # First try to resolve as a string (e.g., from a variable)
        body_str = self._resolve_str(body_node)
        if body_str:
            try:
                import json

                body_dict = json.loads(body_str)
                # Look for common prompt fields
                if "prompt" in body_dict:
                    return str(body_dict["prompt"]) if body_dict["prompt"] else None
                elif "inputText" in body_dict:
                    return str(body_dict["inputText"]) if body_dict["inputText"] else None
                elif "messages" in body_dict and isinstance(body_dict["messages"], list):
                    # Extract system prompt from messages
                    return self._extract_system_prompt_from_messages(body_dict["messages"])
            except (json.JSONDecodeError, KeyError, TypeError):
                pass

        # If body is a json.dumps() call, extract the dictionary
        if isinstance(body_node, ast.Call):
            body_dict = self._extract_dict_from_json_dumps(body_node)
            if body_dict:
                # Look for prompt fields
                if "prompt" in body_dict:
                    return str(body_dict["prompt"]) if body_dict["prompt"] else None
                elif "inputText" in body_dict:
                    return str(body_dict["inputText"]) if body_dict["inputText"] else None

        return None

    def _extract_from_aws_call(self, node: ast.Call) -> list[ASTModelResult]:
        models = []
        model_name = None
        version = "latest"
        source = "aws"
        usage = None
        messages = None
        system_prompt = None

        for kw in node.keywords:
            if kw.arg == "modelId":
                model_name = self._resolve_str(kw.value)
            elif kw.arg == "messages":
                if isinstance(kw.value, ast.List):
                    # Direct list of messages
                    messages = self._extract_messages_from_list(kw.value)
                elif isinstance(kw.value, ast.Name):
                    # Variable reference - check if it's in list_constants
                    var_name = kw.value.id
                    if var_name in self.list_constants:
                        # Try to construct messages from the stored list
                        # This is a simplified approach - in reality, the list might contain dicts
                        pass
                if messages:
                    system_prompt = self._extract_system_prompt_from_messages(messages)
            elif kw.arg == "body":
                # For invoke_model API - pass the AST node, not resolved string
                prompt = self._extract_prompt_from_aws_body(kw.value)
                if prompt and not system_prompt:
                    system_prompt = prompt

        usage = self._get_attr_chain(node.func)

        if model_name is not None:
            models.append(
                ASTModelResult(
                    name=model_name,
                    version=version,
                    source=source,
                    usage=usage,
                    system_prompt=system_prompt,
                    messages=messages,
                )
            )

        return models

    def _is_cohere_call(self, func: ast.AST) -> bool:
        chain = self._get_attr_chain(func)
        return any(attr in chain for attr in self._cohere_attributes)

    def _extract_from_cohere_call(self, node: ast.Call) -> list[ASTModelResult]:
        models = []
        model_name = None
        version = "latest"
        source = "cohere"
        usage = None

        for kw in node.keywords:
            if kw.arg == "model":
                model_name = self._resolve_str(kw.value)

            if model_name is not None:
                break

        usage = self._get_attr_chain(node.func)

        if model_name is not None:
            models.append(ASTModelResult(name=model_name, version=version, source=source, usage=usage))

        return models

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        # Track class-level constant assignments
        class_name = node.name
        old_class = self.current_class
        self.current_class = class_name

        if class_name not in self.class_constants:
            self.class_constants[class_name] = {}

        for item in node.body:
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                # Handle annotated assignments like MODEL_PATH: Path = "..."
                target = item.target.id
                if item.value:
                    if isinstance(item.value, ast.Constant) and isinstance(item.value.value, str):
                        self.class_constants[class_name][target] = item.value.value

            elif isinstance(item, ast.Assign):
                # Handle regular assignments like MODEL_PATH = "..."
                if len(item.targets) == 1 and isinstance(item.targets[0], ast.Name):
                    target = item.targets[0].id
                    val = item.value
                    if isinstance(val, ast.Constant) and isinstance(val.value, str):
                        self.class_constants[class_name][target] = val.value

                    # Also handle dict/list literals at class level
                    elif isinstance(val, ast.Dict):
                        dict_val = self._extract_dict_literal(val)
                        if dict_val:
                            # Store as class-scoped dict constant
                            self.dict_constants[f"{class_name}.{target}"] = dict_val
                    elif isinstance(val, ast.List):
                        list_val = self._extract_list_literal(val)
                        if list_val:
                            # Store as class-scoped list constant
                            self.list_constants[f"{class_name}.{target}"] = list_val

        # Continue visiting
        self.generic_visit(node)

        # Restore previous class context
        self.current_class = old_class

    def _extract_messages_from_list(self, node: ast.List) -> list[dict[str, str]] | None:
        """Extract messages from a list of message dictionaries in AST form."""
        messages = []

        for item in node.elts:
            if isinstance(item, ast.Dict):
                msg_dict = {}
                # Process key-value pairs
                for key, value in zip(item.keys, item.values):
                    if key and isinstance(key, ast.Constant) and isinstance(key.value, str):
                        key_str = key.value
                        # Try to resolve the value
                        value_str = self._resolve_str(value)
                        if value_str:
                            msg_dict[key_str] = value_str

                # Only add if we have both role and content
                if "role" in msg_dict and "content" in msg_dict:
                    messages.append(msg_dict)

        return messages if messages else None

    def _extract_system_prompt_from_messages(self, messages: list[dict[str, str]]) -> str | None:
        """Extract system prompt from a list of messages."""
        for msg in messages:
            if msg.get("role") == "system":
                return msg.get("content")
        return None
