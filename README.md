# HL-SCAI: AI Model Usage Scanner

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A library for scanning and analyzing AI model usage in Python codebases using Abstract Syntax Tree (AST) analysis.

## Features

- 🔍 **AST-based scanning** - Detects AI model usage through code analysis
- 🤖 **Multi-provider support** - OpenAI, Anthropic, HuggingFace, AWS Bedrock, Cohere
- 📊 **Detailed analysis** - Extracts model names, versions, system prompts, and usage patterns
- 🎯 **Smart resolution** - Resolves model names from variables, constants, and class attributes
- 📦 **HuggingFace enrichment** - Fetches model metadata, licenses, and file information
- 📈 **Comprehensive reporting** - JSON output with detailed usage statistics
- 🚀 **GitHub Action** - Integrate AI model scanning into your CI/CD workflows

## Installation

### From PyPI (when published)
```bash
pip install hl-scai
```

### From Source
```bash
# Clone the repository
git clone https://github.com/hiddenlayerai/hl-scai.git
cd hl-scai

# Install in development mode
pip install -e ".[dev]"
```

### Install Options
```bash
# Production installation
pip install hl-scai

# With test dependencies
pip install "hl-scai[test]"

# With development dependencies
pip install "hl-scai[dev]"

# With documentation dependencies
pip install "hl-scai[docs]"
```

## Usage

### Command Line Interface

```bash
# Scan a directory
hl-scai scan -d /path/to/your/project

# Scan with specific output
hl-scai scan -d ./src > analysis.json
```

### Python API

```python
from hl_scai import Agent
from hl_scai.config import get_config

# Create an agent
config = get_config()
agent = Agent(config)

# Analyze a directory
report = agent.analyze_directory("/path/to/project")

# Access results
for asset in report.ai_assets:
    print(f"Model: {asset.metadata.name}")
    print(f"Provider: {asset.metadata.provider.name}")
    print(f"Usages: {', '.join(asset.metadata.usages)}")
```

### GitHub Action

Use HL-SCAI in your GitHub workflows:

```yaml
name: AI Model Scan
on: [push, pull_request]

jobs:
  scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: hiddenlayerai/hl-scai@main
        with:
          directory: .
          fail-on-detection: true
```

See [ACTION_README.md](ACTION_README.md) for detailed GitHub Action documentation.

## Supported AI Providers

- **OpenAI**: GPT-3.5, GPT-4, embeddings, DALL-E
- **Anthropic**: Claude models
- **HuggingFace**: Transformers, pipelines, model hub
- **AWS Bedrock**: Various model providers
- **Cohere**: Generation models

## Development

### Project Structure
```
hl-scai/
├── hl_scai/              # Main package
│   ├── clients/          # External API clients
│   ├── config/           # Configuration management
│   ├── constants/        # Provider constants
│   ├── core.py           # Core Agent class
│   ├── entrypoint/       # CLI entry points
│   ├── models/           # Pydantic data models
│   └── scanners/         # AST scanning logic
├── tests/                # Test suite
│   ├── conftest.py       # Shared fixtures
│   └── unit/             # Unit tests
├── pyproject.toml        # Project configuration
├── Makefile              # Development commands
└── README.md             # This file
```

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/hiddenlayerai/hl-scai.git
cd hl-scai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
make install-dev

# Run development setup
make dev-setup

# Install pre-commit hooks
pre-commit install
```

### Pre-commit Hooks

This project uses pre-commit hooks to automatically format code and catch issues before committing:

- **black** - Auto-formats Python code
- **isort** - Auto-sorts imports
- **flake8** - Checks code style
- **mypy** - Performs type checking
- **bandit** - Security linting
- **pyupgrade** - Upgrades Python syntax

The hooks will run automatically on `git commit`. To run manually:

```bash
# Run on all files
pre-commit run --all-files

# Run on specific files
pre-commit run --files hl_scai/core.py tests/test_core.py
```

### Running Tests

```bash
# Run all tests
make test

# Run with coverage
make coverage

# View coverage report
make coverage-html

# Run specific test
pytest tests/unit/test_ast_scanner.py::TestASTModelScanner::test_scan_file_success
```

### Code Quality

```bash
# Run all linters
make lint

# Format code
make format

# Type checking
make type-check

# Security checks
make security
```

### Available Make Commands

Run `make help` to see all available commands:
- `make install` - Install package in production mode
- `make install-dev` - Install with development dependencies
- `make test` - Run all tests
- `make coverage` - Run tests with coverage report
- `make lint` - Run all linters
- `make format` - Format code with black and isort
- `make clean` - Clean up generated files
- `make build` - Build distribution packages

## Configuration

### Environment Variables

- `HUGGINGFACE_TOKEN` - HuggingFace API token for fetching model metadata

### Configuration File

Create a `.env` file in your project root:
```env
HUGGINGFACE_TOKEN=your_token_here
```

## Testing

The project includes comprehensive unit tests with 81% code coverage.

### Test Structure
- `tests/unit/` - Unit tests for all components
- `tests/conftest.py` - Shared fixtures
- `pytest.ini` - Pytest configuration
- `.coveragerc` - Coverage settings

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=hl_scai

# Run specific marker
pytest -m unit
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Write tests for new features
- Maintain code coverage above 80%
- Follow PEP 8 and use the project's linting configuration
- Add type hints to all functions
- Update documentation as needed

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [Pydantic](https://pydantic-docs.helpmanual.io/) for data validation
- Uses [Click](https://click.palletsprojects.com/) for CLI
- AST analysis powered by Python's built-in `ast` module
