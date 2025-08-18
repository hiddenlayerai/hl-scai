# HL-SCAI Testing Documentation

This directory contains the unit test suite for the `hl_scai` library.

## Test Structure

```
tests/
├── conftest.py          # Shared fixtures and pytest configuration
├── unit/                # Unit tests
│   ├── test_ast_scanner.py      # Tests for AST scanning functionality
│   ├── test_ast_visitors.py     # Tests for AST visitor patterns
│   ├── test_core.py             # Tests for core Agent functionality
│   ├── test_huggingface_client.py # Tests for HuggingFace API client
│   ├── test_models.py           # Tests for Pydantic models
│   └── test_cli.py              # Tests for CLI interface
└── integration/         # Integration tests (future)
```

## Running Tests

### Install Test Dependencies

```bash
pip install -r requirements.txt
```

### Run All Tests

```bash
# Using pytest directly
pytest

# Using make
make test
```

### Run Specific Test Categories

```bash
# Run only unit tests
pytest -m unit
# or
make test-unit

# Run tests in verbose mode
pytest -vvs
# or
make test-verbose

# Run tests and stop on first failure
pytest -x
# or
make test-fast
```

### Run Specific Test Files

```bash
# Run a specific test file
pytest tests/unit/test_ast_scanner.py

# Run a specific test class
pytest tests/unit/test_ast_scanner.py::TestASTModelScanner

# Run a specific test method
pytest tests/unit/test_ast_scanner.py::TestASTModelScanner::test_scan_file_success
```

## Code Coverage

### Generate Coverage Report

```bash
# Generate coverage report in terminal
pytest --cov=hl_scai --cov-report=term-missing

# Generate HTML coverage report
pytest --cov=hl_scai --cov-report=html

# Using make (generates terminal, HTML, and XML reports)
make coverage
```

### View Coverage Report

```bash
# Open HTML coverage report in browser
make coverage-html

# Or manually open the file
open htmlcov/index.html
```

### Coverage Configuration

Coverage settings are defined in `.coveragerc`:
- Source directory: `hl_scai`
- Excluded: test files, virtual environments, cache directories
- HTML output: `htmlcov/`
- XML output: `coverage.xml`

## Writing Tests

### Test Markers

- `@pytest.mark.unit` - Unit tests (fast, isolated)
- `@pytest.mark.integration` - Integration tests (may require external services)
- `@pytest.mark.slow` - Slow tests

### Using Fixtures

Common fixtures are available in `conftest.py`:

```python
def test_example(temp_dir, sample_python_file, mock_hf_client):
    # temp_dir: Temporary directory that's cleaned up after test
    # sample_python_file: Python file with AI model usage examples
    # mock_hf_client: Mocked HuggingFace client
    pass
```

### Example Test

```python
import pytest
from hl_scai.scanners.ast.scanner import ASTModelScanner

class TestASTModelScanner:
    @pytest.mark.unit
    def test_scan_file_success(self, sample_python_file):
        """Test successful file scanning."""
        scanner = ASTModelScanner()
        result = scanner.scan_file(sample_python_file)

        assert isinstance(result, ASTScanResult)
        assert len(result.results) > 0
        assert len(result.errors) == 0
```

## Continuous Integration

The test suite is designed to be run in CI/CD pipelines. The XML coverage report (`coverage.xml`) can be used by tools like Codecov or Coveralls.

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure you're running tests from the project root directory
2. **Missing dependencies**: Run `pip install -r requirements.txt`
3. **Coverage not showing**: Check that `__init__.py` files exist in all packages

### Clean Up

To clean up test artifacts:

```bash
make clean
```

This removes:
- Coverage reports
- Pytest cache
- Python cache files
