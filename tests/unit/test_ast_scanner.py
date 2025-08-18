"""Unit tests for AST scanner."""

from pathlib import Path

import pytest

from hl_scai.models.ast import ASTScanResult, ASTScanResults
from hl_scai.scanners.ast.scanner import ASTModelScanner


class TestASTModelScanner:
    """Test cases for ASTModelScanner."""

    @pytest.mark.unit
    def test_scanner_initialization(self):
        """Test scanner initialization with default and custom exclude patterns."""
        # Default initialization
        scanner = ASTModelScanner()
        assert "venv" in scanner.exclude_patterns
        assert "__pycache__" in scanner.exclude_patterns

        # Custom patterns
        custom_patterns = ["custom_dir", "*.backup"]
        scanner = ASTModelScanner(exclude_patterns=custom_patterns)
        assert "custom_dir" in scanner.exclude_patterns
        assert "*.backup" in scanner.exclude_patterns
        assert "venv" in scanner.exclude_patterns  # Default patterns still included

    @pytest.mark.unit
    def test_should_exclude(self):
        """Test file exclusion logic."""
        scanner = ASTModelScanner(exclude_patterns=["test_exclude"])

        # Should exclude paths containing excluded directories
        assert scanner._should_exclude("/path/to/venv/file.py")
        assert scanner._should_exclude("/path/to/__pycache__/file.py")
        assert scanner._should_exclude("/path/to/test_exclude/file.py")

        # Should not exclude valid paths
        assert not scanner._should_exclude("/path/to/valid/file.py")
        assert not scanner._should_exclude("/path/to/src/main.py")

    @pytest.mark.unit
    def test_scan_file_success(self, sample_python_file):
        """Test successful file scanning."""
        scanner = ASTModelScanner()
        result = scanner.scan_file(sample_python_file)

        assert isinstance(result, ASTScanResult)
        assert len(result.results) > 0
        assert len(result.errors) == 0

        # Check detected models
        model_names = [r.name for r in result.results]

        # The sample file should have detected at least the HuggingFace model
        assert len(model_names) > 0
        assert "bert-base-uncased" in model_names

        # Note: OpenAI models using instance attributes (self.model)
        # cannot be resolved by static AST analysis

    @pytest.mark.unit
    def test_scan_empty_file(self, empty_python_file):
        """Test scanning an empty file."""
        scanner = ASTModelScanner()
        result = scanner.scan_file(empty_python_file)

        assert isinstance(result, ASTScanResult)
        assert len(result.results) == 0
        assert len(result.errors) == 0

    @pytest.mark.unit
    def test_scan_syntax_error_file(self, syntax_error_file):
        """Test scanning a file with syntax errors."""
        scanner = ASTModelScanner()
        result = scanner.scan_file(syntax_error_file)

        assert isinstance(result, ASTScanResult)
        assert len(result.results) == 0
        assert len(result.errors) > 0
        assert "Syntax error" in result.errors[0]

    @pytest.mark.unit
    def test_scan_nonexistent_file(self):
        """Test scanning a non-existent file."""
        scanner = ASTModelScanner()
        result = scanner.scan_file("/path/to/nonexistent/file.py")

        assert isinstance(result, ASTScanResult)
        assert len(result.errors) > 0

    @pytest.mark.unit
    def test_scan_directory_success(self, temp_dir, sample_python_file):
        """Test successful directory scanning."""
        scanner = ASTModelScanner()
        results = scanner.scan_directory(temp_dir)

        assert isinstance(results, ASTScanResults)
        assert results.usage.scanned_files == 1
        assert results.usage.total_results > 0
        assert results.usage.total_errors == 0
        assert sample_python_file in results.scanned_results

    @pytest.mark.unit
    def test_scan_directory_with_multiple_files(self, temp_dir):
        """Test scanning directory with multiple Python files."""
        # Create multiple Python files
        for i in range(3):
            file_path = Path(temp_dir) / f"file{i}.py"
            file_path.write_text(
                """
import openai

client = openai.OpenAI()
response = client.chat.completions.create(model="gpt-3.5-turbo", messages=[])
"""
            )

        scanner = ASTModelScanner()
        results = scanner.scan_directory(temp_dir)

        assert results.usage.scanned_files == 3
        assert results.usage.total_results == 3  # One model per file

    @pytest.mark.unit
    def test_scan_directory_excludes_patterns(self, temp_dir):
        """Test that excluded directories are not scanned."""
        # Create files in excluded directories
        venv_dir = Path(temp_dir) / "venv"
        venv_dir.mkdir()
        (venv_dir / "excluded.py").write_text("import openai")

        # Create file in valid directory
        src_dir = Path(temp_dir) / "src"
        src_dir.mkdir()
        (src_dir / "included.py").write_text("import openai")

        scanner = ASTModelScanner()
        results = scanner.scan_directory(temp_dir)

        # Only the file in src should be scanned
        assert results.usage.scanned_files == 1
        assert any("src" in path for path in results.scanned_results.keys())
        assert not any("venv" in path for path in results.scanned_results.keys())

    @pytest.mark.unit
    def test_scan_nonexistent_directory(self):
        """Test scanning a non-existent directory."""
        scanner = ASTModelScanner()

        with pytest.raises(ValueError, match="does not exist"):
            scanner.scan_directory("/path/to/nonexistent/directory")

    @pytest.mark.unit
    def test_scan_file_not_directory(self, sample_python_file):
        """Test scanning a file path instead of directory."""
        scanner = ASTModelScanner()

        with pytest.raises(ValueError, match="does not exist"):
            scanner.scan_directory(sample_python_file)

    @pytest.mark.unit
    def test_should_exclude_glob_patterns(self):
        """Test exclusion with glob patterns."""
        scanner = ASTModelScanner(exclude_patterns=["*.backup", "test_*.py"])

        # Test glob patterns
        assert scanner._should_exclude("/path/to/file.backup")
        assert scanner._should_exclude("/path/to/test_file.py")
        assert not scanner._should_exclude("/path/to/file.py")

    @pytest.mark.unit
    def test_should_exclude_glob_pattern_exception(self, mocker):
        """Test glob pattern matching with exception."""
        scanner = ASTModelScanner(exclude_patterns=["[invalid"])

        # Mock fnmatch to raise exception
        mocker.patch("fnmatch.fnmatch", side_effect=Exception("Invalid pattern"))

        # Should not crash, just return False
        assert not scanner._should_exclude("/path/to/file.py")

    @pytest.mark.unit
    def test_scan_file_with_logging(self, syntax_error_file, caplog):
        """Test that syntax errors are logged."""
        scanner = ASTModelScanner()

        with caplog.at_level("WARNING"):
            _ = scanner.scan_file(syntax_error_file)

        # Check that warning was logged
        assert "Syntax error" in caplog.text
        assert syntax_error_file in caplog.text

    @pytest.mark.unit
    def test_scan_file_general_exception(self, mocker, temp_dir):
        """Test handling of general exceptions during file scanning."""
        scanner = ASTModelScanner()
        file_path = Path(temp_dir) / "test.py"
        file_path.write_text("valid python code")

        # Mock ast.parse to raise a general exception
        mocker.patch("ast.parse", side_effect=Exception("Unexpected error"))

        result = scanner.scan_file(str(file_path))

        assert len(result.errors) == 1
        assert "Unexpected error" in result.errors[0]
        assert len(result.results) == 0
