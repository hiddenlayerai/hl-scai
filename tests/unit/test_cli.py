"""Unit tests for CLI functionality."""

import json
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from hl_scai.entrypoint.cli import cli, scan
from hl_scai.models.analysis import AnalysisMetadata, AnalysisReport


class TestCLI:
    """Test cases for CLI commands."""

    @pytest.fixture
    def runner(self):
        """Create a Click test runner."""
        return CliRunner()

    @pytest.fixture
    def mock_analysis_report(self):
        """Create a mock analysis report."""
        return AnalysisReport(metadata=AnalysisMetadata(path="/test/path"), ast_scanner={}, ai_assets=[], usage={})

    @pytest.mark.unit
    def test_cli_help(self, runner):
        """Test CLI help command."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Show this message and exit" in result.output

    @pytest.mark.unit
    def test_scan_help(self, runner):
        """Test scan command help."""
        result = runner.invoke(scan, ["--help"])
        assert result.exit_code == 0
        assert "--directory" in result.output or "-d" in result.output

    @pytest.mark.unit
    def test_scan_missing_directory(self, runner):
        """Test scan command without required directory argument."""
        result = runner.invoke(scan, [])
        assert result.exit_code != 0
        assert "Missing option" in result.output

    @pytest.mark.unit
    @patch("hl_scai.entrypoint.cli.get_config")
    @patch("hl_scai.entrypoint.cli.Agent")
    def test_scan_success(self, mock_agent_class, mock_get_config, runner, temp_dir, mock_analysis_report):
        """Test successful scan command execution."""
        # Setup mocks
        mock_config = Mock()
        mock_get_config.return_value = mock_config

        mock_agent = Mock()
        mock_agent.analyze_directory.return_value = mock_analysis_report
        mock_agent_class.return_value = mock_agent

        # Run command
        result = runner.invoke(scan, ["-d", temp_dir])

        # Verify
        assert result.exit_code == 0
        mock_get_config.assert_called_once()
        mock_agent_class.assert_called_once_with(mock_config)
        mock_agent.analyze_directory.assert_called_once_with(temp_dir)

        # Check output is valid JSON
        output_json = json.loads(result.output)
        assert "metadata" in output_json
        assert output_json["metadata"]["path"] == "/test/path"

    @pytest.mark.unit
    @patch("hl_scai.entrypoint.cli.get_config")
    @patch("hl_scai.entrypoint.cli.Agent")
    def test_scan_with_short_option(self, mock_agent_class, mock_get_config, runner, temp_dir, mock_analysis_report):
        """Test scan command with short option -d."""
        # Setup mocks
        mock_config = Mock()
        mock_get_config.return_value = mock_config

        mock_agent = Mock()
        mock_agent.analyze_directory.return_value = mock_analysis_report
        mock_agent_class.return_value = mock_agent

        # Run command with short option
        result = runner.invoke(scan, ["-d", temp_dir])

        assert result.exit_code == 0
        mock_agent.analyze_directory.assert_called_once_with(temp_dir)

    @pytest.mark.unit
    def test_scan_nonexistent_directory(self, runner):
        """Test scan command with non-existent directory."""
        # Click validates that the directory exists before calling our code
        # because we use type=click.Path(exists=True, file_okay=False, dir_okay=True)
        result = runner.invoke(scan, ["-d", "/nonexistent/path"])

        # Click returns exit code 2 for validation errors
        assert result.exit_code == 2
        # Check for Click's error message about invalid path
        assert "does not exist" in result.output or "Invalid value" in result.output

    @pytest.mark.unit
    @patch("hl_scai.entrypoint.cli.get_config")
    @patch("hl_scai.entrypoint.cli.Agent")
    def test_scan_complex_output(self, mock_agent_class, mock_get_config, runner, temp_dir):
        """Test scan command with complex analysis results."""
        from hl_scai.models.analysis import (
            AnalysisAIAssetArtifacts,
            AnalysisAIAssetMetadata,
            AnalysisAIAssetProvider,
            AnalysisAIAssetResult,
        )

        # Create a more complex report
        complex_report = AnalysisReport(
            metadata=AnalysisMetadata(path=temp_dir),
            ast_scanner={
                "/test/file.py": {
                    "results": [
                        {"name": "gpt-4", "version": "latest", "source": "openai", "usage": "chat.completions.create"}
                    ],
                    "errors": [],
                }
            },
            ai_assets=[
                AnalysisAIAssetResult(
                    metadata=AnalysisAIAssetMetadata(
                        name="gpt-4",
                        provider=AnalysisAIAssetProvider(name="openai"),
                        version="latest",
                        source="openai",
                        usages=["chat.completions.create"],
                    ),
                    artifacts=AnalysisAIAssetArtifacts(system_prompts=["You are a helpful assistant."]),
                )
            ],
        )

        # Setup mocks
        mock_config = Mock()
        mock_get_config.return_value = mock_config

        mock_agent = Mock()
        mock_agent.analyze_directory.return_value = complex_report
        mock_agent_class.return_value = mock_agent

        # Run command
        result = runner.invoke(scan, ["-d", temp_dir])

        # Verify
        assert result.exit_code == 0

        # Check output structure
        output_json = json.loads(result.output)
        assert "metadata" in output_json
        assert "ai_assets" in output_json
        assert len(output_json["ai_assets"]) == 1
        assert output_json["ai_assets"][0]["metadata"]["name"] == "gpt-4"
