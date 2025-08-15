"""Unit tests for Pydantic models."""

from datetime import datetime

import pytest

from hl_scai.models.analysis import (
    AnalysisAIAssetArtifacts,
    AnalysisAIAssetDetails,
    AnalysisAIAssetFileArtifact,
    AnalysisAIAssetLicense,
    AnalysisAIAssetMetadata,
    AnalysisAIAssetProvider,
    AnalysisAIAssetResult,
    AnalysisMetadata,
    AnalysisReport,
    AnalysisReportUsage,
)
from hl_scai.models.ast import ASTModelResult, ASTScanResult, ASTScanResults, ASTUsage


class TestASTModels:
    """Test cases for AST-related models."""

    @pytest.mark.unit
    def test_ast_model_result(self):
        """Test ASTModelResult model."""
        result = ASTModelResult(
            name="gpt-4",
            version="latest",
            source="openai",
            usage="chat.completions.create",
            system_prompt="You are helpful.",
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert result.name == "gpt-4"
        assert result.version == "latest"
        assert result.source == "openai"
        assert result.usage == "chat.completions.create"
        assert result.system_prompt == "You are helpful."
        assert len(result.messages) == 1

    @pytest.mark.unit
    def test_ast_model_result_minimal(self):
        """Test ASTModelResult with minimal fields."""
        result = ASTModelResult(name="bert-base", version="main", source="huggingface", usage="from_pretrained")

        assert result.name == "bert-base"
        assert result.system_prompt is None
        assert result.messages is None

    @pytest.mark.unit
    def test_ast_scan_result(self):
        """Test ASTScanResult model."""
        model_results = [
            ASTModelResult(name="gpt-3.5-turbo", version="latest", source="openai", usage="completions.create")
        ]

        scan_result = ASTScanResult(results=model_results, errors=["Some error"])

        assert len(scan_result.results) == 1
        assert scan_result.results[0].name == "gpt-3.5-turbo"
        assert len(scan_result.errors) == 1
        assert scan_result.errors[0] == "Some error"

    @pytest.mark.unit
    def test_ast_usage(self):
        """Test ASTUsage model."""
        usage = ASTUsage(scanned_files=10, total_results=25, total_errors=2)

        assert usage.scanned_files == 10
        assert usage.total_results == 25
        assert usage.total_errors == 2

    @pytest.mark.unit
    def test_ast_usage_defaults(self):
        """Test ASTUsage model defaults."""
        usage = ASTUsage()

        assert usage.scanned_files == 0
        assert usage.total_results == 0
        assert usage.total_errors == 0

    @pytest.mark.unit
    def test_ast_scan_results(self):
        """Test ASTScanResults model."""
        results = ASTScanResults()

        # Add scan results
        results.scanned_results["/path/to/file.py"] = ASTScanResult(
            results=[ASTModelResult(name="claude-3", version="latest", source="anthropic", usage="messages.create")],
            errors=[],
        )

        assert len(results.scanned_results) == 1
        assert "/path/to/file.py" in results.scanned_results
        assert isinstance(results.usage, ASTUsage)


class TestAnalysisModels:
    """Test cases for analysis-related models."""

    @pytest.mark.unit
    def test_analysis_metadata(self):
        """Test AnalysisMetadata model."""
        metadata = AnalysisMetadata(path="/test/path")

        assert metadata.path == "/test/path"
        assert isinstance(metadata.id, str)
        assert isinstance(metadata.created_at, datetime)

    @pytest.mark.unit
    def test_analysis_ai_asset_provider(self):
        """Test AnalysisAIAssetProvider model."""
        provider = AnalysisAIAssetProvider(name="openai", origin="https://openai.com")

        assert provider.name == "openai"
        assert provider.origin == "https://openai.com"

    @pytest.mark.unit
    def test_analysis_ai_asset_metadata(self):
        """Test AnalysisAIAssetMetadata model."""
        provider = AnalysisAIAssetProvider(name="anthropic")
        metadata = AnalysisAIAssetMetadata(
            name="claude-3-opus",
            provider=provider,
            version="20240229",
            source="anthropic",
            usages=["messages.create", "completions.create"],
        )

        assert metadata.name == "claude-3-opus"
        assert metadata.provider.name == "anthropic"
        assert metadata.version == "20240229"
        assert metadata.source == "anthropic"
        assert len(metadata.usages) == 2

    @pytest.mark.unit
    def test_analysis_ai_asset_details(self):
        """Test AnalysisAIAssetDetails model."""
        details = AnalysisAIAssetDetails(
            task="text-generation",
            parameters=175000000000,
            library="transformers",
            sequence_length=4096,
            chat_template="{{messages}}",
        )

        assert details.task == "text-generation"
        assert details.parameters == 175000000000
        assert details.library == "transformers"
        assert details.sequence_length == 4096
        assert details.chat_template == "{{messages}}"

    @pytest.mark.unit
    def test_analysis_ai_asset_license(self):
        """Test AnalysisAIAssetLicense model."""
        license = AnalysisAIAssetLicense(name="apache-2.0", url="https://example.com/license")

        assert license.name == "apache-2.0"
        assert license.url == "https://example.com/license"

    @pytest.mark.unit
    def test_analysis_ai_asset_file_artifact(self):
        """Test AnalysisAIAssetFileArtifact model."""
        artifact = AnalysisAIAssetFileArtifact(name="model.safetensors", size=1000000, sha1="abc123", sha256="def456")

        assert artifact.name == "model.safetensors"
        assert artifact.size == 1000000
        assert artifact.sha1 == "abc123"
        assert artifact.sha256 == "def456"

    @pytest.mark.unit
    def test_analysis_ai_asset_artifacts(self):
        """Test AnalysisAIAssetArtifacts model."""
        artifacts = AnalysisAIAssetArtifacts(
            files=[AnalysisAIAssetFileArtifact(name="model.bin", size=500000)],
            datasets=["dataset1", "dataset2"],
            system_prompts=["You are helpful.", "You are an expert."],
        )

        assert len(artifacts.files) == 1
        assert artifacts.files[0].name == "model.bin"
        assert len(artifacts.datasets) == 2
        assert len(artifacts.system_prompts) == 2

    @pytest.mark.unit
    def test_analysis_ai_asset_result(self):
        """Test AnalysisAIAssetResult model."""
        provider = AnalysisAIAssetProvider(name="openai")
        metadata = AnalysisAIAssetMetadata(
            name="gpt-4", provider=provider, version="latest", source="openai", usages=["chat.completions.create"]
        )
        details = AnalysisAIAssetDetails(task="text-generation")
        artifacts = AnalysisAIAssetArtifacts()
        license = AnalysisAIAssetLicense(name="proprietary")

        result = AnalysisAIAssetResult(metadata=metadata, details=details, artifacts=artifacts, license=license)

        assert result.metadata.name == "gpt-4"
        assert result.details.task == "text-generation"
        assert result.artifacts is not None
        assert result.license.name == "proprietary"

    @pytest.mark.unit
    def test_analysis_report(self):
        """Test AnalysisReport model."""
        metadata = AnalysisMetadata(path="/test/path")
        ast_scanner = {"/file.py": ASTScanResult(results=[], errors=[])}
        ai_assets = [
            AnalysisAIAssetResult(
                metadata=AnalysisAIAssetMetadata(
                    name="gpt-3.5-turbo",
                    provider=AnalysisAIAssetProvider(name="openai"),
                    version="latest",
                    source="openai",
                )
            )
        ]
        usage = AnalysisReportUsage()

        report = AnalysisReport(metadata=metadata, ast_scanner=ast_scanner, ai_assets=ai_assets, usage=usage)

        assert report.metadata.path == "/test/path"
        assert len(report.ast_scanner) == 1
        assert len(report.ai_assets) == 1
        assert report.ai_assets[0].metadata.name == "gpt-3.5-turbo"

    @pytest.mark.unit
    def test_model_serialization(self):
        """Test model serialization to JSON."""
        provider = AnalysisAIAssetProvider(name="anthropic")
        metadata = AnalysisAIAssetMetadata(name="claude-3", provider=provider, version="latest", source="anthropic")
        result = AnalysisAIAssetResult(metadata=metadata)

        # Test model_dump_json
        json_str = result.model_dump_json()
        assert isinstance(json_str, str)
        assert "claude-3" in json_str
        assert "anthropic" in json_str

        # Test model_dump
        dict_data = result.model_dump()
        assert isinstance(dict_data, dict)
        assert dict_data["metadata"]["name"] == "claude-3"
