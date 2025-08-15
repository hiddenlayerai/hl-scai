"""Unit tests for core Agent functionality."""

import pytest

from hl_scai.config.settings import AgentConfig, HuggingFaceConfig
from hl_scai.core import Agent
from hl_scai.models.analysis import AnalysisReport
from hl_scai.models.ast import ASTModelResult, ASTScanResult, ASTScanResults, ASTUsage


class TestAgent:
    """Test cases for Agent class."""

    @pytest.fixture
    def agent_config(self):
        """Create a test agent configuration."""
        return AgentConfig(hf_config=HuggingFaceConfig(huggingface_token="test_token"))

    @pytest.fixture
    def mock_scan_results(self):
        """Create mock scan results."""
        results = ASTScanResults()
        results.scanned_results = {
            "/test/file1.py": ASTScanResult(
                results=[
                    ASTModelResult(
                        name="gpt-4",
                        version="latest",
                        source="openai",
                        usage="chat.completions.create",
                        system_prompt="You are helpful.",
                    ),
                    ASTModelResult(
                        name="bert-base-uncased", version="main", source="huggingface", usage="from_pretrained"
                    ),
                ],
                errors=[],
            )
        }
        results.usage = ASTUsage(scanned_files=1, total_results=2, total_errors=0)
        return results

    @pytest.mark.unit
    def test_agent_initialization(self, agent_config):
        """Test agent initialization with config."""
        agent = Agent(agent_config)

        assert agent.ast_scanner is not None
        assert agent.hf_client is not None

    @pytest.mark.unit
    def test_agent_with_custom_scanner(self, agent_config, mocker):
        """Test agent initialization with custom scanner and client."""
        mock_scanner = mocker.Mock()
        mock_hf_client = mocker.Mock()

        agent = Agent(agent_config, ast_scanner=mock_scanner, hf_client=mock_hf_client)

        assert agent.ast_scanner == mock_scanner
        assert agent.hf_client == mock_hf_client

    @pytest.mark.unit
    def test_analyze_directory_not_exists(self, agent_config):
        """Test analyzing non-existent directory."""
        agent = Agent(agent_config)

        with pytest.raises(FileNotFoundError, match="does not exist"):
            agent.analyze_directory("/nonexistent/path")

    @pytest.mark.unit
    def test_analyze_directory_not_directory(self, agent_config, sample_python_file):
        """Test analyzing a file instead of directory."""
        agent = Agent(agent_config)

        with pytest.raises(NotADirectoryError, match="is not a directory"):
            agent.analyze_directory(sample_python_file)

    @pytest.mark.unit
    def test_analyze_directory_success(self, agent_config, temp_dir, mocker, mock_scan_results, mock_hf_client):
        """Test successful directory analysis."""
        # Mock the scanner
        mock_scanner = mocker.Mock()
        mock_scanner.scan_directory.return_value = mock_scan_results

        agent = Agent(agent_config, ast_scanner=mock_scanner, hf_client=mock_hf_client)

        # Analyze directory
        report = agent.analyze_directory(temp_dir)

        # Verify report structure
        assert isinstance(report, AnalysisReport)
        assert report.metadata.path == temp_dir
        assert len(report.ai_assets) == 2

        # Check OpenAI model
        openai_assets = [a for a in report.ai_assets if a.metadata.source == "openai"]
        assert len(openai_assets) == 1
        assert openai_assets[0].metadata.name == "gpt-4"
        assert openai_assets[0].artifacts.system_prompts == ["You are helpful."]

        # Check HuggingFace model
        hf_assets = [a for a in report.ai_assets if a.metadata.source == "huggingface"]
        assert len(hf_assets) == 1
        assert hf_assets[0].metadata.name == "bert-base-uncased"

    @pytest.mark.unit
    def test_consolidate_duplicate_models(self, agent_config, temp_dir, mocker):
        """Test that duplicate models are consolidated with multiple usages."""
        # Create scan results with duplicate models
        results = ASTScanResults()
        results.scanned_results = {
            "/test/file1.py": ASTScanResult(
                results=[
                    ASTModelResult(name="gpt-4", version="latest", source="openai", usage="chat.completions.create")
                ],
                errors=[],
            ),
            "/test/file2.py": ASTScanResult(
                results=[ASTModelResult(name="gpt-4", version="latest", source="openai", usage="completions.create")],
                errors=[],
            ),
        }

        mock_scanner = mocker.Mock()
        mock_scanner.scan_directory.return_value = results

        agent = Agent(agent_config, ast_scanner=mock_scanner)
        report = agent.analyze_directory(temp_dir)

        # Should have only one gpt-4 asset with multiple usages
        assert len(report.ai_assets) == 1
        assert report.ai_assets[0].metadata.name == "gpt-4"
        assert len(report.ai_assets[0].metadata.usages) == 2
        assert "chat.completions.create" in report.ai_assets[0].metadata.usages
        assert "completions.create" in report.ai_assets[0].metadata.usages

    @pytest.mark.unit
    def test_huggingface_model_enrichment(self, agent_config, temp_dir, mocker, mock_hf_client):
        """Test enrichment of HuggingFace models with metadata."""
        # Create scan results with HF model
        results = ASTScanResults()
        results.scanned_results = {
            "/test/file1.py": ASTScanResult(
                results=[
                    ASTModelResult(
                        name="microsoft/deberta-v3-base", version="main", source="huggingface", usage="from_pretrained"
                    )
                ],
                errors=[],
            )
        }

        mock_scanner = mocker.Mock()
        mock_scanner.scan_directory.return_value = results

        agent = Agent(agent_config, ast_scanner=mock_scanner, hf_client=mock_hf_client)
        report = agent.analyze_directory(temp_dir)

        # Verify HuggingFace API was called
        mock_hf_client.get_model_tree.assert_called_once_with("microsoft/deberta-v3-base", "main")
        mock_hf_client.get_model_details.assert_called_once_with("microsoft/deberta-v3-base", version="main")

        # Check enriched data
        hf_asset = report.ai_assets[0]
        assert hf_asset.metadata.provider.name == "microsoft"
        assert len(hf_asset.artifacts.files) == 2  # From mock data
        assert hf_asset.details.task == "text-generation"
        assert hf_asset.details.library == "transformers"
        assert hf_asset.license.name == "apache-2.0"
        assert hf_asset.license.url == "https://huggingface.co/microsoft/deberta-v3-base/blob/main/LICENSE"

    @pytest.mark.unit
    def test_openai_model_constants(self, agent_config, temp_dir, mocker):
        """Test OpenAI model constants are properly applied."""
        # Create scan results with OpenAI model
        results = ASTScanResults()
        results.scanned_results = {
            "/test/file1.py": ASTScanResult(
                results=[
                    ASTModelResult(
                        name="gpt-4-turbo", version="latest", source="openai", usage="chat.completions.create"
                    )
                ],
                errors=[],
            )
        }

        mock_scanner = mocker.Mock()
        mock_scanner.scan_directory.return_value = results

        agent = Agent(agent_config, ast_scanner=mock_scanner)
        report = agent.analyze_directory(temp_dir)

        # Check that OpenAI constants were applied
        openai_asset = report.ai_assets[0]
        assert openai_asset.details is not None
        assert openai_asset.license is not None

    @pytest.mark.unit
    def test_anthropic_model_constants(self, agent_config, temp_dir, mocker):
        """Test Anthropic model constants are properly applied."""
        # Create scan results with Anthropic model
        results = ASTScanResults()
        results.scanned_results = {
            "/test/file1.py": ASTScanResult(
                results=[
                    ASTModelResult(name="claude-3-opus", version="latest", source="anthropic", usage="messages.create")
                ],
                errors=[],
            )
        }

        mock_scanner = mocker.Mock()
        mock_scanner.scan_directory.return_value = results

        agent = Agent(agent_config, ast_scanner=mock_scanner)
        report = agent.analyze_directory(temp_dir)

        # Check that Anthropic constants were applied
        anthropic_asset = report.ai_assets[0]
        assert anthropic_asset.details is not None
        assert anthropic_asset.license is not None

    @pytest.mark.unit
    def test_system_prompt_consolidation(self, agent_config, temp_dir, mocker):
        """Test that system prompts are consolidated across usages."""
        # Create scan results with same model but different system prompts
        results = ASTScanResults()
        results.scanned_results = {
            "/test/file1.py": ASTScanResult(
                results=[
                    ASTModelResult(
                        name="gpt-4",
                        version="latest",
                        source="openai",
                        usage="chat.completions.create",
                        system_prompt="You are a helpful assistant.",
                    )
                ],
                errors=[],
            ),
            "/test/file2.py": ASTScanResult(
                results=[
                    ASTModelResult(
                        name="gpt-4",
                        version="latest",
                        source="openai",
                        usage="chat.completions.create",
                        system_prompt="You are an expert coder.",
                    )
                ],
                errors=[],
            ),
        }

        mock_scanner = mocker.Mock()
        mock_scanner.scan_directory.return_value = results

        agent = Agent(agent_config, ast_scanner=mock_scanner)
        report = agent.analyze_directory(temp_dir)

        # Should have one model with both system prompts
        assert len(report.ai_assets) == 1
        assert len(report.ai_assets[0].artifacts.system_prompts) == 2
        assert "You are a helpful assistant." in report.ai_assets[0].artifacts.system_prompts
        assert "You are an expert coder." in report.ai_assets[0].artifacts.system_prompts

    @pytest.mark.unit
    def test_huggingface_version_latest_to_main(self, agent_config, temp_dir, mocker, mock_hf_client):
        """Test that HuggingFace 'latest' version is converted to 'main'."""
        # Create scan results with HF model using 'latest' version
        results = ASTScanResults()
        results.scanned_results = {
            "/test/file1.py": ASTScanResult(
                results=[
                    ASTModelResult(
                        name="bert-base-uncased",
                        version="latest",  # This should be converted to "main"
                        source="huggingface",
                        usage="from_pretrained",
                    )
                ],
                errors=[],
            )
        }

        mock_scanner = mocker.Mock()
        mock_scanner.scan_directory.return_value = results

        agent = Agent(agent_config, ast_scanner=mock_scanner, hf_client=mock_hf_client)
        _ = agent.analyze_directory(temp_dir)

        # Verify that 'latest' was converted to 'main' in API calls
        mock_hf_client.get_model_tree.assert_called_once_with("bert-base-uncased", "main")
        mock_hf_client.get_model_details.assert_called_once_with("bert-base-uncased", version="main")

    @pytest.mark.unit
    def test_aws_model_with_anthropic_constants(self, agent_config, temp_dir, mocker):
        """Test AWS source with Anthropic model constants."""
        # Create scan results with AWS Bedrock Anthropic model
        results = ASTScanResults()
        results.scanned_results = {
            "/test/file1.py": ASTScanResult(
                results=[
                    ASTModelResult(
                        name="anthropic.claude-3-sonnet",
                        version="latest",
                        source="aws",  # AWS source
                        usage="invoke_model",
                    )
                ],
                errors=[],
            )
        }

        mock_scanner = mocker.Mock()
        mock_scanner.scan_directory.return_value = results

        agent = Agent(agent_config, ast_scanner=mock_scanner)
        report = agent.analyze_directory(temp_dir)

        # Should match Anthropic constants even though source is AWS
        assert len(report.ai_assets) == 1
        asset = report.ai_assets[0]
        assert asset.metadata.source == "aws"
        assert asset.details is not None  # Should have details from ANTHROPIC_ASSETS
        assert asset.license is not None  # Should have AnthropicAIAssetLicense

    @pytest.mark.unit
    def test_unknown_provider_model(self, agent_config, temp_dir, mocker):
        """Test model from unknown provider (not OpenAI, Anthropic, or HuggingFace)."""
        # Create scan results with Cohere model
        results = ASTScanResults()
        results.scanned_results = {
            "/test/file1.py": ASTScanResult(
                results=[ASTModelResult(name="command-r-plus", version="latest", source="cohere", usage="generate")],
                errors=[],
            )
        }

        mock_scanner = mocker.Mock()
        mock_scanner.scan_directory.return_value = results

        agent = Agent(agent_config, ast_scanner=mock_scanner)
        report = agent.analyze_directory(temp_dir)

        # Should create asset with minimal details
        assert len(report.ai_assets) == 1
        asset = report.ai_assets[0]
        assert asset.metadata.provider.name == "cohere"
        assert asset.metadata.source == "cohere"
        # Should have empty details and license
        assert asset.details.task is None
        assert asset.license.name is None
