import logging
import os

from .clients.huggingface import HuggingFaceClient
from .config.settings import AgentConfig
from .constants.anthropic import ANTHROPIC_ASSETS, AnthropicAIAssetLicense
from .constants.openai import OPENAI_ASSETS, OpenAIAIAssetLicense
from .models.analysis import (
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
from .scanners.ast.scanner import ASTModelScanner

logger = logging.getLogger(__name__)


class Agent:
    def __init__(
        self,
        config: AgentConfig,
        ast_scanner: ASTModelScanner | None = None,
        hf_client: HuggingFaceClient | None = None,
    ):
        self.ast_scanner = ast_scanner or ASTModelScanner()
        hf_token = config.hf_config.huggingface_token if config.hf_config else None
        self.hf_client = hf_client or HuggingFaceClient(hf_token=hf_token)

    def analyze_directory(self, directory: str) -> AnalysisReport:
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory {directory} does not exist")
        elif not os.path.isdir(directory):
            raise NotADirectoryError(f"Path {directory} is not a directory")

        results = self.ast_scanner.scan_directory(directory)

        ai_assets: list[AnalysisAIAssetResult] = []
        for _, scan_result in results.scanned_results.items():
            for result in scan_result.results:
                artifacts = AnalysisAIAssetArtifacts()
                if result.system_prompt:
                    artifacts.system_prompts = [result.system_prompt]
                analysis_model_details = AnalysisAIAssetDetails()
                analysis_model_license = AnalysisAIAssetLicense()

                # if result.name is equal to one of the model names, add the usage to the model's usages,
                # else create a new model
                for model in ai_assets:
                    if result.name == model.metadata.name:
                        if result.usage and result.usage not in model.metadata.usages:
                            model.metadata.usages.append(result.usage)
                        # Add unique system prompts to the list
                        if (
                            result.system_prompt
                            and model.artifacts
                            and result.system_prompt not in model.artifacts.system_prompts
                        ):
                            model.artifacts.system_prompts.append(result.system_prompt)
                        break
                else:
                    if result.source == "huggingface":
                        has_license = False
                        license_path = None

                        provider = AnalysisAIAssetProvider(name=result.name.split("/")[0], origin=None)
                        if result.version == "latest":
                            result.version = "main"

                        # Try to get enriched data from HuggingFace, but don't fail if API is down
                        try:
                            # get file artifacts from huggingface
                            model_tree = self.hf_client.get_model_tree(result.name, result.version or "main")
                            for asset in model_tree:
                                artifacts.files.append(
                                    AnalysisAIAssetFileArtifact(
                                        name=asset["path"],
                                        size=asset["size"],
                                        sha1=asset["oid"],
                                        sha256=asset.get("lfs", {}).get("oid"),
                                    )
                                )
                                if asset["path"] in ["LICENSE", "LICENSE.txt", "LICENSE.md"]:
                                    has_license = True
                                    license_path = asset["path"]
                        except Exception as e:
                            logger.warning(f"Failed to fetch model tree for {result.name}: {e}")

                        try:
                            # get model details from huggingface
                            model_details = self.hf_client.get_model_details(
                                result.name, version=result.version or "main"
                            )
                            if model_details:
                                analysis_model_details.task = model_details.get("pipeline_tag")
                                analysis_model_details.parameters = model_details.get("safetensors", {}).get("total")
                                analysis_model_details.library = model_details.get("library_name")
                                analysis_model_details.chat_template = model_details.get("teokenizer_config", {}).get(
                                    "chat_template"
                                )

                                analysis_model_license.name = model_details.get("cardData", {}).get("license")
                                if analysis_model_license.name is not None and has_license:
                                    analysis_model_license.url = (
                                        f"https://huggingface.co/{result.name}/blob/{result.version}/{license_path}"
                                    )
                        except Exception as e:
                            logger.warning(f"Failed to fetch model details for {result.name}: {e}")

                    else:
                        provider = AnalysisAIAssetProvider(name=result.source, origin=None)

                        if result.source == "openai":
                            for name, details in OPENAI_ASSETS.items():
                                if name in result.name:
                                    analysis_model_details = details
                                    analysis_model_license = OpenAIAIAssetLicense()
                                    break

                        if result.source == "aws" or result.source == "anthropic":
                            for name, details in ANTHROPIC_ASSETS.items():
                                if name in result.name:
                                    analysis_model_details = details
                                    analysis_model_license = AnthropicAIAssetLicense()
                                    break

                    analysis_model_metadata = AnalysisAIAssetMetadata(
                        name=result.name,
                        provider=provider,
                        version=result.version or "latest",
                        source=result.source,
                        usages=[result.usage] if result.usage else [],
                    )

                    analysis_model_result = AnalysisAIAssetResult(
                        metadata=analysis_model_metadata,
                        details=analysis_model_details,
                        license=analysis_model_license,
                        artifacts=artifacts,
                    )

                    ai_assets.append(analysis_model_result)

        return AnalysisReport(
            metadata=AnalysisMetadata(path=directory),
            ast_scanner=results.scanned_results,
            ai_assets=ai_assets,
            usage=AnalysisReportUsage(ast_scanner=results.usage),
        )
