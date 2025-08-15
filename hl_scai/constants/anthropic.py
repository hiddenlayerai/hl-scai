from ..models.analysis import AnalysisAIAssetDetails, AnalysisAIAssetLicense


class AnthropicAIAssetLicense(AnalysisAIAssetLicense):
    name: str = "proprietary"
    url: str = "https://www.anthropic.com/legal/aup"


ANTHROPIC_ASSETS = {
    "claude": AnalysisAIAssetDetails(
        task="text-generation", parameters=None, library="anthropic", sequence_length=200000, chat_template=None
    ),
}
