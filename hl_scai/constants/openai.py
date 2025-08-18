from ..models.analysis import AnalysisAIAssetDetails, AnalysisAIAssetLicense


class OpenAIAIAssetLicense(AnalysisAIAssetLicense):
    name: str = "proprietary"
    url: str = "https://openai.com/policies/"


OPENAI_ASSETS = {
    "gpt-3": AnalysisAIAssetDetails(
        task="text-generation", parameters=None, library="openai", sequence_length=16385, chat_template=None
    ),
    "gpt-4o": AnalysisAIAssetDetails(
        task="text-generation", parameters=None, library="openai", sequence_length=128000, chat_template=None
    ),
    "gpt-4.1": AnalysisAIAssetDetails(
        task="text-generation", parameters=None, library="openai", sequence_length=1047576, chat_template=None
    ),
    "o3": AnalysisAIAssetDetails(
        task="text-generation", parameters=None, library="openai", sequence_length=200000, chat_template=None
    ),
    "o1": AnalysisAIAssetDetails(
        task="text-generation", parameters=None, library="openai", sequence_length=200000, chat_template=None
    ),
    "o4": AnalysisAIAssetDetails(
        task="text-generation", parameters=None, library="openai", sequence_length=200000, chat_template=None
    ),
}
