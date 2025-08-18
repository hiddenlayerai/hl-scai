import uuid
from datetime import datetime

from pydantic import BaseModel

from .ast import ASTScanResult, ASTUsage


# Analysis metadata
class AnalysisMetadata(BaseModel):
    id: str = str(uuid.uuid4())
    created_at: datetime = datetime.now()
    path: str


# Analysis model
class AnalysisAIAssetProvider(BaseModel):
    name: str
    origin: str | None = None


class AnalysisAIAssetMetadata(BaseModel):
    name: str
    provider: AnalysisAIAssetProvider
    version: str
    source: str
    usages: list[str] = []


class AnalysisAIAssetDetails(BaseModel):
    task: str | None = None
    parameters: int | None = None
    library: str | None = None
    sequence_length: int | None = None
    chat_template: str | None = None


class AnalysisAIAssetLicense(BaseModel):
    name: str | None = None
    url: str | None = None


class AnalysisAIAssetFileArtifact(BaseModel):
    name: str
    size: int | None = None
    sha1: str | None = None
    sha256: str | None = None


class AnalysisAIAssetArtifacts(BaseModel):
    files: list[AnalysisAIAssetFileArtifact] = []
    datasets: list[str] = []
    system_prompts: list[str] = []


class AnalysisAIAssetResult(BaseModel):
    metadata: AnalysisAIAssetMetadata
    details: AnalysisAIAssetDetails | None = None
    artifacts: AnalysisAIAssetArtifacts | None = None
    license: AnalysisAIAssetLicense | None = None


class AnalysisReportUsage(BaseModel):
    ast_scanner: ASTUsage = ASTUsage()


class AnalysisReport(BaseModel):
    metadata: AnalysisMetadata
    ast_scanner: dict[str, ASTScanResult] = {}
    ai_assets: list[AnalysisAIAssetResult] = []
    usage: AnalysisReportUsage = AnalysisReportUsage()
