from pydantic import BaseModel


class ASTModelResult(BaseModel):
    name: str
    version: str | None = None
    source: str
    usage: str | None = None
    system_prompt: str | None = None
    messages: list[dict[str, str]] | None = None


class ASTScanResult(BaseModel):
    # file: str
    results: list[ASTModelResult]
    errors: list[str]


class ASTUsage(BaseModel):
    scanned_files: int = 0
    total_results: int = 0
    total_errors: int = 0


class ASTScanResults(BaseModel):
    usage: ASTUsage = ASTUsage()
    scanned_results: dict[str, ASTScanResult] = {}
