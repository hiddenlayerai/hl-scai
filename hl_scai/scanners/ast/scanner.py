import ast
import logging
import os
from pathlib import Path

from ...models.ast import ASTScanResult, ASTScanResults
from .visitors import ModelVisitor

logger = logging.getLogger(__name__)


class ASTModelScanner:
    def __init__(self, exclude_patterns: list[str] | None = None):
        """
        Initialize the scanner with optional exclusion patterns.

        Args:
            exclude_patterns: List of directory names or glob patterns to exclude
                           (e.g., ['venv', '.venv', 'node_modules', '__pycache__'])
        """
        self.exclude_patterns = exclude_patterns or []
        # Add common virtual environment and cache directory names by default
        default_exclude_dirs = [
            "venv",
            ".venv",
            "env",
            ".env",
            "virtualenv",
            ".virtualenv",
            "__pycache__",
            ".pytest_cache",
            "node_modules",
            ".git",
            ".svn",
            ".hg",
        ]
        self.exclude_patterns.extend(default_exclude_dirs)
        # Remove duplicates
        self.exclude_patterns = list(set(self.exclude_patterns))

    def _should_exclude(self, file_path: str) -> bool:
        """
        Check if a file path should be excluded based on the exclusion patterns.

        Args:
            file_path: The path to check

        Returns:
            True if the file should be excluded, False otherwise
        """
        path = Path(file_path)

        # Get all parts of the path
        path_parts = path.parts

        # Check if any excluded directory name appears in the path
        for part in path_parts:
            if part in self.exclude_patterns:
                return True

        # Also support glob patterns for more complex cases
        for pattern in self.exclude_patterns:
            if "*" in pattern or "?" in pattern or "[" in pattern:
                # This is a glob pattern, try to match it
                try:
                    # Convert to string for pattern matching
                    path_str = str(path)
                    import fnmatch

                    if fnmatch.fnmatch(path_str, f"*{pattern}*"):
                        return True
                except Exception:
                    pass

        return False

    def scan_file(self, file_path: str) -> ASTScanResult:
        result = ASTScanResult(results=[], errors=[])

        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content, filename=str(file_path))

            visitor = ModelVisitor()
            visitor.visit(tree)

            result.results = visitor.get_results()
        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_path}: {e}")
            result.errors.append(f"Syntax error: {e}")
        except Exception as e:
            logger.error(f"Error scanning {file_path}: {e}")
            result.errors.append(str(e))

        return result

    def scan_directory(self, directory_path: str) -> ASTScanResults:
        if not Path(directory_path).is_dir():
            raise ValueError(f"Directory {directory_path} does not exist")

        results = ASTScanResults()

        for root, dirs, files in os.walk(directory_path):
            # Filter out excluded directories to prevent os.walk from entering them
            dirs[:] = [d for d in dirs if d not in self.exclude_patterns]

            for file in files:
                if file.endswith(".py"):
                    full_path = os.path.join(root, file)

                    # Still check file path in case of glob patterns
                    if self._should_exclude(full_path):
                        logger.debug(f"Skipping excluded file: {full_path}")
                        continue

                    scan_result = self.scan_file(full_path)

                    results.usage.scanned_files += 1
                    results.usage.total_results += len(scan_result.results)
                    results.usage.total_errors += len(scan_result.errors)

                    results.scanned_results[full_path] = scan_result

        return results
