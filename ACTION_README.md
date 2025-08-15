# HL-SCAI GitHub Action

A GitHub Action for scanning and analyzing AI model usage in Python codebases.

## Features

- 🔍 Automatically scan your codebase for AI model usage during CI/CD
- 📊 Generate detailed reports with model metadata and usage statistics
- 🚨 Optionally fail workflows when AI models are detected
- 💬 Comment on pull requests with scan results
- 📁 Save results as artifacts for further analysis
- 🔧 Configurable inputs and outputs for workflow integration

## Usage

### Basic Example

```yaml
name: AI Model Scan
on: [push, pull_request]

jobs:
  scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Scan for AI models
        uses: hiddenlayerai/hl-scai@main
        with:
          directory: .
```

### Advanced Example

```yaml
name: AI Model Compliance Check
on: [push, pull_request]

jobs:
  scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Scan for AI models
        id: scan
        uses: hiddenlayerai/hl-scai@main
        with:
          directory: ./src
          huggingface-token: ${{ secrets.HUGGINGFACE_TOKEN }}
          output-file: scan-results.json
          fail-on-detection: true

      - name: Process results
        if: always()
        run: |
          echo "Found ${{ steps.scan.outputs.models-found }} models"
          echo "${{ steps.scan.outputs.summary }}"
```

## Inputs

| Input | Description | Required | Default |
|-------|-------------|----------|---------|
| `directory` | Directory to scan for AI model usage | No | `.` |
| `huggingface-token` | HuggingFace API token for fetching model metadata | No | - |
| `output-file` | Path to save the analysis results (JSON format) | No | `hl-scai-results.json` |
| `fail-on-detection` | Fail the workflow if AI models are detected | No | `false` |

## Outputs

| Output | Description |
|--------|-------------|
| `models-found` | Number of AI models detected |
| `report-path` | Path to the generated report file |
| `summary` | Summary of detected AI models |
| `ai-assets` | JSON string of detected AI assets |

## Examples

### PR Comment Integration

Automatically comment on pull requests with scan results:

```yaml
- name: Scan for AI models
  id: scan
  uses: hiddenlayerai/hl-scai@main

- name: Comment on PR
  if: github.event_name == 'pull_request' && steps.scan.outputs.models-found > 0
  uses: actions/github-script@v6
  with:
    script: |
      github.rest.issues.createComment({
        issue_number: context.issue.number,
        owner: context.repo.owner,
        repo: context.repo.repo,
        body: `Found ${steps.scan.outputs.models-found} AI models:\n${steps.scan.outputs.summary}`
      });
```

### Policy Enforcement

Enforce AI usage policies in your codebase:

```yaml
- name: Scan for AI models
  id: scan
  uses: hiddenlayerai/hl-scai@main

- name: Check policies
  run: |
    models_count=${{ steps.scan.outputs.models-found }}
    if [ $models_count -gt 5 ]; then
      echo "Too many AI models detected ($models_count > 5)"
      exit 1
    fi
```

### Integration with Other Actions

Use the scan results in subsequent workflow steps:

```yaml
- name: Scan for AI models
  id: scan
  uses: hiddenlayerai/hl-scai@main

- name: Upload to security dashboard
  if: steps.scan.outputs.models-found > 0
  run: |
    # Use the ai-assets output for further processing
    echo '${{ steps.scan.outputs.ai-assets }}' | jq '.' > models.json
    # Upload to your security dashboard
    curl -X POST https://your-dashboard.com/api/upload \
      -H "Authorization: Bearer ${{ secrets.DASHBOARD_TOKEN }}" \
      -F "file=@models.json"
```

## Using in Your Repository

There are two ways to use this action:

### 1. Direct Usage (Recommended for testing)

Reference the action directly from the source repository:

```yaml
uses: hiddenlayerai/hl-scai@main
```

### 2. Publishing to GitHub Marketplace

To publish this action to the GitHub Marketplace:

1. Ensure your repository is public
2. Add topics: `actions`, `ai`, `security`, `python`
3. Create a release with a semantic version tag (e.g., `v1.0.0`)
4. Submit to the GitHub Marketplace

Once published, users can reference it as:

```yaml
uses: hiddenlayerai/hl-scai@v1
```

## Local Development

To test the action locally:

```bash
# Clone the repository
git clone https://github.com/hiddenlayerai/hl-scai.git
cd hl-scai

# Test the scanner
pip install -e .
hl-scai scan -d /path/to/test/project

# Test the action processing script
python .github/scripts/process-results.py --input test-results.json
```

## Security Considerations

- Store sensitive tokens (like `HUGGINGFACE_TOKEN`) as GitHub Secrets
- Review the scan results before making them public
- Consider the security implications of detected AI models
- Use the `fail-on-detection` flag for strict compliance

## Output JSON Schema

The action produces a JSON file with the following structure:

```json
{
  "metadata": {
    "id": "unique-uuid",
    "created_at": "2025-06-23T17:19:42.982359",
    "path": "scanned/directory/path"
  },
  "ast_scanner": {
    "path/to/file.py": {
      "results": [
        {
          "name": "gpt-4",
          "version": "latest",
          "source": "openai",
          "usage": "client.chat.completions.create",
          "system_prompt": null,
          "messages": [
            {
              "role": "user",
              "content": "Hello!"
            }
          ]
        }
      ],
      "errors": []
    }
  },
  "ai_assets": [
    {
      "metadata": {
        "name": "gpt-4",
        "provider": {
          "name": "openai",
          "origin": null
        },
        "version": "latest",
        "source": "openai",
        "usages": ["client.chat.completions.create"]
      },
      "details": {
        "task": "text-generation",
        "parameters": null,
        "library": "openai",
        "sequence_length": 128000,
        "chat_template": null
      },
      "artifacts": {
        "files": [],
        "datasets": [],
        "system_prompts": []
      },
      "license": {
        "name": "proprietary",
        "url": "https://openai.com/policies/"
      }
    }
  ],
  "usage": {
    "ast_scanner": {
      "scanned_files": 1,
      "total_results": 7,
      "total_errors": 0
    }
  }
}
```

### Schema Field Descriptions

- **metadata**: Information about the scan
  - `id`: Unique identifier for this scan
  - `created_at`: ISO timestamp of when the scan was performed
  - `path`: Directory that was scanned

- **ast_scanner**: Raw AST scan results by file
  - Each file contains `results` array and `errors` array
  - Results include model details, usage context, and messages

- **ai_assets**: Aggregated and enriched model information
  - `metadata`: Basic model information (name, provider, version, usages)
  - `details`: Extended information (task type, parameters, library)
  - `artifacts`: Associated files, datasets, and system prompts
  - `license`: Model license information

- **usage**: Statistics about the scan
  - `scanned_files`: Number of Python files scanned
  - `total_results`: Total number of AI model usages found
  - `total_errors`: Number of errors during scanning

### Accessing Output in Workflows

You can access specific fields from the JSON output:

```yaml
- name: Parse JSON output
  id: parse
  run: |
    # Get all model names
    models=$(jq -r '.ai_assets[].metadata.name' ${{ steps.scan.outputs.report-path }})
    echo "Models: $models"

    # Check for specific provider
    has_openai=$(jq -r '.ai_assets[] | select(.metadata.provider.name == "openai") | .metadata.name' ${{ steps.scan.outputs.report-path }})
    if [ ! -z "$has_openai" ]; then
      echo "OpenAI models found: $has_openai"
    fi

    # Get system prompts
    prompts=$(jq -r '.ai_assets[].artifacts.system_prompts[]' ${{ steps.scan.outputs.report-path }})
    echo "System prompts: $prompts"
```

## Troubleshooting

### Action fails with "command not found"

Ensure the action has proper permissions and Python 3.12+ is available.

### No models detected when expected

1. Check that the scanned directory contains Python files
2. Verify the AI usage patterns match supported providers
3. Review the action logs for any errors

### HuggingFace metadata not fetched

Provide a valid `HUGGINGFACE_TOKEN` in your secrets:

```yaml
with:
  huggingface-token: ${{ secrets.HUGGINGFACE_TOKEN }}
```

## Contributing

Contributions are welcome! Please see the main [CONTRIBUTING.md](CONTRIBUTING.md) guide.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
