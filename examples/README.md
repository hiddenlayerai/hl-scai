# HL-SCAI GitHub Action Examples

This directory contains example workflows demonstrating how to use the HL-SCAI GitHub Action in various scenarios.

## 📁 Directory Structure

- `workflows/` - Example GitHub Actions workflows
  - `example-basic.yml` - Basic usage example (from this repository)
  - `example-advanced.yml` - Advanced features demonstration (from this repository)
  - `external-repo-basic.yml` - Basic usage for external repositories
  - `external-repo-pr-comment.yml` - PR commenting example
  - `external-repo-policy.yml` - Policy enforcement example
  - `external-repo-scheduled.yml` - Scheduled scanning example

## 🚀 Using These Examples

### For External Repositories

1. Choose an example workflow that fits your needs
2. Copy the workflow file to your repository's `.github/workflows/` directory
3. Customize the configuration as needed
4. Commit and push to trigger the workflow

### Quick Start

The simplest way to get started is with the basic example:

```bash
# In your repository
mkdir -p .github/workflows
curl -o .github/workflows/ai-scan.yml https://raw.githubusercontent.com/hiddenlayerai/hl-scai/main/examples/workflows/external-repo-basic.yml
```

## 📋 Example Descriptions

### 1. Basic Usage (`external-repo-basic.yml`)

Simple AI model scanning on push and PR events.

**Use when:** You want basic AI model detection without additional features.

### 2. PR Comments (`external-repo-pr-comment.yml`)

Automatically comments scan results on pull requests.

**Use when:** You want visibility of AI model changes during code review.

**Required permissions:**
```yaml
permissions:
  contents: read
  pull-requests: write
```

### 3. Policy Enforcement (`external-repo-policy.yml`)

Enforces AI usage policies with customizable rules.

**Use when:** You need to enforce specific AI model usage policies.

**Customizable policies:**
- Maximum number of models allowed
- Approved/denied model lists
- Provider restrictions
- License compatibility checks

### 4. Scheduled Audits (`external-repo-scheduled.yml`)

Regular scheduled scans with audit reports.

**Use when:** You want periodic AI usage audits for compliance.

**Features:**
- Weekly scheduled scans
- Markdown audit reports
- Long-term artifact retention
- Optional notifications

## 🔧 Common Customizations

### Scanning Specific Directories

```yaml
with:
  directory: ./src  # Only scan the src directory
```

### Using HuggingFace Token

```yaml
with:
  huggingface-token: ${{ secrets.HUGGINGFACE_TOKEN }}
```

### Failing on Detection

```yaml
with:
  fail-on-detection: true  # Fail the workflow if any models are found
```

### Custom Output File

```yaml
with:
  output-file: my-scan-results.json
```

## 💡 Tips

1. **Secrets Management**: Store sensitive tokens as GitHub Secrets
2. **Artifact Retention**: Adjust retention days based on your compliance needs
3. **Notifications**: Integrate with Slack, email, or other notification systems
4. **Branch Protection**: Use the action as a required status check

## 🔗 Resources

- [HL-SCAI Documentation](https://github.com/hiddenlayerai/hl-scai)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Action Configuration Reference](../ACTION_README.md)
