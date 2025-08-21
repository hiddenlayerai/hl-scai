#!/usr/bin/env python3
"""
Process HL-SCAI scan results and set GitHub Action outputs.
"""
import argparse
import json
import os
import sys
from pathlib import Path


def set_github_output(name: str, value: str):
    """Set a GitHub Action output variable."""
    output_file = os.environ.get("GITHUB_OUTPUT")
    if output_file:
        with open(output_file, "a") as f:
            f.write(f"{name}={value}\n")
    else:
        print(f"::set-output name={name}::{value}")


def process_results(input_file: str, fail_on_detection: bool):
    """Process the HL-SCAI results and set GitHub Action outputs."""
    # Read the results
    with open(input_file) as f:
        results = json.load(f)

    # Extract information
    ai_assets = results.get("ai_assets", [])
    models_found = len(ai_assets)

    # Set outputs
    set_github_output("models-found", str(models_found))
    set_github_output("report-path", input_file)

    # Create summary
    if models_found == 0:
        summary = "No AI models detected in the codebase."
    else:
        summary_lines = [f"Found {models_found} AI model(s) in the codebase:"]
        for asset in ai_assets[:10]:  # Limit to first 10 for summary
            metadata = asset.get("metadata", {})
            provider = metadata.get("provider", {}).get("name", "Unknown")
            model_name = metadata.get("name", "Unknown")
            usages = metadata.get("usages", [])
            usage_count = len(usages)
            summary_lines.append(f"- {provider}: {model_name} ({usage_count} usage(s))")

        if models_found > 10:
            summary_lines.append(f"... and {models_found - 10} more")

        summary = "\n".join(summary_lines)

    set_github_output("summary", summary.replace("\n", "%0A"))  # Escape newlines for GitHub

    # Set AI assets as JSON string
    set_github_output("ai-assets", json.dumps(ai_assets))

    # Print summary to console
    print("\n" + "=" * 60)
    print("HL-SCAI Scan Results")
    print("=" * 60)
    print(summary)
    print("=" * 60 + "\n")

    # Write GitHub step summary if available
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:
        with open(summary_file, "w") as f:
            f.write("## HL-SCAI Scan Results\n\n")
            if models_found == 0:
                f.write("✅ No AI models detected in the codebase.\n")
            else:
                f.write(f"⚠️ Found **{models_found}** AI model(s) in the codebase:\n\n")
                f.write("| Provider | Model | Usages |\n")
                f.write("|----------|-------|--------|\n")
                for asset in ai_assets:
                    metadata = asset.get("metadata", {})
                    provider = metadata.get("provider", {}).get("name", "Unknown")
                    model_name = metadata.get("name", "Unknown")
                    usage_count = len(metadata.get("usages", []))
                    f.write(f"| {provider} | {model_name} | {usage_count} |\n")

            f.write(f"\n📄 [View full report]({input_file})\n")

    # Fail if requested and models were found
    if fail_on_detection and models_found > 0:
        print(f"❌ Failing workflow: {models_found} AI models detected")
        sys.exit(1)

    return models_found


def main():
    parser = argparse.ArgumentParser(description="Process HL-SCAI results")
    parser.add_argument("--input", required=True, help="Input JSON file")
    parser.add_argument("--fail-on-detection", default="false", help="Fail if models detected")

    args = parser.parse_args()

    fail_on_detection = args.fail_on_detection.lower() == "true"

    try:
        process_results(args.input, fail_on_detection)
    except Exception as e:
        print(f"Error processing results: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
