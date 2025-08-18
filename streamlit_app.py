import json
import os
import subprocess

import pandas as pd
import plotly.express as px
import streamlit as st

# Page configuration
st.set_page_config(page_title="Model Scanner Analysis", page_icon="🔍", layout="wide")

# Title and description
st.title("🔍 Model Scanner Analysis")
st.markdown("Analyze your codebase to discover AI model usage patterns")

# Sidebar for directory selection
with st.sidebar:
    st.header("Configuration")

    # Directory input
    directory_path = st.text_input(
        "Directory Path", value="test_repo", help="Enter the path to the directory you want to analyze"
    )

    # Run analysis button
    run_analysis = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

# Main content area
if run_analysis and directory_path:
    # Check if directory exists
    if not os.path.exists(directory_path):
        st.error(f"Directory '{directory_path}' does not exist!")
    else:
        with st.spinner(f"Analyzing directory: {directory_path}..."):
            try:
                # Run the analysis command
                result = subprocess.run(
                    ["python", "-m", "hl_scai.entrypoint.cli", "scan", "-d", directory_path],
                    capture_output=True,
                    text=True,
                    check=True,
                )

                # Parse the JSON output
                analysis_data = json.loads(result.stdout)

                # Store in session state
                st.session_state["analysis_data"] = analysis_data
                st.session_state["analyzed_directory"] = directory_path

                st.success(f"✅ Analysis completed for: {directory_path}")

            except subprocess.CalledProcessError as e:
                st.error(f"Error running analysis: {e.stderr}")
            except json.JSONDecodeError as e:
                st.error(f"Error parsing analysis results: {e}")
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")

# Display results if analysis data exists
if "analysis_data" in st.session_state:
    data = st.session_state["analysis_data"]

    # Display directory analyzed
    st.info(f"📁 Showing results for: **{st.session_state['analyzed_directory']}**")

    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Statistics", "🔍 File Details", "📋 Model Details"])

    with tab1:
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Files Scanned", data["usage"]["ast_scanner"]["scanned_files"])

        with col2:
            st.metric("Total Model Usages", data["usage"]["ast_scanner"]["total_results"])

        with col3:
            st.metric("Unique Models", len(data["models"]))

        with col4:
            st.metric("Total Errors", data["usage"]["ast_scanner"]["total_errors"])

        # Model sources distribution
        st.subheader("Model Sources Distribution")

        # Count models by source
        sources = {}
        for model in data["models"]:
            source = model["metadata"]["source"]
            sources[source] = sources.get(source, 0) + 1

        # Create pie chart
        fig_pie = px.pie(values=list(sources.values()), names=list(sources.keys()), title="Models by Source", hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)

    with tab2:
        # Statistics tab
        st.subheader("Model Usage Statistics")

        # Create a DataFrame of model usages
        model_usage_data = []
        for model in data["models"]:
            usage_count = len(model["metadata"]["usages"])
            model_usage_data.append(
                {
                    "Model": model["metadata"]["name"],
                    "Source": model["metadata"]["source"],
                    "Usage Count": usage_count,
                    "Version": model["metadata"]["version"],
                }
            )

        df_models = pd.DataFrame(model_usage_data)

        # Bar chart of top used models
        if not df_models.empty:
            df_top_models = df_models.nlargest(10, "Usage Count")
            fig_bar = px.bar(
                df_top_models,
                x="Model",
                y="Usage Count",
                color="Source",
                title="Top 10 Most Used Models",
                labels={"Usage Count": "Number of Usages"},
            )
            fig_bar.update_xaxes(tickangle=-45)
            st.plotly_chart(fig_bar, use_container_width=True)

        # Model table
        st.subheader("All Models")
        st.dataframe(df_models, use_container_width=True)

    with tab3:
        # File details tab
        st.subheader("File Analysis Details")

        # Create expandable sections for each file
        for file_path, file_data in data["ast_scanner"].items():
            with st.expander(f"📄 {file_path}"):
                if file_data["errors"]:
                    st.error(f"Errors: {', '.join(file_data['errors'])}")

                if file_data["results"]:
                    # Create a DataFrame for the results
                    results_df = pd.DataFrame(file_data["results"])
                    st.dataframe(results_df, use_container_width=True)
                else:
                    st.info("No models found in this file")

    with tab4:
        # Model details tab
        st.subheader("Detailed Model Information")

        # Search/filter
        search_term = st.text_input("🔍 Search models", placeholder="Enter model name...")

        # Filter models based on search
        filtered_models = data["models"]
        if search_term:
            filtered_models = [m for m in data["models"] if search_term.lower() in m["metadata"]["name"].lower()]

        # Display filtered models
        for model in filtered_models:
            with st.expander(f"🤖 {model['metadata']['name']}"):
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Metadata:**")
                    st.json(
                        {
                            "Provider": model["metadata"]["provider"]["name"],
                            "Source": model["metadata"]["source"],
                            "Version": model["metadata"]["version"],
                            "Usages": model["metadata"]["usages"],
                        }
                    )

                with col2:
                    st.markdown("**Details:**")
                    details = model.get("details", {})
                    if any(v is not None for v in details.values()):
                        st.json({k: v for k, v in details.items() if v is not None})
                    else:
                        st.info("No additional details available")

                # License information
                if model.get("license", {}).get("name"):
                    st.markdown(f"**License:** {model['license']['name']}")
                    if model["license"].get("url"):
                        st.markdown(f"[View License]({model['license']['url']})")

                # File artifacts section
                artifacts = model.get("artifacts", {})
                files = artifacts.get("files", [])
                if files:
                    st.markdown("**📁 File Artifacts:**")

                    # Summary metrics
                    col3, col4, col5 = st.columns(3)
                    with col3:
                        st.metric("Total Files", len(files))
                    with col4:
                        total_size = sum(f.get("size", 0) for f in files)
                        # Convert to human readable format
                        if total_size > 1e9:
                            size_str = f"{total_size / 1e9:.2f} GB"
                        elif total_size > 1e6:
                            size_str = f"{total_size / 1e6:.2f} MB"
                        elif total_size > 1e3:
                            size_str = f"{total_size / 1e3:.2f} KB"
                        else:
                            size_str = f"{total_size} B"
                        st.metric("Total Size", size_str)
                    with col5:
                        # Count file types
                        extensions = {}
                        for f in files:
                            ext = f["name"].split(".")[-1] if "." in f["name"] else "no extension"
                            extensions[ext] = extensions.get(ext, 0) + 1
                        st.metric("File Types", len(extensions))

                    # File list with checkbox to show/hide
                    show_files = st.checkbox(
                        f"Show file list ({len(files)} files)", key=f"files_{model['metadata']['name']}"
                    )
                    if show_files:
                        file_data = []
                        for f in files:
                            size = f.get("size", 0)
                            if size > 1e9:
                                size_str = f"{size / 1e9:.2f} GB"
                            elif size > 1e6:
                                size_str = f"{size / 1e6:.2f} MB"
                            elif size > 1e3:
                                size_str = f"{size / 1e3:.2f} KB"
                            else:
                                size_str = f"{size} B"

                            file_data.append(
                                {
                                    "File Name": f.get("name", "Unknown"),
                                    "Size": size_str,
                                    "SHA1": f.get("sha1", "")[:8] + "..." if f.get("sha1") else "",
                                    "SHA256": f.get("sha256", "")[:8] + "..." if f.get("sha256") else "",
                                }
                            )

                        df_files = pd.DataFrame(file_data)
                        st.dataframe(df_files, use_container_width=True, height=300)

                # Datasets section
                datasets = artifacts.get("datasets", [])
                if datasets:
                    st.markdown("**📊 Datasets:**")
                    for dataset in datasets:
                        st.write(f"- {dataset}")

# Footer
st.markdown("---")
st.markdown("🔍 Model Scanner - Analyze AI model usage in your codebase")
