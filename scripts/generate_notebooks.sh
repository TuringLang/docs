#!/bin/bash

# Script to generate Jupyter notebooks from QMD files without executing code
# This runs as part of the Quarto post-render process

echo "Generating Jupyter notebooks from QMD files..."

# Check if quarto is available
if ! command -v quarto &> /dev/null; then
    echo "Warning: quarto command not found. Skipping notebook generation."
    exit 0
fi

# Find all tutorial QMD files that have Julia engine
tutorial_files=$(find tutorials -name "index.qmd" -type f 2>/dev/null | grep -v "_site" || true)

for qmd_file in $tutorial_files; do
    # Check if file contains "engine: julia"
    if [ -f "$qmd_file" ] && grep -q "engine: julia" "$qmd_file" 2>/dev/null; then
        # Get the directory path
        dir_path=$(dirname "$qmd_file")
        base_name=$(basename "$dir_path")

        # Generate notebook without executing code
        echo "Converting $qmd_file to notebook..."

        # Use quarto to render to ipynb format without execution
        # Using --no-execute to skip code execution entirely
        if quarto render "$qmd_file" --to ipynb --no-execute 2>/dev/null; then
            # The notebook will be created as index.ipynb in the same directory
            ipynb_file="${dir_path}/index.ipynb"

            if [ -f "$ipynb_file" ]; then
                # Create the _site directory structure if it doesn't exist
                site_dir="_site/${dir_path}"
                mkdir -p "$site_dir"

                # Copy the notebook to the _site directory
                cp "$ipynb_file" "$site_dir/index.ipynb"
                echo "Created notebook: $site_dir/index.ipynb"

                # Remove the local ipynb file (keep only in _site)
                rm "$ipynb_file"
            fi
        else
            echo "Warning: Failed to convert $qmd_file to notebook. Skipping."
        fi
    fi
done

# Also handle special pages if they have Julia code
special_pages=("core-functionality/index.qmd" "faq/index.qmd")

for qmd_file in "${special_pages[@]}"; do
    if [ -f "$qmd_file" ] && grep -q "engine: julia" "$qmd_file" 2>/dev/null; then
        echo "Converting $qmd_file to notebook..."

        # Generate notebook without executing
        if quarto render "$qmd_file" --to ipynb --no-execute 2>/dev/null; then
            # Get paths
            dir_path=$(dirname "$qmd_file")
            ipynb_file="${dir_path}/index.ipynb"

            if [ -f "$ipynb_file" ]; then
                site_dir="_site/${dir_path}"
                mkdir -p "$site_dir"
                cp "$ipynb_file" "$site_dir/index.ipynb"
                echo "Created notebook: $site_dir/index.ipynb"
                rm "$ipynb_file"
            fi
        else
            echo "Warning: Failed to convert $qmd_file to notebook. Skipping."
        fi
    fi
done

echo "Notebook generation complete!"