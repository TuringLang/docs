#!/bin/bash
# Generate Jupyter notebooks from .qmd files without re-executing code
# This script converts .qmd files to .ipynb format with proper cell structure

set -e

echo "Generating Jupyter notebooks from .qmd files..."

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Find all .qmd files in tutorials, usage, and developers directories
find tutorials usage developers -name "index.qmd" | while read qmd_file; do
    dir=$(dirname "$qmd_file")
    ipynb_file="${dir}/index.ipynb"

    echo "Converting $qmd_file to $ipynb_file"

    # Convert qmd to ipynb using our custom Python script
    python3 "${SCRIPT_DIR}/qmd_to_ipynb.py" "$qmd_file" "$ipynb_file"

    # Check if conversion was successful
    if [ -f "$ipynb_file" ]; then
        # Move the notebook to the _site directory
        mkdir -p "_site/${dir}"
        cp "$ipynb_file" "_site/${ipynb_file}"
        echo "  ✓ Generated _site/${ipynb_file}"
    else
        echo "  ✗ Failed to generate $ipynb_file"
    fi
done

echo "Notebook generation complete!"
echo "Generated notebooks are in _site/ directory alongside HTML files"