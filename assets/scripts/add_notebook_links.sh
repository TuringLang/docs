#!/bin/bash
# Add Jupyter notebook download links to rendered HTML files
# This injects download links for .ipynb files into the HTML pages

set -e

echo "Adding notebook download links to HTML pages..."

# Find all HTML files that have corresponding .ipynb files
find _site/tutorials _site/usage _site/developers -name "index.html" 2>/dev/null | while read html_file; do
    dir=$(dirname "$html_file")
    ipynb_file="${dir}/index.ipynb"

    # Check if the corresponding .ipynb file exists
    if [ -f "$ipynb_file" ]; then
        # Check if link is already present
        if ! grep -q 'quarto-alternate-formats' "$html_file"; then
            # Use perl for portable in-place editing (works on both Linux and macOS)
            perl -i -pe 'BEGIN{undef $/;} s/(<main class="content"[^>]*>)/$1\n<div class="quarto-alternate-formats"><h2>Other Formats<\/h2><ul><li><a href="index.ipynb"><i class="bi bi-journal-code"><\/i>Jupyter<\/a><\/li><\/ul><\/div>\n/sm' "$html_file"
            echo "  ✓ Added notebook link to $html_file"
        fi
    fi
done

echo "Notebook links added successfully!"