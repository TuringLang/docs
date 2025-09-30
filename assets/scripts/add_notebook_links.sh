#!/bin/bash
# Add Jupyter notebook download links to rendered HTML files
# This adds a download link to the toc-actions section (next to "Edit this page" and "Report an issue")

set -e

echo "Adding notebook download links to HTML pages..."

# Find all HTML files that have corresponding .ipynb files
find _site/tutorials _site/usage _site/developers -name "index.html" 2>/dev/null | while read html_file; do
    dir=$(dirname "$html_file")
    ipynb_file="${dir}/index.ipynb"

    # Check if the corresponding .ipynb file exists
    if [ -f "$ipynb_file" ]; then
        # Check if link is already present
        if ! grep -q 'Download notebook' "$html_file"; then
            # Add as a new <li> item in the toc-actions <ul>
            # This appears alongside "Edit this page" and "Report an issue"
            # We need to add it to BOTH occurrences (sidebar and mobile footer)
            perl -i -pe 's/(<div class="toc-actions"><ul>)/$1<li><a href="index.ipynb" class="toc-action" download><i class="bi bi-journal-code"><\/i>Download notebook<\/a><\/li>/g' "$html_file"
            echo "  ✓ Added notebook link to $html_file"
        fi
    fi
done

echo "Notebook links added successfully!"