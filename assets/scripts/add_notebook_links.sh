#!/bin/bash
# Add Jupyter notebook download links to rendered HTML files
# This adds a download link to the toc-actions section (next to "Edit this page" and "Report an issue")

set -e

echo "Adding notebook download links to HTML pages..."

# Link text variable
LINK_TEXT="Download notebook"

# Find all HTML files that have corresponding .ipynb files
find _site/tutorials _site/usage _site/developers -name "index.html" 2>/dev/null | while read html_file; do
    dir=$(dirname "$html_file")
    ipynb_file="${dir}/index.ipynb"

    # Check if the corresponding .ipynb file exists
    if [ -f "$ipynb_file" ]; then
        # Check if link is already present
        if ! grep -q "$LINK_TEXT" "$html_file"; then
            # Insert the notebook link AFTER the "Report an issue" link
            # This ensures it goes in the right place in the sidebar toc-actions
            # The download="index.ipynb" attribute forces browser to download instead of navigate
            perl -i -pe "s/(<a href=\"[^\"]*issues\/new\"[^>]*><i class=\"bi[^\"]*\"><\/i>Report an issue<\/a><\/li>)/\$1<li><a href=\"index.ipynb\" class=\"toc-action\" download=\"index.ipynb\"><i class=\"bi bi-journal-code\"><\/i>$LINK_TEXT<\/a><\/li>/g" "$html_file"
            echo "  ✓ Added notebook link to $html_file"
        fi
    fi
done

echo "Notebook links added successfully!"