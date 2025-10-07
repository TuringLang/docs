#!/bin/bash
# Add Jupyter notebook download links to rendered HTML files
# This adds a download link to the toc-actions section (next to "Edit this page" and "Report an issue")

set -e

echo "Adding notebook download links to HTML pages..."

# Link text variables
DOWNLOAD_TEXT="Download notebook"
COLAB_TEXT="Open in Colab"

# Colab URL configuration (can be overridden via environment variables)
COLAB_REPO="${COLAB_REPO:-TuringLang/docs}"
COLAB_BRANCH="${COLAB_BRANCH:-gh-pages}"
COLAB_PATH_PREFIX="${COLAB_PATH_PREFIX:-}"

# Find all HTML files that have corresponding .ipynb files
find _site/tutorials _site/usage _site/developers -name "index.html" 2>/dev/null | while read html_file; do
    dir=$(dirname "$html_file")
    ipynb_file="${dir}/index.ipynb"

    # Check if the corresponding .ipynb file exists
    if [ -f "$ipynb_file" ]; then
        # Check if link is already present
        if ! grep -q "$DOWNLOAD_TEXT" "$html_file"; then
            # Get relative path from _site/ directory
            relative_path="${html_file#_site/}"
            relative_path="${relative_path%/index.html}"

            # Construct Colab URL
            if [ -n "$COLAB_PATH_PREFIX" ]; then
                colab_url="https://colab.research.google.com/github/${COLAB_REPO}/blob/${COLAB_BRANCH}/${COLAB_PATH_PREFIX}/${relative_path}/index.ipynb"
            else
                colab_url="https://colab.research.google.com/github/${COLAB_REPO}/blob/${COLAB_BRANCH}/${relative_path}/index.ipynb"
            fi

            # Insert both download and Colab links AFTER the "Report an issue" link
            # The download="index.ipynb" attribute forces browser to download instead of navigate
            perl -i -pe "s/(<a href=\"[^\"]*issues\/new\"[^>]*><i class=\"bi[^\"]*\"><\/i>Report an issue<\/a><\/li>)/\$1<li><a href=\"index.ipynb\" class=\"toc-action\" download=\"index.ipynb\"><i class=\"bi bi-journal-code\"><\/i>$DOWNLOAD_TEXT<\/a><\/li><li><a href=\"$colab_url\" class=\"toc-action\" target=\"_blank\" rel=\"noopener\"><i class=\"bi bi-google\"><\/i>$COLAB_TEXT<\/a><\/li>/g" "$html_file"
            echo "  ✓ Added notebook links to $html_file"
        fi
    fi
done

echo "Notebook links added successfully!"