#!/bin/bash

REPO_URL="https://api.github.com/repos/TuringLang/Turing.jl/tags"

# Fetch the tags
TAGS=$(curl -s $REPO_URL | grep 'name' | sed 's/.*: "\(.*\)",/\1/')

# Filter out pre-release version tags (e.g., 0.33.0-rc.1) and keep only stable version tags
STABLE_TAGS=$(echo "$TAGS" | grep -Eo 'v[0-9]+\.[0-9]+\.[0-9]+$')

# Find the latest version (including bug fix versions)
LATEST_VERSION=$(echo "$STABLE_TAGS" | head -n 1)

# Find the latest minor version (without bug fix)
STABLE_VERSION=$(echo "$STABLE_TAGS" | grep -Eo 'v[0-9]+\.[0-9]+(\.0)?$' | head -n 1)

# Filter out bug fix version tags from STABLE_TAGS to get only minor version tags
MINOR_TAGS=$(echo "$STABLE_TAGS" | grep -Eo 'v[0-9]+\.[0-9]+(\.0)?$')

# Set the minimum version to include in the "Previous Versions" section
MIN_VERSION="v0.31.0"

# Remove bug-fix version number from the display of a version
remove_bugfix() {
  echo "$1" | sed -E 's/\.[0-9]$//'
}

# versions.qmd file will be generated from this content
VERSIONS_CONTENT="---
pagetitle: Versions
repo-actions: false
include-in-header:
  - text: |
      <style>
        a {
          text-decoration: none;
        }
        a:hover {
          text-decoration: underline;
        }
      </style>
---

# Latest Version
| | | |
| --- | --- | --- |
| $(remove_bugfix "$LATEST_VERSION") | [Documentation](versions/${LATEST_VERSION}/) | [Changelog](changelog.qmd) |

# Previous Versions
| | |
| --- | --- |
"
# Add previous versions, excluding the latest and stable versions
for MINOR_TAG in $MINOR_TAGS; do
  if [ "$MINOR_TAG" != "$LATEST_VERSION" ] && [ "$MINOR_TAG" != "$STABLE_VERSION" ] && [ "$MINOR_TAG" \> "$MIN_VERSION" ]; then
    # Find the latest bug fix version for the current minor version
    LATEST_BUG_FIX=$(echo "$STABLE_TAGS" | grep "^${MINOR_TAG%.*}" | sort -r | head -n 1)
    # Remove trailing .0 from display version
    DISPLAY_MINOR_TAG=$(remove_bugfix "$MINOR_TAG")
    VERSIONS_CONTENT="${VERSIONS_CONTENT}| ${DISPLAY_MINOR_TAG} | [Documentation](versions/${LATEST_BUG_FIX}/) |
"
  fi
done

# Add the Archived Versions section manually
VERSIONS_CONTENT="${VERSIONS_CONTENT}
# Archived Versions
Documentation for archived versions is available on our deprecated documentation site.

| | |
| --- | --- |
| v0.31 | [Documentation](../v0.31.4/) |
| v0.30 | [Documentation](../v0.30.9/) |
| v0.29 | [Documentation](../v0.29.3/) |
| v0.28 | [Documentation](../v0.28.3/) |
| v0.27 | [Documentation](../v0.27.0/) |
| v0.26 | [Documentation](../v0.26.6/) |
| v0.25 | [Documentation](../v0.25.3/) |
| v0.24 | [Documentation](../v0.24.4/) |
"

# Write the content to the versions.qmd file
echo "$VERSIONS_CONTENT" > versions.qmd
