#!/bin/bash

# Define the repository URL
REPO_URL="https://api.github.com/repos/TuringLang/Turing.jl/tags"

# Fetch the tags from the repository
TAGS=$(curl -s $REPO_URL | grep 'name' | sed 's/.*: "\(.*\)",/\1/' | sort -r)

# Filter out bug fix versions (e.g., 0.33.1) and keep only minor versions (e.g., 0.33)
MINOR_TAGS=$(echo "$TAGS" | grep -Eo 'v[0-9]+\.[0-9]+(\.0)?$' | sort -r | uniq)

# Start building the new versions section
VERSIONS_SECTION=""
for TAG in $MINOR_TAGS; do
  VERSIONS_SECTION=$(cat << EOF
$VERSIONS_SECTION
              - text: $TAG
                href: versions/$TAG/
EOF
  )
done

# Use awk to replace the existing versions section between the comments
awk -v versions="$VERSIONS_SECTION" '
  BEGIN { in_versions = 0 }
  /# Auto-generated versions section, do not remove these comments/ {
    print $0
    print versions
    in_versions = 1
    next
  }
  /# The versions list ends here/ {
    in_versions = 0
  }
  !in_versions { print $0 }
' _quarto.yml > _quarto.yml.tmp && mv _quarto.yml.tmp _quarto.yml
