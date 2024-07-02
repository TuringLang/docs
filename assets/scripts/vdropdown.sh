#!/bin/sh

# Read the current version from the VERSION file
CURRENT_VERSION=$(cat VERSION)

# Define the versions section to be inserted
VERSIONS_SECTION=$(cat << EOF
# The versions will be inserted here by the script
    right:
      - text: "$CURRENT_VERSION"
        menu:
          - text: Changelog
            href: changelog.qmd
          - text: All Versions
            href: https://turinglang.org/docs/versions.html
# The versions list ends here
EOF
)

# Use awk to replace the existing versions section between the comments
awk -v versions="$VERSIONS_SECTION" '
  BEGIN { in_versions = 0 }
  /# The versions will be inserted here by the script/ {
    print versions
    in_versions = 1
    next
  }
  /# The versions list ends here/ {
    in_versions = 0
    next
  }
  !in_versions { print $0 }
' _quarto.yml > _quarto.yml.tmp && mv _quarto.yml.tmp _quarto.yml
