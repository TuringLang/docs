url="https://raw.githubusercontent.com/TuringLang/Turing.jl/main/HISTORY.md"

changelog_content=$(curl -s "$url")

cat << EOF > changelog.qmd
---
title: Changelog
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

$changelog_content
EOF
