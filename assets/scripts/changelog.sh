url="https://raw.githubusercontent.com/TuringLang/Turing.jl/master/HISTORY.md"

changelog_content=$(curl -s "$url")

cat << EOF > changelog.qmd
---
title: Changelog
repo-actions: false
---

$changelog_content
EOF
