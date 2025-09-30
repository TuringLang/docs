#!/usr/bin/env python3
"""
Convert Quarto .qmd files to Jupyter .ipynb notebooks with proper cell structure.
Each code block becomes a code cell, and markdown content becomes markdown cells.
"""

import sys
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional


class QmdToIpynb:
    def __init__(self, qmd_path: str):
        self.qmd_path = Path(qmd_path)
        self.cells: List[Dict[str, Any]] = []
        self.kernel_name = "julia-1.11"  # Default kernel

    def parse(self) -> None:
        """Parse the .qmd file and extract cells."""
        with open(self.qmd_path, 'r', encoding='utf-8') as f:
            content = f.read()

        lines = content.split('\n')
        i = 0

        # Skip YAML frontmatter
        if lines[0].strip() == '---':
            i = 1
            while i < len(lines) and lines[i].strip() != '---':
                # Check for engine specification
                if lines[i].strip().startswith('engine:'):
                    engine = lines[i].split(':', 1)[1].strip()
                    if engine == 'julia':
                        self.kernel_name = "julia-1.11"
                    elif engine == 'python':
                        self.kernel_name = "python3"
                i += 1
            i += 1  # Skip the closing ---

        # Parse the rest of the document
        current_markdown = []

        while i < len(lines):
            line = lines[i]

            # Check for code block start
            code_block_match = re.match(r'^```\{(\w+)\}', line)
            if code_block_match:
                # Save any accumulated markdown
                if current_markdown:
                    self._add_markdown_cell(current_markdown)
                    current_markdown = []

                # Extract code block
                lang = code_block_match.group(1)
                i += 1
                code_lines = []
                cell_options = []

                # Collect code and options
                while i < len(lines) and not lines[i].startswith('```'):
                    if lines[i].startswith('#|'):
                        cell_options.append(lines[i])
                    else:
                        code_lines.append(lines[i])
                    i += 1

                # Add code cell (with options as comments at the top)
                full_code = cell_options + code_lines
                self._add_code_cell(full_code, lang)

                i += 1  # Skip closing ```
            else:
                # Accumulate markdown
                current_markdown.append(line)
                i += 1

        # Add any remaining markdown
        if current_markdown:
            self._add_markdown_cell(current_markdown)

    def _add_markdown_cell(self, lines: List[str]) -> None:
        """Add a markdown cell, stripping leading/trailing empty lines."""
        # Strip leading empty lines
        while lines and not lines[0].strip():
            lines.pop(0)

        # Strip trailing empty lines
        while lines and not lines[-1].strip():
            lines.pop()

        if not lines:
            return

        content = '\n'.join(lines)
        cell = {
            "cell_type": "markdown",
            "metadata": {},
            "source": content
        }
        self.cells.append(cell)

    def _add_code_cell(self, lines: List[str], lang: str) -> None:
        """Add a code cell."""
        content = '\n'.join(lines)

        # For non-Julia code blocks (like dot/graphviz), add as markdown with code formatting
        # since Jupyter notebooks typically use Julia kernel for these docs
        if lang != 'julia' and lang != 'python':
            # Convert to markdown with code fence
            markdown_content = f"```{lang}\n{content}\n```"
            cell = {
                "cell_type": "markdown",
                "metadata": {},
                "source": markdown_content
            }
        else:
            cell = {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": content
            }

        self.cells.append(cell)

    def to_notebook(self) -> Dict[str, Any]:
        """Convert parsed cells to Jupyter notebook format."""
        notebook = {
            "cells": self.cells,
            "metadata": {
                "kernelspec": {
                    "display_name": "Julia 1.11",
                    "language": "julia",
                    "name": self.kernel_name
                },
                "language_info": {
                    "file_extension": ".jl",
                    "mimetype": "application/julia",
                    "name": "julia",
                    "version": "1.11.0"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 5
        }
        return notebook

    def write(self, output_path: str) -> None:
        """Write the notebook to a file."""
        notebook = self.to_notebook()
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=2, ensure_ascii=False)


def main():
    if len(sys.argv) < 2:
        print("Usage: qmd_to_ipynb.py <input.qmd> [output.ipynb]")
        sys.exit(1)

    qmd_path = sys.argv[1]

    # Determine output path
    if len(sys.argv) >= 3:
        ipynb_path = sys.argv[2]
    else:
        ipynb_path = Path(qmd_path).with_suffix('.ipynb')

    # Convert
    converter = QmdToIpynb(qmd_path)
    converter.parse()
    converter.write(ipynb_path)

    print(f"Converted {qmd_path} -> {ipynb_path}")


if __name__ == "__main__":
    main()