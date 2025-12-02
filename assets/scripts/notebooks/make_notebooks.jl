using JSON
import JuliaSyntax

abstract type Cell end
struct JuliaCodeCell <: Cell
    code::String
end
function JSON.lower(cell::JuliaCodeCell)
    return Dict(
        "cell_type" => "code",
        "source" => cell.code,
        "metadata" => Dict(),
        "outputs" => Any[],
        "execution_count" => nothing,
    )
end
struct MarkdownCell <: Cell
    content::String
end
function JSON.lower(cell::MarkdownCell)
    return Dict(
        "cell_type" => "markdown",
        "source" => cell.content,
        "metadata" => Dict(),
    )
end

struct Notebook
    cells::Vector{Cell}
end
function JSON.lower(nb::Notebook)
    return Dict(
        "cells" => [JSON.lower(cell) for cell in nb.cells],
        "metadata" => Dict(
            "kernelspec" => Dict(
                "display_name" => "Julia",
                "language" => "julia",
                "name" => "julia"
            ),
            "language_info" => Dict(
                "file_extension" => ".jl",
                "mimetype" => "application/julia",
                "name" => "julia"
            )
        ),
        "nbformat" => 4,
        "nbformat_minor" => 5
    )
end

"""
    parse_cells(qmd_path::String)::Notebook

Parse a .qmd file. Returns a vector of `Cell` objects representing the code and markdown
cells, as well as a set of imported packages found in Julia code cells.
"""
function parse_cells(qmd_path::String)::Notebook
    content = read(qmd_path, String)

    # Remove YAML front matter.
    yaml_front_matter_regex = r"^---\n(.*?)\n---\n"s
    content = replace(content, yaml_front_matter_regex => "")
    content = strip(content)

    packages = Set{Symbol}()
    # Extract code blocks.
    executable_content_regex = r"```\{(\w+)\}(.*?)```"s
    # These are Markdown cells.
    markdown_cell_contents = split(content, executable_content_regex; keepempty=true)
    # These are code cells
    code_cell_contents = collect(eachmatch(executable_content_regex, content))
    # Because we set `keepempty=true`, `splits` will always have one more element than `matches`.
    # We can interleave them to reconstruct the document structure.
    cells = Cell[]
    for (i, md_content) in enumerate(markdown_cell_contents)
        md_content = strip(md_content)
        if !isempty(md_content)
            push!(cells, MarkdownCell(md_content))
        end
        if i <= length(code_cell_contents)
            match = code_cell_contents[i]
            lang = match.captures[1]
            code = strip(match.captures[2])
            if lang == "julia"
                cell = JuliaCodeCell(code)
                push!(cells, cell)
                union!(packages, extract_imports(cell))
            else
                # There are some code cells that are not Julia for example
                # dot and mermaid. You can see what cells there are with
                #     git grep -E '```\{.+\}' | grep -v julia
                # For these cells we'll just convert to Markdown.
                push!(cells, MarkdownCell("```$lang\n$code\n```"))
            end
        end
    end

    # Prepend a cell to install the necessary packages
    imports_as_string = join(["\"" * string(pkg) * "\"" for pkg in packages], ", ")
    new_cell = JuliaCodeCell("# Install necessary dependencies.\nusing Pkg\nPkg.activate(; temp=true)\nPkg.add([$imports_as_string])")
    cells = [new_cell, cells...]

    # And we're done!
    return Notebook(cells)
end

"""
    extract_imports(cell::JuliaCodeCell)::Set{Symbol}

Extract all packages that are imported inside `cell`.
"""
function extract_imports(cell::JuliaCodeCell)::Set{Symbol}
    toplevel_expr = JuliaSyntax.parseall(Expr, cell.code)
    imports = Set{Symbol}()
    for expr in toplevel_expr.args
        if expr isa Expr && (expr.head == :using || expr.head == :import)
            for arg in expr.args
                if arg isa Expr && arg.head == :.
                    push!(imports, arg.args[1])
                elseif arg isa Expr && arg.head == :(:)
                    subarg = arg.args[1]
                    if subarg isa Expr && subarg.head == :.
                        push!(imports, subarg.args[1])
                    end
                elseif arg isa Expr && arg.head == :as
                    subarg = arg.args[1]
                    if subarg isa Expr && subarg.head == :.
                        push!(imports, subarg.args[1])
                    elseif subarg isa Symbol
                        push!(imports, subarg)
                    end
                end
            end
        end
    end
    return imports
end

function convert_qmd_to_ipynb(in_qmd_path::String, out_ipynb_path::String)
    @info "converting $in_qmd_path to $out_ipynb_path..."
    notebook = parse_cells(in_qmd_path)
    JSON.json(out_ipynb_path, notebook; pretty=true)
    @info "done."
end

function main(args)
    if length(args) == 0
        # Get the list of .qmd files from the _quarto.yml file. This conveniently
        # also checks that we are at the repo root.
        try
            quarto_config = split(read("_quarto.yml", String), '\n')
            qmd_files = String[]
            for line in quarto_config
                m = match(r"^\s*-\s*(.+\.qmd)\s*$", line)
                if m !== nothing
                    push!(qmd_files, m.captures[1])
                end
            end
            for file in qmd_files
                dir = "_site/" * dirname(file)
                base = replace(basename(file), r"\.qmd$" => ".ipynb")
                isdir(dir) || mkpath(dir)  # mkpath is essentially mkdir -p
                out_ipynb_path = joinpath(dir, base)
                convert_qmd_to_ipynb(file, out_ipynb_path)
            end
        catch e
            if e isa SystemError
                error("Could not find _quarto.yml; please run this script from the repo root.")
            else
                rethrow(e)
            end
        end

    elseif length(args) == 2
        in_qmd_path, out_ipynb_path = args
        convert_qmd_to_ipynb(in_qmd_path, out_ipynb_path)
    end
end
@main
