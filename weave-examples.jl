using Documenter, Turing
using LibGit2: clone
using Weave

# Get paths.
markdown_path = joinpath(@__DIR__, "markdown")

function polish_latex(path::String)
    txt = open(f -> read(f, String), path)
    open(path, "w+") do f
        txt = replace(txt, raw"$$" => raw"\$\$")

        inline = r"(?<=[^\\])\$"
        txt = replace(txt, inline => raw"$$")

        write(f, txt)
    end
end

function add_yaml(path)
    lines = readlines(path, keep=true)

    title = ""
    permalink = "/:collection/:name/"

    # Find the ID sections.
    for i = 1:length(lines)
        line = lines[i]

        if startswith(line, "# ")
            title = replace(line, "# " => "")
            title = replace(title, "\n" => "")

            lines[i] = ""
            break
        end
    end

    yaml_lines = ["---\n", "title: $title\n", "permalink: $permalink\n", "---\n"]

    # Write lines back.
    open(path, "w+") do f
        # Prepend YAML.
        for line in yaml_lines
            write(f, line)
        end

        # Add remaining content.
        for line in lines
            write(f, replace(line, "![](figure" => "![](/tutorials/figure"))
        end
    end
end

# Weave all examples
try
    for file in readdir(@__DIR__)
        if endswith(file, ".ipynb") || endswith(file, ".jmd")
            out_name = split(file, ".")[1] * ".md"
            out_path = joinpath(markdown_path, out_name)

            full_path = joinpath(@__DIR__, file)

            if mtime(out_path) < mtime(full_path)
                @warn "Weaving $full_path as it has been updated since the least weave."
                Weave.weave(full_path,
                    doctype = "github",
                    out_path = out_path,
                    mod = Main)

                polish_latex(out_path)
                add_yaml(out_path)
            else
                @warn "Skipping $full_path as it has not been updated."
            end
        end
    end
catch e
    println("Weaving error: $e")
    rethrow(e)
end
