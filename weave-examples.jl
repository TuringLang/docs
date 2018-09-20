using Documenter, Turing
using LibGit2: clone
using Weave

# Get paths.
markdown_path = joinpath(@__DIR__, "markdown")

function polish_latex(path::String)
    txt = open(f -> read(f, String), path)
    open(path, "w+") do f
        write(f, replace(txt, raw"$$" => raw"\$\$"))
    end
end

# Weave all examples
try
    for file in readdir(@__DIR__)
        if endswith(file, ".ipynb")
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
            else
                @warn "Not weaving $full_path as it has not been updated."
            end
        end
    end
catch e
    println("Weaving error: $e")
    rethrow(e)
end
