module TuringTutorials

using IJulia
using InteractiveUtils
using Pkg
using Plots
using Requires
using Weave

const REPO_DIR = pkgdir(TuringTutorials)::String
const CSS_FILE = joinpath(REPO_DIR, "templates", "skeleton_css.css")
const LATEX_FILE = joinpath(REPO_DIR, "templates", "julia_tex.tpl")

# Not building PDF, because it is fragile. Maybe later.
const DEFAULT_BUILD_LIST = (:script, :html, :github, :notebook)

function __init__()
    @require Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80" include("weaveplots.jl")
end

function polish_latex(path::String)
    # TODO: Is it maybe better to overload https://github.com/JunoLab/Weave.jl/blob/b5ba227e757520f389a6d6e0f2cacb731eab8b12/src/WeaveMarkdown/markdown.jl#L10-L17
    # and replace the `tex.formula` there? Only negative part is that this of course
    # will affect all of the markdown parsers, which is not necessarily desirable.
    txt = replace(read(path, String), "\\\\\n" => "\\\\\\\\\n")
    write(path, txt)
end

function weave_file(
    folder, file, build_list=DEFAULT_BUILD_LIST;
    kwargs...
)
    target = joinpath(folder, file)
    @info "weaving $target"

    if isfile(joinpath(folder, "Project.toml"))
        @info "instantiating" folder
        Pkg.activate(folder)
        Pkg.instantiate()
        Pkg.build()
    end

    args = Dict{Symbol,String}(:folder => folder, :file => file)
    if :script in build_list
        println("building script")
        dir = joinpath(REPO_DIR, "script", basename(folder))
        mkpath(dir)
        tangle(target; out_path=dir)
    end
    if :html ∈ build_list
        println("building HTML")
        dir = joinpath(REPO_DIR, "html", basename(folder))
        mkpath(dir)
        weave(
            target, doctype="md2html", out_path=dir, args=args;
            fig_ext=".svg", css=CSS_FILE, kwargs...
        )
    end
    if :pdf in build_list
        println("building PDF")
        dir = joinpath(REPO_DIR, "pdf", basename(folder))
        mkpath(dir)
        try
            weave(
                target, doctype="md2pdf", out_path=dir, args=args;
                template=LATEX_FILE, kwargs...
            )
        catch ex
            @warn "PDF generation failed" exception=(ex, catch_backtrace())
        end
    end
    if :github in build_list
        println("building Github markdown")
        dir = joinpath(REPO_DIR, "markdown", basename(folder))
        mkpath(dir)
        weave(target, doctype="github", out_path=dir, args=args; kwargs...)
        polish_latex(dir)
    end
    if :notebook in build_list
        println("building notebook")
        dir = joinpath(REPO_DIR, "notebook", basename(folder))
        mkpath(dir)
        Weave.convert_doc(target, joinpath(dir, first(splitext(file)) * ".ipynb"))
    end
end

function weave_folder(folder, build_list=DEFAULT_BUILD_LIST; kwargs...)
    for file in readdir(folder)
        # Skip non-`.jmd` files
        endswith(file, ".jmd") || continue

        try
            weave_file(folder, file, build_list; kwargs...)
        catch ex
            @error(ex)
        end
    end
end

function tutorial_footer(folder=nothing, file=nothing; remove_homedir=true)
    display("text/markdown", """
        ## Appendix
         This tutorial is part of the TuringTutorials repository, found at: <https://github.com/TuringLang/TuringTutorials>.
        """)
    if folder !== nothing && file !== nothing
        display("text/markdown", """
            To locally run this tutorial, do the following commands:
            ```julia, eval = false
            using TuringTutorials
            TuringTutorials.weave_file("$folder", "$file")
            ```
            """)
    end
    display("text/markdown", "Computer Information:")
    vinfo = sprint(InteractiveUtils.versioninfo)
    display("text/markdown",  """
        ```
        $(vinfo)
        ```
        """)

    ctx = Pkg.API.Context()

    pkg_status = let io = IOBuffer()
        Pkg.status(Pkg.API.Context(); io = io)
        String(take!(io))
    end
    projfile = ctx.env.project_file
    remove_homedir && (projfile = replace(projfile, homedir() => "~"))

    display("text/markdown","""
        Package Information:
        """)

    md = "```\n$(pkg_status)\n```"
    display("text/markdown", md)
end
end # module
