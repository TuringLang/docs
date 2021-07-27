module TuringTutorials

import IOCapture

using IJulia
using InteractiveUtils
using Pkg
using Plots
using Requires
using Weave

const REPO_DIR = string(pkgdir(TuringTutorials))::String

include("cache.jl")
include("build.jl")

export build_folder, tutorial_path, folder2filename
export build, build_and_exit, verify_logs, tutorials, changed_tutorials

# Not building PDF, because it is fragile. Maybe later.
default_build_list = (:script, :html, :github, :notebook)

cssfile = joinpath(@__DIR__, "..", "templates", "skeleton_css.css")
latexfile = joinpath(@__DIR__, "..", "templates", "julia_tex.tpl")

function __init__()
    @require Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80" include("weaveplots.jl")
end

"""
    tutorial_path(folder)

Return the absolute path to a tutorial `folder`.
"""
tutorial_path(folder) = joinpath(REPO_DIR, "tutorials", folder)

function polish_latex(path::String)
    # TODO: Is it maybe better to overload https://github.com/JunoLab/Weave.jl/blob/b5ba227e757520f389a6d6e0f2cacb731eab8b12/src/WeaveMarkdown/markdown.jl#L10-L17
    # and replace the `tex.formula` there? Only negative part is that this of course
    # will affect all of the markdown parsers, which is not necessarily desirable.
    txt = open(f -> read(f, String), path)
    open(path, "w+") do f
        txt = replace(txt, "\\\\\n" => "\\\\\\\\\n")
        write(f, txt)
    end
end

function weave_file(
    folder, file, build_list=default_build_list;
    kwargs...
)
    tmp = joinpath(tutorial_path(folder), file)
    Pkg.activate(dirname(tmp))
    Pkg.instantiate()
    args = Dict{Symbol,String}(:folder => folder, :file => file)
    if :script ∈ build_list
        println("Building Script")
        dir = joinpath(REPO_DIR, "script", folder)
        isdir(dir) || mkpath(dir)
        args[:doctype] = "script"
        tangle(tmp;out_path=dir)
    end
    if :html ∈ build_list
        println("Building HTML")
        dir = joinpath(REPO_DIR, "html", folder)
        isdir(dir) || mkpath(dir)
        args[:doctype] = "html"
        weave(
            tmp, doctype = "md2html", out_path=dir, args=args;
            fig_ext=".svg", css=cssfile, kwargs...
        )
    end
    if :pdf ∈ build_list
        println("Building PDF")
        dir = joinpath(REPO_DIR, "pdf", folder)
        isdir(dir) || mkpath(dir)
        args[:doctype] = "pdf"
        try
            weave(
                tmp, doctype="md2pdf", out_path=dir, args=args;
                template=latexfile, kwargs...
            )
        catch ex
            @warn "PDF generation failed" exception=(ex, catch_backtrace())
        end
    end
    if :github ∈ build_list
        println("Building Github Markdown")
        dir = joinpath(REPO_DIR, "markdown", folder)
        isdir(dir) || mkpath(dir)
        args[:doctype] = "github"
        out_path = weave(tmp,doctype = "github",out_path=dir,args=args; kwargs...)
        polish_latex(out_path)
    end
    if :notebook ∈ build_list
        println("Building Notebook")
        dir = joinpath(REPO_DIR, "notebook", folder)
        isdir(dir) || mkpath(dir)
        args[:doctype] = "notebook"
        Weave.convert_doc(tmp, joinpath(dir, file[1:end-4]*".ipynb"))
    end
end

"""
    tutorials::Vector{String}

Return names of the tutorials.
"""
function tutorials()::Vector{String}
    dirs = readdir(joinpath(REPO_DIR, "tutorials"))
    dirs = filter(!=("test.jmd"), dirs)
    # This DiffEq one has to be done manually, because it takes about 12 hours.
    dirs = filter(!=("10-bayesian-differential-equations"), dirs)
    dirs = filter(!=("99-test"), dirs)
end

function weave_all(build_list=default_build_list; kwargs...)
    for tutorial in tutorials()
        weave_folder(folder, build_list; kwargs...)
    end
end

function weave_md(; kwargs...)
    for tutorial in tutorials()
        weave_folder(folder, (:github,); kwargs...)
    end
end

function weave_folder(
    folder, build_list=default_build_list;
    ext = r"^\.[Jj]md", kwargs...
)
    for file in readdir(tutorial_path(folder))
        try
            # HACK: only weave (j)md files
            if occursin(ext, splitext(file)[2])
                println("Building $(joinpath(folder, file))")
                weave_file(folder, file, build_list; kwargs...)
            else
                println("Skipping $(joinpath(folder, file))")
            end
        catch ex
            rethrow(ex)
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
