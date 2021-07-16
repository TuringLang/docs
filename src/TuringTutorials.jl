module TuringTutorials

import IOCapture

using IJulia
using InteractiveUtils
using Pkg
using Plots
using Requires
using Weave

export build_folder, build_all

# Not using PDF, because it is fragile.
default_build_list = (:script, :html, :github, :notebook)

repo_directory = pkgdir(TuringTutorials)
cssfile = joinpath(@__DIR__, "..", "templates", "skeleton_css.css")
latexfile = joinpath(@__DIR__, "..", "templates", "julia_tex.tpl")

function polish_latex(path::String)
    # TODO: Is it maybe better to overload https://github.com/JunoLab/Weave.jl/blob/b5ba227e757520f389a6d6e0f2cacb731eab8b12/src/WeaveMarkdown/markdown.jl#L10-L17
    # and replace the `tex.formula` there? Only negative part is that this of course
    # will affect all of the markdown parsers, which is not necessarily desirable.
    txt = open(f -> read(f, String), path)
    open(path, "w+") do f
        txt = replace(txt, raw"\\\\\n" => "\\\\\\\\\n")
        write(f, txt)
    end
end

function weave_file(
    folder, file, build_list=default_build_list;
    kwargs...
)
    tmp = joinpath(repo_directory,"tutorials",folder,file)
    Pkg.activate(dirname(tmp))
    Pkg.instantiate()
    args = Dict{Symbol,String}(:folder => folder, :file => file)
    if :script ∈ build_list
        println("Building Script")
        dir = joinpath(repo_directory,"script",folder)
        isdir(dir) || mkpath(dir)
        args[:doctype] = "script"
        tangle(tmp;out_path=dir)
    end
    if :html ∈ build_list
        println("Building HTML")
        dir = joinpath(repo_directory,"html",folder)
        isdir(dir) || mkpath(dir)
        args[:doctype] = "html"
        weave(
            tmp, doctype = "md2html", out_path=dir, args=args;
            fig_ext=".svg", css=cssfile, kwargs...
        )
    end
    if :pdf ∈ build_list
        println("Building PDF")
        dir = joinpath(repo_directory,"pdf",folder)
        isdir(dir) || mkpath(dir)
        args[:doctype] = "pdf"
        try
            weave(
                tmp, doctype="md2pdf", out_path=dir, args=args;
                template=latexfile, latex_cmd, kwargs...
            )
        catch ex
            @warn "PDF generation failed" exception=(ex, catch_backtrace())
        end
    end
    if :github ∈ build_list
        println("Building Github Markdown")
        dir = joinpath(repo_directory,"markdown",folder)
        isdir(dir) || mkpath(dir)
        args[:doctype] = "github"
        out_path = weave(tmp,doctype = "github",out_path=dir,args=args; kwargs...)
        polish_latex(out_path)
    end
    if :notebook ∈ build_list
        println("Building Notebook")
        dir = joinpath(repo_directory,"notebook",folder)
        isdir(dir) || mkpath(dir)
        args[:doctype] = "notebook"
        Weave.convert_doc(tmp,joinpath(dir,file[1:end-4]*".ipynb"))
    end
end

"""
    tutorials::Vector{String}

Names of the tutorials; for example, "02-logistic-regression".
"""
function tutorials()::Vector{String}
    dirs = readdir(joinpath(repo_directory, "tutorials"))
    dirs = filter(!=("test.jmd"), dirs)
    # This DiffEq one has to be done manually, because it takes about 12 hours.
    dirs = filter(!=("10-bayesian-differential-equations"), dirs)
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
    for file in readdir(joinpath(repo_directory, "tutorials", folder))
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

"""
    clean_cache()

Manually clean cache just to be sure.
Otherwise, cache files committed to the repo could break the build.
"""
function clean_cache()
    for (root, dirs, files) in walkdir(pkgdir(TuringTutorials); onerror=x->())
        if "cache" in dirs
            cache_dir = joinpath(root, "cache")
            rm(cache_dir; force=true, recursive=true)
        end
    end
end

"""
    error_occurred(log)

Return `true` if an error occurred which is important enough to fail CI.
"""
function error_occurred(log)
    weave_error = contains(log, "ERROR")
end

"""
    build_folder(folder, build_list=default_build_list)

It seems that Weave has no option to fail on error, so we handle errors ourselves.
Also, this method only shows the necessary information in the CI logs.
If something crashes, then show the logs immediately.
If all goes well, then store the logs in a file, but don't show them.
"""
function build_folder(folder, build_list=default_build_list)
    println("Starting to build $folder")
    cache = :all
    c = IOCapture.capture() do
        @timed weave_folder(folder; cache)
    end
    stats = c.value
    gib = round(stats.bytes / 1024^3, digits=2)
    min = round(stats.time / 60, digits=2)
    println("Building of $folder took $min minutes and allocated $gib GiB:")
    log = c.output
    if error_occurred(log)
        @error "Error occured when building $folder:\n$log"
    end
    path = joinpath(repo_directory, "tutorials", folder, "weave_folder.log")
    println("Writing log to $path")
    write(path, log)
end

"""
    parallel_build(folders)

Build `folders` in parallel inside new Julia instances.
This has two benefits, namely that it ensures that globals are reset and reduces the
running time.
"""
function parallel_build(folders)
    # The static schedule creates one task per thread.
    Threads.@threads :static for folder in folders
        ex = """using TuringTutorials; build_folder("$folder")"""
        cmd = `$(Base.julia_cmd()) --project -e $ex`
        run(cmd)
    end
end

"""
    build_all(; debug=false)

Build all outputs. This method is used in the CI job.
Set `debug` to `true` to debug the CI deployment.
"""
function build_all(; debug=false)
    clean_cache()
    if debug
        folders = ["00-introduction", "02-logistic-regression"]
        parallel_build(folders)
    else
        parallel_build(tutorials())
    end
end

function __init__()
    @require Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80" include("weaveplots.jl")
end

end # module
