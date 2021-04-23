module TuringTutorials

using Weave, Pkg, InteractiveUtils, IJulia

# HACK: So Weave.jl has a submodule `WeavePlots` which is loaded using Requires.jl if Plots.jl is available.
# This means that if we want to overload methods in that submodule we need to wait until `Plots.jl` has been loaded.
using Requires, Plots
function __init__()
    @require Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80" begin
        # HACK
        function Weave.WeavePlots.add_plots_figure(report::Weave.Report, plot::Plots.AnimatedGif, ext)
            chunk = report.cur_chunk
            full_name, rel_name = Weave.get_figname(report, chunk, ext = ext)

            # A `AnimatedGif` has been saved somewhere temporarily, so make a copy to `full_name`.
            cp(plot.filename, full_name; force = true)
            push!(report.figures, rel_name)
            report.fignum += 1
            return full_name
        end

        function Base.display(report::Weave.Report, m::MIME"text/plain", plot::Plots.AnimatedGif)
            Weave.WeavePlots.add_plots_figure(report, plot, ".gif")
        end
    end
end

repo_directory = joinpath(@__DIR__,"..")
cssfile = joinpath(@__DIR__, "..", "templates", "skeleton_css.css")
latexfile = joinpath(@__DIR__, "..", "templates", "julia_tex.tpl")

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
    folder, file, build_list=(:script ,:html, :github, :notebook);
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
                template=latexfile, kwargs...
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

function weave_all(build_list=(:script,:html,:pdf,:github,:notebook); kwargs...)
    for folder in readdir(joinpath(repo_directory,"tutorials"))
        folder == "test.jmd" && continue
        weave_folder(folder, build_list; kwargs...)
    end
end

function weave_md(; kwargs...)
    for folder in readdir(joinpath(repo_directory,"tutorials"))
        folder == "test.jmd" && continue
        weave_folder(folder, (:github,); kwargs...)
    end
end

function weave_folder(
    folder, build_list=(:script,:html,:pdf,:github,:notebook);
    ext = r"^\.[Jj]md", kwargs...
)
    for file in readdir(joinpath(repo_directory,"tutorials",folder))
        try
            # HACK: only weave (j)md files
            if occursin(ext, splitext(file)[2])
                println("Building $(joinpath(folder,file))")
                weave_file(folder, file, build_list; kwargs...)
            else
                println("Skipping $(joinpath(folder,file))")
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

end
