module TuringTutorials

using Weave, Pkg, InteractiveUtils, IJulia

repo_directory = joinpath(@__DIR__,"..")
cssfile = joinpath(@__DIR__, "..", "templates", "skeleton_css.css")
latexfile = joinpath(@__DIR__, "..", "templates", "julia_tex.tpl")

function weave_file(
    folder, file, build_list=(:script ,:html, :github, :notebook);
    kwargs...
)
    tmp = joinpath(repo_directory,"tutorials",folder,file)
    Pkg.activate(dirname(tmp))
    Pkg.instantiate()
    args = Dict{Symbol,String}(:folder=>folder,:file=>file)
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
        weave(tmp,doctype = "github",out_path=dir,args=args; kwargs...)
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
            @warn "Weave failed" ex
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
