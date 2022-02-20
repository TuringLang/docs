module TuringTutorials

using Requires
using Weave

using InteractiveUtils
using Markdown
using Pkg

const REPO_DIR = dirname(@__DIR__)
const CSS_FILE = joinpath(REPO_DIR, "templates", "skeleton_css.css")
const LATEX_FILE = joinpath(REPO_DIR, "templates", "julia_tex.tpl")

const DEFAULT_BUILD_LIST = (:script, :html, :github)

function __init__()
    @require Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80" include("weaveplots.jl")
end

function weave_file(folder, file, build_list=DEFAULT_BUILD_LIST)
  target = joinpath(REPO_DIR, "tutorials", folder, file)
  @info("Weaving $(target)")
  
  if isfile(joinpath(REPO_DIR, "tutorials", folder, "Project.toml"))
    @info("Instantiating", folder)
    Pkg.activate(joinpath(REPO_DIR, "tutorials", folder))
    Pkg.instantiate()
    Pkg.build()
    
    @info("Printing out `Pkg.status()`")
    Pkg.status()
  end

  args = Dict{Symbol,String}(:folder => folder, :file => file)
  if :script in build_list
    println("Building Script")
    dir = joinpath(REPO_DIR, "script", basename(folder))
    mkpath(dir)
    tangle(target; out_path=dir)
  end
  if :html in build_list
    println("Building HTML")
    dir = joinpath(REPO_DIR, "html", basename(folder))
    mkpath(dir)
    weave(target, doctype = "md2html", out_path=dir, args=args, css=CSS_FILE, fig_ext=".svg")
  end
  if :pdf in build_list
    println("Building PDF")
    dir = joinpath(REPO_DIR, "pdf", basename(folder))
    mkpath(dir)
    try
      weave(target, doctype="md2pdf", out_path=dir, template=LATEX_FILE, args=args)
    catch ex
      @warn "PDF generation failed" exception=(ex, catch_backtrace())
    end
  end
  if :github in build_list
    println("Building Github Markdown")
    dir = joinpath(REPO_DIR, "markdown", basename(folder))
    mkpath(dir)
    weave(target, doctype = "github", out_path=dir, args=args)
  end
  if :notebook in build_list
    println("Building Notebook")
    dir = joinpath(REPO_DIR, "notebook", basename(folder))
    mkpath(dir)
    Weave.convert_doc(target, joinpath(dir, first(splitext(file)) * ".ipynb"))
  end
end

function weave_all(build_list=DEFAULT_BUILD_LIST)
  for folder in readdir(joinpath(REPO_DIR, "tutorials"))
    weave_folder(folder,build_list)
  end
end

function weave_folder(folder,build_list=DEFAULT_BUILD_LIST)
  for file in readdir(joinpath(REPO_DIR, "tutorials", folder))
    # Skip non-`.jmd` files
    endswith(file, ".jmd") || continue

    try
      weave_file(folder, file, build_list)
    catch e
      @error(e)
    end
  end
end

function tutorial_footer(folder=nothing, file=nothing)
    display(md"""
    ## Appendix
    These tutorials are a part of the TuringTutorials repository, found at: <https://github.com/TuringLang/TuringTutorials>.
    """)
    if folder !== nothing && file !== nothing
        display(Markdown.parse("""
        To locally run this tutorial, do the following commands:
        ```
        using TuringTutorials
        TuringTutorials.weave_file("$folder", "$file")
        ```
        """))
    end
    display(md"Computer Information:")
    vinfo = sprint(InteractiveUtils.versioninfo)
    display(Markdown.parse("""
    ```
    $(vinfo)
    ```
    """))

    display(md"""
    Package Information:
    """)

    proj = sprint(io -> Pkg.status(io=io))
    mani = sprint(io -> Pkg.status(io=io, mode = Pkg.PKGMODE_MANIFEST))

    md = """
    ```
    $(chomp(proj))
    ```
    And the full manifest:
    ```
    $(chomp(mani))
    ```
    """
    display(Markdown.parse(md))
end
end # module
