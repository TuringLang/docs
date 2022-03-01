module TuringTutorials

import Requires
import Weave

import InteractiveUtils
import Markdown
import Pkg

const REPO_DIR = dirname(@__DIR__)
const CSS_FILE = joinpath(REPO_DIR, "templates", "skeleton_css.css")
const LATEX_FILE = joinpath(REPO_DIR, "templates", "julia_tex.tpl")

const DEFAULT_BUILD = (:script, :html, :github)

function __init__()
    Requires.@require Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80" include("weaveplots.jl")
end

function weave(folder::AbstractString, file::AbstractString; out_path_root::AbstractString=pwd(), build::Tuple{Vararg{Symbol}}=DEFAULT_BUILD)
  if !issubset(build, (:script, :html, :pdf, :github, :notebook)) 
    throw(ArgumentError("only build types :script, :html, :pdf, :github, and :notebook are supported"))
  end

  target = joinpath(REPO_DIR, "tutorials", folder, file)
  @info("Weaving $(target)")
  
  # Activate project
  # TODO: use separate Julia process?
  if isfile(joinpath(REPO_DIR, "tutorials", folder, "Project.toml")) && (:github in build || :html in build || :pdf in build)
    @info("Instantiating", folder)
    Pkg.activate(joinpath(REPO_DIR, "tutorials", folder))
    Pkg.instantiate()
    Pkg.build()

    @info("Printing out `Pkg.status()`")
    Pkg.status()
  end

  args = Dict{Symbol,String}(:folder => folder, :file => file)
  if :script in build
    println("Building Script")
    dir = joinpath(out_path_root, "script", basename(folder))
    mkpath(dir)
    Weave.tangle(target; out_path=dir)
  end
  if :html in build
    println("Building HTML")
    dir = joinpath(out_path_root, "html", basename(folder))
    mkpath(dir)
    Weave.weave(target; doctype = "md2html", out_path=dir, args=args, css=CSS_FILE, fig_ext=".svg")
  end
  if :pdf in build
    println("Building PDF")
    dir = joinpath(out_path_root, "pdf", basename(folder))
    mkpath(dir)
    try
        Weave.weave(target; doctype="md2pdf", out_path=dir, template=LATEX_FILE, args=args)
    catch ex
      @warn "PDF generation failed" exception=(ex, catch_backtrace())
    end
  end
  if :github in build
    println("Building Github Markdown")
    dir = joinpath(out_path_root, "markdown", basename(folder))
    mkpath(dir)
    Weave.weave(target; doctype="github", out_path=dir, args=args)
  end
  if :notebook in build
    println("Building Notebook")
    dir = joinpath(out_path_root, "notebook", basename(folder))
    mkpath(dir)
    Weave.convert_doc(target, joinpath(dir, first(splitext(file)) * ".ipynb"))
  end
end

# Weave all tutorials
function weave(; kwargs...)
  for folder in readdir(joinpath(REPO_DIR, "tutorials"))
    weave(folder; kwargs...)
  end
end

# Weave a folder of tutorials
function weave(folder::AbstractString; kwargs...)
  for file in readdir(joinpath(REPO_DIR, "tutorials", folder))
    # Skip non-`.jmd` files
    endswith(file, ".jmd") || continue

    weave(folder, file; kwargs...)
  end
end

function tutorial_footer(folder=nothing, file=nothing)
    display(Markdown.md"""
    ## Appendix
    These tutorials are a part of the TuringTutorials repository, found at: <https://github.com/TuringLang/TuringTutorials>.
    """)
    if folder !== nothing && file !== nothing
        display(Markdown.parse("""
        To locally run this tutorial, do the following commands:
        ```
        using TuringTutorials
        TuringTutorials.weave("$folder", "$file")
        ```
        """))
    end
    display(Markdown.md"Computer Information:")
    vinfo = sprint(InteractiveUtils.versioninfo)
    display(Markdown.parse("""
    ```
    $(vinfo)
    ```
    """))

    display(Markdown.md"""
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
