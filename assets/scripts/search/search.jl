using Pkg
Pkg.instantiate()

using HTTP
using JSON

"""
    DocumenterSearchEntry

JSON schema that Documenter.jl uses for its search index. For an example, see:
https://github.com/TuringLang/DynamicPPL.jl/blob/gh-pages/v0.39.1/search_index.js
"""
struct DocumenterSearchEntry
    location::String
    page::String
    title::String
    text::String
    category::String
end

"""
    QuartoSearchEntry

JSON schema that Quarto uses for its search index. For an example, see:
https://github.com/TuringLang/docs/blob/gh-pages/search_original.json
"""
struct QuartoSearchEntry
    objectID::String
    href::String
    "title of page"
    title::String
    "section name if applicable"
    section::String
    text::String
    crumbs::Union{Vector{String},Nothing}
end

"""
    QuartoSearchEntry(doc_entry::DocumenterSearchEntry) -> QuartoSearchEntry

Converts a `DocumenterSearchEntry` to a `QuartoSearchEntry`.
"""
function QuartoSearchEntry(doc_entry::DocumenterSearchEntry, repo::String)::QuartoSearchEntry
    # Because our links are relative to turinglang.org/docs/, an entry from say
    # DynamicPPL.jl will need to be prepended with `../DynamicPPL.jl/stable` to work
    # correctly.
    location = if occursin("#", doc_entry.location)
        # When opening a Documenter.jl page, if the query parameter `q` is nonempty, it will
        # open up a search bar with that query prefilled. In contrast Quarto stores the
        # query parameter in case the search bar is reopened, but it doesn't actually
        # open the actual search bar.
        #
        # Now, if you search for `your_search_term`, Quarto always adds in
        # `?q=your_search_term` to its search bar links. This allows search bars to be
        # 'prefilled' the next time you open it up. We want to _disable_ this behaviour for
        # Documenter.jl links, because it's quite annoying navigating somewhere else and
        # having an ugly search bar pop up, so we sneak in an empty query parameter into the
        # URL. This is a real hack, but the alternative would be to modify Quarto's JS code
        # itself, which is probably worse.
        #
        # Note that query params should come before anchors, hence the order here.
        before_anchor, anchor = split(doc_entry.location, "#"; limit=2)
        location = before_anchor * "?q=#" * anchor
        joinpath("..", repo, "stable", location)
    else
        # See above for reasoning.
        joinpath("..", repo, "stable", doc_entry.location, "?q=")
    end
    return QuartoSearchEntry(
        # objectID
        location,
        # href
        location,
        # title
        "[$repo] $(doc_entry.page)",
        # section
        doc_entry.title,
        # text
        doc_entry.text,
        # crumbs (no idea what to put here)
        nothing,
    )
end

"""
    get_quarto_search_index() -> Vector{QuartoSearchEntry}

Fetches the Quarto search index either from a local file (if the docs have already been
built); if not, fetches it from the TuringLang/docs GitHub repository.
"""
function get_quarto_search_index()
    search_index = if isfile("_site/search.json")
        @info "Using local search index..."
        JSON.parsefile("_site/search.json", Vector{QuartoSearchEntry})
    else
        @info "Downloading search index from GitHub..."
        resp = HTTP.get(
            "https://raw.githubusercontent.com/TuringLang/docs/refs/heads/gh-pages/search_original.json"
        )
        JSON.parse(String(resp.body), Vector{QuartoSearchEntry})
    end
    # Based on manual inspection of the search index, it appears that the `objectID` and
    # `href` attributes should match. I don't know if Quarto guarantees this, so we warn
    # just in case they don't.
    for entry in search_index
        if entry.objectID != entry.href
            @warn "mismatched objectID and href" objectID=entry.objectID href=entry.href
        end
    end
    return search_index
end

"""
    get_documenter_search_index(repo::String) -> Vector{DocumenterSearchEntry}

Fetches the Documenter.jl search index for the given repository from the published
documentation. This assumes that there is a 'stable' version of the docs (if this isn't the
case, it should definitely be fixed in the upstream repo by triggering a new release with a
working Documenter build.)
"""
function get_documenter_search_index(repo::String)
    url = "https://turinglang.org/$repo/stable/search_index.js"
    @info "Downloading Documenter.jl search index from $url"
    contents = String(HTTP.get(url).body)
    # This file is actually a JavaScript file that says
    #    var documenterSearchIndex = {"docs": [ ... ]};
    # We only want the dictionary, but we should probably check that that file does actually
    # start with that.
    prefix = r"^var documenterSearchIndex = "
    if !occursin(prefix, contents)
        error("Unexpected format of search_index.js file")
    end
    json = replace(contents, prefix => "")
    return JSON.parse(json, Dict{String, Vector{DocumenterSearchEntry}})["docs"]
end

# TODO: Do we also want to include search results from main site? It generally doesn't seem
# like a very meaningful thing to include in the search, and it can clutter actual useful
# results. See e.g. https://github.com/TuringLang/docs/issues/634
# I'm going to say no for now.

repos = [
    "Turing.jl",
    "DynamicPPL.jl",
    "Bijectors.jl",
    "JuliaBUGS.jl",
    "AbstractMCMC.jl",
    "AdvancedMH.jl",
    "AdvancedHMC.jl",
    "AdvancedVI.jl",
    "MCMCChains.jl",
    "MCMCDiagnosticTools.jl",
    "SliceSampling.jl",
    "EllipticalSliceSampling.jl",
]
# Get docs entries
all_entries = get_quarto_search_index()
@info "Fetched $(length(all_entries)) entries from main docs"
# Get entries from other repos
for repo in repos
    doc_entries = get_documenter_search_index(repo)
    @info "Fetched $(length(doc_entries)) entries from $repo"
    quarto_entries = QuartoSearchEntry.(doc_entries, repo)
    append!(all_entries, quarto_entries)
end

# Check that we are running from repo root
if !isdir("_site")
    error("This script must be run from the root of the repository")
end
# Move the old search index out of the way and write the new combined one
output_file = "_site/search.json"
Base.rename(output_file, "_site/search_original.json"; force=true)
JSON.json(output_file, all_entries; pretty=2)
@info "Wrote $(length(all_entries)) entries to $output_file"
