# Set up a temporary environment just to run this script
using Pkg
Pkg.activate(temp=true)
Pkg.add(["YAML", "TOML", "JSON", "HTTP"])
import YAML
import TOML
import JSON
import HTTP

PROJECT_TOML_PATH = "Project.toml"
QUARTO_YML_PATH = "_quarto.yml"
MANIFEST_TOML_PATH = "Manifest.toml"

function major_minor_match(vs...)
    first = vs[1]
    all(v.:major == first.:major && v.:minor == first.:minor for v in vs)
end

function major_minor_patch_match(vs...)
    first = vs[1]
    all(v.:major == first.:major && v.:minor == first.:minor && v.:patch == first.:patch for v in vs)
end

"""
Update the version number in Project.toml to match `target_version`.

This uses a naive regex replacement on lines, i.e. sed-like behaviour. Parsing
the file, editing the TOML and then re-serialising also works and would be more
correct, but the entries in the output file can end up being scrambled, which
would lead to unnecessarily large diffs in the PR.
"""
function update_project_toml(filename, target_version::VersionNumber)
    lines = readlines(filename)
    open(filename, "w") do io
        for line in lines
            if occursin(r"^Turing\s*=\s*\"\d+\.\d+\"\s*$", line)
                println(io, "Turing = \"$(target_version.:major).$(target_version.:minor)\"")
            else
                println(io, line)
            end
        end
    end
end

"""
Update the version number in _quarto.yml to match `target_version`.

See `update_project_toml` for implementation rationale.
"""
function update_quarto_yml(filename, target_version::VersionNumber)
    # Don't deserialise/serialise as this will scramble lines
    lines = readlines(filename)
    open(filename, "w") do io
        for line in lines
            m = match(r"^(\s+)- text:\s*\"v\d+\.\d+\"\s*$", line)
            if m !== nothing
                println(io, "$(m[1])- text: \"v$(target_version.:major).$(target_version.:minor)\"")
            else
                println(io, line)
            end
        end
    end
end

# Retain the original version number string for error messages, as
# VersionNumber() will tack on a patch version of 0
quarto_yaml = YAML.load_file(QUARTO_YML_PATH)
quarto_version_str = quarto_yaml["website"]["navbar"]["right"][1]["text"]
quarto_version = VersionNumber(quarto_version_str)
println("_quarto.yml version: ", quarto_version_str)

project_toml = TOML.parsefile(PROJECT_TOML_PATH)
project_version_str = project_toml["compat"]["Turing"]
project_version = VersionNumber(project_version_str)
println("Project.toml version: ", project_version_str)

manifest_toml = TOML.parsefile(MANIFEST_TOML_PATH)
manifest_version = VersionNumber(manifest_toml["deps"]["Turing"][1]["version"])
println("Manifest.toml version: ", manifest_version)

errors = []

if ENV["TARGET_IS_MASTER"] == "true"
    # This environment variable is set by the GitHub Actions workflow. If it is
    # true, fetch the latest version from GitHub and update files to match this
    # version if necessary.

    resp = HTTP.get("https://api.github.com/repos/TuringLang/Turing.jl/releases/latest")
    latest_version = VersionNumber(JSON.parse(String(resp.body))["tag_name"])
    println("Latest Turing.jl version: ", latest_version)

    if !major_minor_match(latest_version, project_version)
        push!(errors, "$(PROJECT_TOML_PATH) out of date")
        println("$(PROJECT_TOML_PATH) is out of date; updating")
        update_project_toml(PROJECT_TOML_PATH, latest_version)
    end

    if !major_minor_match(latest_version, quarto_version)
        push!(errors, "$(QUARTO_YML_PATH) out of date")
        println("$(QUARTO_YML_PATH) is out of date; updating")
        update_quarto_yml(QUARTO_YML_PATH, latest_version)
    end

    if !major_minor_patch_match(latest_version, manifest_version)
        push!(errors, "$(MANIFEST_TOML_PATH) out of date")
        # Attempt to automatically update Manifest
        println("$(MANIFEST_TOML_PATH) is out of date; updating")
        old_env = Pkg.project().path
        Pkg.activate(".")
        Pkg.update()
        # Check if versions match now, error if not
        Pkg.activate(old_env)
        manifest_toml = TOML.parsefile(MANIFEST_TOML_PATH)
        manifest_version = VersionNumber(manifest_toml["deps"]["Turing"][1]["version"])
        if !major_minor_patch_match(latest_version, manifest_version)
            push!(errors, "Failed to update $(MANIFEST_TOML_PATH) to match latest Turing.jl version")
        end
    end

    if isempty(errors)
        println("All good")
    else
        error("The following errors occurred during version checking: \n", join(errors, "\n"))
    end

else
    # If this is not true, then we are running on a backport-v* branch, i.e. docs
    # for a non-latest version. In this case we don't attempt to fetch the latest
    # patch version from GitHub to check the Manifest (we could, but it is more
    # work as it would involve paging through the list of releases). Instead,
    # we just check that the minor versions match.
    if !major_minor_match(quarto_version, project_version, manifest_version)
        error("The minor versions of Turing.jl in _quarto.yml, Project.toml, and Manifest.toml are inconsistent:
              - _quarto.yml: $quarto_version_str
              - Project.toml: $project_version_str
              - Manifest.toml: $manifest_version
              ")
    end
end
