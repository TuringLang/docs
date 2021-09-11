const REPO_URL = "https://github.com/TuringLang/TuringTutorials"
const CLONED_DIR = joinpath(REPO_DIR, "ClonedTuringTutorials")

"""
    clean_weave_cache()

On the one hand, we need `cache = :all` to have quick builds.
On the other hand, we don't need cache files committed to the repo which break the build.
Therefore, this method manually cleans the cache just to be sure.
"""
function clean_weave_cache()
    for (root, dirs, files) in walkdir(pkgdir(TuringTutorials); onerror=x->())
        if "cache" in dirs
            cache_dir = joinpath(root, "cache")
            rm(cache_dir; force=true, recursive=true)
        end
    end
end

"""
    clone_tutorials_output()

Ensure that `$CLONED_DIR` exists and contains the latest commit from the output branch for `$REPO_URL`.
"""
function clone_tutorials_output()
    branch = "artifacts"
    args = [
        "clone",
        "--depth=1",
        "--branch=$branch"
    ]
    if isdir(CLONED_DIR)
        try
            cd(CLONED_DIR) do
                run(`git checkout $branch`)
                run(`git pull --ff --allow-unrelated-histories`)
            end
        catch
            rm(CLONED_DIR; recursive=true, force=true)
            run(`git $args $REPO_URL $CLONED_DIR`)
            cd(CLONED_DIR) do
                run(`git checkout $branch`)
            end
        end
    else
        run(`git $args $REPO_URL $CLONED_DIR`)
    end
end

"""
    download_artifacts()

Explicitly copy all the updated tutorials from the artifacts branch.
This allows orphaning the artifacts branch on each deploy to ease debugging, cleanup old and
unused files, and have a smaller branch.
"""
function download_artifacts()
    if !isdir(CLONED_DIR)
        clone_tutorials_output()
    end
    T = tutorials()
    for tutorial in T
        for dir in ["html", "markdown", "notebook", "script"]
            from_dir = joinpath(CLONED_DIR, dir, tutorial)
            to_dir = joinpath(REPO_DIR, dir, tutorial)
            mkpath(to_dir)
            # from_dir is missing for new/renamed tutorials.
            if isdir(from_dir)
                cp(from_dir, to_dir; force=true)
            end
        end
    end
end


function file_changed(old_dir, new_dir, file)
    old_path = joinpath(old_dir, file)
    new_path = joinpath(new_dir, file)
    old = read(old_path, String)
    new = isfile(new_path) ? read(new_path, String) : ""
    return old != new
end

"""
    any_changes(tutorial::String)

Return whether there are any changes for the local source files, such as `.jmd` and `Manifest.toml`,
compared to the files in `$CLONED_DIR`.
"""
function any_changes(tutorial::String)
    old_dir = joinpath(CLONED_DIR, "tutorials", tutorial)
    new_dir = joinpath(REPO_DIR, "tutorials", tutorial)
    if isdir(old_dir)
        files = readdir(old_dir)
        files = filter(!=(WEAVE_LOG_FILE), files)
        return any(file_changed.(old_dir, new_dir, files))
    else
        # A newly added tutorial.
        return true
    end
end

"""
    changed_tutorials()

Return the tutorials which have changed compared to the output branch at $REPO_URL.
"""
function changed_tutorials()
    clone_tutorials_output()
    T = tutorials()
    changed = filter(any_changes, T)
    n = length(changed)
    println("Found changes for the tutorials $changed ($(n)/$(length(T)))")
    if length(T) == 0
        changed = first(T)
        println("Running the first tutorial to be able to verify that the CI job works.")
    end
    changed
end
