
"""
    error_occurred(log)

Return `true` if an error occurred.
It would be more stable if Weave would have a fail on error option or something similar.
"""
function error_occurred(log)
    weave_error = contains(log, "ERROR")
end

const WEAVE_LOG_FILE = "weave.log"

log_path(folder) = joinpath(repo_dir, "tutorials", folder, WEAVE_LOG_FILE)

"""
    markdown_output(folder)

Returns the Markdown output for a folder.
The output seems to be the only place where Weave prints the full stacktrace.
"""
function markdown_output(folder)
    file = replace(folder, '-' => '_'; count=1)
    file = "$file.md"
    path = joinpath(repo_dir, "markdown", folder, file)
    text = read(path, String)
    """
    Markdown output (contains stacktrace):
    $text
    """
end

"""
    build_folder(folder)

It seems that Weave has no option to fail on error, so we handle errors ourselves.
Also, this method only shows the necessary information in the CI logs.
If something crashes, then show the logs immediately.
If all goes well, then store the logs in a file, but don't show them.
"""
function build_folder(folder)
    println("$folder - Starting build")
    cache = :all
    c = IOCapture.capture() do
        @timed weave_folder(folder; cache)
    end
    stats = c.value
    gib = round(stats.bytes / 1024^3, digits=2)
    min = round(stats.time / 60, digits=2)
    println("$folder - Build took $min minutes and allocated $gib GiB:")
    log = c.output
    md_out = markdown_output(folder)
    if error_occurred(log)
        @error """
        $folder - Error occured:
        $log

        $md_out
        """
    end
    path = log_path(folder)
    println("$folder - Writing log to $path")
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

function log_has_error(folder)::Bool
    path = log_path(folder)
    if isfile(path)
        println("$folder - Verifying the log")
        log = read(path, String)
        has_error = error_occurred(log)
        println("""$folder - Log contains $(has_error ? "an" : "no") error""")
        return has_error
    else
        println("$folder - No file found to verify")
        return false
    end
end

"""
    verify_logs(T::Vector)

Exits with 1 if one of the log files for the tutorials `T` contain errors.
This method is used at the end of the CI in order to allow the CI to run all the tutorials.
"""
function verify_logs(T::Vector)
    outcomes = log_has_error.(T)
    if any(outcomes)
        println("One of the logs contains an error. Exiting.")
        exit(1)
    end
end

"""
    build(T::Vector=changed_tutorials())

Build all changed outputs. This method is used in the CI job.
Pass `tutorials()` to build all tutorials or `["00-introduction"]` to build only the first.
"""
function build(T::Vector=changed_tutorials())
    clean_weave_cache()
    parallel_build(T)
    verify_logs(T)
end
build(tutorial::AbstractString) = build([string(tutorial)::String])
