using Test
using TuringTutorials

function clean_folder(folder::String)
    dir = tutorial_path(folder)
    rm(dir; force=true, recursive=true)
    mkpath(dir)
end

function write_test_tutorial(folder::String, should_fail::Bool)
    jmd = """
        ---
        title: This is a $(should_fail ? "failing" : "") test file
        ---

        ```julia
        using Distributions

        $(should_fail ? "error()" : "Normal()")
        ```
        """

    filename = TuringTutorials.folder2filename(folder)
    path = joinpath(tutorial_path(folder), "$filename.jmd")
    write(path, jmd)

    project = """
        [deps]
        Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
        """

    path = joinpath(tutorial_path(folder), "Project.toml")
    write(path, project)
end

@testset "build.jl" begin
    folder = "99-test"
    clean_folder(folder)

    should_fail = false
    write_test_tutorial(folder, should_fail)
    build_folder(folder)
end
