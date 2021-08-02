using Test
using TuringTutorials

function cleanup_folder(folder::String)
    dir = tutorial_path(folder)
    rm(dir; force=true, recursive=true)
    mkpath(dir)
end

function write_test_tutorial(folder::String, should_fail::Bool)
    cleanup_folder(folder)
    # The assertion tests whether we can use assertions to verify the tutorial output.
    jmd = """
        ---
        title: This tutorial should $(should_fail ? "fail" : "pass")
        ---

        ```julia
        x = 1 + 1
        ```

        ```julia; echo=false
        $(should_fail ? "@assert x == 3" : "@assert x == 2")
        ```
        """

    filename = folder2filename(folder)
    path = joinpath(tutorial_path(folder), "$filename.jmd")
    write(path, jmd)
end

function remove_test_files(test_folder)
    for dir in ["html", "markdown", "notebook", "script", "tutorials"]
        test_path = joinpath(TuringTutorials.REPO_DIR, dir, test_folder)
        rm(test_path; force=true, recursive=true)
    end
end

@testset "build.jl" begin
    test_folder = "99-test"

    should_fail = false
    write_test_tutorial(test_folder, should_fail)
    @test build(test_folder)
    markdown = TuringTutorials.markdown_output(test_folder)
    @test contains(markdown, "2")

    should_fail = true
    write_test_tutorial(test_folder, should_fail)
    @test !build(test_folder)

    remove_test_files(test_folder)
    dir = joinpath(TuringTutorials.REPO_DIR, "html", test_folder)
    @test !isdir(dir)
end
