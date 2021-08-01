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

@testset "build.jl" begin
    folder = "99-test"

    should_fail = false
    write_test_tutorial(folder, should_fail)
    @test build(folder)
    markdown = TuringTutorials.markdown_output(folder)
    @test contains(markdown, "2")

    should_fail = true
    write_test_tutorial(folder, should_fail)
    @test !build(folder)
end
