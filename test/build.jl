using Test
using TuringTutorials

function cleanup_folder(folder::String)
    dir = tutorial_path(folder)
    rm(dir; force=true, recursive=true)
    mkpath(dir)
end

function write_test_tutorial(folder::String, should_fail::Bool)
    cleanup_folder(folder)
    jmd = """
        ---
        title: This is a $(should_fail ? "failing" : "") test file
        ---

        ```julia
        using Distributions
        using Test # hide

        $(should_fail ? "error()" : "Normal()")
        ```
        """

    filename = folder2filename(folder)
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

    should_fail = false
    write_test_tutorial(folder, should_fail)
    @test build(folder)
    markdown = TuringTutorials.markdown_output(folder)
    @test contains(markdown, "Distributions.Normal{Float64}(μ=0.0, σ=1.0)")

    should_fail = true
    write_test_tutorial(folder, should_fail)
    @test !build(folder)
end
