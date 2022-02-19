using TuringTutorials

# (Re-)create an empty folder of tutorials for testing
pkgdir = dirname(dirname(pathof(TuringTutorials)))
tutorials_dir = joinpath(pkgdir, "tutorials", "Testing")
rm(tutorials_dir; force=true, recursive=true)
mkpath(tutorials_dir)

# Add to the folder of tutorials a Project.toml and a jmd file
write(
    joinpath(tutorials_dir, "Project.toml"),
    """
    [deps]
    TuringTutorials = "09eb8af7-3c66-4d0b-a457-e0c10c662b2b"
    """
)
write(
    joinpath(tutorials_dir, "test.jmd"),
    """
    ---
    title: Test
    author: TuringLang team
    ---

    This is a test of the builder system.

    ```julia, echo=false, skip="notebook"
    using TuringTutorials
    TuringTutorials.tutorial_footer(WEAVE_ARGS[:folder], WEAVE_ARGS[:file])
    ```
    """
)

# Generate default output
TuringTutorials.weave_file(joinpath(tutorials_dir, "Testing"), "test.jmd")
