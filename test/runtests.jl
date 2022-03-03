using TuringTutorials
using Test

# (Re-)create an empty folder of tutorials for testing
pkgdir = dirname(dirname(pathof(TuringTutorials)))
tutorials_dir = joinpath(pkgdir, "tutorials", "Testing")
rm(tutorials_dir; force=true, recursive=true)
mkpath(tutorials_dir)

# Add to the folder of tutorials an empty Project.toml and a jmd file
touch(joinpath(tutorials_dir, "Project.toml"))
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
    """,
)

# Incorrect build types
@test_throws ArgumentError TuringTutorials.weave(tutorials_dir, "test.jmd"; build=(:doc,))

# Generate default output
TuringTutorials.weave(
    tutorials_dir,
    "test.jmd";
    out_path_root=pkgdir,
    build=(:script, :html, :github, :notebook),
)
