
using TuringTutorials

target = ARGS[1]

if isdir(target)
    if !isfile(joinpath(target, "Project.toml"))
        error("cannot weave folder ", target, " without Project.toml!")
    end
    println("weaving the ", target, " folder")
    TuringTutorials.weave_folder(target)
elseif isfile(target)
    println("weaving ", target)
    TuringTutorials.weave_file(dirname(target), basename(target))
else
    error("unable to find weaving target ", target)
end
