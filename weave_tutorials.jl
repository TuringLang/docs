using TuringTutorials
target = ARGS[1]
if isdir(target)
    if !isfile(joinpath(target, "Project.toml"))
        error("Cannot weave folder $(target) without Project.toml!")
    end
    println("Weaving the $(target) folder")
    TuringTutorials.weave_folder(target)
elseif isfile(target)
    folder = dirname(target)[11:end] # remove the tutorials/
    file = basename(target)
    println("Weaving $(folder)/$(file)")
    TuringTutorials.weave_file(folder, file)
else
    error("Unable to find weaving target $(target)!")
end
