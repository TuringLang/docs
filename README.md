# Turing Tutorials

This repository contains tutorials on the the universal probabilistic programming language **Turing**.

The tutorials are defined in the `tutorials` folder.
All the outputs are generated automatically from that.

Additional educational materials can be found at [StatisticalRethinkingJulia/TuringModels.jl](https://github.com/StatisticalRethinkingJulia/TuringModels.jl), which contains Turing adaptations of models from Richard McElreath's [Statistical Rethinking](https://xcelab.net/rm/statistical-rethinking/).
It is a highly recommended resource if you are looking for a greater breadth of examples.

## Contributing

First of all, make sure that your current directory is `TuringTutorials`.
All of the files are generated from the jmd files in the `tutorials` folder.
So to change a tutorial, change one of the `.jmd` file in the `tutorials` folder.

To run the generation process, do for example:

```julia
using TuringTutorials
using Pkg

cd(dirname(dirname(pathof(TuringTutorials))))
Pkg.activate(".")
Pkg.instantiate()

TuringTutorials.weave_file("00-introduction", "00_introduction.jmd")
```

To generate all files do:

```julia
TuringTutorials.weave_all()
```


If you add new tutorials which require new packages, simply updating your local environment will change the project and manifest files.
When this occurs, the updated environment files should be included in the PR.
