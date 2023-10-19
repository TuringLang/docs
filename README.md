# Turing Tutorials

[![Build status](https://badge.buildkite.com/ffe577bc0ee60b5514a50dbe464a7abb9f2a02c0f35be8ca43.svg?branch=master)](https://buildkite.com/julialang/turingtutorials/builds?branch=master)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)

This repository contains tutorials and docs on the probabilistic programming language **Turing**.

The tutorials are defined in the `tutorials` folder.
All the outputs are generated automatically from that.

Additional educational materials can be found at [StatisticalRethinkingJulia/SR2TuringPluto.jl](https://github.com/StatisticalRethinkingJulia/SR2TuringPluto.jl), which contains Turing adaptations of models from Richard McElreath's [Statistical Rethinking](https://xcelab.net/rm/statistical-rethinking/).
It is a highly recommended resource if you are looking for a greater breadth of examples.

## Interactive Notebooks

To run the tutorials interactively via Jupyter notebooks, install the package and open the tutorials like:

```julia
# Install TuringTutorials
using Pkg
pkg"add https://github.com/TuringLang/TuringTutorials"

# Generate notebooks in subdirectory "notebook"
using TuringTutorials
TuringTutorials.weave(; build=(:notebook,))

# Start Jupyter in "notebook" subdirectory
using IJulia
IJulia.notebook(; dir="notebook")
```

You can weave the notebooks to a different folder with

```julia
TuringTutorials.weave(; build=(:notebook,), out_path_root="my/custom/directory")
```

Then the notebooks will be generated in the folder `my/custom/directory/notebook` and you can start Jupyter with

```julia
IJulia.notebook(; dir="my/custom/directory/notebook")
```

## Contributing

First of all, make sure that your current directory is `TuringTutorials`.
All of the files are generated from the jmd files in the `tutorials` folder.
So to change a tutorial, change one of the `.jmd` file in the `tutorials` folder.

To run the generation process, do for example:

```julia
using TuringTutorials
TuringTutorials.weave("00-introduction", "00_introduction.jmd")
```

To generate all files do:

```julia
TuringTutorials.weave()
```

If you add new tutorials which require new packages, simply updating your local environment will change the project and manifest files.
When this occurs, the updated environment files should be included in the PR.

## Credits

The structure of this repository is mainly based on [SciMLTutorials.jl](https://github.com/SciML/SciMLTutorials.jl).
