# Turing Tutorials

This repository contains tutorials on the the universal probabilistic programming language **Turing**.

The tutorials are defined in the `tutorials` folder; all the outputs are automatically generated from that and put in the `artifacts` branch.
So, to change a tutorial, change one of the `.jmd` file in the `tutorials` folder.
At the time of writing, there is one exception to this and that is tutorial 10: Bayesian differential equations.
That tutorial takes ~10 hours to run, and has therefore been excluded from the GitHub CI job.
To update that tutorial, update the `.jmd` file and run `using TuringTutorials; build_folder("10-bayesian-differential-equations")`.

Additional educational materials can be found at [StatisticalRethinkingJulia/TuringModels.jl](https://github.com/StatisticalRethinkingJulia/TuringModels.jl), which contains Turing adaptations of models from Richard McElreath's [Statistical Rethinking](https://xcelab.net/rm/statistical-rethinking/).
It is a highly recommended resource if you are looking for a greater breadth of examples.
