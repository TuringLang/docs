# Turing Tutorials

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/TuringLang/TuringTutorials/master)

This repository contains tutorials on the the universal probabilistic programming language **Turing**. The `markdown` folder contains files converted to markdown from the Jupyter notebooks at the root of this repository.

Note that if you have added or updated a tutorial, you must run the `weave-examples.jl` script in order to publish the markdown version. This is so that we can build moderately sophisticated models but handle the computation on our machines rather than during a Travis build.

Additional educational materials can be found at [StatisticalRethinkingJulia/TuringModels.jl](https://github.com/StatisticalRethinkingJulia/TuringModels.jl), which contains Turing adaptations of models from Richard McElreath's [Statistical Rethinking](https://xcelab.net/rm/statistical-rethinking/). It is a highly recommended resource if you are looking for a greater breadth of examples.
