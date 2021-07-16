# Turing Tutorials

This repository contains tutorials on the the universal probabilistic programming language **Turing**. The `markdown` folder contains files converted to markdown from the Jupyter notebooks at the root of this repository.

Note that if you have added or updated a tutorial, you must run the `weave-examples.jl` script in order to publish the markdown version. This is so that we can build moderately sophisticated models but handle the computation on our machines rather than during a Travis build.

Additional educational materials can be found at [StatisticalRethinkingJulia/TuringModels.jl](https://github.com/StatisticalRethinkingJulia/TuringModels.jl), which contains Turing adaptations of models from Richard McElreath's [Statistical Rethinking](https://xcelab.net/rm/statistical-rethinking/). It is a highly recommended resource if you are looking for a greater breadth of examples.

## Running times

These are some temporary notes to keep track of the running times of scripts.
Be sure to pass only one output format, such as `(:html,)` when running the scripts.
Otherwise, the model runs multiple times.
Also, do not set `cache = :all`, because that would skip steps.

### 00-introduction

```
julia> @time TuringTutorials.weave_folder("00-introduction", (:html,); cache=:off)
```

```
98.789259 seconds (321.66 M allocations: 17.292 GiB, 4.68% gc time, 6.76% compilation time)
```

### 01-gaussian-mixture-model

```
julia> @time TuringTutorials.weave_folder("01-gaussian-mixture-model", (:html,); cache=:off)
```

Original:

```
297.173680 seconds (1.10 G allocations: 98.250 GiB, 5.46% gc time, 0.14% compilation time)
```

I've compared the plots in the pdf output and this change doesn't affect the outcome.

### 02-logistic-regression

```
julia> @time TuringTutorials.weave_folder("02-logistic-regression", (:html,); cache=:off)
```

With for loop and `sample(model, HMC(0.05, 10), MCMCThreads(), 1_500, 3)` (second run):

```
7.463169 seconds (17.60 M allocations: 1.060 GiB, 3.65% gc time, 47.69% compilation time)
```

With broadcasting and `BernoulliLogit` (second run):

```
9.370234 seconds (69.55 M allocations: 9.505 GiB, 7.04% gc time, 27.17% compilation time)
```

With `LazyArray(@~)` (second run):

```
9.867191 seconds (69.23 M allocations: 6.540 GiB, 6.43% gc time, 26.37% compilation time)
```

These changes do not affect the outcome.

### 03-bayesian-neural-network

```
@time TuringTutorials.weave_folder("03-bayesian-neural-network", (:html,); cache=:off)
```

Original (second run):

```
368.964917 seconds (554.86 M allocations: 33.734 GiB, 1.45% gc time, 1.55% compilation time)
```

With Zygote and failures (look at the allocations; not good):

```
225.582076 seconds (964.61 M allocations: 90.918 GiB, 8.01% gc time, 0.94% compilation time)
```

this gives errors.

By using forwarddiff instead of backwarddiff (first run):

```
# first run
347.347928 seconds (594.56 M allocations: 35.799 GiB, 1.60% gc time, 0.57% compilation time)
# second run
338.123706 seconds (553.09 M allocations: 33.286 GiB, 1.64% gc time, 0.01% compilation time)
```

ReverseDiff with `Turing.setrdcache(true)` (second run):

```
# first run
339.383844 seconds (553.05 M allocations: 33.284 GiB, 1.63% gc time, 0.00% compilation time)
# second run
334.417546 seconds (553.01 M allocations: 33.285 GiB, 1.71% gc time)
```

### 04-hidden-markov-models

```
@time TuringTutorials.weave_folder("04-hidden-markov-model", (:html,))
```

With for loop:

```
288.180491 seconds (1.01 G allocations: 87.493 GiB, 5.49% gc time, 2.29% compilation time)
```

### 05-linear-regression

Is already using broadcasting.

### 06-infinite-mixture-model

```
@time TuringTutorials.weave_folder("06-infinite-mixture-model", (:html,))
```

Original:

```
101.354853 seconds (364.89 M allocations: 16.737 GiB, 4.33% gc time, 6.08% compilation time)
```

### 07-poisson-regression

```
@time TuringTutorials.weave_folder("07-poisson-regression", (:html,))
```

With for loop:

```
140.085894 seconds (394.90 M allocations: 23.635 GiB, 4.23% gc time, 0.22% compilation time)
```

The output is roughly the same as https://turing.ml/dev/tutorials/07-poisson-regression/.
Only minor differences in the plots.

With broadcasting:

```
147.403359 seconds (395.80 M allocations: 41.793 GiB, 4.71% gc time, 4.63% compilation time)
```

The output is, again, roughly the same as before.

### Build first two

```
@time build_all(; debug=true)
```

In a new session, without trying to run the jobs in parallel.
