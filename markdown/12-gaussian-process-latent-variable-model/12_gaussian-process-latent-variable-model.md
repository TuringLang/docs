---
redirect_from: "tutorials/12-gaussian-process-latent-variable-model/"
title: "Gaussian Process Latent Variable Model"
permalink: "/:collection/:name/"
---


# Gaussian Process Latent Variable Model

In a previous tutorial, we have discussed latent variable models, in particular probabilistic principal component analysis (pPCA).
Here, we show how we can extend the mapping provided by pPCA to non-linear mappings between input and output.
For more details about the Gaussian Process Latent Variable Model (GPLVM),
we refer the reader to the [original publication](https://jmlr.org/papers/v6/lawrence05a.html) and a [further extension](http://proceedings.mlr.press/v9/titsias10a/titsias10a.pdf).

In short, the GPVLM is a dimensionality reduction technique that allows us to embed a high-dimensional dataset in a lower-dimensional embedding.
Importantly, it provides the advantage that the linear mappings from the embedded space can be non-linearised through the use of Gaussian Processes.

Let's start by loading some dependencies.
```julia
using Turing
using AbstractGPs, KernelFunctions, Random, Plots

using Distributions, LinearAlgebra
using VegaLite, DataFrames, StatsPlots, StatsBase
using RDatasets

Random.seed!(1789);
```




We demonstrate the GPLVM with a very small dataset: [Fisher's Iris data set](https://en.wikipedia.org/wiki/Iris_flower_data_set).
This is mostly for reasons of run time, so the tutorial can be run quickly.
As you will see, one of the major drawbacks of using GPs is their speed,
although this is an active area of research.
We will briefly touch on some ways to speed things up at the end of this tutorial.
We transform the original data with non-linear operations in order to demonstrate the power of GPs to work on non-linear relationships, while keeping the problem reasonably small.

```julia
data = dataset("datasets", "iris")
species = data[!, "Species"]
index = shuffle(1:150)
# we extract the four measured quantities,
# so the dimension of the data is only d=4 for this toy example
dat = Matrix(data[index, 1:4])
labels = data[index, "Species"]

# non-linearize data to demonstrate ability of GPs to deal with non-linearity
dat[:, 1] = 0.5 * dat[:, 1].^2 + 0.1 * dat[:,1].^3
dat[:, 2] = dat[:, 2].^3 + 0.2 * dat[:,2].^4
dat[:, 3] = 0.1 * exp.(dat[:, 3]) - 0.2 * dat[:,3].^2
dat[:, 4] = 0.5 * log.(dat[:, 4]).^2 + 0.01 * dat[:,3].^5

# normalize data
dt = fit(ZScoreTransform, dat, dims=1);
StatsBase.transform!(dt, dat);
```




We will start out by demonstrating the basic similarity between pPCA (see the tutorial on this topic) and the GPLVM model.
Indeed, pPCA is basically equivalent to running the GPLVM model with an automatic relevance determination (ARD) linear kernel.

First, we re-introduce the pPCA model (see the tutorial on pPCA for details)
```julia
@model function pPCA(x, ::Type{TV} = Array{Float64}) where {TV}
  # Dimensionality of the problem.
  N, D = size(x)
  # latent variable z
  z ~ filldist(Normal(), D, N)
  # weights/loadings W
  w ~ filldist(Normal(), D, D)
  mu = (w * z)'
  for d in 1:D
    x[:, d] ~ MvNormal(mu[:, d], 1.)
  end
end;
```





We define two different kernels, a simple linear kernel with an Automatic Relevance Determination transform and a
squared exponential kernel.
```julia
linear_kernel(α) = LinearKernel() ∘ ARDTransform(α)
sekernel(α, σ) = σ * SqExponentialKernel() ∘ ARDTransform(α);
```



And here is the GPLVM model.
We create separate models for the two types of kernel.
```julia
@model function GPLVM_linear(Y, K=4,::Type{T} = Float64) where {T}

  # Dimensionality of the problem.
  N, D = size(Y)
  # K is the dimension of the latent space
  @assert K <= D
  noise = 1e-3

  # Priors
  α ~ MvLogNormal(MvNormal(K, 1.0))
  Z ~ filldist(Normal(0.0, 1.0), K, N)
  mu ~ filldist(Normal(0.0, 1.0), N)

  kernel = linear_kernel(α)

  gp = GP(mu, kernel)
  cv = cov(gp(ColVecs(Z), noise))
  Y ~ filldist(MvNormal(mu, cv), D)
end;

@model function GPLVM(Y, K=4,::Type{T} = Float64) where {T}

  # Dimensionality of the problem.
  N, D = size(Y)
  # K is the dimension of the latent space
  @assert K <= D
  noise = 1e-3

  # Priors
  α ~ MvLogNormal(MvNormal(K, 1.0))
  σ ~ LogNormal(0.0, 1.0)
  Z ~ filldist(Normal(0.0, 1.0), K, N)
  mu ~ filldist(Normal(0.0, 1.0), N)

  kernel = sekernel(α, σ)

  gp = GP(mu, kernel)
  cv = cov(gp(ColVecs(Z), noise))
  Y ~ filldist(MvNormal(mu, cv), D)
end;
```


```julia
# Standard GPs don't scale very well in n, so we use a small subsample for the purpose of this tutorial
n_data = 40
# number of features to use from dataset
n_features = 4
# latent dimension for GP case
ndim = 4;
```


```julia
ppca = pPCA(dat[1:n_data, 1:n_features])
chain_ppca = sample(ppca, NUTS(), 1000);
```


```julia
# we extract the posterior mean estimates of the parameters from the chain
w = reshape(mean(group(chain_ppca, :w))[:, 2], (n_features,n_features))
z = reshape(mean(group(chain_ppca, :z))[:, 2], (n_features, n_data))
X = w * z

df_pre = DataFrame(z', :auto)
rename!(df_pre, Symbol.( ["z"*string(i) for i in collect(1:n_features)]))
df_pre[!,:type] = labels[1:n_data]
p_ppca = df_pre |>  @vlplot(:point, x=:z1, y=:z2, color="type:n")
```

![](figures/12_gaussian-process-latent-variable-model_8_1.png)



We can see that the pPCA fails to distinguish the groups.
In particular, the `setosa` species is not clearly separated from `versicolor` and `virginica`.
This is due to the non-linearities that we introduced, as without them the two groups can be clearly distinguished
using pPCA (see the pPCA tutorial).

Let's try the same with our linear kernel GPLVM model.
```julia
gplvm_linear = GPLVM_linear(dat[1:n_data, 1:n_features], ndim)

chain_linear = sample(gplvm_linear, NUTS(), 500)
# we extract the posterior mean estimates of the parameters from the chain
z_mean = reshape(mean(group(chain_linear, :Z))[:, 2], (n_features, n_data))
alpha_mean = mean(group(chain_linear, :α))[:, 2]
```

```
4-element Vector{Float64}:
 0.4573916332788623
 0.5244420375120039
 0.5032902616561209
 0.4548029438488051
```



```julia
df_gplvm_linear = DataFrame(z_mean', :auto)
rename!(df_gplvm_linear, Symbol.( ["z"*string(i) for i in collect(1:ndim)]))
df_gplvm_linear[!,:sample] = 1:n_data
df_gplvm_linear[!,:labels] = labels[1:n_data]
alpha_indices = sortperm(alpha_mean, rev=true)[1:2]
println(alpha_indices)
df_gplvm_linear[!,:ard1] = z_mean[alpha_indices[1], :]
df_gplvm_linear[!,:ard2] = z_mean[alpha_indices[2], :]

p_linear = df_gplvm_linear |>  @vlplot(:point, x=:ard1, y=:ard2, color="labels:n")
p_linear
```

```
[2, 3]
```


![](figures/12_gaussian-process-latent-variable-model_10_1.png)


We can see that similar to the pPCA case, the linear kernel GPLVM fails to distinguish between the two groups
(`setosa` on the one hand, and `virginica` and `verticolor` on the other).

Finally, we demonstrate that by changing the kernel to a non-linear function, we are able to separate the data again.
```julia
gplvm = GPLVM(dat[1:n_data, 1:n_features], ndim)

chain_gplvm = sample(gplvm, NUTS(), 500)
# we extract the posterior mean estimates of the parameters from the chain
z_mean = reshape(mean(group(chain_gplvm, :Z))[:, 2], (ndim, n_data))
alpha_mean = mean(group(chain_gplvm, :α))[:, 2]
```

```
4-element Vector{Float64}:
 0.15699922659921323
 0.7887930375112809
 0.17187239953779038
 0.15627072713559162
```



```julia
df_gplvm = DataFrame(z_mean', :auto)
rename!(df_gplvm, Symbol.( ["z"*string(i) for i in collect(1:ndim)]))
df_gplvm[!,:sample] = 1:n_data
df_gplvm[!,:labels] = labels[1:n_data]
alpha_indices = sortperm(alpha_mean, rev=true)[1:2]
println(alpha_indices)
df_gplvm[!,:ard1] = z_mean[alpha_indices[1], :]
df_gplvm[!,:ard2] = z_mean[alpha_indices[2], :]

p_gplvm = df_gplvm |>  @vlplot(:point, x=:ard1, y=:ard2, color="labels:n")
p_gplvm
```

```
[2, 3]
```


![](figures/12_gaussian-process-latent-variable-model_12_1.png)




Now, the split between the two groups is visible again.

### Speeding up inference

Gaussian processes tend to be slow, as they naively require operations in the order of $O(n^3)$.
Here, we demonstrate a simple speedup using the Stheno library.
Speeding up Gaussian process inference is an active area of research.

```julia
using Stheno
@model function GPLVM_sparse(Y, K,::Type{T} = Float64) where {T}

  # Dimensionality of the problem.
  N, D = size(Y)
  # dimension of latent space
  @assert K <= D
  # number of inducing points
  n_inducing = 25
  noise = 1e-3

  # Priors
  α ~ MvLogNormal(MvNormal(K, 1.0))
  σ ~ LogNormal(1.0, 1.0)
  Z ~ filldist(Normal(0.0, 1.0), K, N)
  mu ~ filldist(Normal(0.0, 1.0), N)

  kernel = σ * SqExponentialKernel() ∘ ARDTransform(α)

  ## Standard
  # gpc = GPC()
  # f = Stheno.wrap(GP(kernel), gpc)
  # gp = f(ColVecs(Z), noise)
  # Y ~ filldist(gp, D)

  ## SPARSE GP
  #  xu = reshape(repeat(locations, K), :, K) # inducing points
  #  xu = reshape(repeat(collect(range(-2.0, 2.0; length=20)), K), :, K) # inducing points
  lbound = minimum(Y) + 1e-6
  ubound = maximum(Y) - 1e-6
  #  locations ~ filldist(Uniform(lbound, ubound), n_inducing)
  #  locations = [-2., -1.5 -1., -0.5, -0.25, 0.25, 0.5, 1., 2.]
  #  locations = collect(LinRange(lbound, ubound, n_inducing))
  locations = quantile(vec(Y), LinRange(0.01, 0.99, n_inducing))
  xu = reshape(locations, 1, :)
  gp = Stheno.wrap(GP(kernel), GPC())
  fobs = gp(ColVecs(Z), noise)
  finducing = gp(xu, 1e-12)
  sfgp = SparseFiniteGP(fobs, finducing)
  cv = cov(sfgp.fobs)
  Y ~ filldist(MvNormal(mu, cv), D)

end
```

```
GPLVM_sparse (generic function with 3 methods)
```



```julia
n_data = 50
gplvm_sparse = GPLVM_sparse(dat[1:n_data, :], ndim)

chain_gplvm_sparse = sample(gplvm_sparse, NUTS(), 500)
# we extract the posterior mean estimates of the parameters from the chain
z_mean = reshape(mean(group(chain_gplvm_sparse, :Z))[:, 2], (ndim, n_data))
alpha_mean = mean(group(chain_gplvm_sparse, :α))[:, 2]
```

```
4-element Vector{Float64}:
 0.1347245321913309
 0.1886553362408269
 0.6090443132588738
 0.28379259049979433
```



```julia
df_gplvm_sparse = DataFrame(z_mean', :auto)
rename!(df_gplvm_sparse, Symbol.( ["z"*string(i) for i in collect(1:ndim)]))
df_gplvm_sparse[!,:sample] = 1:n_data
df_gplvm_sparse[!,:labels] = labels[1:n_data]
alpha_indices = sortperm(alpha_mean, rev=true)[1:2]
df_gplvm_sparse[!,:ard1] = z_mean[alpha_indices[1], :]
df_gplvm_sparse[!,:ard2] = z_mean[alpha_indices[2], :]
p_sparse = df_gplvm_sparse |>  @vlplot(:point, x=:ard1, y=:ard2, color="labels:n")
p_sparse
```

![](figures/12_gaussian-process-latent-variable-model_16_1.png)



Comparing the runtime, between the two versions, we can observe a clear speed-up with the sparse version.



## Appendix
 This tutorial is part of the TuringTutorials repository, found at: <https://github.com/TuringLang/TuringTutorials>.

To locally run this tutorial, do the following commands:
```julia, eval = false
using TuringTutorials
TuringTutorials.weave_file("12-gaussian-process-latent-variable-model", "12_gaussian-process-latent-variable-model.jmd")
```

Computer Information:
```
Julia Version 1.6.1
Commit 6aaedecc44 (2021-04-23 05:59 UTC)
Platform Info:
  OS: Linux (x86_64-pc-linux-gnu)
  CPU: Intel(R) Core(TM) i7-8550U CPU @ 1.80GHz
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-11.0.1 (ORCJIT, skylake)
Environment:
  JULIA_NUM_THREADS = 8

```

Package Information:

```
      Status `~/TuringDev/TuringTutorials/tutorials/12-gaussian-process-latent-variable-model/Project.toml`
  [99985d1d] AbstractGPs v0.3.9
  [6e4b80f9] BenchmarkTools v1.1.4
  [a93c6f00] DataFrames v1.2.2
  [31c24e10] Distributions v0.25.14
  [ec8451be] KernelFunctions v0.10.13
  [91a5bcdd] Plots v1.21.3
  [ce6b1742] RDatasets v0.7.5
  [2913bbd2] StatsBase v0.33.10
  [f3b207a7] StatsPlots v0.14.26
  [8188c328] Stheno v0.7.12
  [fce5fe82] Turing v0.18.0
  [112f6efa] VegaLite v2.6.0
  [37e2e46d] LinearAlgebra
  [9a3f8284] Random

```
