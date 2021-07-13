# Load Turing.
using Turing
#  using AbstractGPs, Plots
using AbstractGPs, KernelFunctions, Random, Plots

# Background reading:
# https://jmlr.org/papers/v6/lawrence05a.html
# http://proceedings.mlr.press/v9/titsias10a/titsias10a.pdf

#  using Memoization
#  Turing.setrdcache(true)

# Load other dependencies
using Distributions, LinearAlgebra
using VegaLite, DataFrames

using DelimitedFiles
oil_matrix = readdlm("Data.txt", Float64)
labels = readdlm("DataLabels.txt", Float64)
labels = mapslices(x -> findmax(x)[2], labels, dims=2)

# choose AD backend
#  using Zygote # Tracker supported? check it?
#  Turing.setadbackend(:zygote)
#  TODO why does this give an error?

#  SqExponentialKernel is alias of RBFKernel
sekernel(α, σ) = SqExponentialKernel() ∘ ARDTransform(α)

@model function GPLVM(Y, ndim=4, ::Type{T} = Float64) where {T}

  # Dimensionality of the problem.
  N, D = size(Y)
  # dimensions of latent space
  K = ndim
  noise = 1e-6

  # Priors
  α ~ MvLogNormal(MvNormal(K, 1.0))
  σ ~ MvLogNormal(MvNormal(D, 1.0))
  # use filldist for Zygote compatibility
  Z ~ filldist(Normal(0., 1.), K, N)

  kernel = sekernel(α, σ[1])
  K = kernelmatrix(kernel, Z)  # cov matrix
  K += LinearAlgebra.I * (σ[1] + noise)

  Y ~ filldist(MvNormal(zeros(N), K), D)

  ## DENSE GP
  #  gp = GP(kernel, GPC())
  #  prior = gp(ColVecs(Z), noise)
  #  prior = gp(ColVecs(Z))

  #  Y ~ filldist(prior, D)
end

Y = oil_matrix
ndim=4

## DEBUG CODE
#  N, D = size(Y)
#  K = ndim
#  Z = rand(filldist(Normal(0., 1.), K, N))
#  α = rand(MvLogNormal(MvNormal(K, 1.0)))
#  σ = rand(MvLogNormal(MvNormal(D, 1.0)))
#  noise=1e-6
#  kernel = sekernel(α, σ[1])
#  gp = GP(kernel, GPC())
#  prior = gp(ColVecs(Z))#, noise)
#  Y = rand(filldist(prior, D))
#  gp, prior = build_gp(Z, α, σ[d])
#  Y ~ filldist(prior, N, D)
# END DEBUG CODE

n_data=100
gplvm = GPLVM(oil_matrix[1:n_data,:], ndim)

#  # takes a while
chain = sample(gplvm, NUTS(), 500)
z_mean = permutedims(reshape(mean(group(chain, :Z))[:,2], (ndim, n_data)))'
#  z_mean = reshape(mean(group(chain, :Z))[:,2], (n_data, ndim))'
alpha_mean = mean(group(chain, :α))[:,2]

df_gplvm = DataFrame(z_mean', :auto)
rename!(df_gplvm, Symbol.( ["z"*string(i) for i in collect(1:ndim)]))
df_gplvm[!,:sample] = 1:n_data
df_gplvm[!,:labels] = labels[1:n_data]
df_gplvm|>  @vlplot(:point, x=:z1, y=:z2, color="labels:n")

alpha_indices = sortperm(alpha_mean)[1:2]
df_gplvm[!,:ard1] = z_mean[alpha_indices[1], :]
df_gplvm[!,:ard2] = z_mean[alpha_indices[2], :]
df_gplvm |>  @vlplot(:point, x=:ard1, y=:ard2, color="labels:n")

#  advi = ADVI(10, 1000)
#  q = vi(gplvm, advi);

#  z_post = rand(q, 10000);

#  #  awkward way to extract variables
#  _, sym2range = bijector(gplvm, Val(true));
#  sym2range
#
#  α_mean = mean(z_post[union(sym2range[:α]...),:], dims=2)
#
#  Z_mean = reshape(mean(z_post[union(sym2range[:Z]...),:], dims=2), n_data, :)
#
#  df_in = DataFrame(oil_matrix[1:n_data,:], :auto)
#  df_in[!,:sample] = 1:n_data
#  df_in[!,:labels] = labels[1:n_data]
#  df_latent = DataFrames.DataFrame(Z_mean, ["z" * string(i) for i in 1:size(Z_mean)[2]])
#  df_oil = hcat(df_in, df_latent)
#
#  alpha_indices = sortperm(α_mean[:,1])[1:2]
#  p1 = df_oil |>  @vlplot(:point, x="z"*string(alpha_indices[1]), y="z"*string(alpha_indices[2]), color="labels:n")
#
#  p2 = df_oil |>  @vlplot(:point, x="z1", y="z2", color="labels:n")#, shape=:batch)
#  p1
#  p2

#  DataFrames.stack(df_oil, 1:2) |>
    #  @vlplot(:rect, x="sample:o", color=:value, encoding={y={field="variable", type="nominal", sort="-x", axis={title="gene"}}})
#

#  function plot_variational_marginals(z, sym2range)
    #  ps = []
#
    #  for (i, sym) in enumerate(keys(sym2range))
        #  indices = union(sym2range[sym]...)  # <= array of ranges
        #  if sum(length.(indices)) > 1
            #  offset = 1
            #  for r in indices
                #  for j in r
                    #  p = density(z[j, :], title = "$(sym)[$offset]", titlefontsize = 10, label = "")
                    #  push!(ps, p)
#
                    #  offset += 1
                #  end
            #  end
        #  else
            #  p = density(z[first(indices), :], title = "$(sym)", titlefontsize = 10, label = "")
            #  push!(ps, p)
        #  end
    #  end
#
    #  return plot(ps..., layout = (length(ps), 1), size = (500, 1500))
#  end
#
