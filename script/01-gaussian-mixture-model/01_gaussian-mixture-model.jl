
using Distributions
using FillArrays
using StatsPlots

using LinearAlgebra
using Random

# Set a random seed.
Random.seed!(3)

# Define Gaussian mixture model.
w = [0.5, 0.5]
μ = [-3.5, 0.5]
mixturemodel = MixtureModel([MvNormal(Fill(μₖ, 2), I) for μₖ in μ], w)

# We draw the data points.
N = 60
x = rand(mixturemodel, N);


scatter(x[1, :], x[2, :]; legend=false, title="Synthetic Dataset")


using Turing

@model function gaussian_mixture_model(x)
    # Draw the parameters for each of the K=2 clusters from a standard normal distribution.
    K = 2
    μ ~ MvNormal(Zeros(K), I)

    # Draw the weights for the K clusters from a Dirichlet distribution with parameters αₖ = 1.
    w ~ Dirichlet(K, 1.0)
    # Alternatively, one could use a fixed set of weights.
    # w = fill(1/K, K)

    # Construct categorical distribution of assignments.
    distribution_assignments = Categorical(w)

    # Construct multivariate normal distributions of each cluster.
    D, N = size(x)
    distribution_clusters = [MvNormal(Fill(μₖ, D), I) for μₖ in μ]

    # Draw assignments for each datum and generate it from the multivariate normal distribution.
    k = Vector{Int}(undef, N)
    for i in 1:N
        k[i] ~ distribution_assignments
        x[:, i] ~ distribution_clusters[k[i]]
    end

    return k
end

model = gaussian_mixture_model(x);


sampler = Gibbs(PG(100, :k), HMC(0.05, 10, :μ, :w))
nsamples = 100
nchains = 4
burn = 10
chains = sample(model, sampler, MCMCThreads(), nsamples, nchains; discard_initial = burn);


let
    # Verify that the output of the chain is as expected
    for i in MCMCChains.chains(chains)
        # In this case, we *want* to see the degenerate behaviour
        # So error if Rhat is *small*.
        rhat = MCMCChains.rhat(chains)
        @assert maximum(rhat[:, :rhat]) > 2 "Example intended to demonstrate multi-modality likely failed to find both modes!"
    end
end


plot(chains[["μ[1]", "μ[2]"]], legend=true)


using Bijectors: ordered

@model function gaussian_mixture_model_ordered(x)
    # Draw the parameters for each of the K=2 clusters from a standard normal distribution.
    K = 2
    μ ~ ordered(MvNormal(Zeros(K), I))

    # Draw the weights for the K clusters from a Dirichlet distribution with parameters αₖ = 1.
    w ~ Dirichlet(K, 1.0)
    # Alternatively, one could use a fixed set of weights.
    # w = fill(1/K, K)

    # Construct categorical distribution of assignments.
    distribution_assignments = Categorical(w)

    # Construct multivariate normal distributions of each cluster.
    D, N = size(x)
    distribution_clusters = [MvNormal(Fill(μₖ, D), I) for μₖ in μ]

    # Draw assignments for each datum and generate it from the multivariate normal distribution.
    k = Vector{Int}(undef, N)
    for i in 1:N
        k[i] ~ distribution_assignments
        x[:, i] ~ distribution_clusters[k[i]]
    end

    return k
end

model = gaussian_mixture_model_ordered(x);


chains = sample(model, sampler, MCMCThreads(), nsamples, nchains; discard_initial = burn);


let
    # Verify that the output of the chain is as expected
    for i in MCMCChains.chains(chains)
        # μ[1] and μ[2] can no longer switch places. Check that they've found the mean
        chain = Array(chains[:, ["μ[1]", "μ[2]"], i])
        μ_mean = vec(mean(chain; dims=1))
        @assert isapprox(sort(μ_mean), μ; rtol=0.4) "Difference between estimated mean of μ ($(sort(μ_mean))) and data-generating μ ($μ) unexpectedly large!"
    end
end


plot(chains[["μ[1]", "μ[2]"]]; legend=true)


plot(chains[["w[1]", "w[2]"]], legend=true)


# Model with mean of samples as parameters.
μ_mean = [mean(chains, "μ[$i]") for i in 1:2]
w_mean = [mean(chains, "w[$i]") for i in 1:2]
mixturemodel_mean = MixtureModel([MvNormal(Fill(μₖ, 2), I) for μₖ in μ_mean], w_mean)

contour(
    range(-7.5, 3; length=1_000),
    range(-6.5, 3; length=1_000),
    (x, y) -> logpdf(mixturemodel_mean, [x, y]);
    widen=false,
)
scatter!(x[1, :], x[2, :]; legend=false, title="Synthetic Dataset")


assignments = [mean(chains, "k[$i]") for i in 1:N]
scatter(
    x[1, :],
    x[2, :];
    legend=false,
    title="Assignments on Synthetic Dataset",
    zcolor=assignments,
)


using StatsFuns

@model function gmm_marginalized(x)
    K = 2
    D, N = size(x)
    μ ~ ordered(MvNormal(Zeros(K), I))
    w ~ Dirichlet(K, 1.0)
    dists = [MvNormal(Fill(μₖ, D), I) for μₖ in μ]

    for i in 1:N
        lvec = Vector(undef, K)
        for k in 1:K
            lvec[k] = (w[k] + logpdf(dists[k], x[:, i]))
        end
        Turing.@addlogprob! logsumexp(lvec)
    end
end

model = gmm_marginalized(x);


@model function gmm_marginalized(x)
    K = 2
    D, _ = size(x)
    μ ~ ordered(MvNormal(Zeros(K), I))
    w ~ Dirichlet(K, 1.0)

    x ~ MixtureModel([MvNormal(Fill(μₖ, D), I) for μₖ in μ], w)
end

model = gmm_marginalized(x);


sampler = NUTS()
chains = sample(model, sampler, MCMCThreads(), nsamples, nchains; discard_initial = burn);


let
    # Verify for marginalized model that the output of the chain is as expected
    for i in MCMCChains.chains(chains)
        # μ[1] and μ[2] can no longer switch places. Check that they've found the mean
        chain = Array(chains[:, ["μ[1]", "μ[2]"], i])
        μ_mean = vec(mean(chain; dims=1))
        @assert isapprox(sort(μ_mean), μ; rtol=0.4) "Difference between estimated mean of μ ($(sort(μ_mean))) and data-generating μ ($μ) unexpectedly large!"
    end
end


plot(chains[["μ[1]", "μ[2]"]], legend=true)


@model function gmm_recover(x)
    K = 2
    D, N =  size(x)
    μ ~ ordered(MvNormal(Zeros(K), I))
    w ~ Dirichlet(K, 1.0)

    dists = [MvNormal(Fill(μₖ, D), I) for μₖ in μ]

    x ~ MixtureModel(dists, w)

    # Return sample_class(yi) for fixed μ, w.
    function sample_class(xi)
        lvec = [(logpdf(d, xi) + log(w[i])) for (i, d) in enumerate(dists)]
        rand(Categorical(exp.(lvec .- logsumexp(lvec))))
    end

    # Return assignment draws for each datapoint.
    return [sample_class(x[:, i]) for i in 1:N]
end


chains = sample(model, sampler, MCMCThreads(), nsamples, nchains; discard_initial = burn);


assignments = mean(generated_quantities(gmm_recover(x), chains))


scatter(
    x[1, :],
    x[2, :];
    legend=false,
    title="Assignments on Synthetic Dataset - Recovered",
    zcolor=assignments,
)

