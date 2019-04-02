# This file shows how to write setup code and model for Negative Binomial
# Regression.

# Import required libraries
using Turing, Distributions, Distributed, MCMCChains

using DataFrames, LinearAlgebra

using Random
Random.seed!(12); # For reproducibility

Turing.turnprogress(false);

# Generating the data
# We start off by creating a toy dataset.

#     We take the case of a person who takes medicines to prevent excessive sneezing. Alcohol consumption increases the rate of sneezing for that person. Thus, the two factors affecting the number of sneezes in a given day are alcohol consumption and whether the person has taken his medicine. Both these variable are taken as boolean valued while the number of sneezes will be a count valued variable. We also take into consideration that the interaction between the two boolean variables will affect the number of sneezes.
    
# We assume that sneezing occurs at some baseline rate, and that consuming alcohol, not taking antihistamines, or doing both, increase its frequency. Also every subject in the dataset had the flu, increasing the variance of their sneezing (and causing an unfortunate few to sneeze over 70 times a day). If the mean number of sneezes stays the same but variance increases, the data might follow a negative binomial distribution.

θ_noalcohol_meds = 1    # no alcohol, took a medicine
θ_alcohol_meds = 3      # alcohol, took a medicine
θ_noalcohol_nomeds = 6  # no alcohol, no medicine
θ_alcohol_nomeds = 36   # alcohol, no medicine

# no of samples for each of the above cases
q = 1000

# Gamma shape parameter
α = 10

function get_nb_vals(μ, α, size)
    θ = μ/α
    g = rand(Gamma(α, θ), size)
    return [rand(Poisson(g[i])) for i = 1:size]
end

#Generate data from different Negative Binomial distribution
q = 1000

df = DataFrame(
        nsneeze=vcat(get_nb_vals(θ_noalcohol_meds, α, q),
                get_nb_vals(θ_alcohol_meds, α, q),
                get_nb_vals(θ_noalcohol_nomeds, α, q),
                get_nb_vals(θ_alcohol_nomeds, α, q)),
        alcohol=vcat(zeros(q),
                ones(q),
                zeros(q),
                ones(q)),
        nomeds=vcat(zeros(q),
                zeros(q),
                ones(q),
                ones(q)));

# We must convert our `DataFrame` data into the `Matrix` form as the manipulations that we are about are designed to work with `Matrix` data. We also separate the features from the labels which will be later used by the Turing sampler to generate samples from the posterior.

data = Matrix(df[[:alcohol, :nomeds]])
data_labels = df[:nsneeze];

# Rescale the matrices

data = (data .- mean(data, dims=1)) ./ std(data, dims=1);

# Our model *negative_binomail_regression* takes two arguments:
# * X: Set of independent variables
# * y: Set we want to predict

@model m(X, y) = begin
    N, D = size(X)
    β ~ MvNormal(zeros(D+1), Diagonal(ones(D+1)))
    r ~ Gamma()
    for n in 1:N
        μ = exp(β[1] + dot(β[2:end], X[n, :]))
        p = (r * μ) / (1 + r * μ)
        y[n] ~ NegativeBinomial(r, p)
    end
end;

# Sampling from the posterior

# This is temporary while the reverse differentiation backend is being improved.
Turing.setadbackend(:forward_diff)

# Sample using NUTS.

num_chains = 1
chns = mapreduce(c -> sample(m(data, data_labels), NUTS(1500, 200, 0.65) ), chainscat, 1:num_chains)