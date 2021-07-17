
using Turing


@model function two_model(x)
    # Hyper-parameters
    μ0 = 0.0
    σ0 = 1.0
    
    # Draw weights.
    π1 ~ Beta(1,1)
    π2 = 1-π1
    
    # Draw locations of the components.
    μ1 ~ Normal(μ0, σ0)
    μ2 ~ Normal(μ0, σ0)
    
    # Draw latent assignment.
    z ~ Categorical([π1, π2])
    
    # Draw observation from selected component.
    if z == 1
        x ~ Normal(μ1, 1.0)
    else
        x ~ Normal(μ2, 1.0)
    end
end


using Turing.RandomMeasures


# Concentration parameter.
α = 10.0

# Random measure, e.g. Dirichlet process.
rpm = DirichletProcess(α)

# Cluster assignments for each observation.
z = Vector{Int}()

# Maximum number of observations we observe.
Nmax = 500

for i in 1:Nmax
    # Number of observations per cluster.
    K = isempty(z) ? 0 : maximum(z)
    nk = Vector{Int}(map(k -> sum(z .== k), 1:K))
    
    # Draw new assignment.
    push!(z, rand(ChineseRestaurantProcess(rpm, nk)))
end


using Plots

# Plot the cluster assignments over time 
@gif for i in 1:Nmax
    scatter(collect(1:i), z[1:i], markersize = 2, xlabel = "observation (i)", ylabel = "cluster (k)", legend = false)
end


@model function infiniteGMM(x)
    # Hyper-parameters, i.e. concentration parameter and parameters of H.
    α = 1.0
    μ0 = 0.0
    σ0 = 1.0
    
    # Define random measure, e.g. Dirichlet process.
    rpm = DirichletProcess(α)
    
    # Define the base distribution, i.e. expected value of the Dirichlet process.
    H = Normal(μ0, σ0)
    
    # Latent assignment.
    z = tzeros(Int, length(x))
        
    # Locations of the infinitely many clusters.
    μ = tzeros(Float64, 0)
    
    for i in 1:length(x)
        
        # Number of clusters.
        K = maximum(z)
        nk = Vector{Int}(map(k -> sum(z .== k), 1:K))

        # Draw the latent assignment.
        z[i] ~ ChineseRestaurantProcess(rpm, nk)
        
        # Create a new cluster?
        if z[i] > K
            push!(μ, 0.0)

            # Draw location of new cluster.
            μ[z[i]] ~ H
        end
                
        # Draw observation.
        x[i] ~ Normal(μ[z[i]], 1.0)
    end
end


using Plots, Random

# Generate some test data.
Random.seed!(1)
data = vcat(randn(10), randn(10) .- 5, randn(10) .+ 10)
data .-= mean(data)
data /= std(data);


# MCMC sampling
Random.seed!(2)
iterations = 1000
model_fun = infiniteGMM(data);
chain = sample(model_fun, SMC(), iterations);


# Extract the number of clusters for each sample of the Markov chain.
k = map(
    t -> length(unique(vec(chain[t, MCMCChains.namesingroup(chain, :z), :].value))),
    1:iterations
);

# Visualize the number of clusters.
plot(k, xlabel = "Iteration", ylabel = "Number of clusters", label = "Chain 1")


histogram(k, xlabel = "Number of clusters", legend = false)


if isdefined(Main, :TuringTutorials)
    Main.TuringTutorials.tutorial_footer(WEAVE_ARGS[:folder], WEAVE_ARGS[:file])
end

