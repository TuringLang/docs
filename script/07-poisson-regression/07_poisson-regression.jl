
#Import Turing, Distributions and DataFrames
using Turing, Distributions, DataFrames, Distributed

# Import MCMCChain, Plots, and StatsPlots for visualizations and diagnostics.
using MCMCChains, Plots, StatsPlots

# Set a seed for reproducibility.
using Random
Random.seed!(12);

# Turn off progress monitor.
Turing.setprogress!(false)


theta_noalcohol_meds = 1    # no alcohol, took medicine
theta_alcohol_meds = 3      # alcohol, took medicine
theta_noalcohol_nomeds = 6  # no alcohol, no medicine
theta_alcohol_nomeds = 36   # alcohol, no medicine

# no of samples for each of the above cases
q = 100

#Generate data from different Poisson distributions
noalcohol_meds = Poisson(theta_noalcohol_meds)
alcohol_meds = Poisson(theta_alcohol_meds)
noalcohol_nomeds = Poisson(theta_noalcohol_nomeds)
alcohol_nomeds = Poisson(theta_alcohol_nomeds)

nsneeze_data = vcat(rand(noalcohol_meds, q), rand(alcohol_meds, q), rand(noalcohol_nomeds, q), rand(alcohol_nomeds, q) )
alcohol_data = vcat(zeros(q), ones(q), zeros(q), ones(q) )
meds_data = vcat(zeros(q), zeros(q), ones(q), ones(q) )

df = DataFrame(nsneeze = nsneeze_data, alcohol_taken = alcohol_data, nomeds_taken = meds_data, product_alcohol_meds = meds_data.*alcohol_data)
df[sample(1:nrow(df), 5, replace = false), :]


#Data Plotting

p1 = Plots.histogram(df[(df[:,:alcohol_taken] .== 0) .& (df[:,:nomeds_taken] .== 0), 1], title = "no_alcohol+meds")  
p2 = Plots.histogram((df[(df[:,:alcohol_taken] .== 1) .& (df[:,:nomeds_taken] .== 0), 1]), title = "alcohol+meds")  
p3 = Plots.histogram((df[(df[:,:alcohol_taken] .== 0) .& (df[:,:nomeds_taken] .== 1), 1]), title = "no_alcohol+no_meds")  
p4 = Plots.histogram((df[(df[:,:alcohol_taken] .== 1) .& (df[:,:nomeds_taken] .== 1), 1]), title = "alcohol+no_meds")  
plot(p1, p2, p3, p4, layout = (2, 2), legend = false)


# Convert the DataFrame object to matrices.
data = Matrix(df[:,[:alcohol_taken, :nomeds_taken, :product_alcohol_meds]])
data_labels = df[:,:nsneeze]
data


# # Rescale our matrices.
data = (data .- mean(data, dims=1)) ./ std(data, dims=1)


# Bayesian poisson regression (LR)
@model poisson_regression(x, y, n, σ²) = begin
    b0 ~ Normal(0, σ²)
    b1 ~ Normal(0, σ²)
    b2 ~ Normal(0, σ²)
    b3  ~ Normal(0, σ²)
    for i = 1:n
        theta = b0 + b1*x[i, 1] + b2*x[i,2] + b3*x[i,3]
        y[i] ~ Poisson(exp(theta))
    end
end;


# Retrieve the number of observations.
n, _ = size(data)

# Sample using NUTS.

num_chains = 4
chain = mapreduce(
    c -> sample(poisson_regression(data, data_labels, n, 10), NUTS(200, 0.65), 2500, discard_adapt=false), 
    chainscat, 
    1:num_chains);


gelmandiag(chain)


# Taking the first chain
c1 = chain[:,:,1]

# Calculating the exponentiated means
b0_exp = exp(mean(c1[:b0]))
b1_exp = exp(mean(c1[:b1]))
b2_exp = exp(mean(c1[:b2]))
b3_exp = exp(mean(c1[:b3]))

print("The exponent of the meaned values of the weights (or coefficients are): \n")
print("b0: ", b0_exp, " \n", "b1: ", b1_exp, " \n", "b2: ", b2_exp, " \n", "b3: ", b3_exp, " \n")
print("The posterior distributions obtained after sampling can be visualised as :\n")


plot(chain)


# Note the standard deviation before removing the warmup samples
describe(chain)


# Removing the first 200 values of the chains.
chains_new = chain[201:2500,:,:]
describe(chains_new)


plot(chains_new)


if isdefined(Main, :TuringTutorials)
    Main.TuringTutorials.tutorial_footer(WEAVE_ARGS[:folder], WEAVE_ARGS[:file])
end

