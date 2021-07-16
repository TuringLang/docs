
# Load libraries.
using Turing, StatsPlots, Random

# Turn off progress monitor.
Turing.setprogress!(false);

# Set a random seed and use the forward_diff AD mode.
Random.seed!(12345678);


# Define the emission parameter.
y = [ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
N = length(y);  K = 3;

# Plot the data we just made.
plot(y, xlim = (0,30), ylim = (-1,5), size = (500, 250))


# Turing model definition.
@model BayesHmm(y, K) = begin
    # Get observation length.
    N = length(y)

    # State sequence.
    s = tzeros(Int, N)

    # Emission matrix.
    m = Vector(undef, K)

    # Transition matrix.
    T = Vector{Vector}(undef, K)

    # Assign distributions to each element
    # of the transition matrix and the
    # emission matrix.
    for i = 1:K
        T[i] ~ Dirichlet(ones(K)/K)
        m[i] ~ Normal(i, 0.5)
    end

    # Observe each point of the input.
    s[1] ~ Categorical(K)
    y[1] ~ Normal(m[s[1]], 0.1)

    for i = 2:N
        s[i] ~ Categorical(vec(T[s[i-1]]))
        y[i] ~ Normal(m[s[i]], 0.1)
    end
end;


g = Gibbs(HMC(0.01, 50, :m, :T), PG(120, :s))
chn = sample(BayesHmm(y, 3), g, 1000);



# Extract our m and s parameters from the chain.
m_set = MCMCChains.group(chn, :m).value
s_set = MCMCChains.group(chn, :s).value

# Iterate through the MCMC samples.
Ns = 1:length(chn)

# Make an animation.
animation = @gif for i in Ns
    m = m_set[i, :];
    s = Int.(s_set[i,:]);
    emissions = m[s]

    p = plot(y, chn = :red,
        size = (500, 250),
        xlabel = "Time",
        ylabel = "State",
        legend = :topright, label = "True data",
        xlim = (0,30),
        ylim = (-1,5));
    plot!(emissions, color = :blue, label = "Sample $i")
end every 3


# Index the chain with the persistence probabilities.
subchain = chn[["T[1][1]", "T[2][2]", "T[3][3]"]]

plot(subchain,
     seriestype = :traceplot,
     title = "Persistence Probability",
     legend=false)



heideldiag(MCMCChains.group(chn, :T))[1]


if isdefined(Main, :TuringTutorials)
    Main.TuringTutorials.tutorial_footer(WEAVE_ARGS[:folder], WEAVE_ARGS[:file])
end

