---
title: Bayesian Hidden Markov Models
permalink: /:collection/:name/
---



This tutorial illustrates training Bayesian [Hidden Markov Models](https://en.wikipedia.org/wiki/Hidden_Markov_model) (HMM) using Turing. The main goals are learning the transition matrix, emission parameter, and hidden states. For a more rigorous academic overview on Hidden Markov Models, see [An introduction to Hidden Markov Models and Bayesian Networks](http://mlg.eng.cam.ac.uk/zoubin/papers/ijprai.pdf) (Ghahramani, 2001).

Let's load the libraries we'll need. We also set a random seed (for reproducibility) and the automatic differentiation backend to forward mode (more [here](http://turing.ml/docs/autodiff/) on why this is useful). 

````julia
# Load libraries.
using Turing, Plots, Random

# Set a random seed and use the forward_diff AD mode.
Random.seed!(1234);
Turing.setadbackend(:forward_diff);
````




## Simple State Detection

In this example, we'll use something where the states and emission parameters are straightforward. 

````julia
# Define the emission parameter.
y = [ 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 2.0, 2.0, 2.0, 1.0, 1.0 ];
N = length(y);  K = 3;

# Plot the data we just made.
plot(y, xlim = (0,15), ylim = (-1,5), size = (500, 250))
````


![](/tutorials/figures/4_BayesHmm_2_1.svg)


We can see that we have three states, one for each height of the plot (1, 2, 3). This height is also our emission parameter, so state one produces a value of one, state two produces a value of two, and so on.

Ultimately, we would like to understand three major parameters:

1. The transition matrix. This is a matrix that assigns a probability of switching from one state to any other state, including the state that we are already in.
2. The emission matrix, which describes a typical value emitted by some state. In the plot above, the emission parameter for state one is simply one.
3. The state sequence is our understanding of what state we were actually in when we observed some data. This is very important in more sophisticated HMM models, where the emission value does not equal our state.

With this in mind, let's set up our model. We are going to use some of our knowledge as modelers to provide additional information about our system. This takes the form of the prior on our emission parameter.

\$\$
m_i \sim Normal(i, 0.5), \space m = \{1,2,3\}
\$\$

Simply put, this says that we expect state one to emit values in a Normally distributed manner, where the mean of each state's emissions is that state's value. The variance of 0.5 helps the model converge more quickly — consider the case where we have a variance of 1 or 2. In this case, the likelihood of observing a 2 when we are in state 1 is actually quite high, as it is within a standard deviation of the true emission value. Applying the prior that we are likely to be tightly centered around the mean prevents our model from being too confused about the state that is generating our observations.

The priors on our transition matrix are noninformative, using `T[i] ~ Dirichlet(ones(K)/K)`. The Dirichlet prior used in this way assumes that the state is likely to change to any other state with equal probability. As we'll see, this transition matrix prior will be overwritten as we observe data.

````julia
# Turing model definition.
@model BayesHmm(y, K) = begin
    # Get observation length.
    N = length(y)
    
    # State sequence.
    s = tzeros(Int, N)
    
    # Emission matrix.
    m = Vector{Real}(undef, K)
    
    # Transition matrix.
    T = Vector{Vector{Real}}(undef, K)
    
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
````




We will use a combination of two samplers ([HMC](http://turing.ml/docs/library/#Turing.HMC) and [Particle Gibbs](http://turing.ml/docs/library/#Turing.PG)) by passing them to the [Gibbs](http://turing.ml/docs/library/#Turing.Gibbs) sampler. The Gibbs sampler allows for compositional inference, where we can utilize different samplers on different parameters. 

In this case, we use HMC for `m` and `T`, representing the emission and transition matrices respectively. We use the Particle Gibbs sampler for `s`, the state sequence. You may wonder why it is that we are not assigning `s` to the HMC sampler, and why it is that we need compositional Gibbs sampling at all.

The parameter `s` is not a continuous variable. It is a vector of **integers**, and thus Hamiltonian methods like HMC and [NUTS](http://turing.ml/docs/library/#-turingnuts--type) won't work correctly. Gibbs allows us to apply the right tools to the best effect. If you are a particularly advanced user interested in higher performance, you may benefit from setting up your Gibbs sampler to use [different automatic differentiation](http://turing.ml/docs/autodiff/#compositional-sampling-with-differing-ad-modes) backends for each parameter space.

Time to run our sampler.

````julia
g = Gibbs(1000, HMC(2, 0.001, 7, :m, :T), PG(20, 1, :s))
c = sample(BayesHmm(y, 3), g);
````




Let's see how well our chain performed. Ordinarily, using the `describe` function from [MCMCChain](https://github.com/TuringLang/MCMCChain.jl) would be a good first step, but we have generated a lot of parameters here (`s[1]`, `s[2]`, `m[1]`, and so on). It's a bit easier to show how our model performed graphically.

The code below generates an animation showing the graph of the data above, and the data our model generates in each sample.

````julia
# Import StatsPlots for animating purposes.
using StatPlots

# Extract our m and s parameters from the chain.
m_set = c[:m]
s_set = c[:s]

# Iterate through the MCMC samples.
Ns = 1:500

# Make an animation.
animation = @animate for (i, N) in enumerate(Ns)
    m = m_set[N]; s = s_set[N];
    p = plot(y, c = :red,
        size = (500, 250), 
        xlabel = "Time", 
        ylabel = "State", 
        legend = :topright, label = "True data",
        xlim = (0,15),
        ylim = (-1,5)); 
    plot!(p, m[s], c = :blue, label = "Sample $$N")
end every 10;
````




![animation](https://user-images.githubusercontent.com/422990/50612436-de588980-0e8e-11e9-8635-4e3e97c0d7f9.gif)


Looks like our model did a pretty good job, but we should also check to make sure our chain converges. A quick check is to examine whether the diagonal (representing the probability of remaining in the current state) of the transition matrix appears to be stationary. The code below extracts the diagonal and shows a traceplot of each persistence probability.

````julia
T_diag_trace = [t[i][i] for t in c[:T], i in 1:K];

plot(T_diag_trace, ylim = (0,1),
     label = ["T[$$i,$$i]" for i = 1:K],
     xlabel = "Sample", ylabel = "Persistence probability")
````


![](/tutorials/figures/4_BayesHmm_6_1.svg)


A cursory examination of the traceplot above indicates that at least `T[3,3]` and possibly `T[2,2]` have converged to something resembling stationary. `T[1,1]`, on the other hand, has a slight "wobble", and seems less consistent than the others. We can use the diagnostic functions provided by [MCMCChain](https://github.com/TuringLang/MCMCChain.jl) to engage in some formal tests, like the Heidelberg and Welch diagnostic:

````julia
heideldiag(c)
````


````
Burn-in Stationarity p-value      Mean                Halfwidth   
       Test
 T[2][1]     500            0  0.0002    0.824615396  0.0089378396465752719
957099    1
 T[2][2]     500            0  0.0012    0.149720276  0.0097615436207997719
614271    1
 T[2][3]     200            1  0.1140    0.024906146  0.0021633912298860756
388830    1
  lf_num     500            0     NaN    7.000000000  0.0000000000000000000
000000    1
    s[4]     500            0     NaN    2.000000000  0.0000000000000000000
000000    1
    s[2]     500            0     NaN    2.000000000  0.0000000000000000000
000000    1
    s[9]     500            0     NaN    1.000000000  0.0000000000000000000
000000    1
    s[1]     500            0     NaN    2.000000000  0.0000000000000000000
000000    1
    s[6]     500            0     NaN    1.000000000  0.0000000000000000000
000000    1
   s[14]     500            0     NaN    2.000000000  0.0000000000000000000
000000    1
 T[1][1]     100            1  0.0969    0.673554899  0.0256616435600021328
133913    1
 T[1][2]       0            1  0.0898    0.280661620  0.0215710069280442805
206288    1
 T[1][3]     300            1  0.5218    0.045508585  0.0033732919452402823
634607    1
 elapsed     500            0  0.0370    0.071396306  0.0066535612176202218
667753    1
   s[12]     500            0     NaN    1.000000000  0.0000000000000000000
000000    1
 T[3][1]       0            1  0.3270    0.475533405  0.0131407523929687724
412174    1
 T[3][2]       0            1  0.3355    0.454722943  0.0120465927195745319
683295    1
 T[3][3]     500            0  0.0176    0.061226845  0.0054629277768717763
447137    1
   s[11]     500            0     NaN    1.000000000  0.0000000000000000000
000000    1
    s[8]     500            0     NaN    1.000000000  0.0000000000000000000
000000    1
 epsilon       0            1  1.0000    0.001000000  0.0000000000000000012
749983    1
    s[7]     500            0     NaN    1.000000000  0.0000000000000000000
000000    1
   s[10]     500            0     NaN    1.000000000  0.0000000000000000000
000000    1
    m[2]       0            1  0.8580    1.486215067  0.0411114349597817563
264357    1
    s[3]     100            1  0.4115    2.154006118  0.0148065294562437690
245593    1
    m[1]       0            1  0.3463    1.676331952  0.0548027962706177027
008003    1
eval_num     500            0     NaN   18.000000000  0.0000000000000000000
000000    1
    s[5]     500            0     NaN    1.000000000  0.0000000000000000000
000000    1
   s[15]     500            0     NaN    2.000000000  0.0000000000000000000
000000    1
   s[13]     500            0     NaN    1.000000000  0.0000000000000000000
000000    1
      lp       0            1  0.1424 -138.879454821 14.2975186657629418363
057994    0
    m[3]     400            1  0.0697    0.089261481  0.1522512995241660505
030268    0
````




The p-values on the test suggest that we cannot reject the hypothesis that the observed sequence comes from a stationary distribution, so we can be somewhat more confident that our transition matrix has converged to something reasonable. 


## Modifying a Model to Generate Synthetic Data

With our learned parameters, we can change our model to generate synthetic data. You can do this from the first time you specify a model, but it is conceptually easier to separate these tasks and avoid muddying the waters.

In order to create a model that supports this synthetic generating feature, there are several changes to your typical model specification that need to be made. A general guide can be found [here](http://turing.ml/docs/guide/#generating-vectors-of-quantities).

1. Any parameter you were interested in learning before (`s`, `m`, `T`) needs to be moved to the argument line of the model.
2. Assign those parameters default values, such as `zeros(Real, 10)`, or whatever is appropriate. 
3. Make sure you add a `return` line at the end of the model containing the variable(s) you want to generate. In our case, this is `y`.

And that's about it! The code below presents the original `BayesHmm` model with the necessary changes included.

````julia
# Generative model.
@model BayesHmm(
    y = Vector{Real}(undef, 15),
    T = Vector{Vector{Real}}(undef, 3),
    m = Vector{Real}(undef, 3),
    K) = begin
    # Get observation length.
    N = length(y)
    
    # State sequence.
    s = tzeros(Int, N)
    
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
    
    return y
end;
````




Let's extract the parameters we learned from our chain. We're only using the samples starting from 200 to discard the burn-in period.

````julia
learned_T = mean(c[:T][200:end]);
learned_m = mean(c[:m][200:end]);
````




Finally, we can call our model by passing `nothing` into our parameter of interest, and taking a look at it. Note that we call our model using `BayesHmm(nothing, learned_T, learned_m, 3)()` with an extra set of parentheses at the end. For more on this behaviour, see [this](http://turing.ml/docs/guide/#sampling-from-the-prior) section of the guide focusing on sampling from the prior.

````julia
# Generate a single sequence.
generated_data = BayesHmm(nothing, learned_T, learned_m, 3)()
plot(generated_data)
````


![](/tutorials/figures/4_BayesHmm_10_1.svg)


It doesn't look exactly like our model, but it should have all the same properties. Notice that the spikes to level 3 are quite rare, not unlike our original data set.
