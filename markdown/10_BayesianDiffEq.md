---
title: Bayesian Estimation of Differential Equations
permalink: /:collection/:name/
---

Most of the scientific community deals with the basic problem of trying to mathematically model the reality around them and this often involves dynamical systems. The general trend to model these complex dynamical systems is through the use of differential equations. Differential equation models often have non-measurable parameters. The popular “forward-problem” of simulation consists of solving the differential equations for a given set of parameters, the “inverse problem” to simulation, known as parameter estimation, is the process of utilizing data to determine these model parameters. Bayesian inference provides a robust approach to parameter estimation with quantified uncertainty.


```julia
using Turing, Distributions, DifferentialEquations 

# Import MCMCChain, Plots, and StatsPlots for visualizations and diagnostics.
using MCMCChains, Plots, StatsPlots

# Set a seed for reproducibility.
using Random
Random.seed!(14);

# Disable Turing's progress meter for this tutorial.
Turing.turnprogress(false)
```

    ┌ Info: Precompiling Turing [fce5fe82-541a-59a6-adf8-730c64b5f9a0]
    └ @ Base loading.jl:1260
    ┌ Info: Precompiling DifferentialEquations [0c46a032-eb83-5123-abaf-570d42b7fbaa]
    └ @ Base loading.jl:1260
    ┌ Info: Precompiling Plots [91a5bcdd-55d7-5caf-9e0b-520d859cae80]
    └ @ Base loading.jl:1260
    ┌ Info: Precompiling StatsPlots [f3b207a7-027a-5e70-b257-86293d7955fd]
    └ @ Base loading.jl:1260
    ┌ Info: [Turing]: progress logging is disabled globally
    └ @ Turing /home/cameron/.julia/packages/Turing/GMBTf/src/Turing.jl:22





false



Set a logger to catch AdvancedHMC warnings.


```julia
using Logging
Logging.disable_logging(Logging.Warn)
```




    LogLevel(1001)



## The Lotka-Volterra Model

The Lotka–Volterra equations, also known as the predator–prey equations, are a pair of first-order nonlinear differential equations, frequently used to describe the dynamics of biological systems in which two species interact, one as a predator and the other as prey. The populations change through time according to the pair of equations:

$$\frac{dx}{dt} = (\alpha - \beta y)x$$
 
$$\frac{dy}{dt} = (\delta x - \gamma)y$$



```julia
function lotka_volterra(du,u,p,t)
  x, y = u
  α, β, γ, δ  = p
  du[1] = (α - β*y)x # dx =
  du[2] = (δ*x - γ)y # dy = 
end
p = [1.5, 1.0, 3.0, 1.0]
u0 = [1.0,1.0]
prob1 = ODEProblem(lotka_volterra,u0,(0.0,10.0),p)
sol = solve(prob1,Tsit5())
plot(sol)
```




![svg](/tutorials/10_BayesianDiffEq_files/10_BayesianDiffEq_5_0.svg)



We'll generate the data to use for the parameter estimation from simulation. 
With the `saveat` [argument](https://docs.sciml.ai/latest/basics/common_solver_opts/) we specify that the solution is stored only at `0.1` time units. To make the data look more realistic, we add random noise using the function `randn`.


```julia
sol1 = solve(prob1,Tsit5(),saveat=0.1)
odedata = Array(sol1) + 0.8 * randn(size(Array(sol1)))
plot(sol1, alpha = 0.3, legend = false); scatter!(sol1.t, odedata')
```




![svg](/tutorials/10_BayesianDiffEq_files/10_BayesianDiffEq_7_0.svg)



## Direct Handling of Bayesian Estimation with Turing

Previously, functions in Turing and DifferentialEquations were not inter-composable, so Bayesian inference of differential equations needed to be handled by another package called [DiffEqBayes.jl](https://github.com/SciML/DiffEqBayes.jl) (note that DiffEqBayes works also with CmdStan.jl, Turing.jl, DynamicHMC.jl and ApproxBayes.jl - see the [DiffEqBayes docs](https://docs.sciml.ai/latest/analysis/parameter_estimation/#Bayesian-Methods-1) for more info).

From now on however, Turing and DifferentialEquations are completely composable and we can write of the differential equation inside a Turing `@model` and it will just work. Therefore, we can rewrite the Lotka Volterra parameter estimation problem with a Turing `@model` interface as below:


```julia
Turing.setadbackend(:forwarddiff)

@model function fitlv(data, prob1)
    σ ~ InverseGamma(2, 3) # ~ is the tilde character
    α ~ truncated(Normal(1.5,0.5),0.5,2.5)
    β ~ truncated(Normal(1.2,0.5),0,2)
    γ ~ truncated(Normal(3.0,0.5),1,4)
    δ ~ truncated(Normal(1.0,0.5),0,2)

    p = [α,β,γ,δ]
    prob = remake(prob1, p=p)
    predicted = solve(prob,Tsit5(),saveat=0.1)

    for i = 1:length(predicted)
        data[:,i] ~ MvNormal(predicted[i], σ)
    end
end

model = fitlv(odedata, prob1)

# This next command runs 3 independent chains without using multithreading. 
chain = mapreduce(c -> sample(model, NUTS(.65),1000), chainscat, 1:3)
```




    Object of type Chains, with data of type 500×17×3 Array{Float64,3}
    
    Iterations        = 1:500
    Thinning interval = 1
    Chains            = 1, 2, 3
    Samples per chain = 500
    internals         = acceptance_rate, hamiltonian_energy, hamiltonian_energy_error, is_accept, log_density, lp, max_hamiltonian_energy_error, n_steps, nom_step_size, numerical_error, step_size, tree_depth
    parameters        = α, β, γ, δ, σ
    
    2-element Array{ChainDataFrame,1}
    
    Summary Statistics
      parameters    mean     std  naive_se    mcse     ess   r_hat
      ──────────  ──────  ──────  ────────  ──────  ──────  ──────
               α  1.6400  0.3788    0.0098  0.0985  6.0241  4.6765
               β  1.2568  0.4553    0.0118  0.1194  6.0241  5.6826
               γ  2.5128  1.0548    0.0272  0.2784  6.0241  6.9037
               δ  1.0579  0.4827    0.0125  0.1269  6.0241  6.1099
               σ  1.6983  0.6397    0.0165  0.1691  6.0241  7.3797
    
    Quantiles
      parameters    2.5%   25.0%   50.0%   75.0%   97.5%
      ──────────  ──────  ──────  ──────  ──────  ──────
               α  1.0757  1.3041  1.5536  2.0408  2.2903
               β  0.6976  0.8965  1.0570  1.8161  1.9813
               γ  1.0084  1.1320  2.8546  3.4381  3.8983
               δ  0.4722  0.5803  0.9334  1.6086  1.8763
               σ  0.7492  0.8334  2.0593  2.1754  2.3950




The estimated parameters are close to the desired parameter values. We can also check that the chains have converged in the plot.


```julia
plot(chain)
```




![svg](/tutorials/10_BayesianDiffEq_files/10_BayesianDiffEq_12_0.svg)



### Data retrodiction 
In Bayesian analysis it is often useful to retrodict the data, i.e. generate simulated data using samples from the posterior distribution, and compare to the original data (see for instance section 3.3.2 - model checking of McElreath's book "Statistical Rethinking"). Here, we solve again the ODE using the output in `chain`, for 300 randomly picked posterior samples. We plot this ensemble of solutions to check if the solution resembles the data. 


```julia
pl = scatter(sol1.t, odedata');
```


```julia
chain_array = Array(chain)
for k in 1:300 
    resol = solve(remake(prob1,p=chain_array[rand(1:1500), 1:4]),Tsit5(),saveat=0.1)
    plot!(resol, alpha=0.1, color = "#BBBBBB", legend = false)
end
# display(pl)
plot!(sol1, w=1, legend = false)
```




![svg](/tutorials/10_BayesianDiffEq_files/10_BayesianDiffEq_15_0.svg)



In the plot above, the 300 retrodicted time courses from the posterior are plotted in gray, and the original data are the blue and red dots, and the solution that was used to generate the data are the green and purple lines. We can see that, even though we added quite a bit of noise to the data (see dot plot above), the posterior distribution reproduces quite accurately the "true" ODE solution.

## Lokta Volterra with missing predator data

Thanks to the known structure of the problem, encoded by the Lokta-Volterra ODEs, one can also fit a model with incomplete data - even without any data for one of the two variables. For instance, let's suppose you have observations for the prey only, but none for the predator. We test this case by fitting the model only to the $$y$$ variable of the system, without providing any data for $$x$$:


```julia
@model function fitlv2(data, prob1) # data should be a Vector
    σ ~ InverseGamma(2, 3) # ~ is the tilde character
    α ~ truncated(Normal(1.5,0.5),0.5,2.5)
    β ~ truncated(Normal(1.2,0.5),0,2)
    γ ~ truncated(Normal(3.0,0.5),1,4)
    δ ~ truncated(Normal(1.0,0.5),0,2)

    p = [α,β,γ,δ]
    prob = remake(prob1, p=p)
    predicted = solve(prob,Tsit5(),saveat=0.1)

    for i = 1:length(predicted)
        data[i] ~ Normal(predicted[i][2], σ) # predicted[i][2] is the data for y - a scalar, so we use Normal instead of MvNormal
    end
end

model2 = fitlv2(odedata[2,:], prob1)
```




    DynamicPPL.Model{var"###evaluator#333",(:data, :prob1),Tuple{Array{Float64,1},ODEProblem{Array{Float64,1},Tuple{Float64,Float64},true,Array{Float64,1},ODEFunction{true,typeof(lotka_volterra),LinearAlgebra.UniformScaling{Bool},Nothing,Nothing,Nothing,Nothing,Nothing,Nothing,Nothing,Nothing,Nothing,Nothing,Nothing,Nothing},Base.Iterators.Pairs{Union{},Union{},Tuple{},NamedTuple{(),Tuple{}}},DiffEqBase.StandardODEProblem}},(),DynamicPPL.ModelGen{var"###generator#334",(:data, :prob1),(),Tuple{}}}(##evaluator#333, (data = [1.0373159410554433, 0.455111997844011, 0.9187133767127944, -0.259115048591982, 0.3514128305537414, 0.5162643092020036, 0.9787322372445835, -0.0006805260449948558, -0.08833057357290974, 0.4636414910986264  …  3.119725220966818, 3.8955494581199934, 4.932912225131781, 2.8100177568591196, 2.925421407352717, 2.2748396927494876, 1.0152713962244975, 2.556317594971266, 2.3096409202477224, -0.30553906640714645], prob1 = [36mODEProblem[0m with uType [36mArray{Float64,1}[0m and tType [36mFloat64[0m. In-place: [36mtrue[0m
    timespan: (0.0, 10.0)
    u0: [1.0, 1.0]), DynamicPPL.ModelGen{var"###generator#334",(:data, :prob1),(),Tuple{}}(##generator#334, NamedTuple()))



Here we use the multithreading functionality [available](https://turing.ml/dev/docs/using-turing/guide#multithreaded-sampling) in Turing.jl to sample 3 independent chains


```julia
Threads.nthreads()
```




16




```julia
# This next command runs 3 independent chains with multithreading. 
chain2 = sample(model2, NUTS(.45), MCMCThreads(), 5000, 3, progress=false)
```




    Object of type Chains, with data of type 4000×17×3 Array{Float64,3}
    
    Iterations        = 1:4000
    Thinning interval = 1
    Chains            = 1, 2, 3
    Samples per chain = 4000
    internals         = acceptance_rate, hamiltonian_energy, hamiltonian_energy_error, is_accept, log_density, lp, max_hamiltonian_energy_error, n_steps, nom_step_size, numerical_error, step_size, tree_depth
    parameters        = α, β, γ, δ, σ
    
    2-element Array{ChainDataFrame,1}
    
    Summary Statistics
      parameters    mean     std  naive_se    mcse      ess   r_hat
      ──────────  ──────  ──────  ────────  ──────  ───────  ──────
               α  1.4217  0.1660    0.0015  0.0133  51.6958  1.1826
               β  0.9862  0.1288    0.0012  0.0102  53.1975  1.1694
               γ  3.1736  0.2961    0.0027  0.0233  51.3341  1.1737
               δ  1.1284  0.2565    0.0023  0.0207  50.3184  1.1978
               σ  0.7962  0.0605    0.0006  0.0047  56.2673  1.0678
    
    Quantiles
      parameters    2.5%   25.0%   50.0%   75.0%   97.5%
      ──────────  ──────  ──────  ──────  ──────  ──────
               α  1.1731  1.3055  1.3872  1.5190  1.8176
               β  0.7926  0.8938  0.9629  1.0606  1.2834
               γ  2.5500  2.9797  3.2000  3.3872  3.7011
               δ  0.6123  0.9426  1.1478  1.3096  1.6027
               σ  0.6933  0.7520  0.7904  0.8358  0.9291





```julia
pl = scatter(sol1.t, odedata');
chain_array2 = Array(chain2)
for k in 1:300 
    resol = solve(remake(prob1,p=chain_array2[rand(1:12000), 1:4]),Tsit5(),saveat=0.1) 
    # Note that due to a bug in AxisArray, the variables from the chain will be returned always in
    # the order it is stored in the array, not by the specified order in the call - :α, :β, :γ, :δ
    plot!(resol, alpha=0.1, color = "#BBBBBB", legend = false)
end
#display(pl)
plot!(sol1, w=1, legend = false)
```




![svg](/tutorials/10_BayesianDiffEq_files/10_BayesianDiffEq_23_0.svg)



Note that here, the data values of $$x$$ (blue dots) were not given to the model! Yet, the model could predict the values of $$x$$ relatively accurately, albeit with a wider distribution of solutions, reflecting the greater uncertainty in the prediction of the $$x$$ values.

## Inference of Delay Differential Equations

Here we show an example of inference with another type of differential equation: a Delay Differential Equation (DDE). A DDE is an DE system where derivatives are function of values at an earlier point in time. This is useful to model a delayed effect, like incubation time of a virus for instance. 

For this, we will define a [`DDEProblem`](https://diffeq.sciml.ai/stable/tutorials/dde_example/), from the package DifferentialEquations.jl. 

Here is a delayed version of the lokta voltera system:

$$\frac{dx}{dt} = \alpha x(t-\tau) - \beta y(t) x(t)$$
 
$$\frac{dy}{dt} = - \gamma y(t) + \delta x(t) y(t) $$

Where $$x(t-\tau)$$ is the variable $$x$$ at an earlier time point. We specify the delayed variable with a function `h(p, t)`, as described in the [DDE example](https://diffeq.sciml.ai/stable/tutorials/dde_example/).


```julia
function delay_lotka_volterra(du, u, h, p, t)
   x, y = u
   α, β, γ, δ = p
   du[1] = α * h(p, t-1; idxs=1) - β * x * y
   du[2] = -γ * y + δ * x * y
   return
end

p = (1.5,1.0,3.0,1.0)
u0 = [1.0; 1.0]
tspan = (0.0,10.0)
h(p, t; idxs::Int) = 1.0
prob1 = DDEProblem(delay_lotka_volterra,u0,h,tspan,p)
```




    [36mDDEProblem[0m with uType [36mArray{Float64,1}[0m and tType [36mFloat64[0m. In-place: [36mtrue[0m
    timespan: (0.0, 10.0)
    u0: [1.0, 1.0]




```julia
sol = solve(prob1,saveat=0.1)
ddedata = Array(sol)
ddedata = ddedata + 0.5 * randn(size(ddedata))
```




    2×101 Array{Float64,2}:
     1.45377  1.01444   1.49355  1.23384   …  2.79479  2.61251  2.43377  3.17567
     0.88201  0.214703  1.05351  0.470845     1.61454  1.31338  1.59865  0.643372



Plot the data:


```julia
scatter(sol.t, ddedata'); plot!(sol)
```




![svg](/tutorials/10_BayesianDiffEq_files/10_BayesianDiffEq_30_0.svg)



Now we define and run the Turing model. 


```julia
Turing.setadbackend(:forwarddiff)
@model function fitlv(data, prob1)
    
    σ ~ InverseGamma(2, 3)
    α ~ Truncated(Normal(1.5,0.5),0.5,2.5)
    β ~ Truncated(Normal(1.2,0.5),0,2)
    γ ~ Truncated(Normal(3.0,0.5),1,4)
    δ ~ Truncated(Normal(1.0,0.5),0,2)
    
    p = [α,β,γ,δ]
    
    #prob = DDEProblem(delay_lotka_volterra,u0,_h,tspan,p)
    prob = remake(prob1, p=p)
    predicted = solve(prob,saveat=0.1)
    for i = 1:length(predicted)
        data[:,i] ~ MvNormal(predicted[i], σ)
    end
end;
model = fitlv(ddedata, prob1)
```




    DynamicPPL.Model{var"###evaluator#417",(:data, :prob1),Tuple{Array{Float64,2},DDEProblem{Array{Float64,1},Tuple{Float64,Float64},Tuple{},Tuple{},true,NTuple{4,Float64},DDEFunction{true,typeof(delay_lotka_volterra),LinearAlgebra.UniformScaling{Bool},Nothing,Nothing,Nothing,Nothing,Nothing,Nothing,Nothing,Nothing,Nothing,Nothing,Nothing,Nothing},typeof(h),Base.Iterators.Pairs{Union{},Union{},Tuple{},NamedTuple{(),Tuple{}}}}},(),DynamicPPL.ModelGen{var"###generator#418",(:data, :prob1),(),Tuple{}}}(##evaluator#417, (data = [1.4537685182006146 1.0144365870095116 … 2.4337746023657396 3.175674933929347; 0.8820102262205037 0.21470268242472768 … 1.59865201888447 0.6433719617612795], prob1 = [36mDDEProblem[0m with uType [36mArray{Float64,1}[0m and tType [36mFloat64[0m. In-place: [36mtrue[0m
    timespan: (0.0, 10.0)
    u0: [1.0, 1.0]), DynamicPPL.ModelGen{var"###generator#418",(:data, :prob1),(),Tuple{}}(##generator#418, NamedTuple()))



Then we draw samples using multithreading; this time, we draw 3 independent chains in parallel using `MCMCThreads`.


```julia
chain = sample(model, NUTS(.65), MCMCThreads(), 300, 3, progress=true)
plot(chain)
```




![svg](/tutorials/10_BayesianDiffEq_files/10_BayesianDiffEq_34_0.svg)



Finally, we select a 100 sets of parameters from the first chain and plot solutions.


```julia
chain
```




    Object of type Chains, with data of type 150×17×3 Array{Float64,3}
    
    Iterations        = 1:150
    Thinning interval = 1
    Chains            = 1, 2, 3
    Samples per chain = 150
    internals         = acceptance_rate, hamiltonian_energy, hamiltonian_energy_error, is_accept, log_density, lp, max_hamiltonian_energy_error, n_steps, nom_step_size, numerical_error, step_size, tree_depth
    parameters        = α, β, γ, δ, σ
    
    2-element Array{ChainDataFrame,1}
    
    Summary Statistics
      parameters    mean     std  naive_se    mcse       ess   r_hat
      ──────────  ──────  ──────  ────────  ──────  ────────  ──────
               α  1.3786  0.0517    0.0024  0.0023  140.8826  1.0137
               β  0.9286  0.0444    0.0021  0.0022  237.4755  1.0020
               γ  3.2104  0.1429    0.0067  0.0083  115.0547  1.0301
               δ  1.0925  0.0512    0.0024  0.0027  122.4686  1.0216
               σ  0.4963  0.0248    0.0012  0.0012  258.5969  0.9972
    
    Quantiles
      parameters    2.5%   25.0%   50.0%   75.0%   97.5%
      ──────────  ──────  ──────  ──────  ──────  ──────
               α  1.2772  1.3417  1.3794  1.4082  1.4890
               β  0.8447  0.8985  0.9291  0.9543  1.0255
               γ  2.9393  3.1148  3.2113  3.2972  3.5028
               δ  1.0030  1.0568  1.0912  1.1266  1.1973
               σ  0.4518  0.4801  0.4949  0.5115  0.5480





```julia
pl = scatter(sol.t, ddedata')
chain_array = Array(chain) 
for k in 1:100
    
    resol = solve(remake(prob1,p=chain_array[rand(1:450),1:4]),Tsit5(),saveat=0.1)
    # Note that due to a bug in AxisArray, the variables from the chain will be returned always in
    # the order it is stored in the array, not by the specified order in the call - :α, :β, :γ, :δ
    
    plot!(resol, alpha=0.1, color = "#BBBBBB", legend = false)
end
#display(pl)
plot!(sol)
```




![svg](/tutorials/10_BayesianDiffEq_files/10_BayesianDiffEq_37_0.svg)



Here again, the dots is the data fed to the model, the continuous colored line is the "true" solution, and the gray lines are solutions from the posterior. The fit is pretty good even though the data was quite noisy to start.

## Scaling to Large Models: Adjoint Sensitivities

DifferentialEquations.jl's efficiency for large stiff models has been shown in multiple [benchmarks](https://github.com/SciML/DiffEqBenchmarks.jl). To learn more about how to optimize solving performance for stiff problems you can take a look at the [docs](https://docs.sciml.ai/latest/tutorials/advanced_ode_example/). 

[Sensitivity analysis](https://docs.sciml.ai/latest/analysis/sensitivity/), or automatic differentiation (AD) of the solver, is provided by the DiffEq suite. The model sensitivities are the derivatives of the solution $$u(t)$$ with respect to the parameters. Specifically, the local sensitivity of the solution to a parameter is defined by how much the solution would change by changes in the parameter. Sensitivity analysis provides a cheap way to calculate the gradient of the solution which can be used in parameter estimation and other optimization tasks.


The AD ecosystem in Julia allows you to switch between forward mode, reverse mode, source to source and other choices of AD and have it work with any Julia code. For a user to make use of this within [SciML](https://sciml.ai), [high level interactions in `solve`](https://docs.sciml.ai/latest/analysis/sensitivity/#High-Level-Interface:-sensealg-1) automatically plug into those AD systems to allow for choosing advanced sensitivity analysis (derivative calculation) [methods](https://docs.sciml.ai/latest/analysis/sensitivity/#Sensitivity-Algorithms-1). 

More theoretical details on these methods can be found at: https://docs.sciml.ai/latest/extras/sensitivity_math/.

While these sensitivity analysis methods may seem complicated (and they are!), using them is dead simple. Here is a version of the Lotka-Volterra model with adjoints enabled.

All we had to do is switch the AD backend to one of the adjoint-compatible backends (ReverseDiff, Tracker, or Zygote) and boom the system takes over and we're using adjoint methods! Notice that on this model adjoints are slower. This is because adjoints have a higher overhead on small parameter models and we suggest only using these methods for models with around 100 parameters or more. For more details, see https://arxiv.org/abs/1812.01892.


```julia
using Zygote, DiffEqSensitivity
Turing.setadbackend(:zygote)
prob1 = ODEProblem(lotka_volterra,u0,(0.0,10.0),p)
```




    [36mODEProblem[0m with uType [36mArray{Float64,1}[0m and tType [36mFloat64[0m. In-place: [36mtrue[0m
    timespan: (0.0, 10.0)
    u0: [1.0, 1.0]




```julia
@model function fitlv(data, prob)
    σ ~ InverseGamma(2, 3)
    α ~ truncated(Normal(1.5,0.5),0.5,2.5)
    β ~ truncated(Normal(1.2,0.5),0,2)
    γ ~ truncated(Normal(3.0,0.5),1,4)
    δ ~ truncated(Normal(1.0,0.5),0,2)
    p = [α,β,γ,δ]    
    prob = remake(prob, p=p)

    predicted = solve(prob,saveat=0.1)
    for i = 1:length(predicted)
        data[:,i] ~ MvNormal(predicted[i], σ)
    end
end;
model = fitlv(odedata, prob1)
chain = sample(model, NUTS(.65),1000)
```




    Object of type Chains, with data of type 500×17×1 Array{Float64,3}
    
    Iterations        = 1:500
    Thinning interval = 1
    Chains            = 1
    Samples per chain = 500
    internals         = acceptance_rate, hamiltonian_energy, hamiltonian_energy_error, is_accept, log_density, lp, max_hamiltonian_energy_error, n_steps, nom_step_size, numerical_error, step_size, tree_depth
    parameters        = α, β, γ, δ, σ
    
    2-element Array{ChainDataFrame,1}
    
    Summary Statistics
      parameters    mean     std  naive_se    mcse     ess   r_hat
      ──────────  ──────  ──────  ────────  ──────  ──────  ──────
               α  2.1258  0.0002    0.0000  0.0001  3.4840  1.5025
               β  0.4210  0.0002    0.0000  0.0001  2.0647  2.9436
               γ  2.7257  0.0007    0.0000  0.0003  2.0080  2.9125
               δ  1.3201  0.0005    0.0000  0.0002  2.2723  2.3902
               σ  0.5172  0.0044    0.0002  0.0022  2.1994  2.5493
    
    Quantiles
      parameters    2.5%   25.0%   50.0%   75.0%   97.5%
      ──────────  ──────  ──────  ──────  ──────  ──────
               α  2.1254  2.1257  2.1258  2.1259  2.1261
               β  0.4207  0.4207  0.4209  0.4212  0.4213
               γ  2.7248  2.7250  2.7260  2.7264  2.7266
               δ  1.3193  1.3195  1.3202  1.3205  1.3208
               σ  0.5095  0.5139  0.5174  0.5210  0.5241




Now we can exercise control of the sensitivity analysis method that is used by using the `sensealg` keyword argument. Let's choose the `InterpolatingAdjoint` from the available AD [methods](https://docs.sciml.ai/latest/analysis/sensitivity/#Sensitivity-Algorithms-1) and enable a compiled ReverseDiff vector-Jacobian product:


```julia
@model function fitlv(data, prob)
    σ ~ InverseGamma(2, 3)
    α ~ truncated(Normal(1.5,0.5),0.5,2.5)
    β ~ truncated(Normal(1.2,0.5),0,2)
    γ ~ truncated(Normal(3.0,0.5),1,4)
    δ ~ truncated(Normal(1.0,0.5),0,2)
    p = [α,β,γ,δ]
    prob = remake(prob, p=p)
    predicted = solve(prob,saveat=0.1,sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))
    for i = 1:length(predicted)
        data[:,i] ~ MvNormal(predicted[i], σ)
    end
end;
model = fitlv(odedata, prob1)
@time chain = sample(model, NUTS(.65),1000)
```

    476.077757 seconds (2.71 G allocations: 124.356 GiB, 5.05% gc time)





    Object of type Chains, with data of type 500×17×1 Array{Float64,3}
    
    Iterations        = 1:500
    Thinning interval = 1
    Chains            = 1
    Samples per chain = 500
    internals         = acceptance_rate, hamiltonian_energy, hamiltonian_energy_error, is_accept, log_density, lp, max_hamiltonian_energy_error, n_steps, nom_step_size, numerical_error, step_size, tree_depth
    parameters        = α, β, γ, δ, σ
    
    2-element Array{ChainDataFrame,1}
    
    Summary Statistics
      parameters    mean     std  naive_se    mcse       ess   r_hat
      ──────────  ──────  ──────  ────────  ──────  ────────  ──────
               α  1.5571  0.0535    0.0024  0.0040   87.3374  1.0194
               β  1.0574  0.0505    0.0023  0.0038  133.5800  1.0140
               γ  2.8557  0.1402    0.0063  0.0090   95.9407  1.0138
               δ  0.9331  0.0505    0.0023  0.0033   95.7488  1.0161
               σ  0.8052  0.0399    0.0018  0.0020  329.7764  0.9981
    
    Quantiles
      parameters    2.5%   25.0%   50.0%   75.0%   97.5%
      ──────────  ──────  ──────  ──────  ──────  ──────
               α  1.4561  1.5205  1.5569  1.5909  1.6727
               β  0.9631  1.0227  1.0556  1.0831  1.1629
               γ  2.5758  2.7561  2.8590  2.9456  3.1454
               δ  0.8384  0.8969  0.9327  0.9640  1.0374
               σ  0.7357  0.7767  0.8017  0.8306  0.8885




For more examples of adjoint usage on large parameter models, consult the [DiffEqFlux documentation](https://diffeqflux.sciml.ai/dev/).

## Inference of a Stochastic Differential Equation
A Stochastic Differential Equation ([SDE](https://diffeq.sciml.ai/stable/tutorials/sde_example/)) is a differential equation that has a stochastic (noise) term in the expression of the derivatives. Here we fit a Stochastic version of the Lokta-Volterra system.

We use a quasi-likelihood approach in which all trajectories of a solution are compared instead of a reduction such as mean, this increases the robustness of fitting and makes the likelihood more identifiable. We use SOSRI to solve the equation. The NUTS sampler is a bit sensitive to the stochastic optimization since the gradient is then changing with every calculation, so we use NUTS with a target acceptance rate of `0.25`.


```julia
u0 = [1.0,1.0]
tspan = (0.0,10.0)
function multiplicative_noise!(du,u,p,t)
  x,y = u
  du[1] = p[5]*x
  du[2] = p[6]*y
end
p = [1.5,1.0,3.0,1.0,0.1,0.1]

function lotka_volterra!(du,u,p,t)
  x,y = u
  α,β,γ,δ = p
  du[1] = dx = α*x - β*x*y
  du[2] = dy = δ*x*y - γ*y
end


prob_sde = SDEProblem(lotka_volterra!,multiplicative_noise!,u0,tspan,p)

ensembleprob = EnsembleProblem(prob_sde)
@time data = solve(ensembleprob,SOSRI(),saveat=0.1,trajectories=1000)
plot(EnsembleSummary(data))
```

     15.129917 seconds (74.32 M allocations: 8.627 GiB, 8.78% gc time)





![svg](/tutorials/10_BayesianDiffEq_files/10_BayesianDiffEq_48_1.svg)




```julia
Turing.setadbackend(:forwarddiff)
@model function fitlv(data, prob)
    σ ~ InverseGamma(2,3)
    α ~ truncated(Normal(1.3,0.5),0.5,2.5)
    β ~ truncated(Normal(1.2,0.25),0.5,2)
    γ ~ truncated(Normal(3.2,0.25),2.2,4.0)
    δ ~ truncated(Normal(1.2,0.25),0.5,2.0)
    ϕ1 ~ truncated(Normal(0.12,0.3),0.05,0.25)
    ϕ2 ~ truncated(Normal(0.12,0.3),0.05,0.25)
    p = [α,β,γ,δ,ϕ1,ϕ2]
    prob = remake(prob, p=p)
    predicted = solve(prob,SOSRI(),saveat=0.1)

    if predicted.retcode != :Success
        Turing.acclogp!(_varinfo, -Inf)
    end
    for j in 1:length(data)
        for i = 1:length(predicted)
            data[j][i] ~ MvNormal(predicted[i],σ)
        end
    end
end;
```

We use NUTS sampler with a low acceptance ratio and initial parameters since estimating the parameters of SDE with HMC poses a challenge. Probabilistic nature of the SDE solution makes the likelihood function noisy which poses a challenge for NUTS since the gradient is then changing with every calculation. SGHMC might be better suited to be used here.


```julia
model = fitlv(data, prob_sde)
chain = sample(model, NUTS(0.25), 5000, init_theta = [1.5,1.3,1.2,2.7,1.2,0.12,0.12])
plot(chain)
```




![svg](/tutorials/10_BayesianDiffEq_files/10_BayesianDiffEq_51_0.svg)




```julia

```
