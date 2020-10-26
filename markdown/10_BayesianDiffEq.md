---
title: Bayesian Estimation of Differential Equations
permalink: /:collection/:name/
---

Most of the scientific community deals with the basic problem of trying to mathematically model the reality around them and this often involves dynamical systems. The general trend to model these complex dynamical systems is through the use of differential equations. Differential equation models often have non-measurable parameters. The popular “forward-problem” of simulation consists of solving the differential equations for a given set of parameters, the “inverse problem” to simulation, known as parameter estimation, is the process of utilizing data to determine these model parameters. Bayesian inference provides a robust approach to parameter estimation with quantified uncertainty.


```julia
using Turing, Distributions, DataFrames, DifferentialEquations, DiffEqSensitivity

# Import MCMCChain, and StatsPlots for visualizations and diagnostics.
using MCMCChains, StatsPlots

# Set a seed for reproducibility.
using Random
Random.seed!(12);
```

## The Lotka-Volterra Model

The Lotka–Volterra equations, also known as the predator–prey equations, are a pair of first-order nonlinear differential equations, frequently used to describe the dynamics of biological systems in which two species interact, one as a predator and the other as prey. The populations change through time according to the pair of equations:

$$\frac{dx}{dt} = (\alpha - \beta y)x$$
 
$$\frac{dy}{dt} = (\delta x - \gamma)y$$



```julia
function lotka_volterra(du,u,p,t)
  x, y = u
  α, β, δ, γ = p
  du[1] = dx = (α - β*y)x
  du[2] = dy = (δ*x - γ)y
end
p = [1.5, 1.0, 3.0, 1.0]
u0 = [1.0,1.0]
prob = ODEProblem(lotka_volterra,u0,(0.0,10.0),p)
sol = solve(prob,Tsit5())
plot(sol)
```




![svg](/tutorials/10_BayesianDiffEq_files/10_BayesianDiffEq_3_0.svg)



We'll generate the data to use for the parameter estimation from simulation. 
With the `saveat` [argument](https://docs.sciml.ai/latest/basics/common_solver_opts/) we specify that the solution is stored only at `0.1` time units. 


```julia
odedata = Array(solve(prob,Tsit5(),saveat=0.1))
```




    2×101 Array{Float64,2}:
     1.0  1.03981  1.05332  1.03247  0.972908  …  0.133965  0.148601  0.165247
     1.0  1.22939  1.52387  1.88714  2.30908      0.476902  0.450153  0.426924



## Fitting Lotka-Volterra with DiffEqBayes

[DiffEqBayes.jl](https://github.com/SciML/DiffEqBayes.jl) is a high level package that set of extension functionality for estimating the parameters of differential equations using Bayesian methods. It allows the choice of using CmdStan.jl, Turing.jl, DynamicHMC.jl and ApproxBayes.jl to perform a Bayesian estimation of a differential equation problem specified via the DifferentialEquations.jl interface. You can read the [docs](https://docs.sciml.ai/latest/analysis/parameter_estimation/#Bayesian-Methods-1) for an understanding of the available functionality.


```julia
using DiffEqBayes
t = 0:0.1:10.0
priors = [truncated(Normal(1.5,0.5),0.5,2.5),truncated(Normal(1.2,0.5),0,2),truncated(Normal(3.0,0.5),1,4),truncated(Normal(1.0,0.5),0,2)]
bayesian_result_turing = turing_inference(prob,Tsit5(),t,odedata,priors,num_samples=10_000)
```

    ┌ Warning: The current proposal will be rejected due to numerical error(s).
    │   isfinite.((θ, r, ℓπ, ℓκ)) = (true, false, false, false)
    └ @ AdvancedHMC /home/cameron/.julia/packages/AdvancedHMC/WJCQA/src/hamiltonian.jl:47
    ┌ Warning: The current proposal will be rejected due to numerical error(s).
    │   isfinite.((θ, r, ℓπ, ℓκ)) = (true, false, false, false)
    └ @ AdvancedHMC /home/cameron/.julia/packages/AdvancedHMC/WJCQA/src/hamiltonian.jl:47
    ┌ Info: Found initial step size
    │   ϵ = 0.00625
    └ @ Turing.Inference /home/cameron/.julia/packages/Turing/GMBTf/src/inference/hmc.jl:629
    ┌ Warning: The current proposal will be rejected due to numerical error(s).
    │   isfinite.((θ, r, ℓπ, ℓκ)) = (true, false, false, false)
    └ @ AdvancedHMC /home/cameron/.julia/packages/AdvancedHMC/WJCQA/src/hamiltonian.jl:47
    ┌ Warning: The current proposal will be rejected due to numerical error(s).
    │   isfinite.((θ, r, ℓπ, ℓκ)) = (true, false, false, false)
    └ @ AdvancedHMC /home/cameron/.julia/packages/AdvancedHMC/WJCQA/src/hamiltonian.jl:47
    ┌ Warning: The current proposal will be rejected due to numerical error(s).
    │   isfinite.((θ, r, ℓπ, ℓκ)) = (true, false, false, false)
    └ @ AdvancedHMC /home/cameron/.julia/packages/AdvancedHMC/WJCQA/src/hamiltonian.jl:47
    ┌ Warning: The current proposal will be rejected due to numerical error(s).
    │   isfinite.((θ, r, ℓπ, ℓκ)) = (true, false, false, false)
    └ @ AdvancedHMC /home/cameron/.julia/packages/AdvancedHMC/WJCQA/src/hamiltonian.jl:47
    ┌ Warning: The current proposal will be rejected due to numerical error(s).
    │   isfinite.((θ, r, ℓπ, ℓκ)) = (true, false, false, false)
    └ @ AdvancedHMC /home/cameron/.julia/packages/AdvancedHMC/WJCQA/src/hamiltonian.jl:47
    ┌ Warning: The current proposal will be rejected due to numerical error(s).
    │   isfinite.((θ, r, ℓπ, ℓκ)) = (true, false, false, false)
    └ @ AdvancedHMC /home/cameron/.julia/packages/AdvancedHMC/WJCQA/src/hamiltonian.jl:47
    ┌ Warning: The current proposal will be rejected due to numerical error(s).
    │   isfinite.((θ, r, ℓπ, ℓκ)) = (true, false, false, false)
    └ @ AdvancedHMC /home/cameron/.julia/packages/AdvancedHMC/WJCQA/src/hamiltonian.jl:47
    ┌ Warning: The current proposal will be rejected due to numerical error(s).
    │   isfinite.((θ, r, ℓπ, ℓκ)) = (true, false, false, false)
    └ @ AdvancedHMC /home/cameron/.julia/packages/AdvancedHMC/WJCQA/src/hamiltonian.jl:47
    ┌ Warning: The current proposal will be rejected due to numerical error(s).
    │   isfinite.((θ, r, ℓπ, ℓκ)) = (true, false, false, false)
    └ @ AdvancedHMC /home/cameron/.julia/packages/AdvancedHMC/WJCQA/src/hamiltonian.jl:47
    ┌ Warning: The current proposal will be rejected due to numerical error(s).
    │   isfinite.((θ, r, ℓπ, ℓκ)) = (true, false, false, false)
    └ @ AdvancedHMC /home/cameron/.julia/packages/AdvancedHMC/WJCQA/src/hamiltonian.jl:47
    ┌ Warning: The current proposal will be rejected due to numerical error(s).
    │   isfinite.((θ, r, ℓπ, ℓκ)) = (true, false, false, false)
    └ @ AdvancedHMC /home/cameron/.julia/packages/AdvancedHMC/WJCQA/src/hamiltonian.jl:47
    ┌ Warning: The current proposal will be rejected due to numerical error(s).
    │   isfinite.((θ, r, ℓπ, ℓκ)) = (true, false, false, false)
    └ @ AdvancedHMC /home/cameron/.julia/packages/AdvancedHMC/WJCQA/src/hamiltonian.jl:47
    ┌ Warning: The current proposal will be rejected due to numerical error(s).
    │   isfinite.((θ, r, ℓπ, ℓκ)) = (true, false, false, false)
    └ @ AdvancedHMC /home/cameron/.julia/packages/AdvancedHMC/WJCQA/src/hamiltonian.jl:47
    ┌ Warning: The current proposal will be rejected due to numerical error(s).
    │   isfinite.((θ, r, ℓπ, ℓκ)) = (true, false, false, false)
    └ @ AdvancedHMC /home/cameron/.julia/packages/AdvancedHMC/WJCQA/src/hamiltonian.jl:47
    ┌ Warning: The current proposal will be rejected due to numerical error(s).
    │   isfinite.((θ, r, ℓπ, ℓκ)) = (true, false, false, false)
    └ @ AdvancedHMC /home/cameron/.julia/packages/AdvancedHMC/WJCQA/src/hamiltonian.jl:47
    ┌ Warning: The current proposal will be rejected due to numerical error(s).
    │   isfinite.((θ, r, ℓπ, ℓκ)) = (true, false, false, false)
    └ @ AdvancedHMC /home/cameron/.julia/packages/AdvancedHMC/WJCQA/src/hamiltonian.jl:47
    ┌ Warning: The current proposal will be rejected due to numerical error(s).
    │   isfinite.((θ, r, ℓπ, ℓκ)) = (true, false, false, false)
    └ @ AdvancedHMC /home/cameron/.julia/packages/AdvancedHMC/WJCQA/src/hamiltonian.jl:47
    ┌ Warning: The current proposal will be rejected due to numerical error(s).
    │   isfinite.((θ, r, ℓπ, ℓκ)) = (true, false, false, false)
    └ @ AdvancedHMC /home/cameron/.julia/packages/AdvancedHMC/WJCQA/src/hamiltonian.jl:47
    ┌ Warning: The current proposal will be rejected due to numerical error(s).
    │   isfinite.((θ, r, ℓπ, ℓκ)) = (true, false, false, false)
    └ @ AdvancedHMC /home/cameron/.julia/packages/AdvancedHMC/WJCQA/src/hamiltonian.jl:47
    ┌ Warning: The current proposal will be rejected due to numerical error(s).
    │   isfinite.((θ, r, ℓπ, ℓκ)) = (true, false, false, false)
    └ @ AdvancedHMC /home/cameron/.julia/packages/AdvancedHMC/WJCQA/src/hamiltonian.jl:47
    ┌ Warning: The current proposal will be rejected due to numerical error(s).
    │   isfinite.((θ, r, ℓπ, ℓκ)) = (true, false, false, false)
    └ @ AdvancedHMC /home/cameron/.julia/packages/AdvancedHMC/WJCQA/src/hamiltonian.jl:47
    ┌ Warning: The current proposal will be rejected due to numerical error(s).
    │   isfinite.((θ, r, ℓπ, ℓκ)) = (true, false, false, false)
    └ @ AdvancedHMC /home/cameron/.julia/packages/AdvancedHMC/WJCQA/src/hamiltonian.jl:47
    ┌ Warning: The current proposal will be rejected due to numerical error(s).
    │   isfinite.((θ, r, ℓπ, ℓκ)) = (true, false, false, false)
    └ @ AdvancedHMC /home/cameron/.julia/packages/AdvancedHMC/WJCQA/src/hamiltonian.jl:47
    ┌ Warning: The current proposal will be rejected due to numerical error(s).
    │   isfinite.((θ, r, ℓπ, ℓκ)) = (true, false, false, false)
    └ @ AdvancedHMC /home/cameron/.julia/packages/AdvancedHMC/WJCQA/src/hamiltonian.jl:47





    Object of type Chains, with data of type 9000×17×1 Array{Float64,3}
    
    Iterations        = 1:9000
    Thinning interval = 1
    Chains            = 1
    Samples per chain = 9000
    internals         = acceptance_rate, hamiltonian_energy, hamiltonian_energy_error, is_accept, log_density, lp, max_hamiltonian_energy_error, n_steps, nom_step_size, numerical_error, step_size, tree_depth
    parameters        = theta[1], theta[2], theta[3], theta[4], σ[1]
    
    2-element Array{ChainDataFrame,1}
    
    Summary Statistics
      parameters    mean     std  naive_se    mcse        ess   r_hat
      ──────────  ──────  ──────  ────────  ──────  ─────────  ──────
        theta[1]  2.3263  0.1073    0.0011  0.0021  2202.3643  1.0000
        theta[2]  1.5434  0.0957    0.0010  0.0019  2575.4033  1.0002
        theta[3]  3.1259  0.1983    0.0021  0.0031  4127.1344  1.0000
        theta[4]  1.8356  0.0827    0.0009  0.0017  2189.2825  1.0000
            σ[1]  0.8569  0.0436    0.0005  0.0005  6856.5421  0.9999
    
    Quantiles
      parameters    2.5%   25.0%   50.0%   75.0%   97.5%
      ──────────  ──────  ──────  ──────  ──────  ──────
        theta[1]  2.1185  2.2428  2.3337  2.4169  2.4916
        theta[2]  1.3655  1.4750  1.5422  1.6075  1.7367
        theta[3]  2.7571  2.9893  3.1166  3.2546  3.5440
        theta[4]  1.6902  1.7708  1.8307  1.9006  1.9868
            σ[1]  0.7755  0.8266  0.8551  0.8847  0.9484




The estimated parameters are clearly very close to the desired parameter values. We can also check that the chains have converged in the plot.


```julia
plot(bayesian_result_turing)
```




![svg](/tutorials/10_BayesianDiffEq_files/10_BayesianDiffEq_9_0.svg)



## Direct Handling of Bayesian Estimation with Turing

You could want to do some sort of reduction with the differential equation's solution or use it in some other way as well. In those cases DiffEqBayes might not be useful. Turing and DifferentialEquations are completely composable and you can write of the differential equation inside a Turing `@model` and it will just work.

We can rewrite the Lotka Volterra parameter estimation problem with a Turing `@model` interface as below


```julia
Turing.setadbackend(:forwarddiff)

@model function fitlv(data)
    σ ~ InverseGamma(2, 3)
    α ~ truncated(Normal(1.5,0.5),0.5,2.5)
    β ~ truncated(Normal(1.2,0.5),0,2)
    γ ~ truncated(Normal(3.0,0.5),1,4)
    δ ~ truncated(Normal(1.0,0.5),0,2)

    p = [α,β,γ,δ]
    prob = ODEProblem(lotka_volterra,u0,(0.0,10.0),p)
    predicted = solve(prob,Tsit5(),saveat=0.1)

    for i = 1:length(predicted)
        data[:,i] ~ MvNormal(predicted[i], σ)
    end
end

model = fitlv(odedata)
chain = sample(model, NUTS(.65),10000)
```

    ┌ Info: Found initial step size
    │   ϵ = 0.2
    └ @ Turing.Inference /home/cameron/.julia/packages/Turing/GMBTf/src/inference/hmc.jl:629
    [32mSampling: 100%|█████████████████████████████████████████| Time: 0:02:48[39m





    Object of type Chains, with data of type 9000×17×1 Array{Float64,3}
    
    Iterations        = 1:9000
    Thinning interval = 1
    Chains            = 1
    Samples per chain = 9000
    internals         = acceptance_rate, hamiltonian_energy, hamiltonian_energy_error, is_accept, log_density, lp, max_hamiltonian_energy_error, n_steps, nom_step_size, numerical_error, step_size, tree_depth
    parameters        = α, β, γ, δ, σ
    
    2-element Array{ChainDataFrame,1}
    
    Summary Statistics
      parameters    mean     std  naive_se    mcse        ess   r_hat
      ──────────  ──────  ──────  ────────  ──────  ─────────  ──────
               α  1.4999  0.0060    0.0001  0.0001  2341.1779  0.9999
               β  0.9999  0.0037    0.0000  0.0001  2440.6968  0.9999
               γ  3.0001  0.0047    0.0000  0.0001  4070.6419  1.0003
               δ  1.0001  0.0032    0.0000  0.0001  2324.4733  0.9999
               σ  0.0151  0.0011    0.0000  0.0000  4591.2728  0.9999
    
    Quantiles
      parameters    2.5%   25.0%   50.0%   75.0%   97.5%
      ──────────  ──────  ──────  ──────  ──────  ──────
               α  1.4881  1.4960  1.4998  1.5038  1.5118
               β  0.9925  0.9975  0.9999  1.0024  1.0074
               γ  2.9911  2.9970  3.0000  3.0032  3.0095
               δ  0.9937  0.9979  1.0001  1.0022  1.0066
               σ  0.0131  0.0143  0.0150  0.0158  0.0173




## Scaling to Large Models: Adjoint Sensitivities

DifferentialEquations.jl's efficiency for large stiff models has been shown in multiple [benchmarks](https://github.com/SciML/DiffEqBenchmarks.jl). To learn more about how to optimize solving performance for stiff problems you can take a look at the [docs](https://docs.sciml.ai/latest/tutorials/advanced_ode_example/). 

[Sensitivity analysis](https://docs.sciml.ai/latest/analysis/sensitivity/), or automatic differentiation (AD) of the solver, is provided by the DiffEq suite. The model sensitivities are the derivatives of the solution $$u(t)$$ with respect to the parameters. Specifically, the local sensitivity of the solution to a parameter is defined by how much the solution would change by changes in the parameter. Sensitivity analysis provides a cheap way to calculate the gradient of the solution which can be used in parameter estimation and other optimization tasks.


The AD ecosystem in Julia allows you to switch between forward mode, reverse mode, source to source and other choices of AD and have it work with any Julia code. For a user to make use of this within [SciML](https://sciml.ai), [high level interactions in `solve`](https://docs.sciml.ai/latest/analysis/sensitivity/#High-Level-Interface:-sensealg-1) automatically plug into those AD systems to allow for choosing advanced sensitivity analysis (derivative calculation) [methods](https://docs.sciml.ai/latest/analysis/sensitivity/#Sensitivity-Algorithms-1). 

More theoretical details on these methods can be found at: https://docs.sciml.ai/latest/extras/sensitivity_math/.

While these sensitivity analysis methods may seem complicated (and they are!), using them is dead simple. Here is a version of the Lotka-Volterra model with adjoints enabled.

All we had to do is switch the AD backend to one of the adjoint-compatible backends (ReverseDiff, Tracker, or Zygote) and boom the system takes over and we're using adjoint methods! Notice that on this model adjoints are slower. This is because adjoints have a higher overhead on small parameter models and we suggest only using these methods for models with around 100 parameters or more. For more details, see https://arxiv.org/abs/1812.01892.


```julia
Turing.setadbackend(:zygote)
@model function fitlv(data)
    σ ~ InverseGamma(2, 3)
    α ~ truncated(Normal(1.5,0.5),0.5,2.5)
    β ~ truncated(Normal(1.2,0.5),0,2)
    γ ~ truncated(Normal(3.0,0.5),1,4)
    δ ~ truncated(Normal(1.0,0.5),0,2)
    p = [α,β,γ,δ]
    prob = ODEProblem(lotka_volterra,u0,(0.0,10.0),p)
    predicted = solve(prob,saveat=0.1)
    for i = 1:length(predicted)
        data[:,i] ~ MvNormal(predicted[i], σ)
    end
end;
model = fitlv(odedata)
chain = sample(model, NUTS(.65),1000)
```

    ┌ Info: Found initial step size
    │   ϵ = 0.2
    └ @ Turing.Inference /home/cameron/.julia/packages/Turing/GMBTf/src/inference/hmc.jl:629
    [32mSampling: 100%|█████████████████████████████████████████| Time: 0:10:42[39m





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
               α  1.4997  0.0052    0.0002  0.0003  201.5277  1.0046
               β  0.9999  0.0033    0.0001  0.0001  219.1974  1.0027
               γ  3.0003  0.0047    0.0002  0.0003  290.3332  1.0014
               δ  1.0002  0.0029    0.0001  0.0002  210.0807  1.0046
               σ  0.0151  0.0010    0.0000  0.0001  246.6502  1.0017
    
    Quantiles
      parameters    2.5%   25.0%   50.0%   75.0%   97.5%
      ──────────  ──────  ──────  ──────  ──────  ──────
               α  1.4892  1.4962  1.5002  1.5030  1.5108
               β  0.9934  0.9978  1.0000  1.0019  1.0066
               γ  2.9910  2.9971  3.0002  3.0039  3.0084
               δ  0.9943  0.9983  1.0000  1.0021  1.0060
               σ  0.0131  0.0143  0.0151  0.0158  0.0172




Now we can exercise control of the sensitivity analysis method that is used by using the `sensealg` keyword argument. Let's choose the `InterpolatingAdjoint` from the available AD [methods](https://docs.sciml.ai/latest/analysis/sensitivity/#Sensitivity-Algorithms-1) and enable a compiled ReverseDiff vector-Jacobian product:


```julia
@model function fitlv(data)
    σ ~ InverseGamma(2, 3)
    α ~ truncated(Normal(1.5,0.5),0.5,2.5)
    β ~ truncated(Normal(1.2,0.5),0,2)
    γ ~ truncated(Normal(3.0,0.5),1,4)
    δ ~ truncated(Normal(1.0,0.5),0,2)
    p = [α,β,γ,δ]
    prob = ODEProblem(lotka_volterra,u0,(0.0,10.0),p)
    predicted = solve(prob,saveat=0.1,sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))
    for i = 1:length(predicted)
        data[:,i] ~ MvNormal(predicted[i], σ)
    end
end;
model = fitlv(odedata)
@time chain = sample(model, NUTS(.65),1000)
```

    ┌ Info: Found initial step size
    │   ϵ = 0.2
    └ @ Turing.Inference /home/cameron/.julia/packages/Turing/GMBTf/src/inference/hmc.jl:629
    [32mSampling:  11%|████▍                                    |  ETA: 0:06:27[39m┌ Warning: The current proposal will be rejected due to numerical error(s).
    │   isfinite.((θ, r, ℓπ, ℓκ)) = (true, false, false, false)
    └ @ AdvancedHMC /home/cameron/.julia/packages/AdvancedHMC/WJCQA/src/hamiltonian.jl:47
    [32mSampling:  13%|█████▍                                   |  ETA: 0:05:58[39m┌ Warning: The current proposal will be rejected due to numerical error(s).
    │   isfinite.((θ, r, ℓπ, ℓκ)) = (true, false, false, false)
    └ @ AdvancedHMC /home/cameron/.julia/packages/AdvancedHMC/WJCQA/src/hamiltonian.jl:47
    [32mSampling:  15%|██████▎                                  |  ETA: 0:05:27[39m┌ Warning: The current proposal will be rejected due to numerical error(s).
    │   isfinite.((θ, r, ℓπ, ℓκ)) = (true, false, false, false)
    └ @ AdvancedHMC /home/cameron/.julia/packages/AdvancedHMC/WJCQA/src/hamiltonian.jl:47
    [32mSampling:  21%|████████▌                                |  ETA: 0:04:20[39m┌ Warning: The current proposal will be rejected due to numerical error(s).
    │   isfinite.((θ, r, ℓπ, ℓκ)) = (true, false, false, false)
    └ @ AdvancedHMC /home/cameron/.julia/packages/AdvancedHMC/WJCQA/src/hamiltonian.jl:47
    [32mSampling:  23%|█████████▎                               |  ETA: 0:04:03[39m┌ Warning: The current proposal will be rejected due to numerical error(s).
    │   isfinite.((θ, r, ℓπ, ℓκ)) = (true, false, false, false)
    └ @ AdvancedHMC /home/cameron/.julia/packages/AdvancedHMC/WJCQA/src/hamiltonian.jl:47
    [32mSampling:  24%|██████████                               |  ETA: 0:03:48[39m┌ Warning: The current proposal will be rejected due to numerical error(s).
    │   isfinite.((θ, r, ℓπ, ℓκ)) = (true, false, false, false)
    └ @ AdvancedHMC /home/cameron/.julia/packages/AdvancedHMC/WJCQA/src/hamiltonian.jl:47
    [32mSampling:  28%|███████████▌                             |  ETA: 0:03:27[39m┌ Warning: The current proposal will be rejected due to numerical error(s).
    │   isfinite.((θ, r, ℓπ, ℓκ)) = (true, false, false, false)
    └ @ AdvancedHMC /home/cameron/.julia/packages/AdvancedHMC/WJCQA/src/hamiltonian.jl:47
    [32mSampling:  29%|███████████▊                             |  ETA: 0:03:24[39m┌ Warning: The current proposal will be rejected due to numerical error(s).
    │   isfinite.((θ, r, ℓπ, ℓκ)) = (true, false, false, false)
    └ @ AdvancedHMC /home/cameron/.julia/packages/AdvancedHMC/WJCQA/src/hamiltonian.jl:47
    [32mSampling:  29%|████████████                             |  ETA: 0:03:20[39m┌ Warning: The current proposal will be rejected due to numerical error(s).
    │   isfinite.((θ, r, ℓπ, ℓκ)) = (true, false, false, false)
    └ @ AdvancedHMC /home/cameron/.julia/packages/AdvancedHMC/WJCQA/src/hamiltonian.jl:47
    [32mSampling:  36%|███████████████                          |  ETA: 0:02:45[39m┌ Warning: The current proposal will be rejected due to numerical error(s).
    │   isfinite.((θ, r, ℓπ, ℓκ)) = (true, false, false, false)
    └ @ AdvancedHMC /home/cameron/.julia/packages/AdvancedHMC/WJCQA/src/hamiltonian.jl:47
    [32mSampling:  37%|███████████████▏                         |  ETA: 0:02:44[39m┌ Warning: The current proposal will be rejected due to numerical error(s).
    │   isfinite.((θ, r, ℓπ, ℓκ)) = (true, false, false, false)
    └ @ AdvancedHMC /home/cameron/.julia/packages/AdvancedHMC/WJCQA/src/hamiltonian.jl:47
    [32mSampling:  39%|████████████████                         |  ETA: 0:02:36[39m┌ Warning: The current proposal will be rejected due to numerical error(s).
    │   isfinite.((θ, r, ℓπ, ℓκ)) = (true, false, false, false)
    └ @ AdvancedHMC /home/cameron/.julia/packages/AdvancedHMC/WJCQA/src/hamiltonian.jl:47
    [32mSampling:  46%|██████████████████▉                      |  ETA: 0:02:08[39m┌ Warning: The current proposal will be rejected due to numerical error(s).
    │   isfinite.((θ, r, ℓπ, ℓκ)) = (true, false, false, false)
    └ @ AdvancedHMC /home/cameron/.julia/packages/AdvancedHMC/WJCQA/src/hamiltonian.jl:47
    [32mSampling:  48%|███████████████████▊                     |  ETA: 0:02:03[39m┌ Warning: The current proposal will be rejected due to numerical error(s).
    │   isfinite.((θ, r, ℓπ, ℓκ)) = (true, false, false, false)
    └ @ AdvancedHMC /home/cameron/.julia/packages/AdvancedHMC/WJCQA/src/hamiltonian.jl:47
    [32mSampling:  49%|████████████████████▏                    |  ETA: 0:02:01[39m┌ Warning: The current proposal will be rejected due to numerical error(s).
    │   isfinite.((θ, r, ℓπ, ℓκ)) = (true, false, false, false)
    └ @ AdvancedHMC /home/cameron/.julia/packages/AdvancedHMC/WJCQA/src/hamiltonian.jl:47
    [32mSampling:  50%|████████████████████▎                    |  ETA: 0:02:00[39m┌ Warning: The current proposal will be rejected due to numerical error(s).
    │   isfinite.((θ, r, ℓπ, ℓκ)) = (true, false, false, false)
    └ @ AdvancedHMC /home/cameron/.julia/packages/AdvancedHMC/WJCQA/src/hamiltonian.jl:47
    [32mSampling: 100%|█████████████████████████████████████████| Time: 0:03:32[39m


    225.663919 seconds (1.41 G allocations: 66.216 GiB, 5.25% gc time)





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
               α  0.9122  0.2810    0.0126  0.0152  211.4497  0.9992
               β  1.8499  0.1141    0.0051  0.0055  302.7650  1.0018
               γ  2.5879  0.3299    0.0148  0.0228  307.5110  0.9997
               δ  0.1259  0.0221    0.0010  0.0007  219.5371  1.0006
               σ  0.8746  0.0437    0.0020  0.0017  342.6660  1.0008
    
    Quantiles
      parameters    2.5%   25.0%   50.0%   75.0%   97.5%
      ──────────  ──────  ──────  ──────  ──────  ──────
               α  0.5060  0.6920  0.8932  1.0874  1.5467
               β  1.5810  1.7796  1.8709  1.9437  1.9873
               γ  1.9519  2.3707  2.5999  2.8158  3.1966
               δ  0.0843  0.1103  0.1245  0.1410  0.1704
               σ  0.7984  0.8444  0.8722  0.9044  0.9651




For more examples of adjoint usage on large parameter models, consult the [DiffEqFlux documentation](https://diffeqflux.sciml.ai/dev/)

## Including Process Noise: Estimation of Stochastic Differential Equations

This can be easily extended to Stochastic Differential Equations as well.

Let's create the Lotka Volterra equation with some noise and try out estimating it with the same framework we have set up before.

Our equations now become:

$$dx = (\alpha - \beta y)xdt + \phi_1 xdW_1$$

$$dy = (\delta x - \gamma)ydt + \phi_2 ydW_2$$


```julia
function lotka_volterra_noise(du,u,p,t)
    du[1] = p[5]*u[1]
    du[2] = p[6]*u[2]
end
p = [1.5, 1.0, 3.0, 1.0, 0.3, 0.3]
prob = SDEProblem(lotka_volterra,lotka_volterra_noise,u0,(0.0,10.0),p)
```




    [36mSDEProblem[0m with uType [36mArray{Float64,1}[0m and tType [36mFloat64[0m. In-place: [36mtrue[0m
    timespan: (0.0, 10.0)
    u0: [1.0, 1.0]



Solving it repeatedly confirms the randomness of the solution


```julia
sol = solve(prob,saveat=0.01)
p1 = plot(sol)
sol = solve(prob,saveat=0.01)
p2 = plot(sol)
sol = solve(prob,saveat=0.01)
p3 = plot(sol)
plot(p1,p2,p3)
```




![svg](/tutorials/10_BayesianDiffEq_files/10_BayesianDiffEq_23_0.svg)



With the `MonteCarloSummary` it is easy to summarize the results from multiple runs through the `EnsembleProblem` interface, here we run the problem for 1000 `trajectories` and visualize the summary:


```julia
sol = solve(EnsembleProblem(prob),SRIW1(),saveat=0.1,trajectories=500)
summ = MonteCarloSummary(sol)
plot(summ)
```




![svg](/tutorials/10_BayesianDiffEq_files/10_BayesianDiffEq_25_0.svg)



Get data from the means to fit:


```julia
using DiffEqBase.EnsembleAnalysis
averagedata = Array(timeseries_steps_mean(sol))
```




    2×101 Array{Float64,2}:
     1.0  1.04218  1.05885  1.03187  0.967422  …  0.190811  0.197071  0.203714
     1.0  1.22803  1.5283   1.89036  2.30967      1.16424   1.11006   1.07984



Now fit the means with Turing.

We will utilize multithreading with the [`EnsembleProblem`](https://docs.sciml.ai/stable/tutorials/sde_example/#Ensemble-Simulations-1) interface to speed up the SDE parameter estimation.


```julia
Threads.nthreads()
```




    16




```julia
Turing.setadbackend(:forwarddiff)

@model function fitlv(data)
    σ ~ InverseGamma(2, 3)
    α ~ truncated(Normal(1.5,0.5),0.5,2.5)
    β ~ truncated(Normal(1.2,0.5),0,2)
    γ ~ truncated(Normal(3.0,0.5),1,4)
    δ ~ truncated(Normal(1.0,0.5),0,2)
    ϕ1 ~ truncated(Normal(1.2,0.5),0.1,1)
    ϕ2 ~ truncated(Normal(1.2,0.5),0.1,1)

    p = [α,β,γ,δ,ϕ1,ϕ2]
    prob = SDEProblem(lotka_volterra,lotka_volterra_noise,u0,(0.0,10.0),p)
    ensemble_predicted = solve(EnsembleProblem(prob),SRIW1(),saveat=0.1,trajectories=500)
    predicted_means = timeseries_steps_mean(ensemble_predicted)

    for i = 1:length(predicted_means)
        data[:,i] ~ MvNormal(predicted_means[i], σ)
    end
end;

model = fitlv(averagedata)
chain = sample(model, NUTS(.65),500)
```

    ┌ Info: Found initial step size
    │   ϵ = 0.2
    └ @ Turing.Inference /home/cameron/.julia/packages/Turing/GMBTf/src/inference/hmc.jl:629
    ┌ Warning: The current proposal will be rejected due to numerical error(s).
    │   isfinite.((θ, r, ℓπ, ℓκ)) = (true, false, false, false)
    └ @ AdvancedHMC /home/cameron/.julia/packages/AdvancedHMC/WJCQA/src/hamiltonian.jl:47
    [32mSampling:   0%|▏                                        |  ETA: 0:03:49[39m┌ Warning: The current proposal will be rejected due to numerical error(s).
    │   isfinite.((θ, r, ℓπ, ℓκ)) = (true, false, false, false)
    └ @ AdvancedHMC /home/cameron/.julia/packages/AdvancedHMC/WJCQA/src/hamiltonian.jl:47
    ┌ Warning: The current proposal will be rejected due to numerical error(s).
    │   isfinite.((θ, r, ℓπ, ℓκ)) = (true, false, false, false)
    └ @ AdvancedHMC /home/cameron/.julia/packages/AdvancedHMC/WJCQA/src/hamiltonian.jl:47
    [32mSampling: 100%|█████████████████████████████████████████| Time: 2:33:35[39m





    Object of type Chains, with data of type 250×19×1 Array{Float64,3}
    
    Iterations        = 1:250
    Thinning interval = 1
    Chains            = 1
    Samples per chain = 250
    internals         = acceptance_rate, hamiltonian_energy, hamiltonian_energy_error, is_accept, log_density, lp, max_hamiltonian_energy_error, n_steps, nom_step_size, numerical_error, step_size, tree_depth
    parameters        = α, β, γ, δ, σ, ϕ1, ϕ2
    
    2-element Array{ChainDataFrame,1}
    
    Summary Statistics
      parameters    mean     std  naive_se    mcse     ess   r_hat
      ──────────  ──────  ──────  ────────  ──────  ──────  ──────
               α  1.6255  0.0000    0.0000  0.0000  2.0325  2.5501
               β  1.1163  0.0000    0.0000  0.0000  2.0325     Inf
               γ  3.2056  0.0000    0.0000  0.0000  2.0325  0.9960
               δ  0.9268  0.0000    0.0000  0.0000  2.0325  2.9880
               σ  0.0669  0.0000    0.0000  0.0000  2.0325  1.1011
              ϕ1  0.2329  0.0000    0.0000  0.0000  2.0325  3.2549
              ϕ2  0.2531  0.0000    0.0000  0.0000  2.0325  0.9960
    
    Quantiles
      parameters    2.5%   25.0%   50.0%   75.0%   97.5%
      ──────────  ──────  ──────  ──────  ──────  ──────
               α  1.6255  1.6255  1.6255  1.6255  1.6255
               β  1.1163  1.1163  1.1163  1.1163  1.1163
               γ  3.2056  3.2056  3.2056  3.2056  3.2056
               δ  0.9268  0.9268  0.9268  0.9268  0.9268
               σ  0.0669  0.0669  0.0669  0.0669  0.0669
              ϕ1  0.2329  0.2329  0.2329  0.2329  0.2329
              ϕ2  0.2531  0.2531  0.2531  0.2531  0.2531





```julia

```
