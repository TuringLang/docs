---
title: Bayesian Estimation of Differential Equations
permalink: /:collection/:name/
---

Most of the scientific community deals with the basic problem of trying to mathematically model the reality around them and this often involves dynamical systems. The general trend to model these complex dynamical systems is through the use of differential equations. Differential equation models often have non-measurable parameters. The popular “forward-problem” of simulation consists of solving the differential equations for a given set of parameters, the “inverse problem” to simulation, known as parameter estimation, is the process of utilizing data to determine these model parameters. Bayesian inference provides a robust approach to parameter estimation with quantified uncertainty.


```julia
using Turing, Distributions, DataFrames, DifferentialEquations, DiffEqSensitivity

# Import MCMCChain, Plots, and StatsPlots for visualizations and diagnostics.
using MCMCChains, Plots, StatsPlots

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
p = [2.2, 1.0, 2.0, 0.4]
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
     1.0  1.11751  1.22172  1.29665  …  1.31071  1.23009  1.0837   0.892889
     1.0  1.18752  1.44214  1.78356     2.53525  3.14507  3.80916  4.46091



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
    │   ϵ = 0.0125
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
      parameters    mean     std  naive_se    mcse       ess   r_hat
      ──────────  ──────  ──────  ────────  ──────  ────────  ──────
        theta[1]  2.3138  0.2445    0.0026  0.0248   36.1446  1.1426
        theta[2]  1.3113  0.0925    0.0010  0.0069   37.6235  1.0754
        theta[3]  2.6787  0.1880    0.0020  0.0075  149.3134  1.0161
        theta[4]  1.7144  0.5741    0.0061  0.0605   36.1446  1.1525
            σ[1]  1.2880  0.0680    0.0007  0.0034   79.4361  1.0392
    
    Quantiles
      parameters    2.5%   25.0%   50.0%   75.0%   97.5%
      ──────────  ──────  ──────  ──────  ──────  ──────
        theta[1]  1.6991  2.3274  2.3977  2.4500  2.4948
        theta[2]  1.1414  1.2623  1.3194  1.3724  1.4819
        theta[3]  2.3634  2.5086  2.6652  2.8011  3.0856
        theta[4]  0.1872  1.8750  1.9278  1.9659  1.9970
            σ[1]  1.1613  1.2373  1.2847  1.3436  1.4048




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
    │   ϵ = 0.05
    └ @ Turing.Inference /home/cameron/.julia/packages/Turing/GMBTf/src/inference/hmc.jl:629
    [32mSampling:   2%|▉                                        |  ETA: 0:02:08[39m┌ Warning: The current proposal will be rejected due to numerical error(s).
    │   isfinite.((θ, r, ℓπ, ℓκ)) = (true, false, false, false)
    └ @ AdvancedHMC /home/cameron/.julia/packages/AdvancedHMC/WJCQA/src/hamiltonian.jl:47
    [32mSampling:   3%|█▏                                       |  ETA: 0:02:00[39m┌ Warning: The current proposal will be rejected due to numerical error(s).
    │   isfinite.((θ, r, ℓπ, ℓκ)) = (true, false, false, false)
    └ @ AdvancedHMC /home/cameron/.julia/packages/AdvancedHMC/WJCQA/src/hamiltonian.jl:47
    [32mSampling:   3%|█▏                                       |  ETA: 0:02:03[39m┌ Warning: The current proposal will be rejected due to numerical error(s).
    │   isfinite.((θ, r, ℓπ, ℓκ)) = (true, false, false, false)
    └ @ AdvancedHMC /home/cameron/.julia/packages/AdvancedHMC/WJCQA/src/hamiltonian.jl:47
    [32mSampling:   4%|█▌                                       |  ETA: 0:01:58[39m┌ Warning: The current proposal will be rejected due to numerical error(s).
    │   isfinite.((θ, r, ℓπ, ℓκ)) = (true, false, false, false)
    └ @ AdvancedHMC /home/cameron/.julia/packages/AdvancedHMC/WJCQA/src/hamiltonian.jl:47
    [32mSampling: 100%|█████████████████████████████████████████| Time: 0:03:31[39m





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
               α  2.2000  0.0060    0.0001  0.0001  2212.6158  1.0001
               β  1.0000  0.0024    0.0000  0.0000  2276.4414  1.0000
               γ  2.0000  0.0028    0.0000  0.0001  2853.0168  1.0023
               δ  0.4000  0.0006    0.0000  0.0000  2108.4542  1.0004
               σ  0.0151  0.0011    0.0000  0.0000  6389.0453  1.0000
    
    Quantiles
      parameters    2.5%   25.0%   50.0%   75.0%   97.5%
      ──────────  ──────  ──────  ──────  ──────  ──────
               α  2.1881  2.1960  2.2000  2.2041  2.2118
               β  0.9952  0.9984  1.0000  1.0017  1.0047
               γ  1.9945  1.9981  2.0000  2.0019  2.0056
               δ  0.3988  0.3996  0.4000  0.4004  0.4013
               σ  0.0131  0.0143  0.0150  0.0158  0.0174




## Scaling to Large Models: Adjoint Sensitivities

DifferentialEquations.jl's efficiency for large stiff models has been shown in multiple [benchmarks](https://github.com/SciML/DiffEqBenchmarks.jl). To learn more about how to optimize solving performance for stiff problems you can take a look at the [docs](https://docs.sciml.ai/latest/tutorials/advanced_ode_example/). 

[Sensitivity analysis](https://docs.sciml.ai/latest/analysis/sensitivity/), or automatic differentiation (AD) of the solver, is provided by the DiffEq suite. The model sensitivities are the derivatives of the solution $$u(t)$$ with respect to the parameters. Specifically, the local sensitivity of the solution to a parameter is defined by how much the solution would change by changes in the parameter. Sensitivity analysis provides a cheap way to calculate the gradient of the solution which can be used in parameter estimation and other optimization tasks.


The AD ecosystem in Julia allows you to switch between forward mode, reverse mode, source to source and other choices of AD and have it work with any Julia code. For a user to make use of this within [SciML](https://sciml.ai) a high level function [`concrete_solve`](https://docs.sciml.ai/latest/analysis/sensitivity/#High-Level-Interface:-concrete_solve-1) plugs into those AD systems to allow for choosing advanced sensitivity analysis (derivative calculation) [methods](https://docs.sciml.ai/latest/analysis/sensitivity/#Sensitivity-Algorithms-1). 

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
    predicted = concrete_solve(prob,nothing,saveat=0.1)
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
    [32mSampling:   0%|                                         |  ETA: 0:04:57[39m┌ Warning: The current proposal will be rejected due to numerical error(s).
    │   isfinite.((θ, r, ℓπ, ℓκ)) = (true, false, false, false)
    └ @ AdvancedHMC /home/cameron/.julia/packages/AdvancedHMC/WJCQA/src/hamiltonian.jl:47
    [32mSampling: 100%|█████████████████████████████████████████| Time: 0:15:27[39m





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
               α  2.2000  0.0065    0.0003  0.0004  158.5633  0.9987
               β  1.0000  0.0027    0.0001  0.0002  159.9610  0.9992
               γ  2.0000  0.0028    0.0001  0.0001  221.2757  0.9999
               δ  0.4000  0.0007    0.0000  0.0000  146.7523  0.9983
               σ  0.0151  0.0011    0.0000  0.0000  219.6837  1.0048
    
    Quantiles
      parameters    2.5%   25.0%   50.0%   75.0%   97.5%
      ──────────  ──────  ──────  ──────  ──────  ──────
               α  2.1875  2.1953  2.2000  2.2046  2.2117
               β  0.9947  0.9982  0.9999  1.0018  1.0049
               γ  1.9950  1.9981  2.0001  2.0019  2.0057
               δ  0.3988  0.3995  0.4000  0.4005  0.4014
               σ  0.0131  0.0144  0.0151  0.0158  0.0175




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
    predicted = concrete_solve(prob,nothing,saveat=0.1,sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))
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
    [32mSampling:  17%|██████▉                                  |  ETA: 0:04:18[39m┌ Warning: The current proposal will be rejected due to numerical error(s).
    │   isfinite.((θ, r, ℓπ, ℓκ)) = (true, false, false, false)
    └ @ AdvancedHMC /home/cameron/.julia/packages/AdvancedHMC/WJCQA/src/hamiltonian.jl:47
    [32mSampling:  35%|██████████████▎                          |  ETA: 0:03:33[39m┌ Warning: The current proposal will be rejected due to numerical error(s).
    │   isfinite.((θ, r, ℓπ, ℓκ)) = (true, false, false, false)
    └ @ AdvancedHMC /home/cameron/.julia/packages/AdvancedHMC/WJCQA/src/hamiltonian.jl:47
    [32mSampling:  36%|██████████████▉                          |  ETA: 0:03:27[39m┌ Warning: The current proposal will be rejected due to numerical error(s).
    │   isfinite.((θ, r, ℓπ, ℓκ)) = (true, false, false, false)
    └ @ AdvancedHMC /home/cameron/.julia/packages/AdvancedHMC/WJCQA/src/hamiltonian.jl:47
    [32mSampling:  46%|██████████████████▊                      |  ETA: 0:02:56[39m┌ Warning: The current proposal will be rejected due to numerical error(s).
    │   isfinite.((θ, r, ℓπ, ℓκ)) = (true, false, false, false)
    └ @ AdvancedHMC /home/cameron/.julia/packages/AdvancedHMC/WJCQA/src/hamiltonian.jl:47
    [32mSampling: 100%|█████████████████████████████████████████| Time: 0:05:39[39m


    351.169146 seconds (2.47 G allocations: 114.429 GiB, 5.59% gc time)





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
               α  1.9989  0.2671    0.0119  0.0281  114.1424  1.0056
               β  1.0435  0.1471    0.0066  0.0145  117.7621  1.0017
               γ  2.0915  0.2014    0.0090  0.0177  149.4003  1.0095
               δ  1.1443  0.1543    0.0069  0.0153  112.7753  1.0100
               σ  1.1358  0.0542    0.0024  0.0018  406.1583  0.9982
    
    Quantiles
      parameters    2.5%   25.0%   50.0%   75.0%   97.5%
      ──────────  ──────  ──────  ──────  ──────  ──────
               α  1.4456  1.8293  2.0288  2.1955  2.4364
               β  0.7431  0.9545  1.0569  1.1495  1.3078
               γ  1.7553  1.9533  2.0653  2.2167  2.5395
               δ  0.9356  1.0369  1.1058  1.2155  1.5289
               σ  1.0348  1.0945  1.1357  1.1764  1.2338




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
     1.0  1.04136  1.05544  1.04192  0.98489  …  0.183854  0.190431  0.199189
     1.0  1.22986  1.52359  1.89059  2.33363     1.07482   1.02333   0.984698



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
    [32mSampling: 100%|█████████████████████████████████████████| Time: 3:05:52[39m





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
               α  1.3007  0.0000    0.0000  0.0000  2.0325  2.6586
               β  0.7204  0.0000    0.0000  0.0000  2.0325  2.1258
               γ  2.7288  0.0000    0.0000  0.0000  2.0325  1.5039
               δ  1.4337  0.0000    0.0000  0.0000  2.0325  0.9960
               σ  0.4002  0.0000    0.0000  0.0000  2.0325  1.7251
              ϕ1  0.9069  0.0000    0.0000  0.0000  2.0325  5.0295
              ϕ2  0.2177  0.0000    0.0000  0.0000  2.0325  3.4342
    
    Quantiles
      parameters    2.5%   25.0%   50.0%   75.0%   97.5%
      ──────────  ──────  ──────  ──────  ──────  ──────
               α  1.3007  1.3007  1.3007  1.3007  1.3007
               β  0.7204  0.7204  0.7204  0.7204  0.7204
               γ  2.7288  2.7288  2.7288  2.7288  2.7288
               δ  1.4337  1.4337  1.4337  1.4337  1.4337
               σ  0.4002  0.4002  0.4002  0.4002  0.4002
              ϕ1  0.9069  0.9069  0.9069  0.9069  0.9069
              ϕ2  0.2177  0.2177  0.2177  0.2177  0.2177



