---
redirect_from: "tutorials/10-bayesiandiffeq/"
title: "Bayesian Estimation of Differential Equations"
permalink: "/:collection/:name/"
---


Most of the scientific community deals with the basic problem of trying to mathematically model the reality around them and this often involves dynamical systems. The general trend to model these complex dynamical systems is through the use of differential equations. Differential equation models often have non-measurable parameters. The popular “forward-problem” of simulation consists of solving the differential equations for a given set of parameters, the “inverse problem” to simulation, known as parameter estimation, is the process of utilizing data to determine these model parameters. Bayesian inference provides a robust approach to parameter estimation with quantified uncertainty.


```julia
using Turing, Distributions, DifferentialEquations

# Import MCMCChain, Plots, and StatsPlots for visualizations and diagnostics.
using MCMCChains, Plots, StatsPlots

# Set a seed for reproducibility.
using Random
Random.seed!(14);
```




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

![](figures/10_bayesian-differential-equations_2_1.png)



We'll generate the data to use for the parameter estimation from simulation. 
With the `saveat` [argument](https://docs.sciml.ai/latest/basics/common_solver_opts/) we specify that the solution is stored only at `0.1` time units. To make the data look more realistic, we add random noise using the function `randn`.


```julia
sol1 = solve(prob1,Tsit5(),saveat=0.1)
odedata = Array(sol1) + 0.8 * randn(size(Array(sol1)))
plot(sol1, alpha = 0.3, legend = false); scatter!(sol1.t, odedata')
```

![](figures/10_bayesian-differential-equations_3_1.png)



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

```
Chains MCMC chain (1000×17×3 Array{Float64,3}):

Iterations        = 1:1000
Thinning interval = 1
Chains            = 1, 2, 3
Samples per chain = 1000
parameters        = α, β, γ, δ, σ
internals         = acceptance_rate, hamiltonian_energy, hamiltonian_energy
_error, is_accept, log_density, lp, max_hamiltonian_energy_error, n_steps, 
nom_step_size, numerical_error, step_size, tree_depth

Summary Statistics
  parameters      mean       std   naive_se      mcse        ess      rhat
      Symbol   Float64   Float64    Float64   Float64    Float64   Float64

           α    1.2120    0.4926     0.0090    0.0911     6.0574   11.8201
           β    1.0672    0.1639     0.0030    0.0098   288.9165    1.0360
           γ    2.8014    0.2742     0.0050    0.0208    46.5485    1.0915
           δ    0.9538    0.0904     0.0017    0.0056   124.1565    1.0487
           σ    1.2712    0.6531     0.0119    0.1205     6.0766    9.8804

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           α    0.5013    0.5243    1.5228    1.5737    1.6590
           β    0.6738    1.0262    1.0815    1.1337    1.4402
           γ    2.0953    2.6893    2.8377    2.9643    3.2437
           δ    0.8076    0.8993    0.9435    0.9901    1.1873
           σ    0.7482    0.7989    0.8394    2.1115    2.3534
```





The estimated parameters are close to the desired parameter values. We can also check that the chains have converged in the plot.


```julia
plot(chain)
```

![](figures/10_bayesian-differential-equations_5_1.png)



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

![](figures/10_bayesian-differential-equations_7_1.png)



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

```
DynamicPPL.Model{Main.##WeaveSandBox#253.var"#5#6",(:data, :prob1),(),(),Tu
ple{Array{Float64,1},SciMLBase.ODEProblem{Array{Float64,1},Tuple{Float64,Fl
oat64},true,Array{Float64,1},SciMLBase.ODEFunction{true,typeof(Main.##Weave
SandBox#253.lotka_volterra),LinearAlgebra.UniformScaling{Bool},Nothing,Noth
ing,Nothing,Nothing,Nothing,Nothing,Nothing,Nothing,Nothing,Nothing,Nothing
,Nothing,typeof(SciMLBase.DEFAULT_OBSERVED),Nothing},Base.Iterators.Pairs{U
nion{},Union{},Tuple{},NamedTuple{(),Tuple{}}},SciMLBase.StandardODEProblem
}},Tuple{}}(:fitlv2, Main.##WeaveSandBox#253.var"#5#6"(), (data = [2.200730
590544725, 0.8584002186440604, 0.3130803892338444, 0.8065538543184622, -0.3
4719524379658445, 0.2827563462601055, 0.4633732909134419, 0.938813994609707
2, -0.029638888419957155, -0.10766570796447744  …  4.484466907306791, 2.276
637854709268, 3.034635398109261, 1.6534146147281914, 2.3126757947633125, 3.
430419239300897, 1.481768351221498, 1.7989355388635417, 1.343881963121325, 
0.25843622408034905], prob1 = SciMLBase.ODEProblem{Array{Float64,1},Tuple{F
loat64,Float64},true,Array{Float64,1},SciMLBase.ODEFunction{true,typeof(Mai
n.##WeaveSandBox#253.lotka_volterra),LinearAlgebra.UniformScaling{Bool},Not
hing,Nothing,Nothing,Nothing,Nothing,Nothing,Nothing,Nothing,Nothing,Nothin
g,Nothing,Nothing,typeof(SciMLBase.DEFAULT_OBSERVED),Nothing},Base.Iterator
s.Pairs{Union{},Union{},Tuple{},NamedTuple{(),Tuple{}}},SciMLBase.StandardO
DEProblem}(SciMLBase.ODEFunction{true,typeof(Main.##WeaveSandBox#253.lotka_
volterra),LinearAlgebra.UniformScaling{Bool},Nothing,Nothing,Nothing,Nothin
g,Nothing,Nothing,Nothing,Nothing,Nothing,Nothing,Nothing,Nothing,typeof(Sc
iMLBase.DEFAULT_OBSERVED),Nothing}(Main.##WeaveSandBox#253.lotka_volterra, 
LinearAlgebra.UniformScaling{Bool}(true), nothing, nothing, nothing, nothin
g, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, 
SciMLBase.DEFAULT_OBSERVED, nothing), [1.0, 1.0], (0.0, 10.0), [1.5, 1.0, 3
.0, 1.0], Base.Iterators.Pairs{Union{},Union{},Tuple{},NamedTuple{(),Tuple{
}}}(), SciMLBase.StandardODEProblem())), NamedTuple())
```





Here we use the multithreading functionality [available](https://turing.ml/dev/docs/using-turing/guide#multithreaded-sampling) in Turing.jl to sample 3 independent chains


```julia
Threads.nthreads()
```

```
16
```



```julia
# This next command runs 3 independent chains with multithreading.
chain2 = sample(model2, NUTS(.45), MCMCThreads(), 5000, 3, progress=false)
```

```
Chains MCMC chain (5000×17×3 Array{Float64,3}):

Iterations        = 1:5000
Thinning interval = 1
Chains            = 1, 2, 3
Samples per chain = 5000
parameters        = α, β, γ, δ, σ
internals         = acceptance_rate, hamiltonian_energy, hamiltonian_energy
_error, is_accept, log_density, lp, max_hamiltonian_energy_error, n_steps, 
nom_step_size, numerical_error, step_size, tree_depth

Summary Statistics
  parameters      mean       std   naive_se      mcse        ess      rhat
      Symbol   Float64   Float64    Float64   Float64    Float64   Float64

           α    1.5612    0.1706     0.0014    0.0082   174.9037    1.0022
           β    1.1232    0.1357     0.0011    0.0063   201.8733    1.0014
           γ    2.9663    0.2786     0.0023    0.0137   167.4011    1.0034
           δ    0.9418    0.2206     0.0018    0.0105   174.6266    1.0012
           σ    0.8129    0.0637     0.0005    0.0039    81.8891    1.0334

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           α    1.2771    1.4407    1.5336    1.6756    1.9369
           β    0.8886    1.0345    1.1018    1.2067    1.4164
           γ    2.4397    2.7648    2.9761    3.1589    3.4934
           δ    0.5388    0.7779    0.9475    1.0809    1.3972
           σ    0.6917    0.7684    0.8159    0.8532    0.9457
```



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

![](figures/10_bayesian-differential-equations_11_1.png)



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

```
DDEProblem with uType Array{Float64,1} and tType Float64. In-place: true
timespan: (0.0, 10.0)
u0: 2-element Array{Float64,1}:
 1.0
 1.0
```



```julia
sol = solve(prob1,saveat=0.1)
ddedata = Array(sol)
ddedata = ddedata + 0.5 * randn(size(ddedata))
```

```
2×101 Array{Float64,2}:
 1.18609   0.0634866  1.19388  …  2.21241  3.28584  3.16553  2.22741
 0.808641  0.944246   1.46414     1.73124  1.19192  1.28923  1.14635
```





Plot the data:

```julia
scatter(sol.t, ddedata'); plot!(sol)
```

![](figures/10_bayesian-differential-equations_14_1.png)



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

```
DynamicPPL.Model{Main.##WeaveSandBox#253.var"#8#9",(:data, :prob1),(),(),Tu
ple{Array{Float64,2},SciMLBase.DDEProblem{Array{Float64,1},Tuple{Float64,Fl
oat64},Tuple{},Tuple{},true,NTuple{4,Float64},SciMLBase.DDEFunction{true,ty
peof(Main.##WeaveSandBox#253.delay_lotka_volterra),LinearAlgebra.UniformSca
ling{Bool},Nothing,Nothing,Nothing,Nothing,Nothing,Nothing,Nothing,Nothing,
Nothing,Nothing,Nothing,Nothing},typeof(Main.##WeaveSandBox#253.h),Base.Ite
rators.Pairs{Union{},Union{},Tuple{},NamedTuple{(),Tuple{}}},SciMLBase.Stan
dardDDEProblem}},Tuple{}}(:fitlv, Main.##WeaveSandBox#253.var"#8#9"(), (dat
a = [1.1860906554746231 0.06348658995769019 … 3.1655347918676964 2.22741117
6264933; 0.8086410615247461 0.9442456964892731 … 1.2892250042736726 1.14634
85009181613], prob1 = SciMLBase.DDEProblem{Array{Float64,1},Tuple{Float64,F
loat64},Tuple{},Tuple{},true,NTuple{4,Float64},SciMLBase.DDEFunction{true,t
ypeof(Main.##WeaveSandBox#253.delay_lotka_volterra),LinearAlgebra.UniformSc
aling{Bool},Nothing,Nothing,Nothing,Nothing,Nothing,Nothing,Nothing,Nothing
,Nothing,Nothing,Nothing,Nothing},typeof(Main.##WeaveSandBox#253.h),Base.It
erators.Pairs{Union{},Union{},Tuple{},NamedTuple{(),Tuple{}}},SciMLBase.Sta
ndardDDEProblem}(SciMLBase.DDEFunction{true,typeof(Main.##WeaveSandBox#253.
delay_lotka_volterra),LinearAlgebra.UniformScaling{Bool},Nothing,Nothing,No
thing,Nothing,Nothing,Nothing,Nothing,Nothing,Nothing,Nothing,Nothing,Nothi
ng}(Main.##WeaveSandBox#253.delay_lotka_volterra, LinearAlgebra.UniformScal
ing{Bool}(true), nothing, nothing, nothing, nothing, nothing, nothing, noth
ing, nothing, nothing, nothing, nothing, nothing), [1.0, 1.0], Main.##Weave
SandBox#253.h, (0.0, 10.0), (1.5, 1.0, 3.0, 1.0), (), (), Base.Iterators.Pa
irs{Union{},Union{},Tuple{},NamedTuple{(),Tuple{}}}(), false, 0, SciMLBase.
StandardDDEProblem())), NamedTuple())
```





Then we draw samples using multithreading; this time, we draw 3 independent chains in parallel using `MCMCThreads`.

```julia
chain = sample(model, NUTS(.65), MCMCThreads(), 300, 3, progress=true)
plot(chain)
```

![](figures/10_bayesian-differential-equations_16_1.png)



Finally, we select a 100 sets of parameters from the first chain and plot solutions.


```julia
chain
```

```
Chains MCMC chain (300×17×3 Array{Float64,3}):

Iterations        = 1:300
Thinning interval = 1
Chains            = 1, 2, 3
Samples per chain = 300
parameters        = α, β, γ, δ, σ
internals         = acceptance_rate, hamiltonian_energy, hamiltonian_energy
_error, is_accept, log_density, lp, max_hamiltonian_energy_error, n_steps, 
nom_step_size, numerical_error, step_size, tree_depth

Summary Statistics
  parameters      mean       std   naive_se      mcse        ess      rhat
      Symbol   Float64   Float64    Float64   Float64    Float64   Float64

           α    1.5142    0.0583     0.0019    0.0030   349.4243    1.0016
           β    0.9921    0.0459     0.0015    0.0027   392.5261    1.0024
           γ    2.9372    0.1188     0.0040    0.0055   383.4066    1.0013
           δ    0.9823    0.0420     0.0014    0.0022   356.1193    1.0018
           σ    0.4869    0.0241     0.0008    0.0007   651.6937    0.9998

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           α    1.4002    1.4730    1.5134    1.5536    1.6272
           β    0.9060    0.9585    0.9920    1.0235    1.0853
           γ    2.7137    2.8563    2.9329    3.0122    3.1963
           δ    0.9055    0.9539    0.9814    1.0093    1.0731
           σ    0.4422    0.4689    0.4859    0.5043    0.5360
```



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

![](figures/10_bayesian-differential-equations_18_1.png)



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

```
ODEProblem with uType Array{Float64,1} and tType Float64. In-place: true
timespan: (0.0, 10.0)
u0: 2-element Array{Float64,1}:
 1.0
 1.0
```



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

```
Chains MCMC chain (1000×17×1 Array{Float64,3}):

Iterations        = 1:1000
Thinning interval = 1
Chains            = 1
Samples per chain = 1000
parameters        = α, β, γ, δ, σ
internals         = acceptance_rate, hamiltonian_energy, hamiltonian_energy
_error, is_accept, log_density, lp, max_hamiltonian_energy_error, n_steps, 
nom_step_size, numerical_error, step_size, tree_depth

Summary Statistics
  parameters      mean       std   naive_se      mcse        ess      rhat
      Symbol   Float64   Float64    Float64   Float64    Float64   Float64

           α    1.5546    0.0541     0.0017    0.0032   211.5538    1.0024
           β    1.0906    0.0533     0.0017    0.0026   264.2906    0.9991
           γ    2.8837    0.1472     0.0047    0.0091   207.6229    1.0023
           δ    0.9397    0.0525     0.0017    0.0031   198.9996    1.0024
           σ    0.8163    0.0412     0.0013    0.0021   458.2821    0.9991

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           α    1.4516    1.5194    1.5551    1.5900    1.6548
           β    0.9938    1.0520    1.0900    1.1240    1.2007
           γ    2.6248    2.7813    2.8758    2.9740    3.1869
           δ    0.8502    0.9037    0.9368    0.9719    1.0482
           σ    0.7411    0.7865    0.8154    0.8419    0.9009
```





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

```
1404.180798 seconds (5.33 G allocations: 283.222 GiB, 4.01% gc time)
Chains MCMC chain (1000×17×1 Array{Float64,3}):

Iterations        = 1:1000
Thinning interval = 1
Chains            = 1
Samples per chain = 1000
parameters        = α, β, γ, δ, σ
internals         = acceptance_rate, hamiltonian_energy, hamiltonian_energy
_error, is_accept, log_density, lp, max_hamiltonian_energy_error, n_steps, 
nom_step_size, numerical_error, step_size, tree_depth

Summary Statistics
  parameters      mean       std   naive_se      mcse        ess      rhat
      Symbol   Float64   Float64    Float64   Float64    Float64   Float64

           α    1.5577    0.0527     0.0017    0.0030   198.5876    0.9991
           β    1.0928    0.0527     0.0017    0.0031   233.5858    0.9991
           γ    2.8761    0.1391     0.0044    0.0080   207.3585    0.9995
           δ    0.9362    0.0489     0.0015    0.0029   208.3950    0.9990
           σ    0.8120    0.0399     0.0013    0.0011   659.3252    1.0020

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           α    1.4573    1.5234    1.5540    1.5896    1.6707
           β    1.0011    1.0560    1.0913    1.1255    1.2084
           γ    2.6005    2.7871    2.8753    2.9665    3.1595
           δ    0.8402    0.9044    0.9370    0.9677    1.0361
           σ    0.7367    0.7843    0.8096    0.8364    0.8995
```





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

```
6.465698 seconds (38.33 M allocations: 1.291 GiB, 3.42% gc time)
```


![](figures/10_bayesian-differential-equations_22_1.png)

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

![](figures/10_bayesian-differential-equations_24_1.png)


## Appendix
 This tutorial is part of the TuringTutorials repository, found at: <https://github.com/TuringLang/TuringTutorials>.

To locally run this tutorial, do the following commands:
```julia, eval = false
using TuringTutorials
TuringTutorials.weave_file("10-bayesian-differential-equations", "10_bayesian-differential-equations.jmd")
```

Computer Information:
```
Julia Version 1.5.3
Commit 788b2c77c1 (2020-11-09 13:37 UTC)
Platform Info:
  OS: Linux (x86_64-pc-linux-gnu)
  CPU: Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-9.0.1 (ORCJIT, skylake)
Environment:
  JULIA_CMDSTAN_HOME = /home/cameron/stan/
  JULIA_NUM_THREADS = 16

```

Package Information:

```
Status `~/.julia/dev/TuringTutorials/tutorials/10-bayesian-differential-equations/Project.toml`
  [a93c6f00] DataFrames v1.0.1
  [2b5f629d] DiffEqBase v6.60.0
  [ebbdde9d] DiffEqBayes v2.23.0
  [41bf760c] DiffEqSensitivity v6.44.1
  [0c46a032] DifferentialEquations v6.16.0
  [31c24e10] Distributions v0.23.12
  [ced4e74d] DistributionsAD v0.6.24
  [c7f686f2] MCMCChains v4.9.0
  [91a5bcdd] Plots v1.12.0
  [f3b207a7] StatsPlots v0.14.19
  [fce5fe82] Turing v0.15.18
  [9a3f8284] Random

```
