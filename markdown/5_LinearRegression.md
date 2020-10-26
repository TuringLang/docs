---
title: Linear Regression
permalink: /:collection/:name/
---

Turing is powerful when applied to complex hierarchical models, but it can also be put to task at common statistical procedures, like [linear regression](https://en.wikipedia.org/wiki/Linear_regression). This tutorial covers how to implement a linear regression model in Turing.

## Set Up

We begin by importing all the necessary libraries.


```julia
# Import Turing and Distributions.
using Turing, Distributions

# Import RDatasets.
using RDatasets

# Import MCMCChains, and StatPlots for visualizations and diagnostics.
using MCMCChains, StatsPlots

# Functionality for splitting and normalizing the data.
using MLDataUtils: shuffleobs, splitobs, rescale!

# Functionality for evaluating the model predictions.
using Distances

# Set a seed for reproducibility.
using Random
Random.seed!(0)

# Hide the progress prompt while sampling.
Turing.turnprogress(false);
```

    ┌ Info: Precompiling Turing [fce5fe82-541a-59a6-adf8-730c64b5f9a0]
    └ @ Base loading.jl:1260
    ┌ Info: Precompiling RDatasets [ce6b1742-4840-55fa-b093-852dadbb1d8b]
    └ @ Base loading.jl:1260
    ┌ Info: Precompiling Plots [91a5bcdd-55d7-5caf-9e0b-520d859cae80]
    └ @ Base loading.jl:1260
    ┌ Info: Precompiling StatsPlots [f3b207a7-027a-5e70-b257-86293d7955fd]
    └ @ Base loading.jl:1260
    ┌ Info: Precompiling MLDataUtils [cc2ba9b6-d476-5e6d-8eaf-a92d5412d41d]
    └ @ Base loading.jl:1260
    ┌ Info: [Turing]: progress logging is disabled globally
    └ @ Turing /home/cameron/.julia/packages/Turing/GMBTf/src/Turing.jl:22


We will use the `mtcars` dataset from the [RDatasets](https://github.com/johnmyleswhite/RDatasets.jl) package. `mtcars` contains a variety of statistics on different car models, including their miles per gallon, number of cylinders, and horsepower, among others.

We want to know if we can construct a Bayesian linear regression model to predict the miles per gallon of a car, given the other statistics it has. Lets take a look at the data we have.


```julia
# Import the "Default" dataset.
data = RDatasets.dataset("datasets", "mtcars");

# Show the first six rows of the dataset.
first(data, 6)
```




<table class="data-frame"><thead><tr><th></th><th>Model</th><th>MPG</th><th>Cyl</th><th>Disp</th><th>HP</th><th>DRat</th><th>WT</th><th>QSec</th><th>VS</th></tr><tr><th></th><th>String</th><th>Float64</th><th>Int64</th><th>Float64</th><th>Int64</th><th>Float64</th><th>Float64</th><th>Float64</th><th>Int64</th></tr></thead><tbody><p>6 rows × 12 columns (omitted printing of 3 columns)</p><tr><th>1</th><td>Mazda RX4</td><td>21.0</td><td>6</td><td>160.0</td><td>110</td><td>3.9</td><td>2.62</td><td>16.46</td><td>0</td></tr><tr><th>2</th><td>Mazda RX4 Wag</td><td>21.0</td><td>6</td><td>160.0</td><td>110</td><td>3.9</td><td>2.875</td><td>17.02</td><td>0</td></tr><tr><th>3</th><td>Datsun 710</td><td>22.8</td><td>4</td><td>108.0</td><td>93</td><td>3.85</td><td>2.32</td><td>18.61</td><td>1</td></tr><tr><th>4</th><td>Hornet 4 Drive</td><td>21.4</td><td>6</td><td>258.0</td><td>110</td><td>3.08</td><td>3.215</td><td>19.44</td><td>1</td></tr><tr><th>5</th><td>Hornet Sportabout</td><td>18.7</td><td>8</td><td>360.0</td><td>175</td><td>3.15</td><td>3.44</td><td>17.02</td><td>0</td></tr><tr><th>6</th><td>Valiant</td><td>18.1</td><td>6</td><td>225.0</td><td>105</td><td>2.76</td><td>3.46</td><td>20.22</td><td>1</td></tr></tbody></table>




```julia
size(data)
```




    (32, 12)



The next step is to get our data ready for testing. We'll split the `mtcars` dataset into two subsets, one for training our model and one for evaluating our model. Then, we separate the targets we want to learn (`MPG`, in this case) and standardize the datasets by subtracting each column's means and dividing by the standard deviation of that column. The resulting data is not very familiar looking, but this standardization process helps the sampler converge far easier.


```julia
# Remove the model column.
select!(data, Not(:Model))

# Split our dataset 70%/30% into training/test sets.
trainset, testset = splitobs(shuffleobs(data), 0.7)

# Turing requires data in matrix form.
target = :MPG
train = Matrix(select(trainset, Not(target)))
test = Matrix(select(testset, Not(target)))
train_target = trainset[:, target]
test_target = testset[:, target]

# Standardize the features.
μ, σ = rescale!(train; obsdim = 1)
rescale!(test, μ, σ; obsdim = 1)

# Standardize the targets.
μtarget, σtarget = rescale!(train_target; obsdim = 1)
rescale!(test_target, μtarget, σtarget; obsdim = 1);
```

## Model Specification

In a traditional frequentist model using [OLS](https://en.wikipedia.org/wiki/Ordinary_least_squares), our model might look like:

\$\$
MPG_i = \alpha + \boldsymbol{\beta}^\mathsf{T}\boldsymbol{X_i}
\$\$

where $$\boldsymbol{\beta}$$ is a vector of coefficients and $$\boldsymbol{X}$$ is a vector of inputs for observation $$i$$. The Bayesian model we are more concerned with is the following:

\$\$
MPG_i \sim \mathcal{N}(\alpha + \boldsymbol{\beta}^\mathsf{T}\boldsymbol{X_i}, \sigma^2)
\$\$

where $$\alpha$$ is an intercept term common to all observations, $$\boldsymbol{\beta}$$ is a coefficient vector, $$\boldsymbol{X_i}$$ is the observed data for car $$i$$, and $$\sigma^2$$ is a common variance term.

For $$\sigma^2$$, we assign a prior of `truncated(Normal(0, 100), 0, Inf)`. This is consistent with [Andrew Gelman's recommendations](http://www.stat.columbia.edu/~gelman/research/published/taumain.pdf) on noninformative priors for variance. The intercept term ($$\alpha$$) is assumed to be normally distributed with a mean of zero and a variance of three. This represents our assumptions that miles per gallon can be explained mostly by our assorted variables, but a high variance term indicates our uncertainty about that. Each coefficient is assumed to be normally distributed with a mean of zero and a variance of 10. We do not know that our coefficients are different from zero, and we don't know which ones are likely to be the most important, so the variance term is quite high. Lastly, each observation $$y_i$$ is distributed according to the calculated `mu` term given by $$\alpha + \boldsymbol{\beta}^\mathsf{T}\boldsymbol{X_i}$$.


```julia
# Bayesian linear regression.
@model function linear_regression(x, y)
    # Set variance prior.
    σ₂ ~ truncated(Normal(0, 100), 0, Inf)
    
    # Set intercept prior.
    intercept ~ Normal(0, sqrt(3))
    
    # Set the priors on our coefficients.
    nfeatures = size(x, 2)
    coefficients ~ MvNormal(nfeatures, sqrt(10))
    
    # Calculate all the mu terms.
    mu = intercept .+ x * coefficients
    y ~ MvNormal(mu, sqrt(σ₂))
end
```




    DynamicPPL.ModelGen{var"###generator#273",(:x, :y),(),Tuple{}}(##generator#273, NamedTuple())



With our model specified, we can call the sampler. We will use the No U-Turn Sampler ([NUTS](http://turing.ml/docs/library/#-turingnuts--type)) here. 


```julia
model = linear_regression(train, train_target)
chain = sample(model, NUTS(0.65), 3_000);
```

    ┌ Info: Found initial step size
    │   ϵ = 1.6
    └ @ Turing.Inference /home/cameron/.julia/packages/Turing/GMBTf/src/inference/hmc.jl:629
    ┌ Warning: The current proposal will be rejected due to numerical error(s).
    │   isfinite.((θ, r, ℓπ, ℓκ)) = (true, false, false, false)
    └ @ AdvancedHMC /home/cameron/.julia/packages/AdvancedHMC/P9wqk/src/hamiltonian.jl:47
    ┌ Warning: The current proposal will be rejected due to numerical error(s).
    │   isfinite.((θ, r, ℓπ, ℓκ)) = (true, false, false, false)
    └ @ AdvancedHMC /home/cameron/.julia/packages/AdvancedHMC/P9wqk/src/hamiltonian.jl:47


As a visual check to confirm that our coefficients have converged, we show the densities and trace plots for our parameters using the `plot` functionality.


```julia
plot(chain)
```




![svg](/tutorials/5_LinearRegression_files/5_LinearRegression_12_0.svg)



It looks like each of our parameters has converged. We can check our numerical esimates using `describe(chain)`, as below.


```julia
describe(chain)
```




    2-element Array{ChainDataFrame,1}
    
    Summary Statistics
            parameters     mean     std  naive_se    mcse       ess   r_hat
      ────────────────  ───────  ──────  ────────  ──────  ────────  ──────
       coefficients[1]  -0.0413  0.5648    0.0126  0.0389  265.1907  1.0010
       coefficients[2]   0.2770  0.6994    0.0156  0.0401  375.2777  1.0067
       coefficients[3]  -0.4116  0.3850    0.0086  0.0160  695.3990  1.0032
       coefficients[4]   0.1805  0.2948    0.0066  0.0126  479.9290  1.0010
       coefficients[5]  -0.2669  0.7168    0.0160  0.0316  373.0291  1.0009
       coefficients[6]   0.0256  0.3461    0.0077  0.0119  571.0954  1.0028
       coefficients[7]   0.0277  0.3899    0.0087  0.0174  637.1596  1.0007
       coefficients[8]   0.1535  0.3050    0.0068  0.0117  579.1998  1.0032
       coefficients[9]   0.1223  0.2839    0.0063  0.0105  587.6752  0.9995
      coefficients[10]  -0.2839  0.3975    0.0089  0.0195  360.9612  1.0019
             intercept   0.0058  0.1179    0.0026  0.0044  580.0222  0.9995
                    σ₂   0.3017  0.1955    0.0044  0.0132  227.2322  1.0005
    
    Quantiles
            parameters     2.5%    25.0%    50.0%    75.0%   97.5%
      ────────────────  ───────  ───────  ───────  ───────  ──────
       coefficients[1]  -1.0991  -0.4265  -0.0199   0.3244  1.1093
       coefficients[2]  -1.1369  -0.1523   0.2854   0.7154  1.6488
       coefficients[3]  -1.1957  -0.6272  -0.3986  -0.1800  0.3587
       coefficients[4]  -0.3896  -0.0155   0.1663   0.3593  0.7818
       coefficients[5]  -1.6858  -0.6835  -0.2683   0.1378  1.1995
       coefficients[6]  -0.6865  -0.1672   0.0325   0.2214  0.7251
       coefficients[7]  -0.7644  -0.1976   0.0090   0.2835  0.8185
       coefficients[8]  -0.4980  -0.0194   0.1451   0.3428  0.7685
       coefficients[9]  -0.4643  -0.0294   0.1237   0.2807  0.7218
      coefficients[10]  -1.0898  -0.5091  -0.2846  -0.0413  0.5163
             intercept  -0.2240  -0.0671   0.0083   0.0746  0.2364
                    σ₂   0.1043   0.1860   0.2525   0.3530  0.8490




## Comparing to OLS

A satisfactory test of our model is to evaluate how well it predicts. Importantly, we want to compare our model to existing tools like OLS. The code below uses the [GLM.jl]() package to generate a traditional OLS multiple regression model on the same data as our probabalistic model.


```julia
# Import the GLM package.
using GLM

# Perform multiple regression OLS.
train_with_intercept = hcat(ones(size(train, 1)), train)
ols = lm(train_with_intercept, train_target)

# Compute predictions on the training data set
# and unstandardize them.
p = GLM.predict(ols)
train_prediction_ols = μtarget .+ σtarget .* p

# Compute predictions on the test data set
# and unstandardize them.
test_with_intercept = hcat(ones(size(test, 1)), test)
p = GLM.predict(ols, test_with_intercept)
test_prediction_ols = μtarget .+ σtarget .* p;
```

    ┌ Info: Precompiling GLM [38e38edf-8417-5370-95a0-9cbb8c7f171a]
    └ @ Base loading.jl:1260


The function below accepts a chain and an input matrix and calculates predictions. We use the samples of the model parameters in the chain starting with sample 200, which is where the warm-up period for the NUTS sampler ended.


```julia
# Make a prediction given an input vector.
function prediction(chain, x)
    p = get_params(chain[200:end, :, :])
    targets = p.intercept' .+ x * reduce(hcat, p.coefficients)'
    return vec(mean(targets; dims = 2))
end
```




    prediction (generic function with 1 method)



When we make predictions, we unstandardize them so they are more understandable.


```julia
# Calculate the predictions for the training and testing sets
# and unstandardize them.
p = prediction(chain, train)
train_prediction_bayes = μtarget .+ σtarget .* p
p = prediction(chain, test)
test_prediction_bayes = μtarget .+ σtarget .* p

# Show the predictions on the test data set.
DataFrame(
    MPG = testset[!, target],
    Bayes = test_prediction_bayes,
    OLS = test_prediction_ols
)
```




<table class="data-frame"><thead><tr><th></th><th>MPG</th><th>Bayes</th><th>OLS</th></tr><tr><th></th><th>Float64</th><th>Float64</th><th>Float64</th></tr></thead><tbody><p>10 rows × 3 columns</p><tr><th>1</th><td>19.2</td><td>18.3766</td><td>18.1265</td></tr><tr><th>2</th><td>15.0</td><td>6.4176</td><td>6.37891</td></tr><tr><th>3</th><td>16.4</td><td>13.9125</td><td>13.883</td></tr><tr><th>4</th><td>14.3</td><td>11.8393</td><td>11.7337</td></tr><tr><th>5</th><td>21.4</td><td>25.3622</td><td>25.1916</td></tr><tr><th>6</th><td>18.1</td><td>20.7687</td><td>20.672</td></tr><tr><th>7</th><td>19.7</td><td>16.03</td><td>15.8408</td></tr><tr><th>8</th><td>15.2</td><td>18.2903</td><td>18.3391</td></tr><tr><th>9</th><td>26.0</td><td>28.5191</td><td>28.4865</td></tr><tr><th>10</th><td>17.3</td><td>14.498</td><td>14.534</td></tr></tbody></table>



Now let's evaluate the loss for each method, and each prediction set. We will use the mean squared error to evaluate loss, given by 
\$\$
\text{MSE} = \frac{1}{n} \sum_{i=1}^n {(y_i - \hat{y_i})^2}
\$\$
where $$y_i$$ is the actual value (true MPG) and $$\hat{y_i}$$ is the predicted value using either OLS or Bayesian linear regression. A lower SSE indicates a closer fit to the data.


```julia
println(
    "Training set:",
    "\n\tBayes loss: ",
    msd(train_prediction_bayes, trainset[!, target]),
    "\n\tOLS loss: ",
    msd(train_prediction_ols, trainset[!, target])
)

println(
    "Test set:",
    "\n\tBayes loss: ",
    msd(test_prediction_bayes, testset[!, target]),
    "\n\tOLS loss: ",
    msd(test_prediction_ols, testset[!, target])
)
```

    Training set:
    	Bayes loss: 4.664508273535872
    	OLS loss: 4.648142085690519
    Test set:
    	Bayes loss: 14.66153554719035
    	OLS loss: 14.796847779051628


As we can see above, OLS and our Bayesian model fit our training and test data set about the same.
