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

# Import MCMCChains, Plots, and StatPlots for visualizations and diagnostics.
using MCMCChains, Plots, StatsPlots

# Set a seed for reproducibility.
using Random
Random.seed!(0);

# Hide the progress prompt while sampling.
Turing.turnprogress(false);
```

    ┌ Info: Precompiling Turing [fce5fe82-541a-59a6-adf8-730c64b5f9a0]
    └ @ Base loading.jl:1260
    ┌ Info: Precompiling Plots [91a5bcdd-55d7-5caf-9e0b-520d859cae80]
    └ @ Base loading.jl:1260
    ┌ Info: Precompiling StatsPlots [f3b207a7-027a-5e70-b257-86293d7955fd]
    └ @ Base loading.jl:1260
    ┌ Info: [Turing]: progress logging is disabled globally
    └ @ Turing /home/cameron/.julia/packages/Turing/cReBm/src/Turing.jl:22


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



The next step is to get our data ready for testing. We'll split the `mtcars` dataset into two subsets, one for training our model and one for evaluating our model. Then, we separate the labels we want to learn (`MPG`, in this case) and standardize the datasets by subtracting each column's means and dividing by the standard deviation of that column.

The resulting data is not very familiar looking, but this standardization process helps the sampler converge far easier. We also create a function called `unstandardize`, which returns the standardized values to their original form. We will use this function later on when we make predictions.


```julia
# Function to split samples.
function split_data(df, at = 0.70)
    r = size(df,1)
    index = Int(round(r * at))
    train = df[1:index, :]
    test  = df[(index+1):end, :]
    return train, test
end

# A handy helper function to rescale our dataset.
function standardize(x)
    return (x .- mean(x, dims=1)) ./ std(x, dims=1), x
end

# Another helper function to unstandardize our datasets.
function unstandardize(x, orig)
    return (x .+ mean(orig, dims=1)) .* std(orig, dims=1)
end

# Remove the model column.
select!(data, Not(:Model))

# Standardize our dataset.
(std_data, data_arr) = standardize(Matrix(data))

# Split our dataset 70%/30% into training/test sets.
train, test = split_data(std_data, 0.7)

# Save dataframe versions of our dataset.
train_cut = DataFrame(train, names(data))
test_cut = DataFrame(test, names(data))

# Create our labels. These are the values we are trying to predict.
train_label = train_cut[:, :MPG]
test_label = test_cut[:, :MPG]

# Get the list of columns to keep.
remove_names = filter(x->!in(x, [:MPG, :Model]), names(data))

# Filter the test and train sets.
train = Matrix(train_cut[:,remove_names]);
test = Matrix(test_cut[:,remove_names]);
```

## Model Specification

In a traditional frequentist model using [OLS](https://en.wikipedia.org/wiki/Ordinary_least_squares), our model might look like:

\$\$
MPG_i = \alpha + \boldsymbol{\beta}^T\boldsymbol{X_i}
\$\$

where $$\boldsymbol{\beta}$$ is a vector of coefficients and $$\boldsymbol{X}$$ is a vector of inputs for observation $$i$$. The Bayesian model we are more concerned with is the following:

\$\$
MPG_i \sim \mathcal{N}(\alpha + \boldsymbol{\beta}^T\boldsymbol{X_i}, \sigma^2)
\$\$

where $$\alpha$$ is an intercept term common to all observations, $$\boldsymbol{\beta}$$ is a coefficient vector, $$\boldsymbol{X_i}$$ is the observed data for car $$i$$, and $$\sigma^2$$ is a common variance term.

For $$\sigma^2$$, we assign a prior of `TruncatedNormal(0,100,0,Inf)`. This is consistent with [Andrew Gelman's recommendations](http://www.stat.columbia.edu/~gelman/research/published/taumain.pdf) on noninformative priors for variance. The intercept term ($$\alpha$$) is assumed to be normally distributed with a mean of zero and a variance of three. This represents our assumptions that miles per gallon can be explained mostly by our assorted variables, but a high variance term indicates our uncertainty about that. Each coefficient is assumed to be normally distributed with a mean of zero and a variance of 10. We do not know that our coefficients are different from zero, and we don't know which ones are likely to be the most important, so the variance term is quite high. The syntax `::Type{T}=Vector{Float64}` allows us to maintain type stability in our model -- for more information, please review the [performance tips](https://turing.ml/dev/docs/using-turing/performancetips#make-your-model-type-stable). Lastly, each observation $$y_i$$ is distributed according to the calculated `mu` term given by $$\alpha + \boldsymbol{\beta}^T\boldsymbol{X_i}$$.


```julia
# Bayesian linear regression.
@model linear_regression(x, y, n_obs, n_vars, ::Type{T}=Vector{Float64}) where {T} = begin
    # Set variance prior.
    σ₂ ~ truncated(Normal(0,100), 0, Inf)
    
    # Set intercept prior.
    intercept ~ Normal(0, 3)
    
    # Set the priors on our coefficients.
    coefficients = T(undef, n_vars)
    
    for i in 1:n_vars
        coefficients[i] ~ Normal(0, 10)
    end
    
    # Calculate all the mu terms.
    mu = intercept .+ x * coefficients
    y ~ MvNormal(mu, σ₂)
end;
```

With our model specified, we can call the sampler. We will use the No U-Turn Sampler ([NUTS](http://turing.ml/docs/library/#-turingnuts--type)) here. 


```julia
n_obs, n_vars = size(train)
model = linear_regression(train, train_label, n_obs, n_vars)
chain = sample(model, NUTS(0.65), 3000);
```

    ┌ Info: Found initial step size
    │   ϵ = 0.8
    └ @ Turing.Inference /home/cameron/.julia/packages/Turing/cReBm/src/inference/hmc.jl:556


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
            parameters     mean     std  naive_se    mcse        ess   r_hat
      ────────────────  ───────  ──────  ────────  ──────  ─────────  ──────
       coefficients[1]   0.4016  0.4323    0.0097  0.0126  1418.0304  1.0027
       coefficients[2]  -0.1277  0.4637    0.0104  0.0135  1181.7039  1.0005
       coefficients[3]  -0.1022  0.4395    0.0098  0.0109  1431.9877  0.9995
       coefficients[4]   0.6234  0.2929    0.0065  0.0084  1283.4786  0.9999
       coefficients[5]   0.0228  0.4389    0.0098  0.0122   952.5486  1.0011
       coefficients[6]   0.0806  0.3023    0.0068  0.0070  1172.3970  1.0005
       coefficients[7]  -0.0882  0.2855    0.0064  0.0107  1301.0501  0.9995
       coefficients[8]   0.1230  0.2741    0.0061  0.0092  1205.5582  0.9996
       coefficients[9]   0.2870  0.4770    0.0107  0.0200  1142.3295  1.0007
      coefficients[10]  -0.8466  0.4473    0.0100  0.0155  1028.5133  0.9999
             intercept   0.0488  0.1879    0.0042  0.0069  1318.5615  0.9998
                    σ₂   0.4690  0.1216    0.0027  0.0065   441.0666  1.0029
    
    Quantiles
            parameters     2.5%    25.0%    50.0%    75.0%   97.5%
      ────────────────  ───────  ───────  ───────  ───────  ──────
       coefficients[1]  -0.4586   0.1308   0.3952   0.6760  1.2983
       coefficients[2]  -1.0837  -0.4147  -0.1183   0.1735  0.7539
       coefficients[3]  -0.9534  -0.3692  -0.0970   0.1754  0.7399
       coefficients[4]   0.0657   0.4384   0.6213   0.8051  1.2116
       coefficients[5]  -0.8295  -0.2490   0.0233   0.2867  0.8957
       coefficients[6]  -0.5154  -0.1071   0.0786   0.2649  0.6782
       coefficients[7]  -0.6629  -0.2700  -0.0815   0.0969  0.4499
       coefficients[8]  -0.4245  -0.0580   0.1305   0.3092  0.6471
       coefficients[9]  -0.6654  -0.0012   0.2929   0.5712  1.2670
      coefficients[10]  -1.7458  -1.1247  -0.8455  -0.5655  0.0667
             intercept  -0.3131  -0.0670   0.0432   0.1618  0.4354
                    σ₂   0.3089   0.3868   0.4441   0.5243  0.7552




## Comparing to OLS

A satisfactory test of our model is to evaluate how well it predicts. Importantly, we want to compare our model to existing tools like OLS. The code below uses the [GLM.jl]() package to generate a traditional OLS multiple regression model on the same data as our probabalistic model.


```julia
# Import the GLM package.
using GLM

# Perform multiple regression OLS.
ols = lm(@formula(MPG ~ Cyl + Disp + HP + DRat + WT + QSec + VS + AM + Gear + Carb), train_cut)

# Store our predictions in the original dataframe.
train_cut.OLSPrediction = unstandardize(GLM.predict(ols), data.MPG);
test_cut.OLSPrediction = unstandardize(GLM.predict(ols, test_cut), data.MPG);
```

The function below accepts a chain and an input matrix and calculates predictions. We use the mean observation of each parameter in the model starting with sample 200, which is where the warm-up period for the NUTS sampler ended.


```julia
# Make a prediction given an input vector.
function prediction(chain, x)
    p = get_params(chain[200:end, :, :])
    α = mean(p.intercept)
    β = collect(mean.(p.coefficients))
    return  α .+ x * β
end
```




    prediction (generic function with 1 method)



When we make predictions, we unstandardize them so they're more understandable. We also add them to the original dataframes so they can be placed in context.


```julia
# Calculate the predictions for the training and testing sets.
train_cut.BayesPredictions = unstandardize(prediction(chain, train), data.MPG);
test_cut.BayesPredictions = unstandardize(prediction(chain, test), data.MPG);

# Unstandardize the dependent variable.
train_cut.MPG = unstandardize(train_cut.MPG, data.MPG);
test_cut.MPG = unstandardize(test_cut.MPG, data.MPG);

# Show the first side rows of the modified dataframe.
first(test_cut, 6)
```




<table class="data-frame"><thead><tr><th></th><th>MPG</th><th>Cyl</th><th>Disp</th><th>HP</th><th>DRat</th><th>WT</th><th>QSec</th><th>VS</th></tr><tr><th></th><th>Float64</th><th>Float64</th><th>Float64</th><th>Float64</th><th>Float64</th><th>Float64</th><th>Float64</th><th>Float64</th></tr></thead><tbody><p>6 rows × 13 columns (omitted printing of 5 columns)</p><tr><th>1</th><td>116.195</td><td>1.01488</td><td>0.591245</td><td>0.0483133</td><td>-0.835198</td><td>0.222544</td><td>-0.307089</td><td>-0.868028</td></tr><tr><th>2</th><td>114.295</td><td>1.01488</td><td>0.962396</td><td>1.4339</td><td>0.249566</td><td>0.636461</td><td>-1.36476</td><td>-0.868028</td></tr><tr><th>3</th><td>120.195</td><td>1.01488</td><td>1.36582</td><td>0.412942</td><td>-0.966118</td><td>0.641571</td><td>-0.446992</td><td>-0.868028</td></tr><tr><th>4</th><td>128.295</td><td>-1.22486</td><td>-1.22417</td><td>-1.17684</td><td>0.904164</td><td>-1.31048</td><td>0.588295</td><td>1.11604</td></tr><tr><th>5</th><td>126.995</td><td>-1.22486</td><td>-0.890939</td><td>-0.812211</td><td>1.55876</td><td>-1.10097</td><td>-0.642858</td><td>-0.868028</td></tr><tr><th>6</th><td>131.395</td><td>-1.22486</td><td>-1.09427</td><td>-0.491337</td><td>0.324377</td><td>-1.74177</td><td>-0.530935</td><td>1.11604</td></tr></tbody></table>



Now let's evaluate the loss for each method, and each prediction set. We will use sum of squared error function to evaluate loss, given by 

\$\$
\text{SSE} = \sum{(y_i - \hat{y_i})^2}
\$\$

where $$y_i$$ is the actual value (true MPG) and $$\hat{y_i}$$ is the predicted value using either OLS or Bayesian linear regression. A lower SSE indicates a closer fit to the data.


```julia
bayes_loss1 = sum((train_cut.BayesPredictions - train_cut.MPG).^2)
ols_loss1 = sum((train_cut.OLSPrediction - train_cut.MPG).^2)

bayes_loss2 = sum((test_cut.BayesPredictions - test_cut.MPG).^2)
ols_loss2 = sum((test_cut.OLSPrediction - test_cut.MPG).^2)

println("Training set:
    Bayes loss: $$bayes_loss1
    OLS loss: $$ols_loss1
Test set: 
    Bayes loss: $$bayes_loss2
    OLS loss: $$ols_loss2")
```

    Training set:
        Bayes loss: 67.61488347514008
        OLS loss: 67.56037474764642
    Test set: 
        Bayes loss: 278.859606131571
        OLS loss: 270.9481307076011


As we can see above, OLS and our Bayesian model fit our training set about the same. This is to be expected, given that it is our training set. However, the Bayesian linear regression model is less able to predict out of sample -- this is likely due to our selection of priors, and that fact that point estimates were used to forecast instead of the true posteriors.
