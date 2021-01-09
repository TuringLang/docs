---
title: Bayesian Multinomial Logistic Regression
permalink: /:collection/:name/
---
[Multinomial logistic regression](https://en.wikipedia.org/wiki/Multinomial_logistic_regression) is an extension of logistic regression. Logistic regression is used to model problems in which there are exactly two possible discrete outcomes. Multinomial logistic regression is used to model problems in which there are two or more possible discrete outcomes.

In our example, we'll be using the iris dataset. The goal of the iris multiclass problem is to predict the species of a flower given measurements (in centimeters) of sepal length and width and petal length and width. There are three possible species: Iris setosa, Iris versicolor, and Iris virginica.

To start, let's import all the libraries we'll need.


```julia
# Load Turing.
using Turing

# Load RDatasets.
using RDatasets

# Load StatsPlots for visualizations and diagnostics.
using StatsPlots

# Functionality for splitting and normalizing the data.
using MLDataUtils: shuffleobs, splitobs, rescale!

# We need a softmax function which is provided by NNlib.
using NNlib: softmax

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
    ┌ Info: Precompiling StatsPlots [f3b207a7-027a-5e70-b257-86293d7955fd]
    └ @ Base loading.jl:1260
    ┌ Info: Precompiling MLDataUtils [cc2ba9b6-d476-5e6d-8eaf-a92d5412d41d]
    └ @ Base loading.jl:1260
    ┌ Info: [Turing]: progress logging is disabled globally
    └ @ Turing /home/cameron/.julia/packages/Turing/3goIa/src/Turing.jl:23
    ┌ Info: [AdvancedVI]: global PROGRESS is set as false
    └ @ AdvancedVI /home/cameron/.julia/packages/AdvancedVI/PaSeO/src/AdvancedVI.jl:15


## Data Cleaning & Set Up

Now we're going to import our dataset. Twenty rows of the dataset are shown below so you can get a good feel for what kind of data we have.


```julia
# Import the "iris" dataset.
data = RDatasets.dataset("datasets", "iris");

# Show twenty random rows.
data[rand(1:size(data, 1), 20), :]
```




<table class="data-frame"><thead><tr><th></th><th>SepalLength</th><th>SepalWidth</th><th>PetalLength</th><th>PetalWidth</th><th>Species</th></tr><tr><th></th><th>Float64</th><th>Float64</th><th>Float64</th><th>Float64</th><th>Cat…</th></tr></thead><tbody><p>20 rows × 5 columns</p><tr><th>1</th><td>5.6</td><td>2.9</td><td>3.6</td><td>1.3</td><td>versicolor</td></tr><tr><th>2</th><td>5.8</td><td>2.7</td><td>3.9</td><td>1.2</td><td>versicolor</td></tr><tr><th>3</th><td>5.5</td><td>2.3</td><td>4.0</td><td>1.3</td><td>versicolor</td></tr><tr><th>4</th><td>6.7</td><td>3.3</td><td>5.7</td><td>2.5</td><td>virginica</td></tr><tr><th>5</th><td>5.1</td><td>3.5</td><td>1.4</td><td>0.2</td><td>setosa</td></tr><tr><th>6</th><td>5.1</td><td>3.8</td><td>1.5</td><td>0.3</td><td>setosa</td></tr><tr><th>7</th><td>4.8</td><td>3.4</td><td>1.9</td><td>0.2</td><td>setosa</td></tr><tr><th>8</th><td>6.0</td><td>2.9</td><td>4.5</td><td>1.5</td><td>versicolor</td></tr><tr><th>9</th><td>6.9</td><td>3.1</td><td>5.4</td><td>2.1</td><td>virginica</td></tr><tr><th>10</th><td>5.4</td><td>3.9</td><td>1.7</td><td>0.4</td><td>setosa</td></tr><tr><th>11</th><td>5.0</td><td>3.6</td><td>1.4</td><td>0.2</td><td>setosa</td></tr><tr><th>12</th><td>5.7</td><td>3.0</td><td>4.2</td><td>1.2</td><td>versicolor</td></tr><tr><th>13</th><td>5.0</td><td>3.3</td><td>1.4</td><td>0.2</td><td>setosa</td></tr><tr><th>14</th><td>7.7</td><td>3.0</td><td>6.1</td><td>2.3</td><td>virginica</td></tr><tr><th>15</th><td>5.8</td><td>2.8</td><td>5.1</td><td>2.4</td><td>virginica</td></tr><tr><th>16</th><td>4.4</td><td>3.0</td><td>1.3</td><td>0.2</td><td>setosa</td></tr><tr><th>17</th><td>6.3</td><td>3.3</td><td>4.7</td><td>1.6</td><td>versicolor</td></tr><tr><th>18</th><td>6.0</td><td>2.7</td><td>5.1</td><td>1.6</td><td>versicolor</td></tr><tr><th>19</th><td>4.6</td><td>3.4</td><td>1.4</td><td>0.3</td><td>setosa</td></tr><tr><th>20</th><td>6.0</td><td>2.2</td><td>4.0</td><td>1.0</td><td>versicolor</td></tr></tbody></table>



In this data set, the outcome `Species` is currently coded as a string. We convert it to a numerical value by using indices `1`, `2`, and `3` to indicate species `setosa`, `versicolor`, and `virginica`, respectively.


```julia
# Recode the `Species` column.
species = ["setosa", "versicolor", "virginica"]
data[!, :Species_index] = indexin(data[!, :Species], species)

# Show twenty random rows of the new species columns
data[rand(1:size(data, 1), 20), [:Species, :Species_index]]
```




<table class="data-frame"><thead><tr><th></th><th>Species</th><th>Species_index</th></tr><tr><th></th><th>Cat…</th><th>Union…</th></tr></thead><tbody><p>20 rows × 2 columns</p><tr><th>1</th><td>versicolor</td><td>2</td></tr><tr><th>2</th><td>setosa</td><td>1</td></tr><tr><th>3</th><td>setosa</td><td>1</td></tr><tr><th>4</th><td>versicolor</td><td>2</td></tr><tr><th>5</th><td>setosa</td><td>1</td></tr><tr><th>6</th><td>virginica</td><td>3</td></tr><tr><th>7</th><td>versicolor</td><td>2</td></tr><tr><th>8</th><td>versicolor</td><td>2</td></tr><tr><th>9</th><td>setosa</td><td>1</td></tr><tr><th>10</th><td>virginica</td><td>3</td></tr><tr><th>11</th><td>setosa</td><td>1</td></tr><tr><th>12</th><td>setosa</td><td>1</td></tr><tr><th>13</th><td>versicolor</td><td>2</td></tr><tr><th>14</th><td>versicolor</td><td>2</td></tr><tr><th>15</th><td>virginica</td><td>3</td></tr><tr><th>16</th><td>versicolor</td><td>2</td></tr><tr><th>17</th><td>setosa</td><td>1</td></tr><tr><th>18</th><td>setosa</td><td>1</td></tr><tr><th>19</th><td>virginica</td><td>3</td></tr><tr><th>20</th><td>setosa</td><td>1</td></tr></tbody></table>



After we've done that tidying, it's time to split our dataset into training and testing sets, and separate the features and target from the data. Additionally, we must rescale our feature variables so that they are centered around zero by subtracting each column by the mean and dividing it by the standard deviation. Without this step, Turing's sampler will have a hard time finding a place to start searching for parameter estimates.


```julia
# Split our dataset 50%/50% into training/test sets.
trainset, testset = splitobs(shuffleobs(data), 0.5)

# Define features and target.
features = [:SepalLength, :SepalWidth, :PetalLength, :PetalWidth]
target = :Species_index

# Turing requires data in matrix and vector form.
train_features = Matrix(trainset[!, features])
test_features = Matrix(testset[!, features])
train_target = trainset[!, target]
test_target = testset[!, target]

# Standardize the features.
μ, σ = rescale!(train_features; obsdim = 1)
rescale!(test_features, μ, σ; obsdim = 1);
```

## Model Declaration

Finally, we can define our model `logistic_regression`. It is a function that takes three arguments where

- `x` is our set of independent variables;
- `y` is the element we want to predict;
- `σ` is the standard deviation we want to assume for our priors.

We select the `setosa` species as the baseline class (the choice does not matter). Then we create the intercepts and vectors of coefficients for the other classes against that baseline. More concretely, we create scalar intercepts `intercept_versicolor` and `intersept_virginica` and coefficient vectors `coefficients_versicolor` and `coefficients_virginica` with four coefficients each for the features `SepalLength`, `SepalWidth`, `PetalLength` and `PetalWidth`. We assume a normal distribution with mean zero and standard deviation `σ` as prior for each scalar parameter. We want to find the posterior distribution of these, in total ten, parameters to be able to predict the species for any given set of features.


```julia
# Bayesian multinomial logistic regression
@model function logistic_regression(x, y, σ)
    n = size(x, 1)
    length(y) == n || throw(DimensionMismatch("number of observations in `x` and `y` is not equal"))

    # Priors of intercepts and coefficients.
    intercept_versicolor ~ Normal(0, σ)
    intercept_virginica ~ Normal(0, σ)
    coefficients_versicolor ~ MvNormal(4, σ)
    coefficients_virginica ~ MvNormal(4, σ)

    # Compute the likelihood of the observations.
    values_versicolor = intercept_versicolor .+ x * coefficients_versicolor
    values_virginica = intercept_virginica .+ x * coefficients_virginica
    for i in 1:n
        # the 0 corresponds to the base category `setosa`
        v = softmax([0, values_versicolor[i], values_virginica[i]])
        y[i] ~ Categorical(v)
    end
end;
```

## Sampling

Now we can run our sampler. This time we'll use [`HMC`](http://turing.ml/docs/library/#Turing.HMC) to sample from our posterior.


```julia
chain = sample(logistic_regression(train_features, train_target, 1), HMC(0.05, 10), MCMCThreads(), 1500, 3)
```




    Chains MCMC chain (1500×19×3 Array{Float64,3}):

    Iterations        = 1:1500
    Thinning interval = 1
    Chains            = 1, 2, 3
    Samples per chain = 1500
    parameters        = coefficients_versicolor[1], coefficients_versicolor[2], coefficients_versicolor[3], coefficients_versicolor[4], coefficients_virginica[1], coefficients_virginica[2], coefficients_virginica[3], coefficients_virginica[4], intercept_versicolor, intercept_virginica
    internals         = acceptance_rate, hamiltonian_energy, hamiltonian_energy_error, is_accept, log_density, lp, n_steps, nom_step_size, step_size

    Summary Statistics
     [0m[1m                 parameters [0m [0m[1m    mean [0m [0m[1m     std [0m [0m[1m naive_se [0m [0m[1m    mcse [0m [0m[1m      ess [0m [0m[1m    rhat [0m [0m
     [0m[90m                     Symbol [0m [0m[90m Float64 [0m [0m[90m Float64 [0m [0m[90m  Float64 [0m [0m[90m Float64 [0m [0m[90m  Float64 [0m [0m[90m Float64 [0m [0m
     [0m                            [0m [0m         [0m [0m         [0m [0m          [0m [0m         [0m [0m          [0m [0m         [0m [0m
     [0m coefficients_versicolor[1] [0m [0m  1.5404 [0m [0m  0.6753 [0m [0m   0.0101 [0m [0m  0.0335 [0m [0m 332.4769 [0m [0m  1.0017 [0m [0m
     [0m coefficients_versicolor[2] [0m [0m -1.4298 [0m [0m  0.5098 [0m [0m   0.0076 [0m [0m  0.0171 [0m [0m 786.5622 [0m [0m  1.0015 [0m [0m
     [0m coefficients_versicolor[3] [0m [0m  1.1382 [0m [0m  0.7772 [0m [0m   0.0116 [0m [0m  0.0398 [0m [0m 328.8508 [0m [0m  1.0091 [0m [0m
     [0m coefficients_versicolor[4] [0m [0m  0.0693 [0m [0m  0.7300 [0m [0m   0.0109 [0m [0m  0.0374 [0m [0m 368.3007 [0m [0m  1.0048 [0m [0m
     [0m  coefficients_virginica[1] [0m [0m  0.4251 [0m [0m  0.6983 [0m [0m   0.0104 [0m [0m  0.0294 [0m [0m 381.6545 [0m [0m  1.0017 [0m [0m
     [0m  coefficients_virginica[2] [0m [0m -0.6744 [0m [0m  0.6036 [0m [0m   0.0090 [0m [0m  0.0250 [0m [0m 654.1030 [0m [0m  1.0012 [0m [0m
     [0m  coefficients_virginica[3] [0m [0m  2.0076 [0m [0m  0.8424 [0m [0m   0.0126 [0m [0m  0.0390 [0m [0m 344.6077 [0m [0m  1.0067 [0m [0m
     [0m  coefficients_virginica[4] [0m [0m  2.6704 [0m [0m  0.7982 [0m [0m   0.0119 [0m [0m  0.0423 [0m [0m 337.9600 [0m [0m  1.0043 [0m [0m
     [0m       intercept_versicolor [0m [0m  0.8408 [0m [0m  0.5257 [0m [0m   0.0078 [0m [0m  0.0167 [0m [0m 874.4821 [0m [0m  1.0044 [0m [0m
     [0m        intercept_virginica [0m [0m -0.7351 [0m [0m  0.6639 [0m [0m   0.0099 [0m [0m  0.0285 [0m [0m 525.8135 [0m [0m  1.0039 [0m [0m

    Quantiles
     [0m[1m                 parameters [0m [0m[1m    2.5% [0m [0m[1m   25.0% [0m [0m[1m   50.0% [0m [0m[1m   75.0% [0m [0m[1m   97.5% [0m [0m
     [0m[90m                     Symbol [0m [0m[90m Float64 [0m [0m[90m Float64 [0m [0m[90m Float64 [0m [0m[90m Float64 [0m [0m[90m Float64 [0m [0m
     [0m                            [0m [0m         [0m [0m         [0m [0m         [0m [0m         [0m [0m         [0m [0m
     [0m coefficients_versicolor[1] [0m [0m  0.2659 [0m [0m  1.0755 [0m [0m  1.5231 [0m [0m  1.9860 [0m [0m  2.9059 [0m [0m
     [0m coefficients_versicolor[2] [0m [0m -2.4714 [0m [0m -1.7610 [0m [0m -1.4109 [0m [0m -1.0749 [0m [0m -0.4921 [0m [0m
     [0m coefficients_versicolor[3] [0m [0m -0.4377 [0m [0m  0.6358 [0m [0m  1.1456 [0m [0m  1.6500 [0m [0m  2.6215 [0m [0m
     [0m coefficients_versicolor[4] [0m [0m -1.3741 [0m [0m -0.4381 [0m [0m  0.0652 [0m [0m  0.5711 [0m [0m  1.4808 [0m [0m
     [0m  coefficients_virginica[1] [0m [0m -0.9452 [0m [0m -0.0487 [0m [0m  0.4287 [0m [0m  0.8991 [0m [0m  1.7973 [0m [0m
     [0m  coefficients_virginica[2] [0m [0m -1.8717 [0m [0m -1.0756 [0m [0m -0.6641 [0m [0m -0.2501 [0m [0m  0.4867 [0m [0m
     [0m  coefficients_virginica[3] [0m [0m  0.3740 [0m [0m  1.4180 [0m [0m  1.9941 [0m [0m  2.5862 [0m [0m  3.6788 [0m [0m
     [0m  coefficients_virginica[4] [0m [0m  1.1985 [0m [0m  2.1347 [0m [0m  2.6359 [0m [0m  3.1795 [0m [0m  4.3502 [0m [0m
     [0m       intercept_versicolor [0m [0m -0.1652 [0m [0m  0.4888 [0m [0m  0.8340 [0m [0m  1.1858 [0m [0m  1.8891 [0m [0m
     [0m        intercept_virginica [0m [0m -2.0101 [0m [0m -1.1944 [0m [0m -0.7453 [0m [0m -0.2834 [0m [0m  0.5836 [0m [0m




Since we ran multiple chains, we may as well do a spot check to make sure each chain converges around similar points.


```julia
plot(chain)
```




![svg](/tutorials/8_MultinomialLogisticRegression_files/8_MultinomialLogisticRegression_13_0.svg)



Looks good!

We can also use the `corner` function from MCMCChains to show the distributions of the various parameters of our multinomial logistic regression. The corner function requires MCMCChains and StatsPlots.


```julia
corner(
    chain, [Symbol("coefficients_versicolor[$i]") for i in 1:4];
    label=[string(i) for i in 1:4], fmt=:png
)
```




![png](/tutorials/8_MultinomialLogisticRegression_files/8_MultinomialLogisticRegression_15_0.png)




```julia
corner(
    chain, [Symbol("coefficients_virginica[$i]") for i in 1:4];
    label=[string(i) for i in 1:4], fmt=:png
)
```




![png](/tutorials/8_MultinomialLogisticRegression_files/8_MultinomialLogisticRegression_16_0.png)



Fortunately the corner plots appear to demonstrate unimodal distributions for each of our parameters, so it should be straightforward to take the means of each parameter's sampled values to estimate our model to make predictions.

## Making Predictions

How do we test how well the model actually predicts whether someone is likely to default? We need to build a `prediction` function that takes the test dataset and runs it through the average parameter calculated during sampling.

The `prediction` function below takes a `Matrix` and a `Chains` object. It computes the mean of the sampled parameters and calculates the species with the highest probability for each observation. Note that we do not have to evaluate the `softmax` function since it does not affect the order of its inputs.


```julia
function prediction(x::Matrix, chain)
    # Pull the means from each parameter's sampled values in the chain.
    intercept_versicolor = mean(chain, :intercept_versicolor)
    intercept_virginica = mean(chain, :intercept_virginica)
    coefficients_versicolor = [mean(chain, "coefficients_versicolor[$i]") for i in 1:4]
    coefficients_virginica = [mean(chain, "coefficients_virginica[$i]") for i in 1:4]

    # Compute the index of the species with the highest probability for each observation.
    values_versicolor = intercept_versicolor .+ x * coefficients_versicolor
    values_virginica = intercept_virginica .+ x * coefficients_virginica
    species_indices = [argmax((0, x, y)) for (x, y) in zip(values_versicolor, values_virginica)]

    return species_indices
end;
```

Let's see how we did! We run the test matrix through the prediction function, and compute the accuracy for our prediction.


```julia
# Make the predictions.
predictions = prediction(test_features, chain)

# Calculate accuracy for our test set.
mean(predictions .== testset[!, :Species_index])
```




    0.8533333333333334



Perhaps more important is to see the accuracy per class.


```julia
for s in 1:3
    rows = testset[!, :Species_index] .== s
    println("Number of `", species[s], "`: ", count(rows))
    println("Percentage of `", species[s], "` predicted correctly: ",
        mean(predictions[rows] .== testset[rows, :Species_index]))
end
```

    Number of `setosa`: 22
    Percentage of `setosa` predicted correctly: 1.0
    Number of `versicolor`: 24
    Percentage of `versicolor` predicted correctly: 0.875
    Number of `virginica`: 29
    Percentage of `virginica` predicted correctly: 0.7241379310344828


This tutorial has demonstrated how to use Turing to perform Bayesian multinomial logistic regression.
