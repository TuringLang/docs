---
title: Bayesian Logistic Regression
permalink: /:collection/:name/
---
[Bayesian logistic regression](https://en.wikipedia.org/wiki/Logistic_regression#Bayesian) is the Bayesian counterpart to a common tool in machine learning, logistic regression. The goal of logistic regression is to predict a one or a zero for a given training item. An example might be predicting whether someone is sick or ill given their symptoms and personal information.

In our example, we'll be working to predict whether someone is likely to default with a synthetic dataset found in the `RDatasets` package. This dataset, `Defaults`, comes from R's [ISLR](https://cran.r-project.org/web/packages/ISLR/index.html) package and contains information on borrowers.

To start, let's import all the libraries we'll need.


```julia
# Import Turing and Distributions.
using Turing, Distributions

# Import RDatasets.
using RDatasets

# Import MCMCChains, Plots, and StatsPlots for visualizations and diagnostics.
using MCMCChains, Plots, StatsPlots

# We need a logistic function, which is provided by StatsFuns.
using StatsFuns: logistic

# Functionality for splitting and normalizing the data
using MLDataUtils: shuffleobs, stratifiedobs, rescale!

# Set a seed for reproducibility.
using Random
Random.seed!(0);

# Turn off progress monitor.
Turing.turnprogress(false)
```

    ┌ Info: [Turing]: progress logging is disabled globally
    └ @ Turing /home/cameron/.julia/packages/Turing/cReBm/src/Turing.jl:22





    false



## Data Cleaning & Set Up

Now we're going to import our dataset. The first six rows of the dataset are shown below so you can get a good feel for what kind of data we have.


```julia
# Import the "Default" dataset.
data = RDatasets.dataset("ISLR", "Default");

# Show the first six rows of the dataset.
first(data, 6)
```




<table class="data-frame"><thead><tr><th></th><th>Default</th><th>Student</th><th>Balance</th><th>Income</th></tr><tr><th></th><th>Categorical…</th><th>Categorical…</th><th>Float64</th><th>Float64</th></tr></thead><tbody><p>6 rows × 4 columns</p><tr><th>1</th><td>No</td><td>No</td><td>729.526</td><td>44361.6</td></tr><tr><th>2</th><td>No</td><td>Yes</td><td>817.18</td><td>12106.1</td></tr><tr><th>3</th><td>No</td><td>No</td><td>1073.55</td><td>31767.1</td></tr><tr><th>4</th><td>No</td><td>No</td><td>529.251</td><td>35704.5</td></tr><tr><th>5</th><td>No</td><td>No</td><td>785.656</td><td>38463.5</td></tr><tr><th>6</th><td>No</td><td>Yes</td><td>919.589</td><td>7491.56</td></tr></tbody></table>



Most machine learning processes require some effort to tidy up the data, and this is no different. We need to convert the `Default` and `Student` columns, which say "Yes" or "No" into 1s and 0s. Afterwards, we'll get rid of the old words-based columns.


```julia
# Convert "Default" and "Student" to numeric values.
data[!,:DefaultNum] = [r.Default == "Yes" ? 1.0 : 0.0 for r in eachrow(data)]
data[!,:StudentNum] = [r.Student == "Yes" ? 1.0 : 0.0 for r in eachrow(data)]

# Delete the old columns which say "Yes" and "No".
select!(data, Not([:Default, :Student]))

# Show the first six rows of our edited dataset.
first(data, 6)
```




<table class="data-frame"><thead><tr><th></th><th>Balance</th><th>Income</th><th>DefaultNum</th><th>StudentNum</th></tr><tr><th></th><th>Float64</th><th>Float64</th><th>Float64</th><th>Float64</th></tr></thead><tbody><p>6 rows × 4 columns</p><tr><th>1</th><td>729.526</td><td>44361.6</td><td>0.0</td><td>0.0</td></tr><tr><th>2</th><td>817.18</td><td>12106.1</td><td>0.0</td><td>1.0</td></tr><tr><th>3</th><td>1073.55</td><td>31767.1</td><td>0.0</td><td>0.0</td></tr><tr><th>4</th><td>529.251</td><td>35704.5</td><td>0.0</td><td>0.0</td></tr><tr><th>5</th><td>785.656</td><td>38463.5</td><td>0.0</td><td>0.0</td></tr><tr><th>6</th><td>919.589</td><td>7491.56</td><td>0.0</td><td>1.0</td></tr></tbody></table>



After we've done that tidying, it's time to split our dataset into training and testing sets, and separate the labels from the data. We separate our data into two halves, `train` and `test`. You can use a higher percentage of splitting (or a lower one) by modifying the `at = 0.05` argument. We have highlighted the use of only a 5% sample to show the power of Bayesian inference with small sample sizes.

We must rescale our variables so that they are centered around zero by subtracting each column by the mean and dividing it by the standard deviation. Without this step, Turing's sampler will have a hard time finding a place to start searching for parameter estimates. To do this we will leverage `MLDataUtils`, which also lets us effortlessly shuffle our observations and perform a stratified split to get a representative test set.


```julia
function split_data(df, target; at = 0.70)
    shuffled = shuffleobs(df)
    trainset, testset = stratifiedobs(row -> row[target], 
                                      shuffled, p = at)
end

features = [:StudentNum, :Balance, :Income]
numerics = [:Balance, :Income]
target = :DefaultNum

trainset, testset = split_data(data, target, at = 0.05)
for feature in numerics
  μ, σ = rescale!(trainset[!, feature], obsdim=1)
  rescale!(testset[!, feature], μ, σ, obsdim=1)
end

# Turing requires data in matrix form, not dataframe
train = Matrix(trainset[:, features])
test = Matrix(testset[:, features])
train_label = trainset[:, target]
test_label = testset[:, target];
```

## Model Declaration 
Finally, we can define our model.

`logistic_regression` takes four arguments:

- `x` is our set of independent variables;
- `y` is the element we want to predict;
- `n` is the number of observations we have; and
- `σ` is the standard deviation we want to assume for our priors.

Within the model, we create four coefficients (`intercept`, `student`, `balance`, and `income`) and assign a prior of normally distributed with means of zero and standard deviations of `σ`. We want to find values of these four coefficients to predict any given `y`.

The `for` block creates a variable `v` which is the logistic function. We then observe the liklihood of calculating `v` given the actual label, `y[i]`.


```julia
# Bayesian logistic regression (LR)
@model logistic_regression(x, y, n, σ) = begin
    intercept ~ Normal(0, σ)

    student ~ Normal(0, σ)
    balance ~ Normal(0, σ)
    income  ~ Normal(0, σ)

    for i = 1:n
        v = logistic(intercept + student*x[i, 1] + balance*x[i,2] + income*x[i,3])
        y[i] ~ Bernoulli(v)
    end
end;
```

## Sampling

Now we can run our sampler. This time we'll use [`HMC`](http://turing.ml/docs/library/#Turing.HMC) to sample from our posterior.


```julia
# Retrieve the number of observations.
n, _ = size(train)

# Sample using HMC.
chain = mapreduce(c -> sample(logistic_regression(train, train_label, n, 1), HMC(0.05, 10), 1500),
    chainscat,
    1:3
)

describe(chain)
```




    2-element Array{ChainDataFrame,1}
    
    Summary Statistics
      parameters     mean     std  naive_se    mcse        ess   r_hat
      ──────────  ───────  ──────  ────────  ──────  ─────────  ──────
         balance   1.6517  0.3099    0.0046  0.0080   110.2122  1.0004
          income  -0.5174  0.3241    0.0048  0.0081  1440.4337  1.0010
       intercept  -3.8265  0.5148    0.0077  0.0148    54.8792  1.0004
         student  -1.8662  0.6088    0.0091  0.0223   840.9122  1.0037
    
    Quantiles
      parameters     2.5%    25.0%    50.0%    75.0%    97.5%
      ──────────  ───────  ───────  ───────  ───────  ───────
         balance   1.1418   1.4534   1.6331   1.8242   2.2196
          income  -1.1678  -0.7300  -0.5094  -0.3006   0.1079
       intercept  -4.6202  -4.0685  -3.7947  -3.5465  -3.0855
         student  -3.0690  -2.2803  -1.8574  -1.4528  -0.7137




Since we ran multiple chains, we may as well do a spot check to make sure each chain converges around similar points.


```julia
plot(chain)
```




![svg](/tutorials/2_LogisticRegression_files/2_LogisticRegression_13_0.svg)



Looks good!

We can also use the `corner` function from MCMCChains to show the distributions of the various parameters of our logistic regression. 


```julia
# The labels to use.
l = [:student, :balance, :income]

# Use the corner function. Requires StatsPlots and MCMCChains.
corner(chain, l)
```




![svg](/tutorials/2_LogisticRegression_files/2_LogisticRegression_15_0.svg)



Fortunately the corner plot appears to demonstrate unimodal distributions for each of our parameters, so it should be straightforward to take the means of each parameter's sampled values to estimate our model to make predictions.

## Making Predictions
How do we test how well the model actually predicts whether someone is likely to default? We need to build a prediction function that takes the `test` object we made earlier and runs it through the average parameter calculated during sampling.

The `prediction` function below takes a `Matrix` and a `Chain` object. It takes the mean of each parameter's sampled values and re-runs the logistic function using those mean values for every element in the test set.


```julia
function prediction(x::Matrix, chain, threshold)
    # Pull the means from each parameter's sampled values in the chain.
    intercept = mean(chain[:intercept].value)
    student = mean(chain[:student].value)
    balance = mean(chain[:balance].value)
    income = mean(chain[:income].value)

    # Retrieve the number of rows.
    n, _ = size(x)

    # Generate a vector to store our predictions.
    v = Vector{Float64}(undef, n)

    # Calculate the logistic function for each element in the test set.
    for i in 1:n
        num = logistic(intercept .+ student * x[i,1] + balance * x[i,2] + income * x[i,3])
        if num >= threshold
            v[i] = 1
        else
            v[i] = 0
        end
    end
    return v
end;
```

Let's see how we did! We run the test matrix through the prediction function, and compute the [mean squared error](https://en.wikipedia.org/wiki/Mean_squared_error) (MSE) for our prediction. The `threshold` variable sets the sensitivity of the predictions. For example, a threshold of 0.07 will predict a defualt value of 1 for any predicted value greater than 0.07 and no default if it is less than 0.07.


```julia
# Set the prediction threshold.
threshold = 0.07

# Make the predictions.
predictions = prediction(test, chain, threshold)

# Calculate MSE for our test set.
loss = sum((predictions - test_label).^2) / length(test_label)
```




    0.1163157894736842



Perhaps more important is to see what percentage of defaults we correctly predicted. The code below simply counts defaults and predictions and presents the results. 


```julia
defaults = sum(test_label)
not_defaults = length(test_label) - defaults

predicted_defaults = sum(test_label .== predictions .== 1)
predicted_not_defaults = sum(test_label .== predictions .== 0)

println("Defaults: $$defaults
    Predictions: $$predicted_defaults
    Percentage defaults correct $$(predicted_defaults/defaults)")

println("Not defaults: $$not_defaults
    Predictions: $$predicted_not_defaults
    Percentage non-defaults correct $$(predicted_not_defaults/not_defaults)")
```

    Defaults: 316.0
        Predictions: 265
        Percentage defaults correct 0.8386075949367089
    Not defaults: 9184.0
        Predictions: 8130
        Percentage non-defaults correct 0.8852351916376306


The above shows that with a threshold of 0.07, we correctly predict a respectable portion of the defaults, and correctly identify most non-defaults. This is fairly sensitive to a choice of threshold, and you may wish to experiment with it.

This tutorial has demonstrated how to use Turing to perform Bayesian logistic regression. 
