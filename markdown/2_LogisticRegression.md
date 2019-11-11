---
title: Bayesian Logistic Regression
permalink: /:collection/:name/
---

[Bayesian logistic regression](https://en.wikipedia.org/wiki/Logistic_regression#Bayesian) is the Bayesian counterpart to a common tool in machine learning, logistic regression. The goal of logistic regression is to predict a one or a zero for a given training item. An example might be predicting whether someone is sick or ill given their symptoms and personal information.

In our example, we'll be working to predict whether someone is likely to default with a synthetic dataset found in the `RDatasets` package. This dataset, `Defaults`, comes from R's [ISLR](https://cran.r-project.org/web/packages/ISLR/index.html) package and contains information on borrowers.

To start, let's import all the libraries we'll need.

````julia
# Import Turing and Distributions.
using Turing, Distributions

# Import RDatasets.
using RDatasets

# Import MCMCChains, Plots, and StatsPlots for visualizations and diagnostics.
using MCMCChains, Plots, StatsPlots

# We need a logistic function, which is provided by StatsFuns.
using StatsFuns: logistic

# Set a seed for reproducibility.
using Random
Random.seed!(0);

# Turn off progress monitor.
Turing.turnprogress(false)
````


````
false
````




## Data Cleaning & Set Up

Now we're going to import our dataset. The first six rows of the dataset are shown below so you capn get a good feel for what kind of data we have.

````julia
# Import the "Default" dataset.
data = RDatasets.dataset("ISLR", "Default");

# Show the first six rows of the dataset.
first(data, 6)
````


````
6×4 DataFrame
│ Row │ Default      │ Student      │ Balance │ Income  │
│     │ Categorical… │ Categorical… │ Float64 │ Float64 │
├─────┼──────────────┼──────────────┼─────────┼─────────┤
│ 1   │ No           │ No           │ 729.526 │ 44361.6 │
│ 2   │ No           │ Yes          │ 817.18  │ 12106.1 │
│ 3   │ No           │ No           │ 1073.55 │ 31767.1 │
│ 4   │ No           │ No           │ 529.251 │ 35704.5 │
│ 5   │ No           │ No           │ 785.656 │ 38463.5 │
│ 6   │ No           │ Yes          │ 919.589 │ 7491.56 │
````




Most machine learning processes require some effort to tidy up the data, and this is no different. We need to convert the `Default` and `Student` columns, which say "Yes" or "No" into 1s and 0s. Afterwards, we'll get rid of the old words-based columns.

````julia
# Convert "Default" and "Student" to numeric values.
data[!,:DefaultNum] = [r.Default == "Yes" ? 1.0 : 0.0 for r in eachrow(data)]
data[!, :StudentNum] = [r.Student == "Yes" ? 1.0 : 0.0 for r in eachrow(data)]

# Delete the old columns which say "Yes" and "No".
select!(data, Not([:Default, :Student]))

# Show the first six rows of our edited dataset.
first(data, 6)
````


````
6×4 DataFrame
│ Row │ Balance │ Income  │ DefaultNum │ StudentNum │
│     │ Float64 │ Float64 │ Float64    │ Float64    │
├─────┼─────────┼─────────┼────────────┼────────────┤
│ 1   │ 729.526 │ 44361.6 │ 0.0        │ 0.0        │
│ 2   │ 817.18  │ 12106.1 │ 0.0        │ 1.0        │
│ 3   │ 1073.55 │ 31767.1 │ 0.0        │ 0.0        │
│ 4   │ 529.251 │ 35704.5 │ 0.0        │ 0.0        │
│ 5   │ 785.656 │ 38463.5 │ 0.0        │ 0.0        │
│ 6   │ 919.589 │ 7491.56 │ 0.0        │ 1.0        │
````




After we've done that tidying, it's time to split our dataset into training and testing sets, and separate the labels from the data. We separate our data into two halves, `train` and `test`. You can use a higher percentage of splitting (or a lower one) by modifying the `at = 0.05` argument. We have highlighted the use of only a 5% sample to show the power of Bayesian inference with small sample sizes.

We must rescale our variables so that they are centered around zero by subtracting each column by the mean and dividing it by the standard deviation. Without this step, Turing's sampler will have a hard time finding a place to start searching for parameter estimates.

````julia
# Function to split samples.
function split_data(df, at = 0.70)
    (r, _) = size(df)
    index = Int(round(r * at))
    train = df[1:index, :]
    test  = df[(index+1):end, :]
    return train, test
end

# Rescale our columns.
data.Balance = (data.Balance .- mean(data.Balance)) ./ std(data.Balance)
data.Income = (data.Income .- mean(data.Income)) ./ std(data.Income)

# Split our dataset 5/95 into training/test sets.
train, test = split_data(data, 0.05);

# Create our labels. These are the values we are trying to predict.
train_label = train[:,:DefaultNum]
test_label = test[:,:DefaultNum]

# Remove the columns that are not our predictors.
train = train[:,[:StudentNum, :Balance, :Income]];
test = test[:,[:StudentNum, :Balance, :Income]];
````




Our `train` and `test` matrices are still in the `DataFrame` format, which tends not to play too well with the kind of manipulations we're about to do, so we convert them into `Matrix` objects.

````julia
# Convert the DataFrame objects to matrices.
train = Matrix(train);
test = Matrix(test);
````




## Model Declaration 
Finally, we can define our model.

`logistic_regression` takes four arguments:

- `x` is our set of independent variables;
- `y` is the element we want to predict;
- `n` is the number of observations we have; and
- `σ` is the standard deviation we want to assume for our priors.

Within the model, we create four coefficients (`intercept`, `student`, `balance`, and `income`) and assign a prior of normally distributed with means of zero and standard deviations of `σ`. We want to find values of these four coefficients to predict any given `y`.

The `for` block creates a variable `v` which is the logistic function. We then observe the liklihood of calculating `v` given the actual label, `y[i]`.

````julia
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
````




## Sampling

Now we can run our sampler. This time we'll use [`HMC`](http://turing.ml/docs/library/#Turing.HMC) to sample from our posterior.

````julia
# Retrieve the number of observations.
n, _ = size(train)

# Sample using HMC.
chain = mapreduce(c -> sample(logistic_regression(train, train_label, n, 1), HMC(0.05, 10), 1500),
    chainscat,
    1:3
)

describe(chain)
````


````
2-element Array{ChainDataFrame,1}

Summary Statistics
. Omitted printing of 1 columns
│ Row │ parameters │ mean      │ std      │ naive_se   │ mcse       │ ess  
   │
│     │ Symbol     │ Float64   │ Float64  │ Float64    │ Float64    │ Any  
   │
├─────┼────────────┼───────────┼──────────┼────────────┼────────────┼──────
───┤
│ 1   │ balance    │ 1.78999   │ 0.331794 │ 0.00494609 │ 0.00848061 │ 186.5
65 │
│ 2   │ income     │ -0.139602 │ 0.332703 │ 0.00495964 │ 0.011117   │ 967.8
64 │
│ 3   │ intercept  │ -4.12687  │ 0.545573 │ 0.00813292 │ 0.0172957  │ 65.36
02 │
│ 4   │ student    │ -0.918551 │ 0.636926 │ 0.00949473 │ 0.0309855  │ 529.7
24 │

Quantiles

│ Row │ parameters │ 2.5%      │ 25.0%    │ 50.0%     │ 75.0%     │ 97.5%  
  │
│     │ Symbol     │ Float64   │ Float64  │ Float64   │ Float64   │ Float64
  │
├─────┼────────────┼───────────┼──────────┼───────────┼───────────┼────────
──┤
│ 1   │ balance    │ 1.20484   │ 1.57879  │ 1.77619   │ 1.9798    │ 2.41041
  │
│ 2   │ income     │ -0.782103 │ -0.36119 │ -0.135722 │ 0.0840768 │ 0.48844
8 │
│ 3   │ intercept  │ -5.01721  │ -4.39229 │ -4.08815  │ -3.81892  │ -3.3217
8 │
│ 4   │ student    │ -2.16432  │ -1.33493 │ -0.911954 │ -0.502552 │ 0.33238
7 │
````




Since we ran multiple chains, we may as well do a spot check to make sure each chain converges around similar points.

````julia
plot(chain)
````


![](/tutorials/figures/2_LogisticRegression_8_1.png)


Looks good!

We can also use the `corner` function from MCMCChains to show the distributions of the various parameters of our logistic regression. 

````julia
# The labels to use.
l = [:student, :balance, :income]

# Use the corner function. Requires StatsPlots and MCMCChains.
corner(chain, l)
````


![](/tutorials/figures/2_LogisticRegression_9_1.png)


Fortunately the corner plot appears to demonstrate unimodal distributions for each of our parameters, so it should be straightforward to take the means of each parameter's sampled values to estimate our model to make predictions.

## Making Predictions
How do we test how well the model actually predicts whether someone is likely to default? We need to build a prediction function that takes the `test` object we made earlier and runs it through the average parameter calculated during sampling.

The `prediction` function below takes a `Matrix` and a `Chain` object. It takes the mean of each parameter's sampled values and re-runs the logistic function using those mean values for every element in the test set.

````julia
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
````




Let's see how we did! We run the test matrix through the prediction function, and compute the [mean squared error](https://en.wikipedia.org/wiki/Mean_squared_error) (MSE) for our prediction. The `threshold` variable sets the sensitivity of the predictions. For example, a threshold of 0.10 will predict a defualt value of 1 for any predicted value greater than 1.0 and no default if it is less than 0.10.

````julia
# Set the prediction threshold.
threshold = 0.10

# Make the predictions.
predictions = prediction(test, chain, threshold)

# Calculate MSE for our test set.
loss = sum((predictions - test_label).^2) / length(test_label)
````


````
0.09231578947368421
````




Perhaps more important is to see what percentage of defaults we correctly predicted. The code below simply counts defaults and predictions and presents the results. 

````julia
defaults = sum(test_label)
not_defaults = length(test_label) - defaults

predicted_defaults = sum(test_label .== predictions .== 1)
predicted_not_defaults = sum(test_label .== predictions .== 0)

println("Defaults: $$defaults
    Predictions: $$predicted_defaults
    Percentage defaults correct $$(predicted_defaults/defaults)")
````


````
Defaults: 317.0
    Predictions: 258
    Percentage defaults correct 0.8138801261829653
````



````julia

println("Not defaults: $$not_defaults
    Predictions: $$predicted_not_defaults
    Percentage non-defaults correct $$(predicted_not_defaults/not_defaults)")
````


````
Not defaults: 9183.0
    Predictions: 8365
    Percentage non-defaults correct 0.9109223565283676
````




The above shows that with a threshold of 0.10, we correctly predict a respectable portion of the defaults, and correctly identify most non-defaults. This is fairly sensitive to a choice of threshold, and you may wish to experiment with it.

This tutorial has demonstrated how to use Turing to perform Bayesian logistic regression. 
