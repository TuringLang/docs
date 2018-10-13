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
````


````
loaded
````



````julia
# Import MCMCChain, Plots, and StatPlots for visualizations and diagnostics.
using MCMCChain, Plots, StatPlots

# We need a logistic function, which is provided by StatsFuns.
using StatsFuns: logistic

# MLDataUtils provides a sample splitting tool that's very handy.
using MLDataUtils

# Set a seed for reproducibility.
using Random
Random.seed!(0);
````




## Data Cleaning & Set Up

Now we're going to import our dataset. The first six rows of the dataset are shown below so you capn get a good feel for what kind of data we have.

````julia
# Import the "Default" dataset.
data = RDatasets.dataset("ISLR", "Default");

# Show the first six rows of the dataset.
head(data)
````



<table class="data-frame"><thead><tr><th></th><th>Default</th><th>Student</th><th>Balance</th><th>Income</th></tr><tr><th></th><th>Categorical…</th><th>Categorical…</th><th>Float64</th><th>Float64</th></tr></thead><tbody><tr><th>1</th><td>No</td><td>No</td><td>729.526</td><td>44361.6</td></tr><tr><th>2</th><td>No</td><td>Yes</td><td>817.18</td><td>12106.1</td></tr><tr><th>3</th><td>No</td><td>No</td><td>1073.55</td><td>31767.1</td></tr><tr><th>4</th><td>No</td><td>No</td><td>529.251</td><td>35704.5</td></tr><tr><th>5</th><td>No</td><td>No</td><td>785.656</td><td>38463.5</td></tr><tr><th>6</th><td>No</td><td>Yes</td><td>919.589</td><td>7491.56</td></tr></tbody></table>


Most machine learning processes require some effort to tidy up the data, and this is no different. We need to convert the `Default` and `Student` columns, which say "Yes" or "No" into 1s and 0s. Afterwards, we'll get rid of the old words-based columns.

````julia
# Create new rows, defualted to zero.
data[:DefaultNum] = 0.0
data[:StudentNum] = 0.0

for i in 1:length(data.Default)
    # If a row's "Default" or "Student" columns say "Yes",
    # set them to 1 in our new columns.
    data[:DefaultNum][i] = data.Default[i] == "Yes" ? 1.0 : 0.0
    data[:StudentNum][i] = data.Student[i] == "Yes" ? 1.0 : 0.0
end

# Delete the old columns which say "Yes" and "No".
delete!(data, :Default)
delete!(data, :Student)

# Show the first six rows of our edited dataset.
head(data)
````



<table class="data-frame"><thead><tr><th></th><th>Balance</th><th>Income</th><th>DefaultNum</th><th>StudentNum</th></tr><tr><th></th><th>Float64</th><th>Float64</th><th>Float64</th><th>Float64</th></tr></thead><tbody><tr><th>1</th><td>729.526</td><td>44361.6</td><td>0.0</td><td>0.0</td></tr><tr><th>2</th><td>817.18</td><td>12106.1</td><td>0.0</td><td>1.0</td></tr><tr><th>3</th><td>1073.55</td><td>31767.1</td><td>0.0</td><td>0.0</td></tr><tr><th>4</th><td>529.251</td><td>35704.5</td><td>0.0</td><td>0.0</td></tr><tr><th>5</th><td>785.656</td><td>38463.5</td><td>0.0</td><td>0.0</td></tr><tr><th>6</th><td>919.589</td><td>7491.56</td><td>0.0</td><td>1.0</td></tr></tbody></table>


After we've done that tidying, it's time to split our dataset into training and testing sets, and separate the labels from the data. We use `MLDataUtils.splitobs` to separate our data into two halves, `train` and `test`. You can use a higher percentage of splitting (or a lower one) by modifying the `at = 0.05` argument. We have highlighted the use of only a 5% sample to show the power of Bayesian inference with small smaple sizes.

````julia
# Split our dataset 5/95 into training/test sets.
train, test = MLDataUtils.splitobs(data, at = 0.05);

# Create our labels. These are the values we are trying to predict.
train_label = train[:DefaultNum]
test_label = test[:DefaultNum]

# Remove the columns that are not our predictors.
train = train[[:StudentNum, :Balance, :Income]]
test = test[[:StudentNum, :Balance, :Income]]
````




Our `train` and `test` matrices are still in the `DataFrame` format, which tends not to play too well with the kind of manipulations we're about to do, so we convert them into `Matrix` objects.

````julia
# Convert the DataFrame objects to matrices.
train = Matrix(train);
test = Matrix(test);
````




This next part is critically important. We must rescale our variables so that they are centered around zero by subtracting each column by the mean and dividing it by the standard deviation. Without this step, Turing's sampler will have a hard time finding a place to start searching for parameter estimates.

````julia
# Rescale our matrices.
train = (train .- mean(train, dims=1)) ./ std(train, dims=1)
test = (test .- mean(test, dims=1)) ./ std(test, dims=1)
````




## Model Declaration 
Finally, we can define our model.

`logistic regression` takes four arguments:

- `x` is our set of independent variables;
- `y` is the element we want to predict;
- `n` is the number of observations we have; and
- `σ²` is the standard deviation we want to assume for our priors.

Within the model, we create four coefficients (`intercept`, `student`, `balance`, and `income`) and assign a prior of normally distributed with means of zero and standard deviations of `σ²`. We want to find values of these four coefficients to predict any given `y`.

The `for` block creates a variable `v` which is the logistic function. We then observe the liklihood of calculating `v` given the actual label, `y[i]`.

````julia
# Bayesian logistic regression (LR)
@model logistic_regression(x, y, n, σ²) = begin
    intercept ~ Normal(0, σ²)

    student ~ Normal(0, σ²)
    balance ~ Normal(0, σ²)
    income  ~ Normal(0, σ²)

    for i = 1:n
        v = logistic(intercept + student*x[i, 1] + balance*x[i,2] + income*x[i,3])
        y[i] ~ Bernoulli(v)
    end
end;
````




## Sampling

Now we can run our sampler. This time we'll use [`HMC`](http://turing.ml/docs/library/#Turing.HMC) to sample from our posterior.

````julia
# This is temporary while the reverse differentiation backend is being improved.
Turing.setadbackend(:forward_diff)

# Retrieve the number of observations.
n, _ = size(train)

# Sample using HMC.
chain = sample(logistic_regression(train, train_label, n, 1), HMC(1500, 0.05, 10))
````


````
[HMC] Finished with
  Running time        = 34.26214442800002;
  Accept rate         = 0.9946666666666667;
  #lf / sample        = 9.993333333333334;
  #evals / sample     = 11.992666666666667;
  pre-cond. diag mat  = [1.0, 1.0, 1.0, 1.0].
````



````julia

describe(chain)
````


````
Iterations = 1:1500
Thinning interval = 1
Chains = 1
Samples per chain = 1500

Empirical Posterior Estimates:
               Mean                  SD                       Naive SE     
                 MCSE                ESS   
   income  -0.033933583  0.380088470602344796756000 0.009813842111522192226
9578 0.0184895930515621559342421  422.58559
  student  -0.284304750  0.387281658118278471203411 0.009999569414557913857
3110 0.0223005213544469442499274  301.59478
   lf_num   9.993333333  0.258198889747160931218417 0.006666666666666661023
0330 0.0066666666666666471452452 1500.00000
intercept  -4.386463650  0.569482001314424723936725 0.014703962047037581403
7522 0.0238168855773724236213340  571.72879
  elapsed   0.022833752  0.087468726769515933727739 0.002258432813948680478
7328 0.0024252788748974000825054 1300.71528
  balance   1.693144952  0.298957918849792336768445 0.007719060272813826895
0886 0.0120984278925268529114589  610.60765
       lp -60.867338542 31.380006637554323845051840 0.810228287407507297146
4806 1.1525762860633543827049152  741.25341
   lf_eps   0.050000000  0.000000000000000048588456 0.000000000000000001254
5485 0.0000000000000000018544974  686.45764

Quantiles:
              2.5%         25.0%         50.0%         75.0%         97.5% 
   
   income  -0.77093056  -0.296231518  -0.037215981   0.215403802   0.679450
856
  student  -0.99822332  -0.518826075  -0.281800004  -0.051259684   0.445303
065
   lf_num  10.00000000  10.000000000  10.000000000  10.000000000  10.000000
000
intercept  -5.20578527  -4.630727198  -4.357935735  -4.105080753  -3.606788
908
  elapsed   0.01706563   0.017446126   0.018251866   0.022908220   0.031010
357
  balance   1.16023654   1.503315029   1.684428890   1.869271334   2.280035
461
       lp -63.62840058 -60.546210910 -59.387116198 -58.629031792 -57.868429
724
   lf_eps   0.05000000   0.050000000   0.050000000   0.050000000   0.050000
000
````




We can use the `cornerplot` function from StatPlots to show the distributions of the various parameters of our logistic regression. 

````julia
# The labels to use.
l = [:student, :balance, :income]

# Extract the parameters we want to plot.
w1 = chain[:student]
w2 = chain[:balance]
w3 = chain[:income]

# Show the corner plot.
cornerplot(hcat(w1, w2, w3), compact=true, labels = l)
````


![](/tutorials/figures/2_LogisticRegression_9_1.svg)


Fortunately the corner plot appears to demonstrate unimodal distributions for each of our parameters, so it should be straightforward to take the means of each parameter's sampled values to estimate our model to make predictions.

## Making Predictions
How do we test how well the model actually predicts whether someone is likely to default? We need to build a prediction function that takes the `test` object we made earlier and runs it through the average parameter calculated during sampling.

The `prediction` function below takes a `Matrix` and a `Chain` object. It takes the mean of each parameter's sampled values and re-runs the logistic function using those mean values for every element in the test set.

````julia
function prediction(x::Matrix, chain, threshold)
    # Pull the means from each parameter's sampled values in the chain.
    intercept = mean(chain[:intercept])
    student = mean(chain[:student])
    balance = mean(chain[:balance])
    income = mean(chain[:income])

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
    Predictions: 247
    Percentage defaults correct 0.7791798107255521
````



````julia

println("Not defaults: $$not_defaults
    Predictions: $$predicted_not_defaults
    Percentage non-defaults correct $$(predicted_not_defaults/not_defaults)")
````


````
Not defaults: 9183.0
    Predictions: 8467
    Percentage non-defaults correct 0.9220298377436568
````




The above shows that with a threshold of 0.10, we correctly predict a respectable portion of the defaults, and correctly identify most non-defaults. This is fairly sensitive to a choice of threshold, and you may wish to experiment with it.

This tutorial has demonstrated how to use Turing to perform Bayesian logistic regression. 
