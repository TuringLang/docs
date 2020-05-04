---
title: Bayesian Multinomial Logistic Regression
permalink: /:collection/:name/
---
[Multinomial logistic regression](https://en.wikipedia.org/wiki/Multinomial_logistic_regression) is an extension of logistic regression. Logistic regression is used to model problems in which there are exactly two possible discrete outcomes. Multinomial logistic regression is used to model problems in which there are two or more possible discrete outcomes.

In our example, we'll be using the iris dataset. The goal of the iris multiclass problem is to predict the species of a flower given measurements (in centimeters) of sepal length and width and petal length and width. There are three possible species: Iris setosa, Iris versicolor, and Iris virginica.

To start, let's import all the libraries we'll need.


```julia
# Import Turing and Distributions.
using Turing, Distributions

# Import RDatasets.
using RDatasets

# Import MCMCChains, Plots, and StatsPlots for visualizations and diagnostics.
using MCMCChains, Plots, StatsPlots

# We need a softmax function, which is provided by NNlin.
using NNlib: softmax

# Set a seed for reproducibility.
using Random
Random.seed!(0);
```

## Data Cleaning & Set Up

Now we're going to import our dataset. Twenty rows of the dataset are shown below so you can get a good feel for what kind of data we have.


```julia
# Import the "iris" dataset.
data = RDatasets.dataset("datasets", "iris");

# Randomly shuffle the rows of the dataset
num_rows = size(data, 1)
data = data[Random.shuffle(1:num_rows), :]

# Show twenty rows
first(data, 20)
```




<table class="data-frame"><thead><tr><th></th><th>SepalLength</th><th>SepalWidth</th><th>PetalLength</th><th>PetalWidth</th><th>Species</th></tr><tr><th></th><th>Float64</th><th>Float64</th><th>Float64</th><th>Float64</th><th>Categorical…</th></tr></thead><tbody><p>20 rows × 5 columns</p><tr><th>1</th><td>6.9</td><td>3.2</td><td>5.7</td><td>2.3</td><td>virginica</td></tr><tr><th>2</th><td>5.8</td><td>2.7</td><td>5.1</td><td>1.9</td><td>virginica</td></tr><tr><th>3</th><td>6.6</td><td>2.9</td><td>4.6</td><td>1.3</td><td>versicolor</td></tr><tr><th>4</th><td>6.3</td><td>2.5</td><td>5.0</td><td>1.9</td><td>virginica</td></tr><tr><th>5</th><td>5.0</td><td>2.0</td><td>3.5</td><td>1.0</td><td>versicolor</td></tr><tr><th>6</th><td>5.8</td><td>4.0</td><td>1.2</td><td>0.2</td><td>setosa</td></tr><tr><th>7</th><td>6.7</td><td>3.1</td><td>4.7</td><td>1.5</td><td>versicolor</td></tr><tr><th>8</th><td>5.7</td><td>2.8</td><td>4.5</td><td>1.3</td><td>versicolor</td></tr><tr><th>9</th><td>6.3</td><td>2.9</td><td>5.6</td><td>1.8</td><td>virginica</td></tr><tr><th>10</th><td>5.6</td><td>3.0</td><td>4.1</td><td>1.3</td><td>versicolor</td></tr><tr><th>11</th><td>5.6</td><td>2.7</td><td>4.2</td><td>1.3</td><td>versicolor</td></tr><tr><th>12</th><td>5.1</td><td>3.4</td><td>1.5</td><td>0.2</td><td>setosa</td></tr><tr><th>13</th><td>6.7</td><td>3.3</td><td>5.7</td><td>2.1</td><td>virginica</td></tr><tr><th>14</th><td>5.8</td><td>2.6</td><td>4.0</td><td>1.2</td><td>versicolor</td></tr><tr><th>15</th><td>6.4</td><td>2.9</td><td>4.3</td><td>1.3</td><td>versicolor</td></tr><tr><th>16</th><td>4.8</td><td>3.0</td><td>1.4</td><td>0.1</td><td>setosa</td></tr><tr><th>17</th><td>6.3</td><td>3.4</td><td>5.6</td><td>2.4</td><td>virginica</td></tr><tr><th>18</th><td>4.9</td><td>2.5</td><td>4.5</td><td>1.7</td><td>virginica</td></tr><tr><th>19</th><td>4.8</td><td>3.4</td><td>1.6</td><td>0.2</td><td>setosa</td></tr><tr><th>20</th><td>5.0</td><td>2.3</td><td>3.3</td><td>1.0</td><td>versicolor</td></tr></tbody></table>



In this data set, the outcome `Species` is currently coded as a string. We need to convert the `Species` into 1s and 0s.

We will create three new columns: `Species_setosa`, `Species_versicolor` and `Species_virginica`.

- If a row has `setosa` as the species, then it will have `Species_setosa = 1`, `Species_versicolor = 0`, and `Species_virginica = 0`.
- If a row has `versicolor` as the species, then it will have `Species_setosa = 0`, `Species_versicolor = 1`, and `Species_virginica = 0`.
- If a row has `virginica` as the species, then it will have `Species_setosa = 0`, `Species_versicolor = 0`, and `Species_virginica = 1`.


```julia
# Recode the `Species` column
data[!, :Species_setosa] = [r.Species == "setosa" ? 1.0 : 0.0 for r in eachrow(data)]
data[!, :Species_versicolor] = [r.Species == "versicolor" ? 1.0 : 0.0 for r in eachrow(data)]
data[!, :Species_virginica] = [r.Species == "virginica" ? 1.0 : 0.0 for r in eachrow(data)]

# Show twenty rows of the new species columns
first(data[!, [:Species, :Species_setosa, :Species_versicolor, :Species_virginica]], 20)
```




<table class="data-frame"><thead><tr><th></th><th>Species</th><th>Species_setosa</th><th>Species_versicolor</th><th>Species_virginica</th></tr><tr><th></th><th>Categorical…</th><th>Float64</th><th>Float64</th><th>Float64</th></tr></thead><tbody><p>20 rows × 4 columns</p><tr><th>1</th><td>virginica</td><td>0.0</td><td>0.0</td><td>1.0</td></tr><tr><th>2</th><td>virginica</td><td>0.0</td><td>0.0</td><td>1.0</td></tr><tr><th>3</th><td>versicolor</td><td>0.0</td><td>1.0</td><td>0.0</td></tr><tr><th>4</th><td>virginica</td><td>0.0</td><td>0.0</td><td>1.0</td></tr><tr><th>5</th><td>versicolor</td><td>0.0</td><td>1.0</td><td>0.0</td></tr><tr><th>6</th><td>setosa</td><td>1.0</td><td>0.0</td><td>0.0</td></tr><tr><th>7</th><td>versicolor</td><td>0.0</td><td>1.0</td><td>0.0</td></tr><tr><th>8</th><td>versicolor</td><td>0.0</td><td>1.0</td><td>0.0</td></tr><tr><th>9</th><td>virginica</td><td>0.0</td><td>0.0</td><td>1.0</td></tr><tr><th>10</th><td>versicolor</td><td>0.0</td><td>1.0</td><td>0.0</td></tr><tr><th>11</th><td>versicolor</td><td>0.0</td><td>1.0</td><td>0.0</td></tr><tr><th>12</th><td>setosa</td><td>1.0</td><td>0.0</td><td>0.0</td></tr><tr><th>13</th><td>virginica</td><td>0.0</td><td>0.0</td><td>1.0</td></tr><tr><th>14</th><td>versicolor</td><td>0.0</td><td>1.0</td><td>0.0</td></tr><tr><th>15</th><td>versicolor</td><td>0.0</td><td>1.0</td><td>0.0</td></tr><tr><th>16</th><td>setosa</td><td>1.0</td><td>0.0</td><td>0.0</td></tr><tr><th>17</th><td>virginica</td><td>0.0</td><td>0.0</td><td>1.0</td></tr><tr><th>18</th><td>virginica</td><td>0.0</td><td>0.0</td><td>1.0</td></tr><tr><th>19</th><td>setosa</td><td>1.0</td><td>0.0</td><td>0.0</td></tr><tr><th>20</th><td>versicolor</td><td>0.0</td><td>1.0</td><td>0.0</td></tr></tbody></table>



After we've done that tidying, it's time to split our dataset into training and testing sets, and separate the labels from the data. We separate our data into two halves, `train` and `test`.

We must rescale our feature variables so that they are centered around zero by subtracting each column by the mean and dividing it by the standard deviation. Without this step, Turing's sampler will have a hard time finding a place to start searching for parameter estimates.


```julia
# Function to split samples.
function split_data(df, at)
    (r, _) = size(df)
    index = Int(round(r * at))
    train = df[1:index, :]
    test  = df[(index+1):end, :]
    return train, test
end

# Rescale our feature variables.
data.SepalLength = (data.SepalLength .- mean(data.SepalLength)) ./ std(data.SepalLength)
data.SepalWidth = (data.SepalWidth .- mean(data.SepalWidth)) ./ std(data.SepalWidth)
data.PetalLength = (data.PetalLength .- mean(data.PetalLength)) ./ std(data.PetalLength)
data.PetalWidth = (data.PetalWidth .- mean(data.PetalWidth)) ./ std(data.PetalWidth)

# Split our dataset 50/50 into training/test sets.
train, test = split_data(data, 0.50);

label_names = [:Species_setosa, :Species_versicolor, :Species_virginica]
feature_names = [:SepalLength, :SepalWidth, :PetalLength, :PetalWidth]

# Create our labels. These are the values we are trying to predict.
train_labels = train[:, label_names]
test_labels = test[:, label_names]

# Create our features. These are our predictors.
train_features = train[:, feature_names];
test_features = test[:, feature_names];
```

Our `train` and `test` matrices are still in the `DataFrame` format, which tends not to play too well with the kind of manipulations we're about to do, so we convert them into `Matrix` objects.


```julia
# Convert the DataFrame objects to matrices.
train_labels = Matrix(train_labels);
test_labels = Matrix(test_labels);

train_features = Matrix(train_features);
test_features = Matrix(test_features);
```

## Model Declaration 
Finally, we can define our model.

`logistic_regression` takes four arguments:

- `x` is our set of independent variables;
- `y` is the element we want to predict;
- `n` is the number of observations we have; and
- `σ` is the standard deviation we want to assume for our priors.

We need to create our coefficients. To do so, we first need to select one of the species as the baseline species. The selection of the baseline class does not matter. Then we create our coefficients against that baseline.

Let us select `"setosa"` as the baseline. We create ten coefficients (`intercept_versicolor`, `intercept_virginica`, `SepalLength_versicolor`, `SepalLength_virginica`, `SepalWidth_versicolor`, `SepalWidth_virginica`, `PetalLength_versicolor`, `PetalLength_virginica`, `PetalWidth_versicolor`, and `PetalWidth_virginica`) and assign a prior of normally distributed with means of zero and standard deviations of `σ`. We want to find values of these ten coefficients to predict any given `y`.

The `for` block creates a variable `v` which is the softmax function. We then observe the liklihood of calculating `v` given the actual label, `y[i]`.


```julia
# Bayesian multinomial logistic regression
@model logistic_regression(x, y, n, σ) = begin
    intercept_versicolor ~ Normal(0, σ)
    intercept_virginica ~ Normal(0, σ)
    
    SepalLength_versicolor ~ Normal(0, σ)
    SepalLength_virginica ~ Normal(0, σ)
    
    SepalWidth_versicolor ~ Normal(0, σ)
    SepalWidth_virginica ~ Normal(0, σ)
    
    PetalLength_versicolor  ~ Normal(0, σ)
    PetalLength_virginica  ~ Normal(0, σ)
    
    PetalWidth_versicolor ~ Normal(0, σ)
    PetalWidth_virginica  ~ Normal(0, σ)


    for i = 1:n
        v = softmax([0, # this 0 corresponds to the base category `setosa`
                     intercept_versicolor + SepalLength_versicolor*x[i, 1] +
                                            SepalWidth_versicolor*x[i, 1] +
                                            PetalLength_versicolor*x[i, 2] +
                                            PetalWidth_versicolor*x[i, 2],
                     intercept_virginica + SepalLength_virginica*x[i, 3] +
                                           SepalWidth_virginica*x[i, 3] +
                                           PetalLength_virginica*x[i, 4] +
                                           PetalWidth_virginica*x[i, 4]])
        y[i, :] ~ Multinomial(1, v)
    end
end;
```

## Sampling

Now we can run our sampler. This time we'll use [`HMC`](http://turing.ml/docs/library/#Turing.HMC) to sample from our posterior.


```julia
# Retrieve the number of observations.
n, _ = size(train_features)

# Sample using HMC.
chain = mapreduce(c -> sample(logistic_regression(train_features, train_labels, n, 1), HMC(0.05, 10), 1500),
    chainscat,
    1:3
)

describe(chain)
```

    [32mSampling: 100%|█████████████████████████████████████████| Time: 0:00:05[39m
    [32mSampling: 100%|█████████████████████████████████████████| Time: 0:00:04[39m
    [32mSampling: 100%|█████████████████████████████████████████| Time: 0:00:04[39m





    2-element Array{ChainDataFrame,1}
    
    Summary Statistics
                  parameters     mean     std  naive_se    mcse        ess   r_hat
      ──────────────────────  ───────  ──────  ────────  ──────  ─────────  ──────
      PetalLength_versicolor  -0.7982  0.7341    0.0109  0.0353   330.7338  1.0022
       PetalLength_virginica   1.7376  0.8532    0.0127  0.0424   415.6667  1.0023
       PetalWidth_versicolor  -0.7018  0.7335    0.0109  0.0339   355.4789  1.0017
        PetalWidth_virginica   1.6843  0.8452    0.0126  0.0382   447.2635  1.0089
      SepalLength_versicolor   0.8642  0.7315    0.0109  0.0357   370.1195  1.0052
       SepalLength_virginica   1.5303  0.8641    0.0129  0.0452   321.9949  1.0078
       SepalWidth_versicolor   0.8227  0.7506    0.0112  0.0363   364.8514  1.0036
        SepalWidth_virginica   1.5765  0.8516    0.0127  0.0468   405.3356  1.0078
        intercept_versicolor   1.0275  0.4539    0.0068  0.0156  1004.6000  1.0029
         intercept_virginica  -0.9449  0.6155    0.0092  0.0246   700.5740  1.0033
    
    Quantiles
                  parameters     2.5%    25.0%    50.0%    75.0%   97.5%
      ──────────────────────  ───────  ───────  ───────  ───────  ──────
      PetalLength_versicolor  -2.2429  -1.2926  -0.7839  -0.2936  0.5790
       PetalLength_virginica   0.0566   1.1492   1.7495   2.3157  3.4072
       PetalWidth_versicolor  -2.1443  -1.2000  -0.7049  -0.2173  0.7632
        PetalWidth_virginica  -0.0565   1.1274   1.6940   2.2695  3.2582
      SepalLength_versicolor  -0.5667   0.3953   0.8762   1.3547  2.2408
       SepalLength_virginica  -0.1067   0.9405   1.5259   2.0997  3.2549
       SepalWidth_versicolor  -0.5795   0.3072   0.8086   1.3063  2.3417
        SepalWidth_virginica  -0.1364   1.0034   1.5684   2.1657  3.2340
        intercept_versicolor   0.1491   0.7267   1.0235   1.3287  1.9327
         intercept_virginica  -2.1602  -1.3516  -0.9434  -0.5233  0.2201




Since we ran multiple chains, we may as well do a spot check to make sure each chain converges around similar points.


```julia
plot(chain)
```




![svg](/tutorials/8_MultinomialLogisticRegression_files/8_MultinomialLogisticRegression_15_0.svg)



Looks good!

We can also use the `corner` function from MCMCChains to show the distributions of the various parameters of our multinomial logistic regression. The corner function requires MCMCChains and StatsPlots.


```julia
corner(chain, [:SepalLength_versicolor, :SepalWidth_versicolor, :PetalLength_versicolor, :PetalWidth_versicolor])
```




![svg](/tutorials/8_MultinomialLogisticRegression_files/8_MultinomialLogisticRegression_17_0.svg)




```julia
corner(chain, [:SepalLength_versicolor, :SepalWidth_versicolor, :PetalLength_versicolor, :PetalWidth_versicolor])
```




![svg](/tutorials/8_MultinomialLogisticRegression_files/8_MultinomialLogisticRegression_18_0.svg)



Fortunately the corner plots appear to demonstrate unimodal distributions for each of our parameters, so it should be straightforward to take the means of each parameter's sampled values to estimate our model to make predictions.

## Making Predictions
How do we test how well the model actually predicts whether someone is likely to default? We need to build a prediction function that takes the `test` object we made earlier and runs it through the average parameter calculated during sampling.

The `prediction` function below takes a `Matrix` and a `Chain` object. It takes the mean of each parameter's sampled values and re-runs the softmax function using those mean values for every element in the test set.


```julia
function prediction(x::Matrix, chain)
    # Pull the means from each parameter's sampled values in the chain.
    intercept_versicolor = mean(chain[:intercept_versicolor].value)
    intercept_virginica = mean(chain[:intercept_virginica].value)
    SepalLength_versicolor = mean(chain[:SepalLength_versicolor].value)
    SepalLength_virginica = mean(chain[:SepalLength_virginica].value)
    SepalWidth_versicolor = mean(chain[:SepalWidth_versicolor].value)
    SepalWidth_virginica = mean(chain[:SepalWidth_virginica].value)
    PetalLength_versicolor = mean(chain[:PetalLength_versicolor].value)
    PetalLength_virginica = mean(chain[:PetalLength_virginica].value)
    PetalWidth_versicolor = mean(chain[:PetalWidth_versicolor].value)
    PetalWidth_virginica = mean(chain[:PetalWidth_virginica].value)

    # Retrieve the number of rows.
    n, _ = size(x)

    # Generate a vector to store our predictions.
    v = Vector{String}(undef, n)

    # Calculate the softmax function for each element in the test set.
    for i in 1:n
        num = softmax([0, # this 0 corresponds to the base category `setosa`
                     intercept_versicolor + SepalLength_versicolor*x[i, 1] +
                                            SepalWidth_versicolor*x[i, 1] +
                                            PetalLength_versicolor*x[i, 2] +
                                            PetalWidth_versicolor*x[i, 2],
                     intercept_virginica + SepalLength_virginica*x[i, 3] +
                                           SepalWidth_virginica*x[i, 3] +
                                           PetalLength_virginica*x[i, 4] +
                                           PetalWidth_virginica*x[i, 4]])
        c = argmax(num) # we pick the class with the highest probability
        if c == 1
            v[i] = "setosa"
        elseif c == 2
            v[i] = "versicolor"
        else # c == 3
            @assert c == 3
            v[i] = "virginica"
        end
    end
    return v
end;
```

Let's see how we did! We run the test matrix through the prediction function, and compute the accuracy for our prediction.


```julia
# Make the predictions.
predictions = prediction(test_features, chain)

# Calculate accuracy for our test set.
mean(predictions .== test[!, :Species])
```




    0.8933333333333333



Perhaps more important is to see the accuracy per class.


```julia
setosa_rows = test[!, :Species] .== "setosa"
versicolor_rows = test[!, :Species] .== "versicolor"
virginica_rows = test[!, :Species] .== "virginica"

println("Number of setosa: $$(sum(setosa_rows))")
println("Number of versicolor: $$(sum(versicolor_rows))")
println("Number of virginica: $$(sum(virginica_rows))")

println("Percentage of setosa predicted correctly: $$(mean(predictions[setosa_rows] .== test[setosa_rows, :Species]))")
println("Percentage of versicolor predicted correctly: $$(mean(predictions[versicolor_rows] .== test[versicolor_rows, :Species]))")
println("Percentage of virginica predicted correctly: $$(mean(predictions[virginica_rows] .== test[virginica_rows, :Species]))")
```

    Number of setosa: 32
    Number of versicolor: 25
    Number of virginica: 18
    Percentage of setosa predicted correctly: 0.96875
    Percentage of versicolor predicted correctly: 0.76
    Percentage of virginica predicted correctly: 0.9444444444444444


This tutorial has demonstrated how to use Turing to perform Bayesian multinomial logistic regression. 
