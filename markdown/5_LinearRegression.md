---
title: Linear Regression
permalink: /:collection/:name/
---


Turing is powerful when applied to complex hierarchical models, but it can also be put to task at common statistical procedures, like [linear regression](https://en.wikipedia.org/wiki/Linear_regression). This tutorial covers how to implement a linear regression model in Turing.

## Set Up

We begin by importing all the necessary libraries.

````julia
# Import Turing and Distributions.
using Turing, Distributions

# Import RDatasets.
using RDatasets

# Import MCMCChain, Plots, and StatPlots for visualizations and diagnostics.
using MCMCChain, Plots, StatsPlots

# Set a seed for reproducibility.
using Random
Random.seed!(0);

# Hide the progress prompt while sampling.
Turing.turnprogress(false);
````




We will use the `mtcars` dataset from the [RDatasets](https://github.com/johnmyleswhite/RDatasets.jl) package. `mtcars` contains a variety of statistics on different car models, including their miles per gallon, number of cylinders, and horsepower, among others.

We want to know if we can construct a Bayesian linear regression model to predict the miles per gallon of a car, given the other statistics it has. Lets take a look at the data we have.

````julia
# Import the "Default" dataset.
data = RDatasets.dataset("datasets", "mtcars");

# Show the first six rows of the dataset.
first(data, 6)
````



<table class="data-frame"><thead><tr><th></th><th>Model</th><th>MPG</th><th>Cyl</th><th>Disp</th><th>HP</th><th>DRat</th><th>WT</th><th>QSec</th><th>VS</th><th>AM</th><th>Gear</th><th>Carb</th></tr><tr><th></th><th>String⍰</th><th>Float64⍰</th><th>Int64⍰</th><th>Float64⍰</th><th>Int64⍰</th><th>Float64⍰</th><th>Float64⍰</th><th>Float64⍰</th><th>Int64⍰</th><th>Int64⍰</th><th>Int64⍰</th><th>Int64⍰</th></tr></thead><tbody><p>6 rows × 12 columns</p><tr><th>1</th><td>Mazda RX4</td><td>21.0</td><td>6</td><td>160.0</td><td>110</td><td>3.9</td><td>2.62</td><td>16.46</td><td>0</td><td>1</td><td>4</td><td>4</td></tr><tr><th>2</th><td>Mazda RX4 Wag</td><td>21.0</td><td>6</td><td>160.0</td><td>110</td><td>3.9</td><td>2.875</td><td>17.02</td><td>0</td><td>1</td><td>4</td><td>4</td></tr><tr><th>3</th><td>Datsun 710</td><td>22.8</td><td>4</td><td>108.0</td><td>93</td><td>3.85</td><td>2.32</td><td>18.61</td><td>1</td><td>1</td><td>4</td><td>1</td></tr><tr><th>4</th><td>Hornet 4 Drive</td><td>21.4</td><td>6</td><td>258.0</td><td>110</td><td>3.08</td><td>3.215</td><td>19.44</td><td>1</td><td>0</td><td>3</td><td>1</td></tr><tr><th>5</th><td>Hornet Sportabout</td><td>18.7</td><td>8</td><td>360.0</td><td>175</td><td>3.15</td><td>3.44</td><td>17.02</td><td>0</td><td>0</td><td>3</td><td>2</td></tr><tr><th>6</th><td>Valiant</td><td>18.1</td><td>6</td><td>225.0</td><td>105</td><td>2.76</td><td>3.46</td><td>20.22</td><td>1</td><td>0</td><td>3</td><td>1</td></tr></tbody></table>

````julia
size(data)
````


````
(32, 12)
````




The next step is to get our data ready for testing. We'll split the `mtcars` dataset into two subsets, one for training our model and one for evaluating our model. Then, we separate the labels we want to learn (`MPG`, in this case) and standardize the datasets by subtracting each column's means and dividing by the standard deviation of that column.

The resulting data is not very familiar looking, but this standardization process helps the sampler converge far easier. We also create a function called `unstandardize`, which returns the standardized values to their original form. We will use this function later on when we make predictions.

````julia
# Function to split samples.
function split_data(df, at = 0.70)
    (r, _) = size(df)
    index = Int(round(r * at))
    train = df[1:index, :]
    test  = df[(index+1):end, :]
    return train, test
end

# Split our dataset 70%/30% into training/test sets.
train, test = split_data(data, 0.7)

# Save dataframe versions of our dataset.
train_cut = DataFrame(train)
test_cut = DataFrame(test)

# Create our labels. These are the values we are trying to predict.
train_label = train[:, :MPG]
test_label = test[:, :MPG]

# Get the list of columns to keep.
remove_names = filter(x->!in(x, [:MPG, :Model]), names(data))

# Filter the test and train sets.
train = Matrix(train[:,remove_names]);
test = Matrix(test[:,remove_names]);

# A handy helper function to rescale our dataset.
function standardize(x)
    return (x .- mean(x, dims=1)) ./ std(x, dims=1), x
end

# Another helper function to unstandardize our datasets.
function unstandardize(x, orig)
    return x .* std(orig, dims=1) .+ mean(orig, dims=1)
end

# Standardize our dataset.
(train, train_orig) = standardize(train)
(test, test_orig) = standardize(test)
(train_label, train_l_orig) = standardize(train_label)
(test_label, test_l_orig) = standardize(test_label);
````




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

For $$\sigma^2$$, we assign a prior of `TruncatedNormal(0,100,0,Inf)`. This is consistent with [Andrew Gelman's recommendations](http://www.stat.columbia.edu/~gelman/research/published/taumain.pdf) on noninformative priors for variance. The intercept term ($$\alpha$$) is assumed to be normally distributed with a mean of zero and a variance of three. This represents our assumptions that miles per gallon can be explained mostly by our assorted variables, but a high variance term indicates our uncertainty about that. Each coefficient is assumed to be normally distributed with a mean of zero and a variance of 10. We do not know that our coefficients are different from zero, and we don't know which ones are likely to be the most important, so the variance term is quite high. Lastly, each observation $$y_i$$ is distributed according to the calculated `mu` term given by $$\alpha + \boldsymbol{\beta}^T\boldsymbol{X_i}$$.

````julia
# Bayesian linear regression.
@model linear_regression(x, y, n_obs, n_vars) = begin
    # Set variance prior.
    σ₂ ~ TruncatedNormal(0,100, 0, Inf)
    
    # Set intercept prior.
    intercept ~ Normal(0, 3)
    
    # Set the priors on our coefficients.
    coefficients = Array{Real}(undef, n_vars)
    coefficients ~ [Normal(0, 10)]
    
    # Calculate all the mu terms.
    mu = intercept .+ x * coefficients
    for i = 1:n_obs
        y[i] ~ Normal(mu[i], σ₂)
    end
end;
````




With our model specified, we can call the sampler. We will use the No U-Turn Sampler ([NUTS](http://turing.ml/docs/library/#-turingnuts--type)) here. 

````julia
n_obs, n_vars = size(train)
model = linear_regression(train, train_label, n_obs, n_vars)
chain = sample(model, NUTS(1500, 200, 0.65));
````


````
[NUTS] Finished with
  Running time        = 52.33233410599997;
  #lf / sample        = 0.0;
  #evals / sample     = 96.34133333333334;
  pre-cond. metric    = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0,....
````




As a visual check to confirm that our coefficients have converged, we show the densities and trace plots for our parameters using the `plot` functionality.

````julia
plot(chain)
````


![](/tutorials/figures/5_LinearRegression_7_1.svg)


It looks like each of our parameters has converged. We can check our numerical esimates using `describe(chain)`, as below.

````julia
describe(chain)
````


````
Iterations = 1:1500
Thinning interval = 1
Chains = 1
Samples per chain = 1500

Empirical Posterior Estimates:
                      Mean          SD        Naive SE       MCSE         E
SS   
          lf_num   0.000000000  0.000000000 0.0000000000 0.0000000000      
  NaN
 coefficients[7]  -0.069379087  0.297111820 0.0076713942 0.0062666679 1500.
00000
 coefficients[9]   0.183295384  0.346104116 0.0089363699 0.0121530202  811.
04526
              σ₂   0.469602793  0.193836072 0.0050048259 0.0163021444  1
41.37731
 coefficients[8]   0.126314007  0.272189289 0.0070278972 0.0043142310 1500.
00000
 coefficients[4]   0.614135907  0.344119004 0.0088851145 0.0120624748  813.
85021
 coefficients[1]   0.382941674  0.457696592 0.0118176752 0.0156500086  855.
31521
       intercept   0.008082493  0.151973153 0.0039239299 0.0066216748  526.
74212
         elapsed   0.034888223  0.042308688 0.0010924056 0.0011282595 1406.
18056
         epsilon   0.047158347  0.038086234 0.0009833823 0.0022651873  282.
70133
        eval_num  96.341333333 50.032726030 1.2918394312 1.0528120470 1500.
00000
 coefficients[2]  -0.088609326  0.493283172 0.0127365167 0.0205509990  576.
13827
coefficients[10]  -0.604518361  0.360273693 0.0093022267 0.0144890516  618.
27998
 coefficients[3]  -0.104194057  0.346855319 0.0089557658 0.0104515672 1101.
37161
 coefficients[6]   0.081634971  0.270333156 0.0069799721 0.0086099287  985.
82430
              lp -53.022593551  4.907915831 0.1267218419 0.4523216235  117.
73335
 coefficients[5]  -0.008293860  0.482175193 0.0124497100 0.0227674972  448.
51724
          lf_eps   0.047158347  0.038086234 0.0009833823 0.0022651873  282.
70133

Quantiles:
                      2.5%          25.0%          50.0%         75.0%     
    97.5%    
          lf_num   0.0000000000   0.000000000   0.0000000000   0.000000000 
  0.000000000
 coefficients[7]  -0.6548774776  -0.256435392  -0.0719427702   0.114995816 
  0.506433692
 coefficients[9]  -0.5279036788  -0.008055657   0.1858021665   0.381595355 
  0.828848599
              σ₂   0.2955408557   0.376066933   0.4384306420   0.5122987
26   0.804123318
 coefficients[8]  -0.3391120463  -0.043125504   0.1063523810   0.279790838 
  0.672932982
 coefficients[4]  -0.0721507597   0.415220069   0.6144661282   0.821120009 
  1.281506693
 coefficients[1]  -0.5443022848   0.111974485   0.3776405387   0.671266258 
  1.293037338
       intercept  -0.2049437856  -0.060838585   0.0038548309   0.069651795 
  0.199756525
         elapsed   0.0053328645   0.023527849   0.0350438500   0.037787749 
  0.072904877
         epsilon   0.0216297253   0.043869934   0.0438699344   0.043869934 
  0.089893126
        eval_num  16.0000000000  46.000000000  94.0000000000  94.000000000 
190.000000000
 coefficients[2]  -1.0837163073  -0.384377079  -0.0823865151   0.201347814 
  0.840527589
coefficients[10]  -1.2762537187  -0.819783906  -0.6115876281  -0.405837555 
  0.141566194
 coefficients[3]  -0.8056741812  -0.305848312  -0.1035610230   0.108768514 
  0.566118540
 coefficients[6]  -0.4815832882  -0.077092069   0.0724022284   0.244972212 
  0.645677684
              lp -62.5878136647 -55.200763798 -52.3556673479 -49.992794821 
-46.801290190
 coefficients[5]  -0.9656042060  -0.296266573  -0.0020491374   0.279551290 
  0.925626893
          lf_eps   0.0216297253   0.043869934   0.0438699344   0.043869934 
  0.089893126
````




## Comparing to OLS

A satisfactory test of our model is to evaluate how well it predicts. Importantly, we want to compare our model to existing tools like OLS. The code below uses the [GLM.jl]() package to generate a traditional OLS multivariate regression on the same data as our probabalistic model.

````julia
# Import the GLM package.
using GLM

# Perform multivariate OLS.
ols = lm(@formula(MPG ~ Cyl + Disp + HP + DRat + WT + QSec + VS + AM + Gear + Carb), train_cut)

# Store our predictions in the original dataframe.
train_cut.OLSPrediction = GLM.predict(ols);
test_cut.OLSPrediction = GLM.predict(ols, test_cut);
````




The function below accepts a chain and an input matrix and calculates predictions. We use the mean observation of each parameter in the model starting with sample 200, which is where the warm-up period for the NUTS sampler ended.

````julia
# Make a prediction given an input vector.
function prediction(chain, x)
    α = chain[:intercept][200:end]
    β = chain[:coefficients][200:end]
    return  mean(α) .+ x * mean(β)
end
````


````
prediction (generic function with 2 methods)
````




When we make predictions, we unstandardize them so they're more understandable. We also add them to the original dataframes so they can be placed in context.

````julia
# Calculate the predictions for the training and testing sets.
train_cut.BayesPredictions = unstandardize(prediction(chain, train), train_l_orig);
test_cut.BayesPredictions = unstandardize(prediction(chain, test), test_l_orig);

# Show the first side rows of the modified dataframe.
first(test_cut, 6)
````



<table class="data-frame"><thead><tr><th></th><th>Model</th><th>MPG</th><th>Cyl</th><th>Disp</th><th>HP</th><th>DRat</th><th>WT</th><th>QSec</th><th>VS</th><th>AM</th><th>Gear</th><th>Carb</th><th>OLSPrediction</th><th>BayesPredictions</th></tr><tr><th></th><th>String⍰</th><th>Float64⍰</th><th>Int64⍰</th><th>Float64⍰</th><th>Int64⍰</th><th>Float64⍰</th><th>Float64⍰</th><th>Float64⍰</th><th>Int64⍰</th><th>Int64⍰</th><th>Int64⍰</th><th>Int64⍰</th><th>Float64⍰</th><th>Float64</th></tr></thead><tbody><p>6 rows × 14 columns</p><tr><th>1</th><td>AMC Javelin</td><td>15.2</td><td>8</td><td>304.0</td><td>150</td><td>3.15</td><td>3.435</td><td>17.3</td><td>0</td><td>0</td><td>3</td><td>2</td><td>19.8583</td><td>17.2538</td></tr><tr><th>2</th><td>Camaro Z28</td><td>13.3</td><td>8</td><td>350.0</td><td>245</td><td>3.73</td><td>3.84</td><td>15.41</td><td>0</td><td>0</td><td>3</td><td>4</td><td>16.0462</td><td>17.2745</td></tr><tr><th>3</th><td>Pontiac Firebird</td><td>19.2</td><td>8</td><td>400.0</td><td>175</td><td>3.08</td><td>3.845</td><td>17.05</td><td>0</td><td>0</td><td>3</td><td>2</td><td>18.5746</td><td>16.0018</td></tr><tr><th>4</th><td>Fiat X1-9</td><td>27.3</td><td>4</td><td>79.0</td><td>66</td><td>4.08</td><td>1.935</td><td>18.9</td><td>1</td><td>1</td><td>4</td><td>1</td><td>29.3233</td><td>25.8621</td></tr><tr><th>5</th><td>Porsche 914-2</td><td>26.0</td><td>4</td><td>120.3</td><td>91</td><td>4.43</td><td>2.14</td><td>16.7</td><td>0</td><td>1</td><td>5</td><td>2</td><td>30.7731</td><td>27.9732</td></tr><tr><th>6</th><td>Lotus Europa</td><td>30.4</td><td>4</td><td>95.1</td><td>113</td><td>3.77</td><td>1.513</td><td>16.9</td><td>1</td><td>1</td><td>5</td><td>2</td><td>25.2892</td><td>21.8802</td></tr></tbody></table>


Now let's evaluate the loss for each method, and each prediction set. We will use sum of squared error function to evaluate loss, given by 

\$\$
\text{SSE} = \sum{(y_i - \hat{y_i})^2}
\$\$

where $$y_i$$ is the actual value (true MPG) and $$\hat{y_i}$$ is the predicted value using either OLS or Bayesian linear regression. A lower SSE indicates a closer fit to the data.

````julia
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
````


````
Training set:
    Bayes loss: 67.62926048530228
    OLS loss: 67.56037474764624
Test set: 
    Bayes loss: 213.08154799539776
    OLS loss: 270.94813070761944
````




As we can see above, OLS and our Bayesian model fit our training set about the same. This is to be expected, given that it is our training set. But when we look at our test set, we see that the Bayesian linear regression model is better able to predict out of sample.
