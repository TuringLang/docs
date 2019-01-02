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
````


````
loaded
````



````julia
# Import MCMCChain, Plots, and StatPlots for visualizations and diagnostics.
using MCMCChain, Plots, StatPlots

# MLDataUtils provides a sample splitting tool that's very handy.
using MLDataUtils

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


The next step is to get our data ready for testing. We'll split the `mtcars` dataset into two subsets, one for training our model and one for evaluating our model. Then, we separate the labels we want to learn (`MPG`, in this case) and standardize the datasets by subtracting each column's means and dividing by the standard deviation of that column.

The resulting data is not very familiar looking, but this standardization process helps the sampler converge far easier. We also create a function called `unstandardize`, which returns the standardized values to their original form. We will use this function later on when we make predictions.

````julia
# Split our dataset 70%/30% into training/test sets.
train, test = MLDataUtils.splitobs(data, at = 0.7);

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
  Running time        = 48.31455570699994;
  #lf / sample        = 0.0013333333333333333;
  #evals / sample     = 166.042;
  pre-cond. metric    = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0,....
````




As a visual check to confirm that our coefficients have converged, we show the densities and trace plots for our parameters using the `plot` functionality.

````julia
plot(chain)
````


![](/tutorials/figures/5_LinearRegression_6_1.svg)


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
                      Mean           SD        Naive SE       MCSE         
ESS   
          lf_num   0.0013333333  0.051639778 0.0013333333 0.0013333333 1500
.00000
 coefficients[7]  -0.0786491557  0.278772607 0.0071978778 0.0067186243 1500
.00000
 coefficients[9]   0.2071839815  0.330939626 0.0085448244 0.0203379836  264
.77792
              σ₂   0.4766792321  0.283946631 0.0073314705 0.0163353046  
302.14747
 coefficients[8]   0.1284803579  0.247850334 0.0063994681 0.0071310842 1208
.00267
 coefficients[4]   0.5986806799  0.318674014 0.0082281277 0.0103136941  954
.69535
 coefficients[1]   0.3622907141  0.476442777 0.0123016996 0.0194642548  599
.16430
       intercept   0.0022922010  0.120060757 0.0030999554 0.0020259848 1500
.00000
         elapsed   0.0322097038  0.084159835 0.0021729976 0.0027334832  947
.93147
         epsilon   0.0460736473  0.051153720 0.0013207834 0.0028309912  326
.49565
        eval_num 166.0420000000 86.146183970 2.2242849057 2.3910870610 1298
.01978
 coefficients[2]  -0.1300890184  0.469373808 0.0121191796 0.0257948068  331
.11065
 coefficients[3]  -0.0532708022  0.369601595 0.0095430722 0.0107190303 1188
.93107
coefficients[10]  -0.6312961836  0.341874420 0.0088271596 0.0153384749  496
.78544
 coefficients[6]   0.0758755978  0.259348063 0.0066963382 0.0089701335  835
.92671
              lp -52.9933052583  4.506305165 0.1163522990 0.2654208904  288
.25111
 coefficients[5]   0.0177791814  0.473669969 0.0122301060 0.0227001182  435
.40706
          lf_eps   0.0460736473  0.051153720 0.0013207834 0.0028309912  326
.49565

Quantiles:
                      2.5%           25.0%           50.0%         75.0%   
       97.5%     
          lf_num   0.0000000000   0.00000000000   0.0000000000   0.00000000
0   0.00000000000
 coefficients[7]  -0.6368049464  -0.24897767580  -0.0695249495   0.08054628
1   0.49442087463
 coefficients[9]  -0.4139831900  -0.00050359548   0.2083519823   0.39043004
9   0.84112760808
              σ₂   0.3039165184   0.38042539549   0.4424127727   0.52150
0540   0.76702056758
 coefficients[8]  -0.3432589148  -0.02928831152   0.1247398431   0.27760285
7   0.62992104755
 coefficients[4]  -0.0454653301   0.40644261040   0.5978327066   0.80116691
2   1.23207920460
 coefficients[1]  -0.5815445572   0.09128700827   0.3509835222   0.63157550
2   1.32299054666
       intercept  -0.2028319916  -0.06259757339   0.0010312832   0.06318343
7   0.19767845203
         elapsed   0.0052388088   0.01740961475   0.0279629675   0.03194519
5   0.05883527522
         epsilon   0.0226028431   0.04217697431   0.0421769743   0.04217697
4   0.09251541804
        eval_num  30.7500000000  76.00000000000 156.0000000000 156.00000000
0 316.00000000000
 coefficients[2]  -0.9776337072  -0.44424549292  -0.1427765579   0.16626936
3   0.82953564684
 coefficients[3]  -0.8059994591  -0.28453622277  -0.0480533671   0.17745567
0   0.68293607246
coefficients[10]  -1.2856992435  -0.82405835339  -0.6158924521  -0.42188041
5  -0.00023074154
 coefficients[6]  -0.4314061452  -0.09084599165   0.0754552914   0.24540950
5   0.57483678139
              lp -62.2494699405 -55.11189346096 -52.2884823257 -50.12194655
1 -47.08724817518
 coefficients[5]  -0.9421219004  -0.25822083928   0.0370508839   0.31716281
3   0.90484765049
          lf_eps   0.0226028431   0.04217697431   0.0421769743   0.04217697
4   0.09251541804
````




## Comparing to OLS

A satisfactory test of our model is to evaluate how well it predicts. Importantly, we want to compare our model to existing tools like OLS. The code below uses the [GLM.jl]() package to generate a traditional OLS multivariate regression on the same data as our probabalistic model.

````julia
# Import the GLM package.
using GLM

# Perform multivariate OLS.
ols = lm(@formula(MPG ~ Cyl + Disp + HP + DRat + WT + QSec + VS + AM + Gear + Carb), train_cut)

# Store our predictions in the original dataframe.
train_cut.OLSPrediction = predict(ols);
test_cut.OLSPrediction = predict(ols, test_cut);
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
prediction (generic function with 1 method)
````




When we make predictions, we unstandardize them so they're more understandable. We also add them to the original dataframes so they can be placed in context.

````julia
# Calculate the predictions for the training and testing sets.
train_cut.BayesPredictions = unstandardize(prediction(chain, train), train_l_orig);
test_cut.BayesPredictions = unstandardize(prediction(chain, test), test_l_orig);

# Show the first side rows of the modified dataframe.
first(test_cut, 6)
````



<table class="data-frame"><thead><tr><th></th><th>Model</th><th>MPG</th><th>Cyl</th><th>Disp</th><th>HP</th><th>DRat</th><th>WT</th><th>QSec</th><th>VS</th><th>AM</th><th>Gear</th><th>Carb</th><th>OLSPrediction</th><th>BayesPredictions</th></tr><tr><th></th><th>String⍰</th><th>Float64⍰</th><th>Int64⍰</th><th>Float64⍰</th><th>Int64⍰</th><th>Float64⍰</th><th>Float64⍰</th><th>Float64⍰</th><th>Int64⍰</th><th>Int64⍰</th><th>Int64⍰</th><th>Int64⍰</th><th>Float64⍰</th><th>Float64</th></tr></thead><tbody><p>6 rows × 14 columns</p><tr><th>1</th><td>AMC Javelin</td><td>15.2</td><td>8</td><td>304.0</td><td>150</td><td>3.15</td><td>3.435</td><td>17.3</td><td>0</td><td>0</td><td>3</td><td>2</td><td>19.8583</td><td>17.049</td></tr><tr><th>2</th><td>Camaro Z28</td><td>13.3</td><td>8</td><td>350.0</td><td>245</td><td>3.73</td><td>3.84</td><td>15.41</td><td>0</td><td>0</td><td>3</td><td>4</td><td>16.0462</td><td>17.1299</td></tr><tr><th>3</th><td>Pontiac Firebird</td><td>19.2</td><td>8</td><td>400.0</td><td>175</td><td>3.08</td><td>3.845</td><td>17.05</td><td>0</td><td>0</td><td>3</td><td>2</td><td>18.5746</td><td>15.7953</td></tr><tr><th>4</th><td>Fiat X1-9</td><td>27.3</td><td>4</td><td>79.0</td><td>66</td><td>4.08</td><td>1.935</td><td>18.9</td><td>1</td><td>1</td><td>4</td><td>1</td><td>29.3233</td><td>25.7076</td></tr><tr><th>5</th><td>Porsche 914-2</td><td>26.0</td><td>4</td><td>120.3</td><td>91</td><td>4.43</td><td>2.14</td><td>16.7</td><td>0</td><td>1</td><td>5</td><td>2</td><td>30.7731</td><td>28.0541</td></tr><tr><th>6</th><td>Lotus Europa</td><td>30.4</td><td>4</td><td>95.1</td><td>113</td><td>3.77</td><td>1.513</td><td>16.9</td><td>1</td><td>1</td><td>5</td><td>2</td><td>25.2892</td><td>21.9803</td></tr></tbody></table>


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

println("Training set:")
````


````
Training set:
````



````julia
println("  Bayes loss: $$bayes_loss1")
````


````
Bayes loss: 67.70847342511051
````



````julia
println("  OLS loss: $$ols_loss1")
````


````
OLS loss: 67.56037474764624
````



````julia

println("Test set:")
````


````
Test set:
````



````julia
println("  Bayes loss: $$bayes_loss2")
````


````
Bayes loss: 206.70849416171862
````



````julia
println("  OLS loss: $$ols_loss2")
````


````
OLS loss: 270.94813070761944
````




As we can see above, OLS and our Bayesian model fit our training set about the same. This is to be expected, given that it is our training set. But when we look at our test set, we see that the Bayesian linear regression model is better able to predict out of sample.
