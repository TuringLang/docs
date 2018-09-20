# Import RDatasets.
using RDatasets
using Distributions
using Turing
using MCMCChain
using StatsFuns: logistic
using MLDataUtils

# Import the "Default" dataset.
data = RDatasets.dataset("ISLR", "Default");

# Show the first six rows of the dataset.
head(data)

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

# Split our dataset 70/30 into training/test sets.
train, test = MLDataUtils.splitobs(data, at = 0.01);

# Create our labels. These are the values we are trying to predict.
train_label = train[:DefaultNum]
test_label = test[:DefaultNum]

# Remove the columns that are not our predictors.
train = train[[:StudentNum, :Balance, :Income]]
test = test[[:StudentNum, :Balance, :Income]]

# Convert the DataFrame objects to matrices.
train = Matrix(train);
test = Matrix(test);

# Bayesian logistic regression (LR)
@model lr_nuts(x, y, d, n, σ²) = begin
    α ~ Normal(0, σ²)
    β ~ MvNormal(zeros(d), σ² * ones(d))
    for i = 1:n
        v = logistic(α + transpose(x[i,:]) * β)
        y[i] ~ Bernoulli(v)
    end
end

train = (train .- mean(train, dims=1)) ./ std(train, dims=1)

n, d = size(train)
chain = sample(lr_nuts(train, train_label, d, n, 1), NUTS(200, 1.5))
describe(chain)
