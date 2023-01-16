# Machine Learning Notes
##### Following: Introduction to Machine Learning with Python By Andreas C. Müller & Sarah Guido

### Chapter 1: Introduction
- There are two kinds of machine learning algorithms. Supervised learning which has pairs of inputs and desired outputs and unsupervised algorithms which only has input data.
- Data used to build the machine learning model is the training data/training set.
- Data used to assess how well a model works is the test data/test set/hold-out set.
- Inspect data to find abnormalities such as inconsistent measurements. These inconsistencies are very common in the real world. You can identify these peculiarities by using visualizer like a scatter plot.

### Chapter 2: Supervised Learning**
- One of the two types of supervised machine learning problems is classification. Classification can be split into binary classification which distinguishes between two classes or multiclass classification which tries to classify between multiple different classes.
- The other type of supervised machine learning problems is regression. Regression types try to predict a continuous number or a floating-point number.
- Overfitting occurs when you have a model that follows the training data too closely and is therefore very accurate retaining to the training data but is unable to generalize to new data.
- Underfitting is when the model is too simple and is not able to capture all the aspects and variability of the data. In this case, the model will do bad even on the training set.
- If a model is able to make good predictions on unseen data, it is able to generalize from the training set to the test set. There is a sweet spot that will yield the best generalization performance.
- Larger datasets allow for more complex models. However, duplicating data points or collecting similar data will not help.

* **k-Nearest Neighbors**: Simplest machine learning algorithm. To make a prediction for a new data point, the algorithm finds the closest data points in the training dataset.
    - Works for binary or multiclass classifications. Works with classifications and regressions.
    - In its simplest form, the algorithm just finds a singular closest neighbor. However, the 'k' can be adjusted to fit what is most appropriate.
    - Strengths: Easy to understand, gives reasonable performance without many adjustments, good baseline method to try before using more advanced techniques, fast to build.
    - Weaknesses: Gets much slower with larger datasets, does not perform well with datasets with many features, does bad on datasets where most features are 0 most of the time.
    
### Chapter 2: Linear Models**
- Makes a prediction using a linear function of the input features.

*Linear Models For Regression: ŷ = w[0] * x[0] + w[1] * x[1] + ... + w[p] * x[p] + b*
    - Works well when there are more features than training data points.
    - Linear regression, 
* **Linear Regression (Ordinary Least Squares)**: Simplest and most classic linear method for regression. Finds *w* and *b* that minimize the mean squared error between predictions and the true regression targets *y* on the training set.
    - Has no parameters which is a benefit but also has no way to control model complexity.
* **Ridge Regression**: The formula used is similar to ordinary least squares except the coefficients are chosen not only so that they predict well but they also fit an additional constraint.
    - Magnitudes of coefficients should be as small as possible (entries of w should be as close to 0 as possible) meaning each feature has a minimal effect on the outcome while still predicting well. This constraint is called regularization which means to explicitly restrict a model while avoiding overfitting.
    - Ridge regression specifically uses L2 regularization.
    - The ridge model makes a trade-off between the simplicity of the model and its performance on the training set. This trade-off is specified by the alpha parameter. Increasing alpha moves the coefficients closer to zero which *generally* decreases set performance but helps generalization.
    - With enough data, regularization becomes less important and eventually, ridge and linear regression will have the same performance.