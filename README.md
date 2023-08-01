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
* **Lasso**: An alternative to Ridge, Lasso also restricts coefficients to be close to 0 - this time using L1 regularization.
    - Sometimes coefficients end up exactly 0, which means features can be completely ignored. This can be seen as a positive as it makes a model easier to interpret and can emphasize the most important features.
    - Lasso is a better choice than Ridge when you have a large amount of features and you only expect a few to be relevant.
    - *ElasticNet* (from scikit-learn) combines the consequences of both Lasso and Ridge at the price of having to adjust for L1 and L2 regularization.

*Linear Models For Classification: ŷ = w[0] * x[0] + w[1] * x[1] + ... + w[p] * x[p] + b > 0*
    - Instead of returning the weighted sum of the features as for regression, classification checks if the value is greater or less than 0.
    - These models use a decision boundary to separate two classes using a line, plane, or hyperplane.
    - The two most common linear classification algorithms are *logistic regression* and *linear support vector machines*. Both use L2 regularization. The trade-off parameter that determines the strength of regularization is called C. The larger the C, the weaker the regularization.
    - Low values of C causes the algorithm to adjust to the majority while high values of C emphasizes the importance of each data point.
    - Many linear classification models are for binary classifications only and don't extend naturally to multiclass classification (except for logistic regression). One technique to extend the binary algorithms to be multiclass is to use the *one-vs.-rest* approach where where a binary model is learned for each class that tries to separate itself from all other classes. To make a prediction, all classifiers are run on a test point where the classifier with the highest score on a single class has its label returned as the classification.

- If you assume only a few of your features are important, use L1 regularization, otherwise use L2 regularization.
- Linear models excel when the number of features is large compared to the number of samples. They are also used on very large datasets where it is not feasible to train other models. However, it suffers in lower-dimensional spaces where other models have a better generalization performance.

**Naive Bayes Classifiers**: Work very similarly to linear models although they tend to train faster and give worse generalization results. The classifiers work well with high-dimensional sparse data and are robust to the parameters. They are a great baseline model and are commonly used on very large datasets where even a linear model might take too long to train.

### Chapter 2: Decision Trees**
- Essentially learns from a hierarchy of if/else questions that eventually lead to a decision.
- One way to think of it is to think of a graph of points. Each test (question) will split the graph into two, separating the points. Each split section will keep on splitting until every section has only homogenous data points. To make a guess, you would just check what section your new data point is in and what class dominates that section.
- Since this method looks to get 100% homogeny, the data will most likely be overfit. In order to control complexity, you can either pre-prune the tree (stopping the creation of the tree early) or post-prune/prune the tree (build the tree and then collapse the nodes that contain little information).
- While decision trees can be built for regression, most tree-based models cannot make predictions outside of the range of the training data.
- While decision trees can be easily visualized and are invariant to the scaling of data (does not require normalization or standardization), they tend to overfit and provide poor generalization performance.