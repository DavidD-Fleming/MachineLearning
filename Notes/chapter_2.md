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

* **Random Forests**: A random forest is a collection of decision trees, where each tree is slightly different from others. By building many trees, all of which work well and overfit, you can reduce the amount of overfitting by averaging their results. In order to randomize the trees, you create a bootstrap sample of the data (same length as the original sample except data points can be randomly repeated). When building the random forest, you can decide the amount of features that the split looks at. When all features are utilized, the trees will be relatively similar whereas when only one feature is used, the forest will be very different.
    - With random forests making up the deficiencies of decision trees, the only real use of decision trees is to visualize the data easier.
    - You can parallelize the process of making random trees by utilizing multiple CPU cores which will speed up the creation process.
    - In order to create forests that are robust against randomness, it's important to use more trees.
    - Random forests don't work as well as linear models on very high dimensional, sparse data. Linear models also make more sense if time and memory are more important in an application.
    - The most important parameters are the number of estimators, max number of features, and pruning options like the max depth. More estimators is always better however they return diminishing results so only use what you can afford. The max number of features decides randomness with smaller max features reducing overfitting.

* **Gradient Boosted Regression Trees**: Instead of using random trees, this method builds the next tree in order to correct the mistakes of the previous one. Instead of randomization, strong pre-pruning methods are used. The trees are also generally shallow which means less memory is used and predictions are faster. Gradient boosted trees are more sensitive to parameter settings but can provide better accuracy if the settings are set correctly.
    - The parameters that affect the results are the number of trees, the amount of pruning, and the learning rate (how much the next tree tries to learn from the previous tree).
    - A general strategy is to use a random tree first as they are quite robust. However, if prediction time is a premium or it is important to get that last drop of accuracy, you can switch to using gradient boosted trees.
    - The main drawback of gradient boosted decision trees is that you have to be very careful when tuning the parameters and it may take a while to train. However, like other trees, the algorithm works well without scaling and on a mixture of binary and continuous features. It does not work well with high-dimensional sparse data.
    - Whereas more estimators is always good for random trees, more estimators may lead to overfitting for gradient boosted trees.

### Chapter 2: Kernelized Support Vector Machines**
- Adding nonlinear features to the representation of our data can make linear models much more powerful.
- Knowing which features to add can be difficult. However, there are kernel tricks you can use like the polynomial kernel and Gaussian kernel that directly compute the distance of the data points for the expanded representation without actually computing the expansion.
- During training, the SVM learns how important each of the training data points are to represent the decision boundary between two classes. Only a subset of the points actually matter for defining the boundary, which are the support vectors.
- Sometimes, datasets will have completely different orders of magnitude which can be devastating for kernel SVMs. One way to fix this problem is to rescale each feature so that they are all approximately on the same scale.
- SVMs work well on low and high dimensional data but don't scale well with the number of samples.
- Another downside is that SVMs require careful preprocessing of the data and tuning of parameters. Furthermore, they are hard to inspect and it can be difficult to understand why a particular prediction was made.

* **Gaussian Kernel (Radial Basis Kernel)**: The distance between data points is k_rbf(x_1, x_2) = exp(-y * ||x_1 - x_2||^2) where x_1 and x_2 are the data points and y (gamma) is the parameter that controls the width of the kernel.
    - The gamma parameter corresponds to the inverse of the width of the Gaussian kernel meaning the lower the value, the farther the reach. In other words, the wider the radius of the kernel, the further the influence of each training example. Gamma acts as the regularization parameter.
    - The C parameter decides how strict the model is, where each data point can only have very limited influence. In creasing C means that points have a stronger influence on the model and tehrefore the decision boundary bends more to correctly classify them.

### Chapter 2: Neural Networks (Deep Learning)**
- There are many ways to control the complexity of a neural network. The number of hidden layers, the number of units in each layer, and the regularization are just a few ways.
- The weights of neural networks are random so you can obtain different models even when using the same parameters. If the networks are large and their complexity are chosen properly, this should not affect accuracy too much.
- The cons of neural networks are that they take a long time to train and require careful preprocessing of data. They also work better with homogeneous data rather than heterogeneous data.
- A common way to adjust parameters is to first create a network that is large enough to overfit. Once you know the training data can be learned, either shrink the network or increase alpha to add regularization.