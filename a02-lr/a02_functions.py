# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
# ---

# %%
import numpy as np

# %%
# normalise the data
def normalize_data(X, Xtest): 
    """Computes z-scores for a training set and a holdout test set.

    Keyword arguments: 
    X, Xtest : two 2D ndarrays
    """
    mu_estimate = np.mean(X, axis=0)
    sigma_estimate = np.std(X, axis=0)

    X_normalized = (X - mu_estimate)/sigma_estimate
    Xtest_normalized = (Xtest - mu_estimate)/sigma_estimate
    
    return X_normalized, Xtest_normalized

# %%
# you don't need to modify this function
def logsumexp(x):
    """Computes log(sum(exp(x)).

    Uses offset trick to reduce risk of numeric over- or underflow. When x is a
    1D ndarray, computes logsumexp of its entries. When x is a 2D ndarray,
    computes logsumexp of each column.

    Keyword arguments:
    x : a 1D or 2D ndarray
    """
    offset = np.max(x, axis=0)
    return offset + np.log(np.sum(np.exp(x - offset), axis=0))


# %%
# Define the logistic function. Make sure it operates on both scalars
# and vectors.
def sigma(x):
    
    return 1 / (1 + np.exp(-x))

# %%
# Define the logarithm of the logistic function. Make sure it operates on both
# scalars and vectors. Perhaps helpful: isinstance(x, np.ndarray).
def logsigma(x):
    # YOUR CODE HERE
    return np.log(sigma(x))


# %%
def l(y, X, w):
    """Log-likelihood of the logistic regression model.

    Parameters
    ----------
    y : ndarray of shape (N,)
        Binary labels (either 0 or 1).
    X : ndarray of shape (N,D)
        Design matrix.
    w : ndarray of shape (D,)
        Weight vector.
    """
    

    pass


# %%
def dl(y, X, w):
    """Gradient of the log-likelihood of the logistic regression model.

    Parameters
    ----------
    y : ndarray of shape (N,)
        Binary labels (either 0 or 1).
    X : ndarray of shape (N,D)
        Design matrix.
    w : ndarray of shape (D,)
        Weight vector.

    Returns
    -------
    ndarray of shape (D,)
    """
    # YOUR CODE HERE
    pass


# %%
# define the objective and update function for one gradient-descent epoch for
# fitting an MLE estimate of logistic regression with gradient descent (should
# return a tuple of two functions; see optimize)
def gd(y, X):
    def objective(w):
        # YOUR CODE HERE
        pass

    def update(w, eps):
        # YOUR CODE HERE
        pass

    return (objective, update)


# %%
def sgdepoch(y, X, w, eps):
    """Run one SGD epoch and return the updated weight vector."""
    # Run N stochastic gradient steps (without replacement). Do not rescale each
    # step by factor N (i.e., proceed differently than in the lecture slides).
    # YOUR CODE HERE
    pass


# %%
# define the objective and update function for one gradient-descent epoch for
# fitting an MLE estimate of logistic regression with stochastic gradient descent
# (should return a tuple of two functions; see optimize)
def sgd(y, X):
    def objective(w):
        # YOUR CODE HERE
        pass

    def update(w, eps):
        return sgdepoch(y, X, w, eps)

    return (objective, update)


# %%
def predict(Xtest, w):
    """Returns vector of predicted confidence values for logistic regression with
    weight vector w."""
    # YOUR CODE HERE
    pass


def classify(Xtest, w):
    """Returns 0/1 vector of predicted class labels for logistic regression with
    weight vector w."""
    # YOUR CODE HERE
    pass


# %%
def l_l2(y, X, w, lambda_):
    """Log-density of posterior of logistic regression with weights w and L2
    regularization parameter lambda_"""
    # YOUR CODE HERE
    pass


# %%
def dl_l2(y, X, w, lambda_):
    """Gradient of log-density of posterior of logistic regression with weights w
    and L2 regularization parameter lambda_."""
    # YOUR CODE HERE
    pass


# %%
def gd_l2(y, X, lambda_):
    # YOUR CODE HERE
    pass


# %%
# you don't need to modify this function
def optimize(obj_up, theta0, nepochs=50, eps0=0.01, verbose=True):
    """Iteratively minimize a function.

    We use it here to run either gradient descent or stochastic gradient
    descent, using arbitrarly optimization criteria.

    Parameters
    ----------
    obj_up  : a tuple of form (f, update) containing two functions f and update.
              f(theta) computes the value of the objective function.
              update(theta,eps) performs an epoch of parameter update with step size
              eps and returns the result.
    theta0  : ndarray of shape (D,)
              Initial parameter vector.
    nepochs : int
              How many epochs (calls to update) to run.
    eps0    : float
              Initial step size.
    verbose : boolean
              Whether to print progress information.

    Returns
    -------
    A triple consisting of the fitted parameter vector, the values of the
    objective function after every epoch, and the step sizes that were used.
    """

    f, update = obj_up

    # initialize results
    theta = theta0
    values = np.zeros(nepochs + 1)
    eps = np.zeros(nepochs + 1)
    values[0] = f(theta0)
    eps[0] = eps0

    # now run the update function nepochs times
    for epoch in range(nepochs):
        if verbose:
            print(
                "Epoch {:3d}: f={:10.3f}, eps={:10.9f}".format(
                    epoch, values[epoch], eps[epoch]
                )
            )
        theta = update(theta, eps[epoch])

        # we use the bold driver heuristic
        values[epoch + 1] = f(theta)
        if values[epoch] < values[epoch + 1]:
            eps[epoch + 1] = eps[epoch] / 2.0
        else:
            eps[epoch + 1] = eps[epoch] * 1.05

    # all done
    if verbose:
        print("Result after {} epochs: f={}".format(nepochs, values[-1]))
    return theta, values, eps

