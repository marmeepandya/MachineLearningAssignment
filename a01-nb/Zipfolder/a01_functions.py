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
from a01_helper import logsumexp


# %%
def nb_train(X, y, alpha=1, K=None, C=None):
    """Train a Naive Bayes model.

    We assume that all features are encoded as integers and have the same domain
    (set of possible values) from 0:(K-1). Similarly, class labels have domain
    0:(C-1).

    Parameters
    ----------
    X : ndarray of shape (N,D)
        Design matrix.
    y : ndarray of shape (N,)
        Class labels.
    alpha : int
        Parameter for symmetric Dirichlet prior (Laplace smoothing) for all
        fitted distributions.
    K : int
        Each feature takes values in [0,K-1]. None means auto-detect.
    C : int
        Each class label takes values in [0,C-1]. None means auto-detect.

    Returns
    -------
    A dictionary with the following keys and values:

    logpriors : ndarray of shape (C,)
        Log prior probabilities of each class such that logpriors[c] contains
        the log prior probability of class c.

    logcls : ndarray of shape(C,D,K)
        A class-by-feature-by-value array of class-conditional log-likelihoods
        such that logcls[c,j,v] contains the conditional log-likelihood of value
        v in feature j given class c.
    """
    N, D = X.shape
    if K is None:
        K = np.max(X) + 1
    if C is None:
        C = np.max(y) + 1

    # Compute class priors and store them in priors
    priors = np.zeros(C)
    # YOUR CODE HERE
    class_counts = np.bincount(y, minlength=C)
    priors = (class_counts + alpha - 1) / (N + C * (alpha - 1))

    # Compute class-conditional densities in a class x feature x value array
    # and store them in cls.
    cls = np.zeros((C, D, K))
    # YOUR CODE HERE
    for c in range(C):
        Xc = X[y == c]
        Nc = Xc.shape[0]
        for j in range(D):
            # if Nc == 0, feature_counts -> zeros; denom = K*(alpha-1)
            feature_counts = np.bincount(Xc[:, j], minlength=K) if Nc > 0 else np.zeros(K, dtype=int)
            denom = Nc + K * (alpha - 1)
            cls[c, j, :] = (feature_counts + alpha - 1) / denom

    # Output result
    return dict(logpriors=np.log(priors), logcls=np.log(cls))

# %%
def nb_predict(model, Xnew):
    """Predict using a Naive Bayes model.

    Parameters
    ----------
    model : dict
        A Naive Bayes model trained with nb_train.
    Xnew : nd_array of shape (Nnew,D)
        New data to predict.

    Returns
    -------
    A dictionary with the following keys and values:

    yhat : nd_array of shape (Nnew,)
        Predicted label for each new data point.

    logprob : nd_array of shape (Nnew,)
        Log-probability of the label predicted for each new data point.
    """
    logpriors = model["logpriors"]
    logcls = model["logcls"]
    Nnew = Xnew.shape[0]
    C, D, K = logcls.shape

    # Compute the unnormalized log joint probabilities P(Y=c, x_i) of each
    # test point (row i) and each class (column c); store in logjoint
    logjoint = np.zeros((Nnew, C))
    # YOUR CODE HERE

    # start with class log-priors broadcasted to shape (Nnew, C)
    logjoint = np.tile(logpriors[None, :], (Nnew, 1))  # (Nnew, C)

    # add up per-feature conditional log-probabilities
    for j in range(D):
        vals = Xnew[:, j]                  # shape (Nnew,)
        # logcls[:, j, vals] -> shape (C, Nnew); transpose to (Nnew, C)
        logjoint += logcls[:, j, vals].T


    # Compute predicted labels (in "yhat") and their log probabilities
    # P(yhat_i | x_i) (in "logprob")
    # YOUR CODE HERE

    # predicted labels: class with highest unnormalized log-joint per example
    yhat = np.argmax(logjoint, axis=1).astype(int)  # shape (Nnew,)

    # compute normalizer log p(x_i) = logsumexp_c logjoint[i,c]
    # our helper computes logsumexp over columns, so pass transpose (C, Nnew)
    lse = logsumexp(logjoint.T)  # shape (Nnew,)

    # log-probability of the predicted label: log p(yhat | x) = log p(yhat,x) - log p(x)
    logprob = logjoint[np.arange(Nnew), yhat] - lse  # shape (Nnew,)

    return dict(yhat=yhat, logprob=logprob)


# %%
def nb_generate(model, ygen):
    """Given a Naive Bayes model, generate some data.

    Parameters
    ----------
    model : dict
        A Naive Bayes model trained with nb_train.
    ygen : nd_array of shape (n,)
        Vector of class labels for which to generate data.

    Returns
    -------
    nd_array of shape (n,D)

    Generated data. The i-th row is a sampled data point for the i-th label in
    ygen.
    """
    logcls = model["logcls"]
    n = len(ygen)
    C, D, K = logcls.shape
    Xgen = np.zeros((n, D))
    for i in range(n):
        c = ygen[i]
        # Generate the i-th example of class c, i.e., row Xgen[i,:]. To sample
        # from a categorical distribution with parameter theta (a probability
        # vector), you can use np.random.choice(range(K),p=theta).
        for j in range(D):
            theta = np.exp(logcls[c, j, :])  # class-conditional probabilities for feature j
            Xgen[i, j] = np.random.choice(range(K), p=theta)

    return Xgen
