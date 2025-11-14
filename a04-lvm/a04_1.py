# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: py3
#     language: python
#     name: py3
# ---

# %%
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import svd

# %load_ext autoreload
# %autoreload 2
from a04_helper import *
from a04_functions import ppca_mle, ppca_nll

# %% [markdown]
# # 1 Probabilistic PCA

# %% [markdown]
# ## 1a) Toy data

# %%
# Generate and plot a toy dataset
toy_ppca = ppca_gen(Z=1, sigma2=0.5, seed=0)
ppca_plot_2d(toy_ppca)
print(np.sum(toy_ppca["X"] ** 3))  # must be 273244.3990646409

# %%
# Impact of noise
# YOUR CODE HERE

# %% [markdown]
# ## 1b) Maximum Likelihood Estimation

# %%
# Implement MLE for PPCA by completing the function `ppca_mle` in a04_functions.py.

# %%
# Test your solution. This should produce:
# {'mu': array([0.96935329, 1.98309575]),
#  'W': array([[-1.72988776], [-0.95974566]]),
#  'sigma2': 0.4838656103694303}
ppca_mle(toy_ppca["X"], 1)

# %%
# Test your solution. This should produce:
# {'mu': array([0.96935329, 1.98309575]),
# 'W': array([[-1.83371058,  0.33746522], [-1.0173468 , -0.60826214]]),
# 'sigma2': 0.0}
ppca_mle(toy_ppca["X"], 2)

# %% [markdown]
# ## 1c) Negative Log-Likelihood

# %%
# Implement the computation of the conditional negative log-likelihood by completing the function `ppca_nll` in a04_functions.py.

# %%
# Test your solution. This should produce: 32154.198760474777
ppca_nll(toy_ppca["X"], ppca_mle(toy_ppca["X"], 1))

# %% [markdown]
# ## 1d) Discover the Secret!

# %%
# Load the secret data
X = np.loadtxt("data/secret_ppca.csv", delimiter=",")

# %%
# Determine a suitable choice of L using a scree plot.
# Your code here

# %%
# Determine a suitable choice of Z using validation data.
split = len(X) * 3 // 4
X_train = X[:split,]
X_valid = X[split:,]

# %%
# YOUR CODE HERE
