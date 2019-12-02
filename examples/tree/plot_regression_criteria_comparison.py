"""
=============================================================
Comparing different regression split criteria on toy datasets

For visual examples of these datasets, see
:ref:`sphx_glr_auto_examples_datasets_plot_nonlinear_regression_datasets.py`.
=============================================================
"""

# Author: Vivek Gopalakrishnan <vgopala4@jhu.edu>
# License: BSD 3 clause

from itertools import product
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.linalg import norm

from sklearn.datasets import (make_independent_noise, make_log_regression,
                              make_multiplicative_noise, make_sin_regression,
                              make_square_regression)
from sklearn.ensemble import RandomForestRegressor

print(__doc__)

###############################################################################
n_dimensions = 10
noise = 10.0
simulations = {
    "Logarithmic": [make_log_regression, noise],
    r"Sine Period $4\pi$": [make_sin_regression, noise],
    "Square": [make_square_regression, noise],
    "Multiplicative": [make_multiplicative_noise, None],
    "Independence": [make_independent_noise, None],
}


###############################################################################
def _train_forest(X, y, criterion):
    """Fit a RandomForestRegressor with default parameters and specific criterion."""
    regr = RandomForestRegressor(
        n_estimators=500, criterion=criterion, max_features="sqrt", max_depth=5)
    regr.fit(X, y)
    return regr


def _test_forest(X, y, regr):
    """Calculate the accuracy of the model on a heldout set."""
    y_pred = regr.predict(X)
    return norm(y_pred - y) / len(y)


###############################################################################
def main(simulation_name, n_samples, criterion, n_dimensions=10, n_iter=10):
    """Measure the performance of RandomForest under simulation conditions.

    Parameters
    ----------
    simulation_name : str
        Key from `simulations` dictionary.
    n_samples : int
        Number of training samples.
    criterion : string
        Split criterion used to train forest. Choose from
        ("mse", "mae", "friedman_mse", "axis", "oblique").
    n_dimensions : int, optional (default=10)
        Number of features and targets to sample.
    n_iter : int, optional (default=10)
        Number of times to test a given parameter configuration.

    Returns
    -------
    (average, error) : np.ndarray
        NumPy array with the average MSE and standard error.
    """

    sim, noise = simulations[simulation_name]
    n_samples = int(n_samples)
    n_dimensions = int(n_dimensions)

    # Make a validation dataset
    if noise is not None:
        X_test, y_test = sim(n_samples=1000,
                             n_dimensions=n_dimensions,
                             noise=noise)
    else:
        X_test, y_test = sim(n_samples=1000,
                             n_dimensions=n_dimensions)

    # For each iteration in `n_iter`, train a forest on a newly sampled
    # training set and save its performance on the validation set
    score = []
    for _ in range(n_iter):

        # Sample training data
        if noise is not None:
            X_train, y_train = sim(n_samples=n_samples,
                                   n_dimensions=n_dimensions,
                                   noise=noise)
        else:
            X_train, y_train = sim(n_samples=n_samples,
                                   n_dimensions=n_dimensions)

        # Train RandomForest
        regr = _train_forest(X_train, y_train, criterion)
        forest_score = _test_forest(X_test, y_test, regr)
        score.append(forest_score)

    # Calculate average and standard deviation
    score = np.array(score)
    average = score.mean()
    error = score.std() / np.sqrt(n_iter)

    return np.array((average, error))


###############################################################################
# Construct the parameter space
simulation_names = simulations.keys()
sample_sizes = np.arange(5, 101, 5)
criteria = ["mae", "mse", "friedman_mse", "axis", "oblique"]
params = product(simulation_names, sample_sizes, criteria)

# Construct validation datasets
print("Constructing validation datasets...")
for simulation_name, (sim, noise) in simulations.items():
    if noise is not None:
        X_test, y_test = sim(n_samples=1000,
                             n_dimensions=n_dimensions,
                             noise=noise)
    else:
        X_test, y_test = sim(n_samples=1000,
                             n_dimensions=n_dimensions)
    simulations[simulation_name].append((X_test, y_test))

with Pool() as pool:

    # Run the simulations in parallel
    scores = pool.starmap(main, params)

    # Save results as a DataFrame
    params = np.array(list(product(sample_sizes, simulation_names, criteria)))
    df = np.concatenate((params, scores), axis=1)
    columns = ["n_samples", "simulation", "criterion", "average", "error"]
    df = pd.DataFrame(df, columns=columns)
    df.to_csv("./results.csv")

    # Plot the results
    sns.relplot(x="n_samples",
                y="average",
                hue="criterion",
                col="simulation",
                kind="line",
                data=df,
                facet_kws={'sharey': False, 'sharex': True})
    plt.show()
