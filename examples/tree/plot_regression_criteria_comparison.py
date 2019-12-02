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
NOISE = 10.0
simulations = {
    "Logarithmic": (make_log_regression, NOISE),
    r"Sine Period $4\pi$": (make_sin_regression, NOISE),
    "Square": (make_square_regression, NOISE),
    "Multiplicative": (make_multiplicative_noise, None),
    "Independence": (make_independent_noise, None),
}


###############################################################################
def train_forest(X, y, criterion):
    """
    Fit a RandomForestRegressor with default parameters and specific criterion.
    """
    regr = RandomForestRegressor(
        n_estimators=500, criterion=criterion, max_features="sqrt")
    regr.fit(X, y)
    return regr


def test_forest(X, y, regr):
    """
    Calculate the accuracy of the model on a heldout set.
    """
    y_pred = regr.predict(X)
    return norm(y_pred - y) / len(y)


###############################################################################
def main(simulation_name, n_samples, n_dimensions, criterion, n_iter=10):

    sim, noise = simulations[simulation_name]
    n_samples = int(n_samples)
    n_dimensions = int(n_dimensions)

    # Make a validation dataset
    if noise is not None:
        X_test, y_test = sim(
            n_samples=1000, n_dimensions=n_dimensions, noise=noise)
    else:
        X_test, y_test = sim(n_samples=1000, n_dimensions=n_dimensions)

    # Train forests and score them
    score = []
    for _ in range(n_iter):

        # Sample training data
        if noise is not None:
            X_train, y_train = sim(n_samples=n_samples,
                                   n_dimensions=n_dimensions, noise=noise)
        else:
            X_train, y_train = sim(n_samples=n_samples,
                                   n_dimensions=n_dimensions)

        # Train RandomForest
        regr = train_forest(X_train, y_train, criterion)
        forest_score = test_forest(X_test, y_test, regr)
        score.append(forest_score)

    # Calculate average and standard deviation
    score = np.array(score)
    average = score.mean()
    error = score.std() / np.sqrt(n_iter)
    out = np.array([average, error])

    return out


###############################################################################
# Start running the simulations
start_time = time.time()

# Save parameter space as a numpy array
params = product(
    range(5, 105, 5), simulations.keys(), ["mae", "mse", "friedman_mse"]
)
params = np.array(list(params))

# Open multiprocessing
with Pool() as pool:

    # Run the pools
    scores = pool.starmap(main, params)

    # Save data to array
    df = np.concatenate((params, scores), axis=1)
    columns = ["n_samples", "simulation", "criterion", "average", "error"]
    df = pd.DataFrame(df, columns=columns)
    print(df.head())

    df.to_csv("./results/sim_noise/results.csv")

# Print runtime
print("All finished!")
print("Took {} minutes".format((time.time() - start_time) / 60))
