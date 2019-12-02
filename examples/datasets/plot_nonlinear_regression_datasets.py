"""
=====================================================
Plot randomly generated nonlinear regression datasets
=====================================================

Plot several randomly generated 1D regression datasets.
This example illustrates the :func:`datasets.make_log_regression`,
:func:`datasets.make_sin_regression`, :func:`datasets.make_square_regression`,
:func:`datasets.make_multiplicative_noise`, and
:func:`datasets.make_independent_noise` functions.

For each, :math:`n = 100` points are sampled with noise to show the actual
sample data used for one-dimensional relationships (gray dots).

For comparison purposes, :math:`n = 1000` points are sampled without noise to
highlight each underlying dependency (black dots). Note that only black points
are plotted for the :func:`datasets.make_multiplicative_noise`, and
:func:`datasets.make_independent_noise` functions, as they do not have a noise
parameter.
"""

print(__doc__)

import matplotlib.pyplot as plt

from sklearn.datasets import (make_independent_noise, make_log_regression,
                              make_multiplicative_noise, make_sin_regression,
                              make_square_regression)


def plot_simulation(simulation_name, ax):

    # Get simulation function
    simulation = simulations[simulation_name]

    # Sample noiseless and noisy versions of the data
    if simulation_name in ["Logarithmic", r"Sine Period $4\pi$", "Square"]:
        X_pure, y_pure = simulation(n_samples=1000, n_dimensions=1, noise=0)
        X_noise, y_noise = simulation(n_samples=100, n_dimensions=1, noise=1)
    else:
        X_pure, y_pure = simulation(n_samples=1000, n_dimensions=1)
        X_noise, y_noise = simulation(n_samples=100, n_dimensions=1)

    # Plot the data points from both data sets
    ax.scatter(X_pure, y_pure, c="#CCD1D1")
    ax.scatter(X_noise, y_noise, c="#17202A")
    ax.set_title(simulation_name)


simulations = {
    "Logarithmic": make_log_regression,
    r"Sine Period $4\pi$": make_sin_regression,
    "Square": make_square_regression,
    "Multiplicative": make_multiplicative_noise,
    "Independence": make_independent_noise,
}

_, axs = plt.subplots(1, 5, sharex='row', sharey='row', figsize=(40, 4))
plt.subplots_adjust(bottom=.15)

for simulation_name, ax in zip(simulations.keys(), axs):
    plot_simulation(simulation_name, ax)

plt.show()
