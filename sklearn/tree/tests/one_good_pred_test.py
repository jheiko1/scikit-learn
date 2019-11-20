import numpy as np 
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import random
import scipy

#sample uniform random matrix
# weight matrix

def generate_w(num_feats = 100,num_pred = 10):
    w = np.random.uniform(size=(num_feats,num_pred))
    sparse = np.array([random.randint(0,2) for i in range(num_feats * num_pred)]).reshape((num_feats,num_pred))
    w = np.multiply(w,sparse)
    return w

def generate_data(w, num_samples=100, num_feats = 100, num_pred = 10, var = 1000):

    X = np.random.uniform(low=0, high=10, size=(num_samples, num_feats))
    y = np.dot(X, w) # num_samples * num_pred
    noise = np.random.normal(loc=0.0, scale=var, size=(num_samples, num_pred))
    noise[:,0] = 0
    y = y + noise
    return X, y

def measure_mse(X_train, y_train, X_test, y_test, max_depth=10, n_features=1, min_leaf_size=5, n_trees=1000, n_bagging=10):
    """
    Return MSE for each split criteria.
    """
    
    # Iterate over different split criteria
    errors = []
    for split in ["mae", "mse", "axis", "oblique"]:

        # Fit model
        rf = RandomForestRegressor(criterion=split, max_depth=max_depth, min_samples_leaf=min_leaf_size, n_estimators=n_trees, random_state=1)
        rf.fit(X_train,y_train)

        # Make predictions and score
        yhat = rf.predict(X_test)
        mse = np.linalg.norm(y_test[:,0]-yhat[:,0])
        errors.append(mse)
        
    return errors




if __name__ == "__main__":
    w = generate_w()
    X_test, y_test = generate_data(w, num_samples=100)
    '''
    plt.scatter(X_test[:, 0], X_test[:, 1], c="blue", label="X_test")
    plt.scatter(y_test[:, 0], y_test[:, 1], c="red", label="y_test")

    # Plot lines between matched pairs of points

    for xi, yi in zip(X, y):
        plt.plot(
            [xi[0], yi[0]], 
            [xi[1], yi[1]], 
            c="black", 
            alpha = 0.15
        )

    plt.legend()
    plt.show()
    '''
    # Test functions on sample data
    measure_mse(X_test, y_test, X_test, y_test)

    # Run simulation
    results = []
    max_n = 80#201
    n_iter = 5#10

    for n in range(10, max_n, 10):
        for i in range(n_iter):
            # Generate sample data
            X_train, y_train = generate_data(w, var=n)

            # Measure MSE
            mse = measure_mse(X_train, y_train, X_test, y_test)

            # Add to dataframe
            mse.insert(0, n)
            results.append(mse)

            print(n, i)

    # Convert to dataframe
    columns = ["mae", "mse", "projection_axis", "projection _oblique"]
    columns.insert(0, "n")
    df = pd.DataFrame(results, columns=columns)
    df = pd.melt(df, id_vars=['n'], value_vars=columns[1:], var_name='split', value_name='mse')
    df["mse"] /= df["n"]
    df.head()

    with sns.plotting_context("talk", font_scale=1):

        f = sns.lineplot(x="n", y="mse", hue="split", data=df)
        f.set(xlabel="n", ylabel="mse / n")
        plt.show()

