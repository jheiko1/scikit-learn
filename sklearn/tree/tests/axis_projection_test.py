from sklearn.ensemble import RandomForestRegressor
# from vivek's 10-16-19 experiment
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def generate_data(
    n=25, 
    mean=[0, 0], 
    cov=[[1, 0], [0, 1]],
    theta=np.pi/4
):
    """
    Generate synthetic data. 
    X ~iid MVN(u=0, cov=I).
    y = AX where A is a rotation matrix.
    """
    
    # Rotation matrix
    A = [
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ]
    
    # Sample random variables
    X = np.random.multivariate_normal(mean, cov, size=n)
    y = np.dot(A, X.T)
    
    return X, y.T


'''
def predict(rf, X):
    """
    Return predictions for every element in X.
    """
    
    yhat = []
    for xi in X:
        yi = np.mean(rf.predict(xi), axis=0)
        yhat.append(yi)
        
    return np.array(yhat)
'''
def measure_mse(X, y, max_depth=10, n_features=1, min_leaf_size=5, n_trees=1000, n_bagging=10):
    """
    Return MSE for each split criteria.
    """
    
    # Iterate over different split criteria
    errors = []
    for split in ["mae", "mse", "axis", "oblique"]:

        # Fit model
        rf = RandomForestRegressor(criterion=split, max_depth=max_depth, min_samples_leaf=min_leaf_size, n_estimators=n_trees, random_state=1)
        rf.fit(X,y)

        # Make predictions and score
        yhat = rf.predict(X)
        mse = np.linalg.norm(y-yhat)
        errors.append(mse)
        
    return errors




if __name__ == "__main__":
	X, y = generate_data()

	plt.scatter(X[:, 0], X[:, 1], c="blue", label="X")
	plt.scatter(y[:, 0], y[:, 1], c="red", label="y")

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
	# Test functions on sample data
	measure_mse(X, y)

	# Run simulation
	results = []
	max_n = 201
	n_iter = 10

	for n in range(10, max_n, 10):
		for i in range(n_iter):
			
			# Generate sample data
			X, y = generate_data(n=n)
			
			# Measure MSE
			mse = measure_mse(X, y)
			
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