import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering

# Generating random data
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Agglomerative clustering
cluster = AgglomerativeClustering(n_clusters=4, linkage='ward')
cluster.fit_predict(X)

# Plotting the clusters
plt.scatter(X[:, 0], X[:, 1], c=cluster.labels_, cmap='viridis')
plt.show()

s