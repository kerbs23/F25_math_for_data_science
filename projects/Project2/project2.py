# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown] id="0gdC70xxFyc4"
# Math 5750/6880: Mathematics of Data Science \
# Project 2
# pyright: basic
# ruff: noqa: e402


# %% [markdown] id="i9_7SnpMGKDJ"
# # 1. Clustering Gaussian Blobs using $k$-means

# %% colab={"base_uri": "https://localhost:8080/"} id="AB136H0PGKq1" outputId="6fcea671-2dcb-4b43-c98a-83ec001de164"
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs

# Generate 5 Gaussian blobs in 10 dimensions
X, y_true = make_blobs(
    n_samples=1000,
    centers=5,
    n_features=10,
    cluster_std=1.5,
    random_state=1)        # reproducibility
X = StandardScaler().fit_transform(X)

print(type(X),X.shape)
print(type(y_true),y_true.shape)

# %% id="5GAsN-dmHjRM"
# Start with a basic k-means analysys of the data
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

df = pd.DataFrame(data=X)
df.columns = [f'X_{i+1}' for i in range(len(df.columns))]
df['true_category'] = y_true

kmeans_5 = KMeans(n_clusters=5).fit(X) # because this is using k-means++ sampling there is only one run, so lowest inertia is just the inertia of the output
print(f'Lowest inertia: {kmeans_5.inertia_}')
df['kmeans_5_pred_category'] = kmeans_5.labels_

centers = kmeans_5.cluster_centers_

# PCA down to 2 dim
X_pca_2d = PCA(n_components=2).fit(X)
df[['X_pca_2d_1', 'X_pca_2d_2']] = X_pca_2d.transform(X)
centers_2d = X_pca_2d.transform(centers)

print(df.head)
df.to_csv("gausian_blob_processed.csv")

# Plot it
plt.scatter(df['X_pca_2d_1'], df['X_pca_2d_2'], c=df['kmeans_5_pred_category'], cmap='viridis')
plt.scatter(centers_2d[:, 0], centers_2d[:, 1], c='red', marker='X', s=200)
plt.savefig('KMeans_plot.png')
plt.show()

# now to re-index to match the predicted and real columns
# going to just take the mode of each match, and if there are no collisions call it good

# Find modal mapping
modal_map = df.groupby('kmeans_5_pred_category')['true_category'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)

# Check if mapping is one-to-one
print(modal_map)

# Apply transformation
df['kmeans_5_pred_category_adj'] = df['kmeans_5_pred_category'].map(modal_map) # pyright: ignore[]

# create the confusion matrix
confusion_matrix_kmean5 = confusion_matrix(df['true_category'], df['kmeans_5_pred_category_adj'])
print(confusion_matrix_kmean5)

# %% [markdown] id="a2qcKggmIH8T"

# # 2. Clustering Fashion-MNIST using $k$-means

# %% colab={"base_uri": "https://localhost:8080/"} id="B9IQwhgcIVOl" outputId="5cc76846-93c1-492c-a1ab-6388f8300da9"
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler

# Load Fashion-MNIST from OpenML
# Classes (0-9): T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot
X, y = fetch_openml("Fashion-MNIST", version=1, as_frame=False, parser="auto", return_X_y=True)
y = y.astype(int)

print(type(X),X.shape)
print(type(y),y.shape)

# %% id="0REsDBunNmEl"
# your code here

# %% [markdown] id="6Bpow7TrZ7iB"
# # 3. Dimensionality reduction for Fashion-MNIST

# %% id="ejYYENCQZ9tj"
# your code here

# %% [markdown] id="fOTFcjWOfCZU"
# # 4. Clustering Fashion-MNIST using spectral clustering

# %% id="MRB_nw21fI24"
# your code here
