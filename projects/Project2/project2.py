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
plt.savefig('latex_jail/plots/KMeans_plot.png')
plt.show()
plt.close()

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
import matplotlib.pyplot as plt

# Load Fashion-MNIST from OpenML
# Classes (0-9): T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot
X, y = fetch_openml("Fashion-MNIST", version=1, as_frame=False, parser="auto", return_X_y=True)
y = y.astype(int)

print(type(X),X.shape)
print(type(y),y.shape)

# %% id="0REsDBunNmEl"
from sklearn.cluster import KMeans
from scipy.stats import mode


# think it would be cool to see the data through the process
def peek_img(X=X, y=y, savepath=None):
    # get one unique sample per class
    unique_indices = []
    for class_id in range(10):
        idx = np.where(y == class_id)[0][0]
        unique_indices.append(idx)
    
    # Create figure with labels
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    side_length = int(np.sqrt(X.shape[1]))


    for i, idx in enumerate(unique_indices):
        ax = axes.flat[i]
        ax.imshow(X[idx].reshape(side_length,side_length), cmap='gray')
        ax.set_title(class_names[y[idx]])
        ax.axis('off')
    
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath)
    plt.show()
    plt.close()

peek_img(savepath='latex_jail/plots/examble_items_fashion.png')

# Scale the data
X = StandardScaler().fit_transform(X)
peek_img()

# It does not appear to need a sample size reduction for the kmeans to run reasonably quickly.
kmeans = KMeans(n_clusters=10, random_state=0).fit(X)

# Going to implement the kmeans mode matching but with the arrays instead of as a df

def kmean_confusion_mtx(kmeans=kmeans, y=y):
    kmeans_labels = kmeans.labels_

    modal_map = {}
    for cluster_id in range(10):
        mask = kmeans_labels == cluster_id
        if np.sum(mask) > 0:
            modal_map[cluster_id], _ = mode(y[mask])
        else:
            modal_map[cluster_id] = -1  # Handle empty clusters
    
    # Apply mapping
    kmeans_labels_mapped = np.array([modal_map[label] for label in kmeans_labels])
    
    # Confusion matrix
    confusion_matrix_kmeans = confusion_matrix(y, kmeans_labels_mapped)
    print(confusion_matrix_kmeans)

    total_samples = len(y)
    correct_predictions = np.trace(confusion_matrix_kmeans)
    confused_entries = total_samples - correct_predictions
    
    print(f"Total samples: {total_samples}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Confused entries: {confused_entries}")
    return confused_entries
    
# %% [markdown] id="6Bpow7TrZ7iB"
# # 3. Dimensionality reduction for Fashion-MNIST


# %% id="ejYYENCQZ9tj"
import time
from sklearn import random_projection
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split

def compare_PCA_vs_Random_Proj(k=10, X = X, y = y):
    print('fitting PCA')
    start = time.time()
    pca = PCA(n_components=k).fit_transform(X)
    pca_time = time.time() - start
    print(f'pca time: {pca_time}')

    print('fitting random projection')
    start = time.time()
    rand = random_projection.GaussianRandomProjection(n_components=k).fit_transform(X)
    rand_time = time.time() - start
    print(f'rand time: {rand_time}')
    # Compute distance correlations

    print('calculating the distances')
    orig_dist = pdist(X, 'euclidean')
    print('orig_dist done')
    pca_dist = pdist(pca, 'euclidean')
    print('pca_dist_done')
    rand_dist = pdist(rand, 'euclidean')
    print('rand_dist done')
    
    pca_corr, _ = pearsonr(orig_dist, pca_dist)
    rand_corr, _ = pearsonr(orig_dist, rand_dist)
    
    print(f'k={k}: PCA time={pca_time:.3f}s (corr={pca_corr:.4f}), RP time={rand_time:.3f}s (corr={rand_corr:.4f})')
    
    return pca_corr, rand_corr, pca_time, rand_time


dimensions = [10, 20, 50, 100, 200]
pca_corrs, rand_corrs = [], []

# There is no way my computer can pairwise thes bad boys, especially by like this afternoon,
# So Im goint go cut the data wayyyy down with a train/test split and only do the "train" data
# Also I know there is a much better way to do all this but this needs to be done 3 days
# ago so IDGAF

X_reasonable, y_reasonable, a, b, = train_test_split(X, y, test_size = .9, random_state = 42)

for k in dimensions:
    pca_corr, rand_corr, _, _ = compare_PCA_vs_Random_Proj(k, X_reasonable, y_reasonable)
    pca_corrs.append(pca_corr)
    rand_corrs.append(rand_corr)

# Plot results
plt.plot(dimensions, pca_corrs, 'o-', label='PCA')
plt.plot(dimensions, rand_corrs, 's-', label='Random Projection')
plt.xlabel('Target dimension k')
plt.ylabel('Distance correlation')
plt.legend()
plt.savefig('latex_jail/plots/comparison_pca_vs_rand.png')
plt.show()



# %% [markdown] id="fOTFcjWOfCZU"
# # 4. Clustering Fashion-MNIST using spectral clustering

# %% id="MRB_nw21fI24"
from sklearn.cluster import SpectralClustering

# The full thing wants 36.5 G of ram.... I could give it a swap file or i could do other things today
# Since this is more then just demonstrating the lemma, Ill try it with half the data

X_reasonable, y_reasonable, a, b = train_test_split(X, y, test_size = .9, random_state = 42)

pca = PCA(n_components=20).fit_transform(X_reasonable)


spectral_clustering = SpectralClustering(n_clusters = 10, assign_labels='kmeans', verbose=True).fit(pca)
print('done with spectral clustering')
kmean_confusion_mtx(spectral_clustering)


