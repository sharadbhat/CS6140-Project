import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

filepath = os.path.join(os.path.dirname(__file__), '../data/train.json') 

with open(filepath) as data_file:    
    data = pd.read_json(data_file)


##########################################
## Get list of cuisines and ingredients ##
##########################################
cuisines_list = data['cuisine'].unique()
ingredients_list = list(set([ingredient for ingredients in data['ingredients'] for ingredient in ingredients]))


#######################################################################
## Count the number of times each ingredient appears in each cuisine ##
#######################################################################
ingredients_per_cuisine = {}
for _, recipe in data.iterrows():
    cuisine = recipe['cuisine']
    if cuisine not in ingredients_per_cuisine:
        ingredients_per_cuisine[cuisine] = {}
    for ingredient in recipe['ingredients']:
        ingredients_per_cuisine[cuisine][ingredient] = ingredients_per_cuisine[cuisine].get(ingredient, 0) + 1

ingredients_per_cuisine_df = pd.DataFrame.from_dict(ingredients_per_cuisine).fillna(0).T


#################################################
## Running TFIDF on ingredients_per_cuisine_df ##
#################################################
tfidf_transformer = TfidfTransformer()
tfidf = tfidf_transformer.fit_transform(ingredients_per_cuisine_df)


##########################################
## Running PCA to reduce dimensionality ##
##########################################
pca = PCA(n_components=2).fit(tfidf.toarray())
data2D = pca.transform(tfidf.toarray())

n_clusters = 3
# Create a subplot with 1 row and 2 columns
fig, ax1 = plt.subplots(1, 1)
# fig.set_size_inches(18, 7)

# The 1st subplot is the silhouette plot
# The silhouette coefficient can range from -1, 1 but in this example all
# lie within [-0.1, 1]
ax1.set_xlim([-0.1, 1])
# The (n_clusters+1)*10 is for inserting blank space between silhouette
# plots of individual clusters, to demarcate them clearly.
ax1.set_ylim([0, len(data2D) + (n_clusters + 1) * 10])

# Initialize the clusterer with n_clusters value and a random generator
# seed of 10 for reproducibility.
clusterer = KMeans(init='k-means++', n_clusters=n_clusters, max_iter=1000)
cluster_labels = clusterer.fit_predict(data2D)

# The silhouette_score gives the average value for all the samples.
# This gives a perspective into the density and separation of the formed
# clusters
silhouette_avg = silhouette_score(data2D, cluster_labels)
print('For n_clusters =', n_clusters,
        'The average silhouette_score is :', silhouette_avg)

# Compute the silhouette scores for each sample
sample_silhouette_values = silhouette_samples(data2D, cluster_labels)

y_lower = 10
for i in range(n_clusters):
    # Aggregate the silhouette scores for samples belonging to
    # cluster i, and sort them
    ith_cluster_silhouette_values = \
        sample_silhouette_values[cluster_labels == i]

    ith_cluster_silhouette_values.sort()

    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

    color = cm.Spectral(float(i) / n_clusters)
    ax1.fill_betweenx(np.arange(y_lower, y_upper),
                        0, ith_cluster_silhouette_values,
                        facecolor=color, edgecolor=color, alpha=0.8)

    # Label the silhouette plots with their cluster numbers at the middle
    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

    # Compute the new y_lower for next plot
    y_lower = y_upper + 10  # 10 for the 0 samples

ax1.set_title('Silhouette Plot')
ax1.set_xlabel('Silhouette Coefficient Values')
ax1.set_ylabel('Cluster Label')

# The vertical line for average silhouette score of all the values
ax1.axvline(x=silhouette_avg, color='red', linestyle='--')

ax1.set_yticks([])  # Clear the yaxis labels / ticks
ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

# 2nd Plot showing the actual clusters formed
# colors = cm.Spectral(cluster_labels.astype(float) / n_clusters)
# ax2.scatter(data2D[:, 0], data2D[:, 1], marker='.', s=30, lw=0, alpha=1, c=colors, edgecolor='k')

# # Labeling the clusters
# centers = clusterer.cluster_centers_
# # Draw white circles at cluster centers
# ax2.scatter(centers[:, 0], centers[:, 1], marker='o', c='white', alpha=1, s=200, edgecolor='k')

# for i, c in enumerate(centers):
#     ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50, edgecolor='k')

# ax2.set_title('Clusters')
# ax2.set_xlabel('Feature space for the 1st feature')
# ax2.set_ylabel('Feature space for the 2nd feature')

plt.suptitle(('Silhouette Analysis for k = %d' % n_clusters),
                fontsize=14, fontweight='bold')

plt.show()