import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

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


#######################################
## Calculating the Silhouette Scores ##
#######################################
silhouette_scores = []
for k in range(2, 20):
    kmeans = KMeans(init='k-means++', n_clusters=k, max_iter=1000).fit(data2D)
    silhouette_scores.append(silhouette_score(data2D, kmeans.labels_))


####################################
## Plotting the Silhouette Scores ##
####################################
plt.plot(range(2, 20), silhouette_scores, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.xticks(range(2, 20))
plt.title('Silhouette Scores')
plt.show()
