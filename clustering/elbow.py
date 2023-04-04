import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
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


##########################################################
## Running KMeans on TFIDF matrix and storing SSE score ##
##########################################################
sse = {}
for k in range(1, 21):
    kmeans = KMeans(init='k-means++', n_clusters=k, max_iter=1000).fit(data2D)
    sse[k] = kmeans.inertia_


#############################
## Plotting the SSE scores ##
#############################
plt.axvline(x=3, color='r', linestyle='--')
plt.plot(list(sse.keys()), list(sse.values()), marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of cluster')
plt.legend(['Elbow Point (k=3)', 'SSE'], reverse=True)
plt.ylabel('SSE')
plt.xticks(list(sse.keys()))
plt.show()
