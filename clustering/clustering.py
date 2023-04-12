import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
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


######################################
## Running KMeans on reduced matrix ##
######################################
kmeans = KMeans(init='k-means++', n_clusters=3, max_iter=1000).fit(data2D)


###########################
## Plotting the clusters ##
###########################
plt.scatter(data2D[:,0], data2D[:,1], c=kmeans.labels_, cmap='rainbow')
for i, txt in enumerate(ingredients_per_cuisine_df.index):
    plt.annotate(txt, (data2D[i,0], data2D[i,1]))
plt.show()
