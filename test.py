import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import mglearn

from sklearn.datasets import load_iris
from sklearn.cluster import DBSCAN


iris = load_iris()
df = pd.DataFrame(iris.data)
df.columns= iris.feature_names
df["type"] = iris.target
df.head()
X = df[df.columns[:-1]]
y = iris.target
escala = MinMaxScaler()
x_escalada = escala.fit_transform(X)
pca =PCA(n_components=2) 
pca.fit(x_escalada)
fig = plt.figure(figsize=(20,6))
transformada = pca.transform(x_escalada)
dbscan = DBSCAN(eps= 0.08, min_samples=6)
dbscan.fit(transformada)
df["cluster"] = dbscan.labels_
plt.subplot(1,2,1)
mglearn.discrete_scatter(transformada[:,0], transformada[:,1], y )
plt.legend(iris.target_names, loc="best")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.subplot(1,2,2)
mglearn.discrete_scatter(transformada[:,0], transformada[:,1], df.cluster )
plt.legend(["Cluster %d"%i for i in sorted(df.cluster.unique())], loc="best")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()


df = pd.read_csv('data/All_Streaming_Shows.csv')
df['Content Rating'] = df['Content Rating'].str.replace('+','').fillna('-1')
df['Content Rating'] = df['Content Rating'].str.replace('all','0').astype(int)
df['IMDB Rating'].fillna(0, inplace=True)
df['No of Seasons'] = df['No of Seasons'].str.replace('Seasons','').str.replace('Season','').astype(int)
df['Crime'] = df['Genre'].str.contains('Crime').astype(int)
df['Drama'] = df['Genre'].str.contains('Drama').astype(int)
df['Action'] = df['Genre'].str.contains('Action').astype(int)
df['Comedy'] = df['Genre'].str.contains('Comedy').astype(int)
df['Fantasy'] = df['Genre'].str.contains('Fantasy').astype(int)
df['Animation'] = df['Genre'].str.contains('Animation').astype(int)
df['Disney'] = df['Streaming Platform'].str.contains('Disney').fillna(0).astype(int)
df['Netflix'] = df['Streaming Platform'].str.contains('Netflix').fillna(0).astype(int)
df['HBO'] = df['Streaming Platform'].str.contains('HBO').fillna(0).astype(int)
features = ['Content Rating', 'IMDB Rating','R Rating','No of Seasons']
df.head()
X = df[features].values
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=6, random_state=0)
kmeans.fit(X_scaled)
labels = kmeans.labels_
df['cluster'] = labels
tsne = TSNE()
X_embedded = tsne.fit_transform(X_scaled)
plt.figure(figsize=(10,8))
ax = sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=labels, legend='full', palette='muted')
title = 'TSNE'
plt.title(title)
plt.legend(title='Cluster')
plt.show()