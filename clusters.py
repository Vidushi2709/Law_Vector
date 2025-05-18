import numpy as np
import pandas as pd
import ast 
from sklearn.cluster import KMeans
import umap.umap_ as umap


data1= pd.read_csv('cases_with_embeddings.csv')
data1['embedding'] = data1['embedding'].apply(ast.literal_eval)

X = np.array(data1['embedding'].tolist())
X_umap = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine').fit_transform(X)

kmeans = KMeans(n_clusters=10, random_state=42).fit(X_umap)

data1["kmeans_cluster"] = kmeans.labels_
data1.to_csv('cases_with_clusters.csv', index=False)

import pandas as pd
km= pd.read_csv("cases_with_clusters.csv")

from sklearn.metrics import silhouette_score
score = silhouette_score(X_umap, km["kmeans_cluster"])
print("Silhouette Score:", score)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(X_umap[:, 0], X_umap[:, 1], c=km["kmeans_cluster"], cmap='tab10')
plt.colorbar(label='Cluster ID')
plt.title("Legal Case Clusters")
plt.show()