import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Mall_Customers.csv')
X= np.array(dataset.iloc[:,[3,4]])

import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distances')

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5 , affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

plt.scatter(X[y_hc == 0,0], X[y_hc == 0,1], s= 100, c='red', label= 'Careful')
plt.scatter(X[y_hc == 1,0], X[y_hc == 1,1], s= 100, c='orange', label= 'Standard')
plt.scatter(X[y_hc == 2,0], X[y_hc == 2,1], s= 100, c='blue', label= 'Target')
plt.scatter(X[y_hc == 3,0], X[y_hc == 3,1], s= 100, c='pink', label= 'Careless')
plt.scatter(X[y_hc == 4,0], X[y_hc == 4,1], s= 100, c='yellow', label= 'Sensible')
plt.title('Clusters of clients')
plt.xlabel('Income')
plt.ylabel('Expenditure')
plt.legend()