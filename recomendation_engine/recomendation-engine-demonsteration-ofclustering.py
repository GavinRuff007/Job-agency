# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 21:57:08 2023

@author: LENOVO
"""

from sklearn.datasets import make_regression, make_classification, make_blobs
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import mpl_toolkits.mplot3d
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report


def ClusterIndicesNumpy(clustNum, labels_array): #numpy 
    return np.where(labels_array == clustNum)[0]

#generating a random database of useres so that we demonstrate that our algorithem works.
center_box = (3, 7) # defines the box that cluster centres are allowed to be in
standard_dev = 1 # defines the standard deviation of clusters
X, y = make_blobs(n_samples=500, centers=3, n_features=10, center_box=center_box , cluster_std=standard_dev,random_state=42)

data = pd.DataFrame({'X0': X[:, 0], 'X1': X[:, 1], 'y': y})
sns.scatterplot(data=data, x='X0', y='X1', hue='y')
plt.show() 

data_ = pd.DataFrame({'X0': X[:, 0], 'X2': X[:, 2], 'y': y})
sns.scatterplot(data=data_, x='X0', y='X2', hue='y')
plt.show() 

#make the model standard
scaler = StandardScaler()
X_std = scaler.fit_transform(X)
#elbow method
inertias = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(X_std)
    inertias.append(kmeans.inertia_)
plt.plot(range(1,11), inertias, marker='o')
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show() 

#training the model
model = KMeans(n_clusters = 4, n_init="auto")
model.fit(X_std)
y_gen = model.labels_
y_predict= model.fit_predict(X_std)
print(y_predict)  
data = pd.DataFrame({'X0': X_std[:, 0], 'X1': X_std[:, 1], 'y': y})
sns.scatterplot(data=data, x='X0', y='X1', hue='y')
plt.show() 

model.cluster_centers_

#3d picture of model
estimators = [
    ("k_means_4", model)]
fig = plt.figure(figsize=(15, 15))
titles = ["4 clusters"]
for idx, ((name, est), title) in enumerate(zip(estimators, titles)):
    ax = fig.add_subplot(2, 2, idx + 1, projection="3d", elev=48, azim=134)
    est.fit(X_std)
    labels = est.labels_
    ax.scatter(X_std[:, 3], X_std[:, 0], X_std[:, 2], c=labels.astype(float), edgecolor="k")
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])
    ax.set_xlabel("Petal width")
    ax.set_ylabel("Sepal length")
    ax.set_zlabel("Petal length")
    ax.set_title(title)

plt.show()

X_train, X_test, y_train, y_test = train_test_split(X_std, y_predict, stratify=y, random_state=0)

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


error = []
# Calculating the error rate for K-values between 1 and 30
for i in range(1, 60):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))

classifier = KNeighborsClassifier(n_neighbors=11)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred))


plt.figure(figsize=(12, 5))
plt.plot(range(1, 60), error, color='red', marker='o',
        markerfacecolor='yellow', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')


random_karjo = [[5,5,6,7,8,9,7,3,2,1]]

random_karjo = scaler.transform(random_karjo)

y_pred_ = classifier.predict(random_karjo)
print(y_pred_)

print(X[ClusterIndicesNumpy(2,y_gen)])