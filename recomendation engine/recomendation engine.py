# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 08:00:16 2023

@author: thomas
"""
OMP_NUM_THREADS=2

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
import math

def ClusterIndicesNumpy(clustNum, labels_array): #numpy 
    return np.where(labels_array == clustNum)[0]

#generating a random database of useres so that we demonstrate that our algorithem works.
center_box = (3, 7) # defines the box that cluster centres are allowed to be in
standard_dev = 1 # defines the standard deviation of clusters
X, y = make_blobs(n_samples=500, centers=3, n_features=10, center_box=center_box , cluster_std=standard_dev,random_state=42)

X_ratings, y = make_blobs(n_samples=500, centers=3, n_features=10, center_box=center_box , cluster_std=standard_dev,random_state=42)

print(X_ratings[1])

data = pd.DataFrame({'X0': X[:, 0], 'X1': X[:, 1], 'y': y})
sns.scatterplot(data=data, x='X0', y='X1', hue='y')
plt.show() 

data_ = pd.DataFrame({'X0': X[:, 0], 'X2': X[:, 2], 'y': y})
sns.scatterplot(data=data_, x='X0', y='X2', hue='y')
plt.show() 

print(y)

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

X_train, X_test, y_train, y_test = train_test_split(X_std, y_predict, stratify=y_predict, random_state=0)

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
#we find which cluster the random user belongs to. 
print(y_pred_)



random_karjo = [[1,5,6,7,4,9,7,3,2,1],[5,6,7,3,4,5,3,4,8,9]]
#base is at one.
random_karjo_baisc_ratings = [1,1,3,1,1,5,9,7,2,3]
mean_karjo = np.mean(random_karjo_baisc_ratings)

##random_karjo = [[5,2,1,7,4,3,7,8,4,1],[5,6,7,3,4,5,3,4,8,9]]
#base is at one.
#random_karjo_baisc_ratings = [2,1,3,1,1,1.5,3,4,4,3]

calculating_wight = np.zeros((500))

random_karjo = scaler.transform(random_karjo)

y_pred_ = classifier.predict(random_karjo)
print(y_pred_[0])

for i in ClusterIndicesNumpy(y_pred_[0],y_gen):
    mean_x = np.mean(X_ratings[i]); 
    a_ = 0;
    b_ = 0;
    c_ = 0;
    for j in range(X_ratings[i].size):
     if(random_karjo_baisc_ratings[j]!=1):
         kar = (random_karjo_baisc_ratings[j]-mean_karjo)*(random_karjo_baisc_ratings[j]-mean_karjo)
         a_ += kar
         kar_x = (X_ratings[i][j]-mean_x)*(X_ratings[i][j]-mean_x)
         b_ +=  kar_x
         c = 1
         c *= (random_karjo_baisc_ratings[j]-mean_karjo)
         c *= (X_ratings[i][j]-mean_x)
         c_ += c
    calculating_wight[i] = c_/math.sqrt((a_*b_))



random_karjo_baisc_ratings_gussed = np.zeros((10))


for i in range(random_karjo_baisc_ratings_gussed.size):
    sum_ = 0;
    weights = 0;
    for j in ClusterIndicesNumpy(y_pred_[0],y_gen):
        mean = np.mean(X_ratings[j]);
        r = X_ratings[j][i];
        sum_ += (r - mean)*calculating_wight[j];
        weights += calculating_wight[j];
    to_add =   sum_/weights
    random_karjo_baisc_ratings_gussed[i] = mean_karjo + to_add

   
for i in range(random_karjo_baisc_ratings_gussed.size):
    if(random_karjo_baisc_ratings[i] ==1 ):
        random_karjo_baisc_ratings[i] = random_karjo_baisc_ratings_gussed[i]
        
print(random_karjo_baisc_ratings)   