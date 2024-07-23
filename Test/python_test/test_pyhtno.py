# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 14:04:31 2024

@author: LENOVO
"""
OMP_NUM_THREADS=2
import unittest
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

def ClusterIndicesNumpy(clustNum, labels_array):
    return np.where(labels_array == clustNum)[0]

class TestYourCode(unittest.TestCase):
    def setUp(self):
        # Set up any necessary data for your tests
        center_box = (3, 7)
        standard_dev = 1
        self.X, self.y = make_blobs(n_samples=500, centers=3, n_features=10, center_box=center_box, cluster_std=standard_dev, random_state=42)
        self.X_ratings, self.y = make_blobs(n_samples=500, centers=3, n_features=10, center_box=center_box, cluster_std=standard_dev, random_state=42)
        self.data = pd.DataFrame({'X0': self.X[:, 0], 'X1': self.X[:, 1], 'y': self.y})
        self.scaler = StandardScaler()
        self.X_std = self.scaler.fit_transform(self.X)
        self.model = KMeans(n_clusters=4, n_init="auto")
        self.model.fit(self.X_std)
        self.y_predict = self.model.labels_
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_std, self.y_predict, stratify=self.y_predict, random_state=0)

    def test_cluster_indices_numpy(self):
        labels_array = self.y_predict

        # Test cluster indices for cluster 0
        self.assertTrue(np.array_equal(ClusterIndicesNumpy(0, labels_array), np.where(labels_array == 0)[0]))

        # Test cluster indices for cluster 1
        self.assertTrue(np.array_equal(ClusterIndicesNumpy(1, labels_array), np.where(labels_array == 1)[0]))

        # Test cluster indices for cluster 2
        self.assertTrue(np.array_equal(ClusterIndicesNumpy(2, labels_array), np.where(labels_array == 2)[0]))

    def test_kmeans_elbow_method(self):
        inertias = []
        for i in range(1, 6):
            kmeans = KMeans(n_clusters=i)
            kmeans.fit(self.X_train)
            inertias.append(kmeans.inertia_)

        # Ensure that the number of inertias is equal to the range
        self.assertEqual(len(inertias), 5)

    def test_knearest_neighbors_classification(self):
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(self.X_train, self.y_train)
        y_pred = knn.predict(self.X_test)

        # Ensure that the number of predictions matches the number of test samples
        self.assertEqual(len(y_pred), len(self.y_test))

    def test_weighted_rating_calculation(self):
        random_karjo_baisc_ratings = np.array([1, 1, 3, 1, 1, 5, 9, 7, 2, 3])
        calculating_wight = np.zeros((500))

        for i in ClusterIndicesNumpy(0, self.y_predict):
            mean_x = np.mean(self.X_ratings[i])
            a_ = 0
            b_ = 0
            c_ = 0
            for j in range(self.X_ratings[i].size):
                if random_karjo_baisc_ratings[j] != 1:
                    kar = (random_karjo_baisc_ratings[j] - np.mean(random_karjo_baisc_ratings)) ** 2
                    a_ += kar
                    kar_x = (self.X_ratings[i][j] - mean_x) ** 2
                    b_ += kar_x
                    c = (random_karjo_baisc_ratings[j] - np.mean(random_karjo_baisc_ratings)) * (self.X_ratings[i][j] - mean_x)
                    c_ += c
            calculating_wight[i] = c_ / np.sqrt((a_ * b_))

        self.assertEqual(len(calculating_wight), len(self.y_predict))    

    # Test ClusterIndicesNumpy function with a single cluster
    def test_cluster_indices_numpy_negative_cluster_indices(self):
        labels_array = np.array([0] * 100)
        with self.assertRaises(ValueError):
            ClusterIndicesNumpy(-1, labels_array)

    # Test KMeans elbow method with floating-point cluster numbers
    def test_kmeans_elbow_method_floating_point_clusters(self):
        X_floating = np.random.rand(100, 10)
        inertias = []
        for i in range(1, 6):
            kmeans = KMeans(n_clusters=i)
            kmeans.fit(X_floating)
            inertias.append(kmeans.inertia_)
        self.assertEqual(len(inertias), 5)

    # Test KNearest Neighbors classification with a dataset containing a single class
    def test_knearest_neighbors_classification_single_class_dataset(self):
        X_single_class = np.random.rand(100, 10)
        y_single_class = np.zeros(100)
        X_train_single, X_test_single, y_train_single, y_test_single = train_test_split(X_single_class, y_single_class, test_size=0.2)
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train_single, y_train_single)
        y_pred_single = knn.predict(X_test_single)
        self.assertEqual(len(y_pred_single), len(y_test_single))

    # Test weighted rating calculation for a single cluster
    def test_weighted_rating_calculation_single_cluster(self):
        y_predict_single = np.zeros(100)
        random_karjo_baisc_ratings_single = np.random.randint(1, 10, size=10)
        calculating_wight_single = np.zeros(100)
        for i in ClusterIndicesNumpy(0, y_predict_single):
            mean_x = np.mean(self.X_ratings[i])
            a_ = 0
            b_ = 0
            c_ = 0
            for j in range(self.X_ratings[i].size):
                if random_karjo_baisc_ratings_single[j] != 1:
                    kar = (random_karjo_baisc_ratings_single[j] - np.mean(random_karjo_baisc_ratings_single)) ** 2
                    a_ += kar
                    kar_x = (self.X_ratings[i][j] - mean_x) ** 2
                    b_ += kar_x
                    c = (random_karjo_baisc_ratings_single[j] - np.mean(random_karjo_baisc_ratings_single)) * (self.X_ratings[i][j] - mean_x)
                    c_ += c
            calculating_wight_single[i] = c_ / np.sqrt((a_ * b_))
        self.assertEqual(len(calculating_wight_single), len(y_predict_single))

    # Test ClusterIndicesNumpy function with empty labels array
    def test_cluster_indices_numpy_empty_labels_array(self):
        labels_array_empty = np.array([])
        with self.assertRaises(ValueError):
            ClusterIndicesNumpy(0, labels_array_empty)

    # Test KMeans elbow method with a large number of clusters
    def test_kmeans_elbow_method_large_number_of_clusters(self):
        X_large_clusters = np.random.rand(100, 10)
        inertias = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i)
            kmeans.fit(X_large_clusters)
            inertias.append(kmeans.inertia_)
        self.assertEqual(len(inertias), 10)

    # Test KNearest Neighbors classification with an invalid test size
    def test_knearest_neighbors_classification_invalid_test_size(self):
        X_invalid_size = np.random.rand(100, 10)
        y_invalid_size = np.random.randint(0, 5, size=100)
        with self.assertRaises(ValueError):
            train_test_split(X_invalid_size, y_invalid_size, test_size=1.5)

    # Test weighted rating calculation for an empty cluster
    def test_weighted_rating_calculation_empty_cluster(self):
        y_predict_empty = np.array([])
        random_karjo_baisc_ratings_empty = np.array([])
        calculating_wight_empty = np.zeros(0)
        for i in ClusterIndicesNumpy(0, y_predict_empty):
            mean_x = np.mean(self.X_ratings[i])
            a_ = 0
            b_ = 0
            c_ = 0
            for j in range(self.X_ratings[i].size):
                if random_karjo_baisc_ratings_empty[j] != 1:
                    kar = (random_karjo_baisc_ratings_empty[j] - np.mean(random_karjo_baisc_ratings_empty)) ** 2
                    a_ += kar
                    kar_x = (self.X_ratings[i][j] - mean_x) ** 2
                    b_ += kar_x
                    c = (random_karjo_baisc_ratings_empty[j] - np.mean(random_karjo_baisc_ratings_empty)) * (self.X_ratings[i][j] - mean_x)
                    c_ += c
            calculating_wight_empty[i] = c_ / np.sqrt((a_ * b_))
        self.assertEqual(len(calculating_wight_empty), len(y_predict_empty))

if __name__ == '__main__':
    unittest.main()