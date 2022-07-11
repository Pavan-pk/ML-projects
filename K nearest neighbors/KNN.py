#!/usr/bin/env python
# encoding: utf-8

__author__ = "Pavan Kumar Raja"
__version__ = "1.0"

import math
import matplotlib.pyplot as plt
import numpy as np
import random
import scipy.io
from collections import defaultdict
from tqdm import tqdm


class KMeansClustering:
    def __init__(self, dataset, initializer='1', title=""):
        self.dataset = dataset
        self.initializer = initializer
        self.k_objective_dict = {}
        self.graph_title = title

    def get_euclidean_distance(self, p, q):
        """
        Calculate euclidean distance of 2 point in d dimensional space.
        :param p: Numpy array of dimension d.
        :param q: Numpy array of dimension d.
        :return: Euclidean distance of point p to q.
        """
        return sum([(p1 - q1) ** 2 for p1, q1 in zip(p, q)]) ** 0.5

    def get_avg_eculidean_distance(self, p, list_q):
        """
        Calculate average distance of a point to a list of points in d dimensional space.
        :param p: Numpy array of dimension d.
        :param list_q: list of Numpy arrays of dimension d.
        :return: Average euclidean distance of p to all point in list_q.
        """
        return np.mean([self.get_euclidean_distance(p, q) for q in list_q])

    def init_strategy(self, k):
        """
        Given K and the strategy the class is initialized to,
        Initialize the initial centroids accordingly.
        If strategy is 1, Initialize centroids as random K points.
        If strategy is 2, Initialize 1st centroid randomly and following centroids are initialized such that
        it has maximum average distance to all the centroids initialized so far.
        :param k: Number of clusters.
        :return: centroids initialized to class strategy.
        """
        centroids = []
        if self.initializer == '1':
            while len(centroids) < k:
                centroids.append(self.dataset[random.randrange(len(self.dataset))])
        elif self.initializer == '2':
            centroids.append(self.dataset[random.randrange(len(self.dataset))])
            while len(centroids) < k:
                max_dist = float('-inf')
                considered = None
                for data_point in self.dataset:
                    if data_point in centroids:
                        continue
                    dist = self.get_avg_eculidean_distance(data_point, centroids)
                    if dist > max_dist:
                        max_dist = dist
                        considered = data_point
                centroids.append(considered)
        return centroids

    def run(self, max_k):
        """
        Given maximum number of clusters,
        1. Starting from 2 to max_K, find the clusters and objective function value for the dataset.
        2. Visualize K vs objective function value graph.
        :param max_k: Maximum number of clusters
        """
        for k in range(2, max_k + 1):
            centroids = self.init_strategy(k)
            clusters = None
            for itr in range(1000):
                clusters = [[] for _ in range(len(centroids))]
                for data in dataset:
                    dists = {self.get_euclidean_distance(centroids[idx], data): idx for idx in range(len(centroids))}
                    clusters[dists[sorted(dists)[0]]].append(data)
                old_centroids = centroids.copy()
                for idx, cluster in enumerate(clusters):
                    if cluster:
                        centroids[idx] = np.mean(cluster, axis=0)
                if not any([np.sum(abs((new - old) / old)) > 0.001 for new, old in zip(centroids, old_centroids)]):
                    break

            objectives = np.sum((clusters[0] - centroids[0]) ** 2)
            for i in range(1, len(centroids)):
                objectives += np.sum((clusters[i] - centroids[i]) ** 2)
            self.k_objective_dict[k] = np.sum(objectives)
        self.visualize()

    def visualize(self):
        """
        Visualize the {k:objective function value} dictionary.
        """
        k, obj = zip(*sorted(self.k_objective_dict.items()))
        plt.plot(k, obj)
        plt.xlabel("Clusters (K)")
        plt.ylabel("Objective function")
        plt.suptitle(self.graph_title)
        plt.title("Clusters(k) vs Objective function value")
        plt.show()


if __name__ == "__main__":
    """
    Driver function
    Note: Graphs are popped up in sequence, so to view next graph we have close the previous one.
    """
    dataset = scipy.io.loadmat('k-means_dataset.mat')["AllSamples"]
    k_means11 = KMeansClustering(dataset, initializer='1', title="Strategy 1, Run 1")
    k_means12 = KMeansClustering(dataset, initializer='1', title="Strategy 1, Run 2")
    k_means21 = KMeansClustering(dataset, initializer='1', title="Strategy 2, Run 1")
    k_means22 = KMeansClustering(dataset, initializer='1', title="Strategy 2, Run 2")
    # Should close a figure for next one to pop up, everything isn't showing up together.
    k_means11.run(10)
    k_means12.run(10)
    k_means21.run(10)
    k_means22.run(10)
