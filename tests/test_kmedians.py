import pytest
from sklearn.datasets import make_blobs
import numpy as np
from src.kMedians import kmedians
from scipy.spatial.distance import euclidean

def _euclidean_dist(self, x, y):
    return euclidean(x, y) ** 2

def _check_validity(X, ):
    '''
    intra = intra-class compactness measure
    inter = inter-cluster distance
    :return: validity = ratio between intra- and inter- class dissimilarity
    '''

    class_compactness = [np.sum([_euclidean_dist(x) for x in cluster]) for cluster in
                         self.labels]  # check self.labels/ replace
    class_compactness = np.sum(class_compactness)
    intra = class_compactness / len(X)

    median_distances = list(set(self.cluster_centers_))
    distances = [[_euclidean_dist(x, median_distances[j]) for j in range(k + 1, self.k)] for k, x in
                 enumerate(median_distances)]

    inter = np.argmin(distances, axis=0)

    validity = intra / inter
    return intra, inter, validity


def test_validity():

def test_k_medians():
    n_samples = 100
    n_features = 2
    X, y = make_blobs(100, 2)
    k_medians = kmedians(n_features)
    k_medians = k_medians.fit(X)
    # clusters =



