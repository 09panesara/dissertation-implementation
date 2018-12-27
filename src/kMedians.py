import numpy as np
from sklearn.utils import check_random_state
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances

class kMedians():
    def __init__(self, k, max_iter=100, random_state=0, tol=1e-4):
        self.k = k
        self.max_iter = max_iter
        self.random_state = random_state
        self.tol = tol
        self.total_no_frames = 100 # TODO

    def _e_step(self, X):
        self.labels_ = manhattan_distances(X, self.cluster_centers_).argmin(axis=1)

    def _average(self, X):
        return np.median(X, axis=0)

    def fit(self, X, y=None):
        n_samples = X.shape[0]
        vdata = np.mean(np.var(X, 0))

        random_state = check_random_state(self.random_state)
        self.labels_ = random_state.permutation(n_samples)[:self.k]
        self.cluster_centers = X[self.labels_]

        for i in range(self.max_iter):
            centers_old = self.cluster_centers_.copy()

            self._e_step(X)
            self._m_step(X)

            if np.sum((centers_old - self.cluster_centers) ** 2) < self.tol * vdata:
                break

        return self

    def _euclidiean_dist(self, x, y=None):
        if y == None:
            y = self.cluster_centers
        return euclidean_distances(x, y, squared=True)

    def _check_validity(self):
        '''
        intra = intra-class compactness measure
        inter = inter-cluster distance
        :return: validity = ratio between intra- and inter- class dissimilarity
        '''

        class_compactness = [np.sum([self._euclidean_dist(x) for x in cluster]) for cluster in self.labels] # check self.labels/ replace
        class_compactness = np.sum(class_compactness)
        intra =  class_compactness/self.total_no_frames

        distances = [cluster.centroid for cluster in self.clusters]
        K = len(distances) # no clusters
        distances = [[self._euclidiean_dist(x, distances[j]) for j in range(k+1, K)] for k, x in enumerate(distances)]

        inter = np.argmin(distances, axis=0)

        validity = intra/inter
