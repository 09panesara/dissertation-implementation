import numpy as np
from sklearn.utils import check_random_state
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances
from scipy.spatial.distance import euclidean

class kMedians():
    def __init__(self, k, max_iter=100, random_state=0):
        self.k = k
        self.max_iter = max_iter
        self.random_state = random_state
        self.total_no_frames = 100 # TODO
        self.validity_thresh = 0.1

    def _e_step(self, X):
        ''' Assign points to clusters '''
        self.labels_ = euclidean(X, self.cluster_centers_).argmin(axis=1)

    def _m_step(self, X):
        ''' Calculate new centers from cluster median '''
        clusters_indices = [[i for i, c in enumerate(self.labels_) if self.cluster_centers_[i] == center] for center in set(self.cluster_centers_)]
        ''' Calculate median from each '''
        for cluster in clusters_indices:
            cluster_pts = [X[i] for i in cluster]
            median = min(map(lambda p1:(p1,sum(map(lambda p2:euclidean(p1,p2),cluster_pts))),cluster_pts), key = lambda x:x[1])[0]
            median_index = [index for index in cluster if index == median][0]
            self.labels_ = [median_index if i in cluster else label for i, label in enumerate(self.labels_)]

        self.cluster_centers_ = X[self.labels_]




    def fit(self, X, y=None):
        random_state = check_random_state(self.random_state)
        self.labels_ = random_state.permutation(self.total_no_frames)[:self.k]
        self.cluster_centers_ = X[self.labels_]

        ''' Iterate until reached maximum number of iterations or validity reaches desired threshold '''
        for i in range(self.max_iter):
            self._e_step(X)
            self._m_step(X)

            if self._check_validity():
                break

        return self

    def _euclidiean_dist(self, x, y):
        return euclidean(x, y)**2

    def _check_validity(self):
        '''
        intra = intra-class compactness measure
        inter = inter-cluster distance
        :return: validity = ratio between intra- and inter- class dissimilarity
        '''

        class_compactness = [np.sum([self._euclidean_dist(x) for x in cluster]) for cluster in self.labels] # check self.labels/ replace
        class_compactness = np.sum(class_compactness)
        intra =  class_compactness/self.total_no_frames

        median_distances = list(set(self.cluster_centers_))
        distances = [[self._euclidiean_dist(x, median_distances[j]) for j in range(k+1, self.k)] for k, x in enumerate(median_distances)]

        inter = np.argmin(distances, axis=0)

        validity = intra/inter
        return validity < self.validity_thresh

    def _get_clusters(self, X):
        clusters_indices = [[i for i, c in enumerate(self.labels_) if self.cluster_centers_[i] == center] for center in set(self.cluster_centers_)]
        clusters = [[X[i] for i in sublist] for sublist in clusters_indices]

        return clusters_indices, clusters
