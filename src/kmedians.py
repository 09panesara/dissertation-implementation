import numpy as np
import pandas as pd
from sklearn.utils import check_random_state
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import euclidean
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

class kmedians():
    def __init__(self, k, no_frames, max_iter=100, random_state=30):
        self.k = k
        self.max_iter = max_iter
        self.random_state = random_state
        self.total_no_frames = no_frames
        self.change_in_validity_thresh = 0.02

    def _e_step(self, X):
        ''' Assign points to clusters '''
        print('E step: Assigning points to clusters')
        print(X.shape)
        self.labels_ = [[euclidean(pt,center) for center in self.cluster_centers_] for pt in X]
        print('Here')
        self.labels_ = self.labels_.argmin(axis=1)
        # self.labels_ = euclidean_distances(X, self.cluster_centers_).argmin(axis=1)

    def _m_step(self, X):
        ''' Calculate new centers from cluster median '''
        print('M step: Calculate new centers from cluster median')
        old_centers = list(set(tuple(row) for row in self.cluster_centers_))
        old_centers = [list(center) for center in old_centers]
        clusters_indices = [[c for i, c in enumerate(self.labels_) if (self.cluster_centers_[i] == center).all()] for center in old_centers]
        ''' Calculate median from each '''
        for cluster in clusters_indices:
            cluster_pts = [X[i] for i in cluster]
            median = min(map(lambda p1:(p1,sum(map(lambda p2:euclidean(p1,p2),cluster_pts))),cluster_pts), key = lambda x:x[1])[0]
            median_index = [index for index in cluster if (X[index] == median).all()][0]
            self.labels_ = [median_index if i in cluster else label for i, label in enumerate(self.labels_)]

        self.cluster_centers_ = X[self.labels_]




    def fit(self, X):
        print('Fitting kmedians to data')
        X = np.array(X)
        random_state = check_random_state(self.random_state)
        center_labels = random_state.permutation(self.total_no_frames)[:self.k]
        self.labels_ = np.array([random_state.choice(center_labels) for i in range(self.total_no_frames)])
        self.cluster_centers_ = X[self.labels_]
        old_validity = 0

        ''' Iterate until reached maximum number of iterations or change in validity is below desired threshold '''
        for i in range(self.max_iter):
            self._e_step(X)
            self._m_step(X)
            validity = self._check_validity(X)
            print(validity)
            print(abs(old_validity - validity))
            if abs(old_validity - validity) < self.change_in_validity_thresh:
                break
            else:
                old_validity = validity

        return self

    def _euclidean_dist(self, x, y):
        return euclidean(x, y)**2

    def _check_validity(self, X):
        '''
        intra = intra-class compactness measure
        inter = inter-cluster distance
        :return: validity = ratio between intra- and inter- class dissimilarity
        '''
        intra = np.sum([self._euclidean_dist(x, self.cluster_centers_[i]) for i, x in enumerate(X)])/self.total_no_frames

        median_distances = list(set(tuple(row) for row in self.cluster_centers_))
        median_distances = [list(tpl) for tpl in median_distances]
        if len(median_distances) > 1:
            distances = [self._euclidean_dist(x, median_distances[j]) for k, x in enumerate(median_distances) for j in range(k + 1, len(median_distances))]
            inter = min(distances)

            validity = intra/inter
            print('Current validity: %s' %(validity))
            return validity

        else: # Only one cluster, return maximum integer
            return float('inf')


    def _get_clusters(self, X):
        centers_set = list(set(tuple(row) for row in self.cluster_centers_))
        centers_set = [list(center) for center in centers_set]
        clusters_indices = [[i for i, c in enumerate(self.labels_) if (self.cluster_centers_[i] == center).all()] for center in centers_set]
        clusters = [np.array([X[i] for i in sublist]) for sublist in clusters_indices]

        return np.array(clusters)


    def _visualise_clusters(self, clusters, plot_name, show_plot=False, paco=False):
        print('Visualising clusters...')
        cmap = plt.cm.get_cmap('Pastel1', len(clusters)).colors
        labels = [[cmap[i] for j in range(len(cluster))] for i, cluster in enumerate(clusters)]
        labels = [l for label_arr in labels for l in label_arr]
        flat_clusters = [pt for cluster in clusters for pt in cluster]

        pca = PCA(n_components=2).fit(flat_clusters)
        pca_2d = pca.transform(flat_clusters)
        principalDf = pd.DataFrame(data=pca_2d
                                   , columns=['principal component 1', 'principal component 2'])

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('Principal Component 1', fontsize=15)
        ax.set_ylabel('Principal Component 2', fontsize=15)
        ax.set_title('2 component PCA', fontsize=20)

        ax.scatter(principalDf.loc[:,'principal component 1'], principalDf.loc[:, 'principal component 2'], c=labels, s=50)

        ax.legend(labels)
        ax.grid()
        print('Saving plot...')
        if paco:
            plt.savefig('../plots/paco_' + plot_name)
        else:
            plt.savefig('../plots/' + plot_name)
        if show_plot:
            plt.show()
        print('Done')


