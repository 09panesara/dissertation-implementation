import pytest
from sklearn.datasets import make_blobs
import numpy as np
from src.kMedians import kmedians
from scipy.spatial.distance import euclidean
from itertools import combinations

def _set_multid(x):
    return set(tuple(item) for item in x)

def _euclidean_dist(x, y):
    return euclidean(x, y) ** 2

def _check_validity(X, cluster_centers):
    '''
    intra = intra-class compactness measure
    inter = inter-cluster distance
    :return: validity = ratio between intra- and inter- class dissimilarity
    '''

    intra = np.sum([_euclidean_dist(x, cluster_centers[i]) for i, x in enumerate(X)]) / len(X)

    median_distances = list(_set_multid(cluster_centers))
    distances = [_euclidean_dist(x, median_distances[j]) for k, x in
                 enumerate(median_distances) for j in range(k + 1, len(median_distances)) ]
    inter = min(distances)

    validity = intra / inter
    return intra, inter, validity

@pytest.mark.skip(reason="Speedup tests by ignoring")
def test_validity():
    X = [[0.5, 1.5], [3.0, 6.0], [2.0, 5.5], [1.2, 1.2]]
    cluster_centers = [[0.5, 1.5], [4.0, 5.0], [4.0, 5.0], [0.5, 1.5]]
    intra, inter, validity = _check_validity(X, cluster_centers)
    assert intra == 1.7075
    assert inter == 24.5
    assert validity == 683/9800

    ''' Test with > 2 centers that inter works '''

    X =               [[0.5, 1.5], [3.0, 6.0], [2.0, 5.5], [1.2, 1.2], [10.0, 12], [9, 14]]
    cluster_centers = [[0.5, 1.5], [4.0, 5.0], [4.0, 5.0], [0.5, 1.5], [10.0, 12], [10.0, 12]]
    intra, inter, validity = _check_validity(X, cluster_centers)
    assert round(intra, 3) == round(1183/600, 3)
    assert round(inter, 3) == round(49/2, 3)
    assert round(validity, 3) == round(169 / 2100, 3)


def _in_same_cluster(x, y, ground_truth):
    for cluster in ground_truth:
        for pair in cluster:
            if (x in pair) and (y in pair):
                return True
    return False

def _accuracy_metrics(predicted, ground_truth):
    '''
    precision = tp / (tp+fp)
    recall = tp / (tp+fn)
    f1score = 2 * precision * recall / (precision + recall)
    '''

    '''
    To evaluate the clustering results, precision, recall, and F-measure were calculated over pairs of points. 
    For each pair of points that share at least one cluster in the overlapping clustering results, 
    these measures try to estimate whether the prediction of this pair as being in the same cluster was correct 
    with respect to the underlying true categories in the data. Precision is calculated as the fraction of pairs correctly 
    put in the same cluster, recall is the fraction of actual pairs that were identified, 
    and F-measure is the harmonic mean of precision and recall.'''

    predicted_pairs = [list(combinations(cluster, 2)) for cluster in predicted]
    gt_pairs = [list(combinations(cluster, 2)) for cluster in ground_truth]

    TP_PLUS_FP = np.sum([len(cluster) for cluster in predicted_pairs])
    TP = np.sum([np.sum([1 for x, y in pairs for pairs in predicted_cluster if _in_same_cluster(x, y, predicted_cluster)]) for predicted_cluster in predicted])
    # TP = np.sum([np.sum([1 for x, y in predicted_cluster if _in_same_cluster(x, y, gt_pairs)]) for predicted_cluster in predicted_pairs])
    FP = TP_PLUS_FP - TP
    FN = np.sum([np.sum([1 for x, y in gt_cluster if not _in_same_cluster(x, y, predicted_pairs)]) for gt_cluster in gt_pairs])


    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1score = 2 * precision * recall / (precision + recall)
    return precision, recall, f1score




def test_k_medians():
    no_samples = 100
    no_features = 5
    no_centers = 4
    X, y = make_blobs(no_samples, no_features, centers=no_centers)
    k_medians = kmedians(no_features, no_samples)
    k_medians = k_medians.fit(X)
    predicted_clusters = k_medians._get_clusters(X)
    actual_clusters = np.array([[x for j, x in enumerate(X) if y[j]==i] for i in range(no_centers)])

    precision, recall, f1score = _accuracy_metrics(predicted_clusters, actual_clusters)
    assert precision >= 0.7
    assert recall >= 0.6
    assert f1score >= 0.5
    print(precision, recall, f1score)




