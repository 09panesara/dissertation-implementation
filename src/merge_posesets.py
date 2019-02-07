''' Merge emotion category posesets to eliminate redundant elements '''
import os
import numpy as np
import random

emotions = emotions = ['ang', 'fea', 'hap', 'sad', 'unt']
def get_centers(dir):
    cluster_centers = [np.load(dir+'/cluster_' + emotion + '.npz', encoding='latin1') for emotion in emotions if os.path.isdir(dir+'/cluster_' + emotion + '.npz')]
    cluster_centers = [clusters['predicted_centers'] for clusters in cluster_centers]
    # Flatten
    cluster_centers = [center for emotion_center_set in cluster_centers for center in emotion_center_set]
    return cluster_centers

def _euclid_distance(c1, c2):
    c1 = np.array(c1)
    c2 = np.array(c2)
    return np.linal.norm(c1-c2)

#TODO test
def merge_centers(centers, thresh):
    ''' Iteratively merge centroids with a distance less than thresh '''
    old_centers = set([])
    new_centers = set(centers)


    while old_centers != new_centers:
        old_centers = new_centers
        no_centers = len(old_centers)
        # choose center at random and eliminate other centroid at < thresh away
        removed = False
        while (not removed):
            center = random.randint(1, len(old_centers))
            distances = [i for i, c in enumerate(old_centers) if c != center and _euclid_distance(c, center) < thresh]
            if len(distances) > 0:
                rmv_index = random.choice(distances)
                new_centers = [center for center in old_centers if center != old_centers[rmv_index]]
            else:
                removed = True
    return new_centers







def find_thresh(posesets):
    dir = '../data/clusters'
    emotion_centers = get_centers(dir)


    thresholds = [3.5, 4, 4.5, 5, 5.5, 6]
    dict_size = []
    for threshold in thresholds:
        merged = merge_centers(emotion_centers, threshold)
        dict_size.append(len(merged))
    file_path = dir + '/merge_thresh_emp.txt'
    with open(file_path, 'w') as file:
        file.write('thresh poseset_size\n')
        for i in range(len(thresholds)):
            file.write(str(thresholds[i]) + ' ' + str(len(merged[i])) + "\n")


