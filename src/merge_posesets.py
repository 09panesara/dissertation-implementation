''' Merge emotion category posesets to eliminate redundant elements '''
import os
import numpy as np
import random
from sklearn import preprocessing

emotions = emotions = ['ang', 'fea', 'hap', 'sad', 'unt']
def get_centers(dir):
    print('Loading centers...')
    cluster_centers = [np.load(dir+'/clusters_' + emotion + '.npz', encoding='latin1') for emotion in emotions if os.path.isfile(dir+'/clusters_' + emotion + '.npz')]
    cluster_centers = [clusters['predicted_centers'] for clusters in cluster_centers]
    # Flatten
    cluster_centers = [center for emotion_center_set in cluster_centers for center in emotion_center_set]

    return _convert_np_arr_to_set(cluster_centers)

def _euclid_distance(c1, c2):
    c1 = np.array(c1)
    c2 = np.array(c2)
    return np.linalg.norm(c1-c2)

def _convert_np_arr_to_set(arr):
    arr = set([tuple(y) for y in arr])
    arr = [list(tpl) for tpl in arr]
    return arr

#TODO test
def merge_centers(centers, thresh):
    ''' Iteratively merge centroids with a distance less than thresh '''
    old_centers = set([])
    new_centers = _convert_np_arr_to_set(centers)
    print('Merging with threshold: ' + str(thresh))

    while old_centers != new_centers:
        old_centers = new_centers
        no_centers = len(old_centers)
        # choose center at random and eliminate other centroid at < thresh away
        removed = False
        picked = []
        while not(removed):
            if len(picked) == no_centers:
                break
            center = random.randint(1, no_centers)
            picked.append(center)
            distances = [i for i, c in enumerate(old_centers) if c != center and _euclid_distance(c, center) < thresh]
            if len(distances) > 0:
                rmv_index = random.choice(distances)
                new_centers = [center for center in old_centers if center != old_centers[rmv_index]]
                removed = True

    return new_centers




def _calc_dist(centers):
    dist = []
    for c1 in centers:
        for c2 in centers:
            if (c1 != c2):
                dist.append(_euclid_distance(c1, c2))
    print(np.mean(dist))


def find_thresh():
    dir = '../data/clusters'
    emotion_centers = get_centers(dir)
    thresholds = [i for i in range(500, 10000, 500)] # pick thresh = 6500
    # thresholds = [3.5, 4, 4.5, 5, 5.5, 6]
    dict_size = []
    for threshold in thresholds:
        merged = merge_centers(emotion_centers, threshold)
        dict_size.append(len(merged))
    file_path = dir + '/merge_thresh_emp.txt'
    with open(file_path, 'w') as file:
        file.write('thresh poseset_size\n')
        for i in range(len(thresholds)):
            file.write(str(thresholds[i]) + ' ' + str(dict_size[i]) + "\n")

if __name__ == '__main__':
    dir = '../data/clusters'
    emotion_centers = get_centers(dir)
    # emotion_centers = preprocessing.scale(emotion_centers)
    thresh = 6500
    new_centers = merge_centers(emotion_centers, thresh=thresh)
    print('Saving merged centers...')
    np.savez_compressed('../data/merged_centers.npz', merged_centers=new_centers, thresh=thresh)
    print('Done.')




