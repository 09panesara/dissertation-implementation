''' Merge emotion category posesets to eliminate redundant elements '''
import numpy as np
import random
from collections import Counter
from sklearn import preprocessing


def get_centers(emotions, dir, keep_by_emotion=False):
    ''' keep_by_emotion = whether to keep separated by emotion '''
    print('Loading centers from dir ' + dir + '...')
    cluster_centers = []
    for emotion in emotions:
        centers_file = dir+'/clusters_' + emotion + '.npz'
        cluster_centers.append(np.load(centers_file, encoding='latin1'))
    # cluster_centers = [np.load(dir+'/clusters_' + emotion + '.npz', encoding='latin1') for emotion in emotions if os.path.isfile(dir+'/clusters_' + emotion + '.npz')]
    cluster_centers = [clusters['predicted_centers'] for clusters in cluster_centers]
    # Flatten
    if keep_by_emotion:
        return cluster_centers
    else:
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
def merge_centers_helper(centers, thresh):
    ''' Iteratively merge centroids with a distance less than thresh '''
    old_centers = set([])
    new_centers = _convert_np_arr_to_set(centers)
    print('Merging with threshold: ' + str(thresh))
    i = 1
    while old_centers != new_centers:
        # print ('Iteration ' + str(i))
        i+= 1
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
            distance_indices = [i for i, c in enumerate(old_centers) if c != center and _euclid_distance(c, center) < thresh]
            if len(distance_indices) > 0:
                rmv_index = random.choice(distance_indices)
                new_centers = old_centers[:rmv_index] + old_centers[rmv_index+1:]
                # new_centers = [center for center in old_centers if center != old_centers[rmv_index]]
                removed = True

    return new_centers




def _calc_dist(centers):
    dist = []
    for c1 in centers:
        for c2 in centers:
            if (c1 != c2):
                dist.append(_euclid_distance(c1, c2))
    print(np.mean(dist))


def find_thresh(emotions, dir='../data/action_db/clusters'):
    emotion_centers = get_centers(emotions, dir)
    thresholds = [i for i in range(100000000, 400000000, 10000000)] # pick thresh = 6500 # for actions_db
    dict_size = []
    for threshold in thresholds:
        merged = merge_centers_helper(emotion_centers, threshold)
        dict_size.append(len(merged))
    file_path = dir + '/merge_thresh_exper.txt'
    with open(file_path, 'w') as file:
        file.write('thresh poseset_size\n')
        for i in range(len(thresholds)):
            file.write(str(thresholds[i]) + ' ' + str(dict_size[i]) + "\n")

def merge_centers(emotions, thresh=0, dir='../data/paco/train-test/clusters'):
    ''' Merge centers '''
    emotion_centers = get_centers(emotions, dir)

    new_centers = merge_centers_helper(emotion_centers, thresh=thresh)
    # get emotion category each merged center belongs to
    emotion_centers = get_centers(emotions, dir, keep_by_emotion=True)
    emotions_of_merged = ['' for c in new_centers]
    print(len(new_centers))
    for i, center in enumerate(new_centers):
        for j in range(len(emotions)):
            for k, c in enumerate(emotion_centers[j]):
                if (c == center).all():
                    emotions_of_merged[i] = emotions[j]

    counter = Counter(emotions_of_merged)
    print(counter)

    print('Saving merged centers...')
    np.savez_compressed(dir +"/../merged_centers.npz", merged_centers=new_centers, thresh=thresh,
                        centers_emotions=emotions_of_merged)
    print('Done.')


def experiment_merge_centers(emotions, thresh=0, dir='../data/paco/train-test/clusters'):
    ''' Merge centers '''
    emotion_centers = get_centers(emotions, dir)

    new_centers = merge_centers_helper(emotion_centers, thresh=thresh)
    # get emotion category each merged center belongs to
    emotion_centers = get_centers(emotions, dir, keep_by_emotion=True)
    emotions_of_merged = ['' for c in new_centers]
    print('thresh: ' + str(thresh))
    print('No. merged centers: ' + str(len(new_centers)))
    for i, center in enumerate(new_centers):
        for j in range(len(emotions)):
            for k, c in enumerate(emotion_centers[j]):
                if (c == center).all():
                    emotions_of_merged[i] = emotions[j]

    counter = Counter(emotions_of_merged)
    counter_arr = list(counter.values())
    print("Mean: " +str(np.mean(counter_arr)))
    print("Std: " + str(np.std(counter_arr)))
    print(counter)
    # latex table format\
    div_by = 100000 if 'paco' in dir else 1000000
    print(str(thresh/div_by) + " & " + str(len(new_centers)) + " & " + str(np.mean(counter_arr)) + " & " + str(round(np.std(counter_arr), 3)) + " \\\\")
    print("\n")



if __name__ == '__main__':
    emotions = ['ang', 'hap', 'neu', 'sad']
    # find_thresh(dir='../data/paco/clusters')
    # emotions = ['ang', 'fea', 'hap', 'sad', 'unt']
    path = '../data/action_db/clusters/v1.3'
    experiment_merge_centers(emotions, thresh=0, dir=path)
    experiment_merge_centers(emotions, thresh=1.5, dir=path)
    experiment_merge_centers(emotions, thresh=3, dir=path)
    experiment_merge_centers(emotions, thresh=4.5, dir=path)
    experiment_merge_centers(emotions, thresh=6, dir=path)
    experiment_merge_centers(emotions, thresh=7.5, dir=path)
    experiment_merge_centers(emotions, thresh=9, dir=path)






