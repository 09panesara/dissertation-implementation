from kmedians import kmedians
import numpy as np
from utils import data_utils
import pandas as pd
import os
from merge_posesets import merge_centers, get_centers
import ast
import numbers
from pyclustering.cluster import kmedians as pyclkmedians
from pyclustering.utils import draw_clusters;

from sklearn.utils import check_random_state
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from utils.data_utils import convert_to_list

joints = [
    'Hip',
    'RHip',
    'RKnee',
    'RFoot',
    'LHip',
    'LKnee',
    'LFoot',
    'Spine',
    'Thorax',
    'Neck/Nose',
    'Head',
    'LShoulder',
    'LElbow',
    'LWrist',
    'RShoulder',
    'RElbow',
    'RWrist'
]

def flatten_by_frame(df, csv=False):
    '''
    :param df: dataframe for particular emotion, containing one row per video
    :return: Flattened dataframe containing 1 row per frame
    '''

    cols_to_ignore = ['emotion', 'subject', 'action']
    if 'intensity' in df.columns:
        cols_to_ignore.append('intensity')
    if 'fold' in df.columns:
        cols_to_ignore.append('fold')
    df = df.drop(cols_to_ignore, axis=1)
    columns = df.columns.values
    split_df = pd.DataFrame(columns=columns)

    for i, row in df.iterrows():
        arr = np.array(row)
        if csv:
            arr = [convert_to_list(r) for r in arr]
        arr = list(zip(*arr))
        arr = [np.array(row) for row in arr]
        df_temp = pd.DataFrame(arr, columns=columns)
        assert(len(df_temp.columns.values)==len(split_df.columns.values))
        split_df = split_df.append(df_temp)

    return split_df


def generate_lexicon(emotion, train, override_existing_clusters=False, visualise_clusters=True):
    '''
    Generates lexicon for reference poseset per emotion category
    :return:
    '''
    # obtain action sequences for emotion
    no_frames = len(train)

    # Check if need to carry out k-medians or clusters exist already
    file_path = '../data/paco/clusters/clusters_' + emotion + '.npz'
    k_medians = kmedians(8, no_frames, max_iter=50)
    if not os.path.isfile(file_path) or override_existing_clusters:
        # Carry out k-medians
        k_medians = k_medians.fit(train)
        predicted_clusters = k_medians._get_clusters(train)
        print('Saving clusters...')
        np.savez_compressed(file_path, predicted_clusters=predicted_clusters, predicted_centers=k_medians.cluster_centers_)
        print('Done.')
    else:
        clusters_data = np.load(file_path, encoding='latin1')
        predicted_clusters = clusters_data['predicted_clusters']

    if visualise_clusters:
        print('Visualising clusters...')
        k_medians._visualise_clusters(clusters=predicted_clusters, plot_name='kmedians_' + emotion + '.png', paco=True)


def generate_lexicon2(emotion, train, override_existing_clusters=False, visualise_clusters=True):
    # obtain action sequences for emotion
    no_frames = len(train)
    k = 8
    # Check if need to carry out k-medians or clusters exist already
    file_path = '../data/paco/clusters/clusters_' + emotion + '.npz'
    if not os.path.isfile(file_path) or override_existing_clusters:
        random_state = check_random_state(19)
        random_state = random_state.permutation(no_frames)[:k]
        initial_centers_ = train[random_state]
        k_medians = pyclkmedians.kmedians(data=train, initial_centers=initial_centers_)
        k_medians.process()
        print('Saving clusters...')
        np.savez_compressed(file_path, predicted_clusters=k_medians._get_clusters(),
                            predicted_centers=k_medians.get_medians())
        print('Done.')
    else:
        clusters_data = np.load(file_path, encoding='latin1')
        predicted_clusters = clusters_data['predicted_clusters']

    if visualise_clusters:
        print('Visualising clusters...')
        k_medians._visualise_clusters(clusters=predicted_clusters, plot_name='kmedians_' + emotion + '.png', paco=True)

def _visualise_clusters(clusters, plot_name, show_plot=False, paco=False):

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

    ax.scatter(principalDf.loc[:, 'principal component 1'], principalDf.loc[:, 'principal component 2'], c=labels,
               s=50)

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

def _visualise_data(X, plot_name, show_plot=False, paco=False):

    cmap = plt.cm.get_cmap('Pastel1', len(X)).colors
    labels = [[cmap[i] for j in range(len(cluster))] for i, cluster in enumerate(X)]
    labels = [l for label_arr in labels for l in label_arr]
    pca = PCA(n_components=2).fit(X)
    pca_2d = pca.transform(X)
    principalDf = pd.DataFrame(data=pca_2d
                               , columns=['principal component 1', 'principal component 2'])

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)

    ax.scatter(principalDf.loc[:, 'principal component 1'], principalDf.loc[:, 'principal component 2'], c=labels,
               s=50)

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


def generate_lexicon2(emotion, train, override_existing_clusters=False, visualise_clusters=True):
    # obtain action sequences for emotion
    no_frames = len(train)
    k = 40
    # Check if need to carry out k-medians or clusters exist already
    file_path = '../data/paco/clusters/clusters_' + emotion + '.npz'
    if not os.path.isfile(file_path) or override_existing_clusters:
        print('Doing clustering')
        random_state = check_random_state(36)
        random_state = random_state.permutation(no_frames)[:k]
        initial_centers_ = train[random_state]
        k_medians = pyclkmedians.kmedians(data=train, initial_centers=initial_centers_, ccore = True)
        k_medians.process()
        predicted_centers=k_medians.get_medians()
        predicted_clusters = k_medians.get_clusters()
        predicted_clusters = [[train[i] for i in cluster] for cluster in predicted_clusters]




        print('Saving clusters...')
        np.savez_compressed(file_path, predicted_clusters=predicted_clusters,
                            predicted_centers=predicted_centers)
        print('Done.')
    else:
        clusters_data = np.load(file_path, encoding='latin1')
        predicted_clusters = clusters_data['predicted_clusters']

    if visualise_clusters:
        print('Visualising clusters for emotion %s...' %(emotion))
        _visualise_clusters(clusters=predicted_clusters, plot_name='kmedians_' + emotion + '.png', paco=True)


PACO = False
if PACO:
    emotions = ['ang', 'hap', 'sad', 'neu']
else:
    emotions = ['ang', 'fea', 'hap', 'sad', 'neu', 'unt']



# LMA_train, LMA_test = data_utils.get_train_test_set(folder='../data/paco')

no_folds = 10

# for i in range(no_folds):
#     # df_path = '../data/paco/training/train_' + emotion + '.h5'
#     if PACO:
#         df_path = '../data/paco/10_fold_cross_val/LMA_features_test_fold_' + str(i) + '.csv'
#     else:
#         df_path = '../data/action_db/10_fold_cross_val/LMA_features_test_fold_' + str(i) + '.csv'
#     LMA_train = pd.read_csv(df_path).iloc[:, 1:]
#     for emotion in emotions:
#         print('Finding clusters for emotion ' + emotion)
#         df = LMA_train.loc[LMA_train['emotion'] == emotion]
#         df = flatten_by_frame(df, paco=True)
#         # Write pandas dataframe to compressed h5.py file
#         # df.to_hdf(df_path, key='df', mode='w')
#
#         df_arr = np.array(df)
#         print(df_arr)
#         # _visualise_data(df_arr, plot_name='data.png', paco=True)
#     #     generate_lexicon(emotion, df_arr)


# ''' Merge centers '''
# dir = '../data/paco/clusters'
# emotion_centers = get_centers(dir)
# # emotion_centers = preprocessing.scale(emotion_centers)
# thresh = 6500
# new_centers = merge_centers(emotion_centers, thresh=thresh)
# print('Saving merged centers...')
# np.savez_compressed('../data/paco/clusters/merged_centers.npz', merged_centers=new_centers, thresh=thresh)
# print('Done.')



# kpts = np.load('../data/action_db/3dpb_keypoints.npz', encoding='latin1')['positions_3d'].item()
# X = []
# for subject in kpts:
#     for emotion in kpts[subject]['walking']:
#         for intensity in kpts[subject]['walking'][emotion]:
#             for data in kpts[subject]['walking'][emotion][intensity]:
#                 for frame in data:
#                     flattened_frame = [i for f in frame for i in f]
#                     X.append(flattened_frame)
LMA_features = pd.read_hdf('../data/action_db/LMA_features.h5')
for emotion in emotions:
    df = LMA_features.loc[LMA_features['emotion']==emotion]
    X = flatten_by_frame(df, csv=False)
    print(len(X))
    pca = PCA(n_components=2).fit(X)
    pca_2d = pca.transform(X)
    principalDf = pd.DataFrame(data=pca_2d
                               , columns=['principal component 1', 'principal component 2'])

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)

    ax.scatter(principalDf.loc[:, 'principal component 1'], principalDf.loc[:, 'principal component 2'])

    plt.show()










