from kMedians import kmedians
import numpy as np
from utils import data_utils
import pandas as pd
import os

def flatten_by_frame(df):
    '''
    :param df: dataframe for particular emotion, containing one row per video
    :return: Flattened dataframe containing 1 row per frame
    '''
    cols_to_ignore = ['emotion', 'intensity', 'subject', 'action']
    df = df.drop(cols_to_ignore, axis=1)
    columns = df.columns.values
    split_df = pd.DataFrame(columns=columns)


    for i, row in df.iterrows():
        timestep = row['timestep_btwn_frame']
        no_frames = len(row['rise_sink'])
        timestep = [timestep for i in range(no_frames)]
        row['timestep_btwn_frame'] = timestep
        arr = np.array(row)
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
    file_path = '../data/clusters/clusters_' + emotion + '.npz'
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
        predicted_centers = clusters_data['predicted_centers']

    if visualise_clusters:
        print('Visualising clusters...')
        k_medians._visualise_clusters(clusters=predicted_clusters, plot_name='kmedians_' + emotion + '.png')



emotions = ['ang', 'fea', 'hap', 'sad', 'unt']
# neu = empty so ignore
LMA_train, LMA_test = data_utils.get_train_test_set()


for emotion in emotions:
    print('Finding clusters for emotion ' + emotion)
    df_path = '../data/train_' + emotion + '.h5'
    if not os.path.isfile(df_path):
        df = LMA_train.loc[LMA_train['emotion'] == emotion]
        df = flatten_by_frame(df)
        # Write pandas dataframe to compressed h5.py file
        df.to_hdf(df_path, key='df', mode='w')
    else:
        df = pd.read_hdf(df_path)
    df_arr = np.array(df)
    generate_lexicon(emotion, df_arr)

