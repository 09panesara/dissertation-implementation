import pandas as pd
import numpy as np
import os
from scipy.spatial.distance import euclidean
from pomegranate import *
import random
from sklearn.externals import joblib

def _flatten_by_frame_by_row(df):
    '''
    :param df: dataframe for particular emotion, containing one row per video
    :return: Flattened numpy array containing [[[frame1],[frame2],...], [...],...]
    '''
    cols_to_ignore = ['emotion', 'intensity', 'subject', 'action']
    df = df.drop(cols_to_ignore, axis=1)
    columns = df.columns.values
    split_df = pd.DataFrame(columns=columns)

    split_arr = []
    for i, row in df.iterrows(): # Each row = different video for corresponding emotion
        timestep = row['timestep_btwn_frame']
        no_frames = len(row['rise_sink'])
        timestep = [timestep for i in range(no_frames)]
        row['timestep_btwn_frame'] = timestep
        arr = np.array(row)
        arr = list(zip(*arr))
        arr = [np.array(row) for row in arr]
        split_arr.append(arr)
    return np.array(split_arr)

def soft_assignment(dict_path='../../data/clusters/merged_centers.npz', train_data_dir='../../data/training'):
    ''' Do soft assignment '''
    global_dict = np.load(dict_path, encoding='latin1')
    LMA_train = pd.read_hdf(train_data_dir + "/train_data.h5")

    thresh = global_dict['thresh']
    global_dict = global_dict['merged_centers']

    for emotion in emotions:
        df = LMA_train.loc[LMA_train['emotion'] == emotion]
        print('Doing soft assignment for emotion: ' + emotion)
        no_subjects = len(df.index)
        soft_assign_v = _flatten_by_frame_by_row(df)
        # over each video, over each frame, calculate o(t) = relative position of vector in space drawn by key poses at frame t
        o_t = [[[euclidean(frame, ref_pose) for ref_pose in global_dict] for frame in vid] for vid in soft_assign_v]
        o_t = [[[d_j/np.sum(frame) for d_j in frame] for frame in vid] for vid in o_t]
        file_path = train_data_dir + "/train_" + emotion + ".npz"
        np.savez_compressed(file_path, soft_assign=o_t, thresh=thresh)
        print('Saving...Done.')

def get_soft_assignment(train_data_dir='../../data/training'):
    ''' Returns soft assignment files '''
    print("Loading emotion train sets...")
    train_set = [(emotion, np.load(train_data_dir + '/train_' + emotion + '.npz')['soft_assign']) for emotion in emotions]

    train_set = [[row + [emotion] for row in emotion_data] for emotion, emotion_data in train_set]
    train_set = [row for emotion_data in train_set for row in emotion_data] # flatten
    random.shuffle(train_set)
    # Split into data and labels
    X = [row[:-1] for row in train_set]
    y = [row[-1] for row in train_set]

    return X, y



if __name__ == '__main__':
    emotions = ['ang', 'fea', 'hap', 'sad', 'unt']
    X, y = get_soft_assignment()
    ''' cross validate for HMM learning - TODO: after run once while HMM is doing learning, need to consider gender, intensity '''
    model = HiddenMarkovModel.from_samples(NormalDistribution, n_components=5, X=X, algorithm='baum-welch')
    model = model.fit(X, labels=y, algorithm='baum-welch') #TODO: check if labels are even used in BW?
    # TODO: use log probabilities
    # multivariate GMM HMM
    # Persist HMM

    joblib.dump(model, "../../models/HMM.pkl")



