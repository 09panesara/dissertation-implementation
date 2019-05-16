import os
import numpy as np
from scipy.spatial.distance import euclidean
import ast
import pandas as pd

def _flatten_by_frame_by_row(df):
    '''
    :param df: dataframe for particular emotion, containing one row per video
    :return: Flattened numpy array containing [[[frame1],[frame2],...], [...],...]
    '''

    def _convert_to_list(s):
        try:
            return ast.literal_eval(s)
        except:
            return [float(item) for item in s]

    cols_to_ignore = ['emotion', 'subject', 'action']
    if 'intensity' in df.columns.values:
        cols_to_ignore.append('intensity')
    if 'fold' in df.columns.values:
        cols_to_ignore.append('fold')
    df = df.drop(cols_to_ignore, axis=1)

    split_arr = []
    for i, row in df.iterrows():  # Each row = different video for corresponding emotion
        no_frames = len(row['rise_sink'])
        if 'timestep_btwn_frame' in df.columns.values:
            timestep = row['timestep_btwn_frame']
            timestep = [timestep for i in range(no_frames)]
            row['timestep_btwn_frame'] = timestep
        arr = np.array(row)
        arr = [_convert_to_list(r) for r in arr]
        arr = list(zip(*arr))
        arr = [np.array(row) for row in arr]
        split_arr.append(arr)
    return np.array(split_arr)



def _soft_assignment(data):
    ''' Do soft assignment
        Default parameters = for train dataset
    '''
    dict_path = '../data/paco/LOSO/ale/merged_centers.npz'
    print('Dictionary path for merged centers is: ' + dict_path)
    global_dict = np.load(dict_path, encoding='latin1')

    thresh = global_dict['thresh']
    centers_emotions = global_dict['centers_emotions']
    global_dict = global_dict['merged_centers']
    output = {}
    print('Doing soft assignment on dataset')
    emotions = ['ang', 'hap', 'neu', 'sad']
    for emotion in emotions:
        print('Doing soft assignment for emotion: ' + emotion)
        df = data.loc[data['emotion'] == emotion]
        soft_assign_v = _flatten_by_frame_by_row(df)
        # over each video, over each frame, calculate o(t) = relative position of vector in space drawn by key poses at frame t
        # +1 to avoid div by 0 error
        o_t = [[[1 / ((euclidean(frame, ref_pose) + 1) ** 2) for ref_pose in global_dict] for frame in vid] for vid in
               soft_assign_v]
        total = 0
        correct = 0
        for vid in o_t:
            for frame in vid:
                total += 1
                max_d = np.argmax(frame)
                if emotion == centers_emotions[max_d]:
                    correct += 1

        o_t = [[[d_j / np.sum(frame) for d_j in frame] for frame in vid] for vid in o_t]
        output[emotion] = o_t

        print(correct, "/", total, "=", (correct / total))

    # np.savez_compressed(soft_assign_path, soft_assign=output, thresh=thresh)
    # print('Saving to ' + soft_assign_path + '.')
    # return output


df = pd.read_csv('../data/paco/LMA_features.csv').iloc[:, 1:]
subject = 'ale'
train = df.loc[df['subject']!=subject]
_soft_assignment(train)