import pandas as pd
import numpy as np
import os
from scipy.spatial.distance import euclidean
from pomegranate import *
import random
from collections import Counter
import simplejson
import matplotlib.pyplot as plt
import seaborn as sns

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

def _soft_assignment(dict_path='../../data/clusters/merged_centers.npz', data_fpath='../../data/training/train_data.h5', output_fname="train_soft_assign.npz"):
    ''' Do soft assignment
        Default parameters = for train dataset
    '''
    global_dict = np.load(dict_path, encoding='latin1')
    LMA_train = pd.read_hdf(data_fpath)

    thresh = global_dict['thresh']
    global_dict = global_dict['merged_centers']
    output = {}
    for emotion in emotions:
        df = LMA_train.loc[LMA_train['emotion'] == emotion]
        print('Doing soft assignment for emotion: ' + emotion)
        soft_assign_v = _flatten_by_frame_by_row(df)
        # over each video, over each frame, calculate o(t) = relative position of vector in space drawn by key poses at frame t
        o_t = [[[euclidean(frame, ref_pose) for ref_pose in global_dict] for frame in vid] for vid in soft_assign_v]
        o_t = [[[d_j/np.sum(frame) for d_j in frame] for frame in vid] for vid in o_t]
        output[emotion] = o_t

    file_path = os.path.dirname(data_fpath) + "/" + output_fname
    np.savez_compressed(file_path, soft_assign=output, thresh=thresh)
    print('Saving...Done.')
    return output


def get_soft_assignment(soft_assign):
    ''' Returns soft assignment files for data set '''
    data = [(emotion, soft_assign[emotion]) for emotion in soft_assign]
    data = [[row + [emotion] for row in emotion_data] for emotion, emotion_data in data]
    data = [row for emotion_data in data for row in emotion_data] # flatten
    random.shuffle(data)
    # Split into data and labels
    X = [row[:-1] for row in data]
    y = [row[-1] for row in data]

    return X, y



def get_train_test_soft_assign(train_fpath='../../data/training/train_soft_assign.npz', test_fpath='../../data/test/test_soft_assign.npz'):
    if not os.path.isfile(train_fpath):
        print("Soft assignment on train dataset")
        train_set = _soft_assignment()
    else:
        print("Loading emotion train sets...")
        train_set = np.load(train_fpath, encoding='latin1')['soft_assign'].item()
    train_X, train_y = get_soft_assignment(train_set)

    if not os.path.isfile(test_fpath):
        print("Soft assignment on test dataset")
        test_set = _soft_assignment(data_fpath=os.path.dirname(test_fpath) + '/test_data.h5', output_fname=test_fpath)
    else:
        print("Loading emotion test sets...")
        test_set = np.load(test_fpath, encoding='latin1')['soft_assign'].item()
    test_X, test_y = get_soft_assignment(test_set)

    return train_X, train_y, test_X, test_y

def train_model(train_X, train_y):
    ''' cross validate for HMM learning - TODO: after run once while HMM is doing learning, need to consider gender, intensity '''
    print('Training model')
    # multivariate GMM HMM
    model = HiddenMarkovModel.from_samples(MultivariateGaussianDistribution, n_components=5, X=train_X, algorithm='baum-welch', min_iterations=10)

    for i in range(5):
        print(model.states[i].distribution.parameters[0])

    model.fit(train_X, labels=train_y, algorithm='baum-welch', min_iterations=1500, verbose=True)  # TODO: check if labels are even used in BW?
    model.bake()

    # TODO: use log probabilities
    # Persist HMM
    model_json = HiddenMarkovModel.to_json(model)
    with open('../../models/HMM.json', 'w') as file:
        simplejson.dump(model_json, file)
    return model


def _top_3_classes(predictions):
    ''' Identify top 3 classes assigned '''
    counter = Counter(predictions)
    top_3 = counter.most_common(3)
    top_3 = [most_common[0] for most_common in top_3]
    top_3 = [emotions[most_common] for most_common in top_3] # Get in 'ang', etc form
    return top_3


def hmm_inference(model, test_X, test_y):
    RR_0 = {emotion: 0 for emotion in emotions}
    RR_1 = {emotion: 0 for emotion in emotions}
    RR_2 = {emotion: 0 for emotion in emotions}
    RR_cum = {emotion: 0 for emotion in emotions}
    I_0 = {emotion: 0 for emotion in emotions}
    I_1 = {emotion: 0 for emotion in emotions}
    I_2 = {emotion: 0 for emotion in emotions}
    I = {emotion:0 for emotion in emotions}

    print('Predicting emotion from model...')
    for i, vid in enumerate(test_X):
        gt = test_y[i]
        predictions = model.predict(vid)
        top_3 = _top_3_classes(predictions)
        I[gt] += 1
        if top_3[0] == gt:
            I_0[gt] += 1
        elif len(top_3) > 1 and top_3[1] == gt: # might only be 1 prediction across timeseries
            I_1[gt] += 1
        elif len(top_3) > 2 and top_3[2] == gt: # might be < 3 predicted emotions across timeseries
            I_2[gt] += 1

    # Calculate RR(0), RR(1), RR(2) for each emotion
    for emotion in emotions:
        RR_0[emotion] = I_0[emotion] / I[emotion]
        RR_1[emotion] = I_1[emotion] / I[emotion]
        RR_2[emotion] = I_2[emotion] / I[emotion]
        RR_cum[emotion] = RR_0[emotion] + RR_1[emotion] + RR_2[emotion]

    with open('../../data/model_output.txt', 'w') as f:
        print("Writing recognition results to data/model_output.txt...")
        f.write('Emotion,RR_0,RR_1,RR_2,RR_cum \n')
        for emotion in emotions:
            f.write(emotion + ',' + str(RR_0[emotion]) + "," + str(RR_1[emotion]) + "," + str(RR_2[emotion]) + "," + str(RR_cum[emotion]) + "\n")
        print('Done')

def hmm():
    train_X, train_y, test_X, test_y = get_train_test_soft_assign()

    # Load model
    if not os.path.isfile('../../models/HMM.json'):
        model = train_model(train_X, train_y)
    else:
        with open('../../models/HMM.json', 'rb') as f:
            model = simplejson.load(f)
            model = HiddenMarkovModel.from_json(model)

    # Test model
    hmm_inference(model, test_X, test_y)

def plot_recog_rate():
    full_emotion_names = {'ang': 'anger', 'fea': 'fear', 'hap': 'happiness', 'sad': 'sadness', 'unt': 'untrustworthiness'}
    d = {'emotion': [], 'Key': [], 'recognition rate (%)': []}
    with open('../../data/model_output.txt', 'r') as f:
        next(f)
        for line in f:
            recog_rates = line.split(",")
            emotion = full_emotion_names[recog_rates[0]]
            d['emotion'].append(emotion)
            d['Key'].append('recognition rates at position 1')
            d['recognition rate (%)'].append(float(recog_rates[1])*100)
            d['emotion'].append(emotion)
            d['Key'].append('recognition rates at position 2')
            d['recognition rate (%)'].append(float(recog_rates[2])*100)
            d['emotion'].append(emotion)
            d['Key'].append('recognition rates at position 3')
            d['recognition rate (%)'].append(float(recog_rates[3])*100)
            d['emotion'].append(emotion)
            d['Key'].append('cumulative recognition rates at pos 1,2,3')
            d['recognition rate (%)'].append(float(recog_rates[4])*100)

    df = pd.DataFrame(data=d)
    sns.factorplot(x='emotion', y='recognition rate (%)', hue='Key', data=df, kind='bar')
    sns.despine(offset=10, trim=True)
    print('Saving plot...')
    plt.savefig('../../plots/recog_rate.png')
    plt.show()



if __name__ == '__main__':
    emotions = ['ang', 'fea', 'hap', 'sad', 'unt']
    # hmm()
    plot_recog_rate()







