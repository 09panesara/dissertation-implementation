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
import pandas as pd
import ast
import pomegranate as pmg
from pomegranate import *



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
    cols_to_ignore = ['emotion', 'subject', 'action', 'timestep_btwn_frame']
    if 'intensity' in df.columns.values:
        cols_to_ignore.append('intensity')
    df = df.drop(cols_to_ignore, axis=1)

    split_arr = []
    for i, row in df.iterrows(): # Each row = different video for corresponding emotion
        no_frames = len(row['rise_sink'])
        if 'timestep_btwn_frame' in df.columns.values:
            timestep = row['timestep_btwn_frame']
            timestep = [timestep for i in range(no_frames)]
            row['timestep_btwn_frame'] = timestep
        arr = np.array(row)
        if PACO:
            arr = [_convert_to_list(r) for r in arr]
        arr = list(zip(*arr))
        arr = [np.array(row) for row in arr]
        split_arr.append(arr)
    return np.array(split_arr)



def _soft_assignment(dict_path='../../data/action_db/clusters/merged_centers.npz', data_fpath='../../data/action_db/training/train_data.h5', output_fname="train_soft_assign.npz"):
    ''' Do soft assignment
        Default parameters = for train dataset
    '''

    if PACO:
        dict_path = '../../data/paco/clusters/merged_centers.npz'
        assert 'paco' in data_fpath
    global_dict = np.load(dict_path, encoding='latin1')
    if data_fpath.endswith('.h5'):
        LMA_train = pd.read_hdf(data_fpath)
    else:
        LMA_train = pd.read_csv(data_fpath).iloc[:, 1:]

    if RMV_EMOTIONS != []:
        for emotion in RMV_EMOTIONS:
            print('Discarding emotion' + emotion)
            LMA_train = LMA_train[LMA_train['emotion']!=emotion]
    thresh = global_dict['thresh']
    centers_emotions = global_dict['centers_emotions']
    global_dict = global_dict['merged_centers']
    output = {}
    print('Doing soft assignment on dataset: ' + data_fpath)
    for emotion in EMOTIONS:
        print('Doing soft assignment for emotion: ' + emotion)
        df = LMA_train.loc[LMA_train['emotion'] == emotion]
        soft_assign_v = _flatten_by_frame_by_row(df)
        # over each video, over each frame, calculate o(t) = relative position of vector in space drawn by key poses at frame t
        # +1 to avoid div by 0 error
        o_t = [[[1/((euclidean(frame, ref_pose)+1)**2) for ref_pose in global_dict] for frame in vid] for vid in soft_assign_v]
        total = 0
        correct = 0
        for vid in o_t:
            for frame in vid:
                total += 1
                max_d = np.argmax(frame)
                if emotion == centers_emotions[max_d]:
                    correct += 1

        o_t = [[[d_j/np.sum(frame) for d_j in frame] for frame in vid] for vid in o_t]
        output[emotion] = o_t

        print(correct , "/" , total , "=" , (correct/total))


    file_path = os.path.dirname(data_fpath) + "/" + output_fname
    np.savez_compressed(file_path, soft_assign=output, thresh=thresh)
    print('Saving to ' + file_path + '.')
    return output


def get_soft_assignment(soft_assign):
    ''' Returns soft assignment files for data set '''
    data = [(emotion, soft_assign[emotion]) for emotion in soft_assign]
    # store emotion so that can retain y_label after random shuffle of data
    data = [[row + [emotion] for row in emotion_data] for emotion, emotion_data in data]
    data = [row for emotion_data in data for row in emotion_data] # flatten
    random.shuffle(data)
    # Split into data and labels
    X = [row[:-1] for row in data]
    y = [row[-1] for row in data]
    return X, y



def get_train_test_soft_assign(train_fpath='../../data/action_db/training/train_soft_assign.npz', test_fpath='../../data/action_db/test/test_soft_assign.npz', override_soft_assign=False):
    if not os.path.isfile(train_fpath) or override_soft_assign:
        print("Soft assignment on train dataset")
        train_data_fpath = os.path.dirname(train_fpath) + '/train_data.h5'
        if not os.path.isfile(train_data_fpath):
            if os.path.isfile(os.path.dirname(train_fpath) + '/train_data.csv'):
                train_data_fpath = os.path.dirname(train_fpath) + '/train_data.csv'
        train_set = _soft_assignment(data_fpath=train_data_fpath, output_fname='train_soft_assign.npz')
    else:
        print("Loading emotion train sets...")
        train_set = np.load(train_fpath, encoding='latin1')
        train_set = train_set['soft_assign'].item()
    train_X, train_y = get_soft_assignment(train_set)

    if not os.path.isfile(test_fpath) or override_soft_assign:
        print("Soft assignment on test dataset")
        test_set = _soft_assignment(data_fpath=os.path.dirname(test_fpath) + '/test_data.h5', output_fname='test_soft_assign.npz')
    else:
        print("Loading emotion test sets...")
        test_set = np.load(test_fpath, encoding='latin1')
        test_set = test_set['soft_assign'].item()

    test_X, test_y = get_soft_assignment(test_set)
    return train_X, train_y, test_X, test_y

def hmm(override_model=False, override_soft_assign=False):
    if PACO:
        print('HMM for paco dataset')
        train_X, train_y, test_X, test_y = get_train_test_soft_assign(train_fpath='../../data/paco/training/train_soft_assign.npz', test_fpath='../../data/paco/test/test_soft_assign.npz', override_soft_assign=override_soft_assign)
    else:
        print('HMM for action db')
        train_X, train_y, test_X, test_y = get_train_test_soft_assign(override_soft_assign=override_soft_assign)
    # Load model
    hmm_json_dir = '../../models/paco' if PACO else '../../models/action_db'
    if not os.path.isdir(hmm_json_dir) or override_model:
        # hidden state sequence for each observation sequence
        # train_labels = _get_labels(train_X)
        # TODO: check_acc(train_y, train_labels)
        models = train_model(train_X, train_y, hmm_json_dir=hmm_json_dir)
    else:
        models = {emotion: None for emotion in EMOTIONS}
        for emotion in EMOTIONS:
            print('Loading model for emotion ' + emotion)
            with open(hmm_json_dir + "/model_"  + emotion + ".json", 'rb') as f:
                model = simplejson.load(f)
                model = pmg.HiddenMarkovModel.from_json(model)
                models[emotion] = model

    # Test model
    model_output_dir = '../../data/paco' if PACO else '../../data/action_db'
    hmm_inference(models, test_X, test_y, model_output_dir=model_output_dir)

def get_train_test(train_fpath='../../data/action_db/training/train_soft_assign.npz', test_fpath='../../data/action_db/test/test_soft_assign.npz'):
    return 0

def train_model(train_X, train_y, hmm_json_dir):
    ''' cross validate for HMM learning - TODO: after run once while HMM is doing learning, need to consider gender, intensity '''
    def _gen_model(data, emotion):
        model = pmg.HiddenMarkovModel.from_samples(pmg.NormalDistribution, n_components=8, X=data)
        print('Training HMM model')
        model.fit(data, algorithm='baum-welch', verbose=True)
        model.bake()
        # Persist HMM
        model_json = pmg.HiddenMarkovModel.to_json(model)
        print('Saving HMM model')
        with open(hmm_json_dir + "/model_"  + emotion + ".json", 'w') as file:
            simplejson.dump(model_json, file)
        model.plot()
        return model

    X = {emotion: [train_X[i] for i, y in enumerate(train_y) if y==emotion] for emotion in EMOTIONS}
    HMM_models = {emotion: _gen_model(X[emotion], emotion) for emotion in EMOTIONS}
    return HMM_models




def hmm_inference(models, test_X, test_y, model_output_dir='../../data/action_db', plot_recog_rate=True):
    RR_1 = {emotion: {emotion: 0 for emotion in EMOTIONS} for emotion in EMOTIONS}
    RR_2= {emotion: {emotion: 0 for emotion in EMOTIONS} for emotion in EMOTIONS}
    RR_cum = {emotion: {emotion: 0 for emotion in EMOTIONS} for emotion in EMOTIONS}
    gt_counts = {emotion: 0 for emotion in EMOTIONS}

    print('Predicting emotion from model...')

    for i, vid in enumerate(test_X):
        gt = test_y[i]
        print("Predicting emotion for gt " + gt)
        gt_counts[gt] += 1
        predictions = {emotion: models[emotion].viterbi(vid) for emotion in EMOTIONS}
        predictions = [predictions[emotion][0] for emotion in EMOTIONS]
        emotion_pred_0 = EMOTIONS[np.argmax(predictions)]
        print("Predicted emotion is: " + emotion_pred_0)
        predictions[np.argmax(predictions)] = 0
        emotion_pred_1 = EMOTIONS[np.argmax(predictions)]
        print("Second predicted emotion is: " + emotion_pred_1)


        RR_1[gt][emotion_pred_0] += 1
        RR_2[gt][emotion_pred_1] += 1

    for emotion1 in RR_1:
        for emotion2 in RR_1[emotion1]:
            RR_1[emotion1][emotion2] = RR_1[emotion1][emotion2] / gt_counts[emotion1]
            RR_2[emotion1][emotion2] = RR_2[emotion1][emotion2] / gt_counts[emotion1]
            RR_cum[emotion1][emotion2] = RR_1[emotion1][emotion2] + RR_2[emotion1][emotion2]

    # normalise RR_cum
    for emotion1 in RR_1:
        for emotion2 in RR_1[emotion1]:
            RR_cum[emotion1][emotion2] = RR_cum[emotion1][emotion2] / np.sum([RR_cum[emotion1][e] for e in EMOTIONS])

    print("Writing recognition results to " + model_output_dir + "/model_output.npz...")
    np.savez_compressed(model_output_dir + '/model_output.npz', RR_1=RR_1, RR_2=RR_2, RR_cum=RR_cum)
    print('Done')




def confusion_mat():
    n_emotions = len(EMOTIONS)
    if PACO:
        recog_rates = np.load('../../data/paco/model_output.npz', encoding='latin1')
    else:
        recog_rates = np.load('../../data/action_db/model_output.npz', encoding='latin1')
    RR_1 = recog_rates['RR_1'].item()
    RR_cum = recog_rates['RR_cum'].item()
    conf_mat_RR_1 = np.zeros((n_emotions,n_emotions))
    conf_mat_RR_cum = np.zeros((n_emotions,n_emotions))

    for i,emotion1 in enumerate(EMOTIONS):
        for j,emotion2 in enumerate(EMOTIONS):
            conf_mat_RR_1[i][j] = RR_1[emotion1][emotion2] # flip indices so that gt is on y axis in confusion matrix plot
            conf_mat_RR_cum[i][j] = RR_cum[emotion1][emotion2]

    df_RR_1 = pd.DataFrame(conf_mat_RR_1, index=EMOTIONS, columns=EMOTIONS)
    df_RR_cum = pd.DataFrame(conf_mat_RR_cum, index=EMOTIONS, columns=EMOTIONS)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,12))
    ax1.title.set_text('Recognition rate at position 1')
    sns.set(font_scale=1.4)  # for label size
    sns.heatmap(df_RR_1, annot=True, annot_kws={"size": 16}, ax=ax1)  # font size

    ax1.set_ylabel('Ground truth emotion')

    ax2.title.set_text('Cumulative recognition rates for position 1, 2')
    sns.set(font_scale=1.4)  # for label size
    sns.heatmap(df_RR_cum, annot=True, annot_kws={"size": 16}, ax=ax2)  # font size
    ax2.set_ylabel('Ground truth emotion')

    plt.show()




def get_emotion_hidden_state_series(X, y):
    def _kth_most_common(lst, k):
        data = Counter(lst)
        return data.most_common(k)[k - 1][0]

    def _get_labels(X, merged_centers_fpath):
        ''' get labels of emotion of cluster each frame in obs is closest to '''
        centers_emotions = np.load(merged_centers_fpath, encoding='latin1')['centers_emotions']

        labels = [[centers_emotions[np.argmax(obs)] for obs in x] for x in X]
        return labels

    if PACO:
        labels = _get_labels(X, '../../data/paco/clusters/merged_centers.npz')
    else:
        labels = _get_labels(X, '../../data/action_db/clusters/merged_centers.npz')
    for i, gt in enumerate(y):
        print(gt)
        print(labels[i])

    top_emotion = [_kth_most_common(y_lbls,1) for y_lbls in labels]
    second_emotion = [_kth_most_common(y_lbls, 1) for y_lbls in labels]

    gt_counts = Counter(y)

    RR_1 = {emotion: 0 for emotion in EMOTIONS}
    RR_2 = {emotion: 0 for emotion in EMOTIONS}

    for emotion in EMOTIONS:
        top_correct = [1 for i, em in enumerate(top_emotion) if em == y[i] and y[i]==emotion]
        second_correct = [1 for i, em in enumerate(second_emotion) if em == y[i] and y[i]==emotion]

        RR_1[emotion] = np.sum(top_correct) / gt_counts[emotion]
        RR_2[emotion] = np.sum(second_correct) / gt_counts[emotion]
    print(RR_1)
    print(RR_2)


if __name__ == '__main__':
    # paco db
    PACO=True
    RMV_EMOTIONS = []
    EMOTIONS = ['ang', 'fea', 'hap', 'sad']
    hmm(override_model=True, override_soft_assign=False)
    confusion_mat()

    # actions db
    # paco=False
    # RMV_EMOTIONS = []
    # EMOTIONS = ['ang', 'fea', 'hap', 'sad', 'unt']
    # hmm(override_model=True, override_soft_assign=False)
    # confusion_mat()

    # Test what emotions are predicted per state from cluster poses
    # PACO=True
    # RMV_EMOTIONS = []
    # EMOTIONS = ['ang', 'fea', 'hap', 'neu', 'sad']
    # train_X, train_y, test_X, test_y = get_train_test_soft_assign(
    #     train_fpath='../../data/paco/training/train_soft_assign.npz',
    #     test_fpath='../../data/paco/test/test_soft_assign.npz')

    # PACO=True
    # RMV_EMOTIONS = []
    # EMOTIONS = ['ang', 'fea', 'hap', 'sad', 'unt']
    # train_X, train_y, test_X, test_y = get_train_test_soft_assign(
    #     train_fpath='../../data/paco/training/train_soft_assign.npz',
    #     test_fpath='../../data/paco/test/test_soft_assign.npz', override_soft_assign=True)
    # get_emotion_hidden_state_series(train_X, train_y)






