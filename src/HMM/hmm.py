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

# TODO: clean up into class


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
    print('here')
    cols_to_ignore = ['emotion', 'subject', 'action', 'timestep_btwn_frame']
    if 'intensity' in df.columns.values:
        cols_to_ignore.append('intensity')
    df = df.drop(cols_to_ignore, axis=1)

    split_arr = []
    print('here')
    for i, row in df.iterrows(): # Each row = different video for corresponding emotion
        no_frames = len(row['rise_sink'])
        if 'timestep_btwn_frame' in df.columns.values:
            timestep = row['timestep_btwn_frame']
            timestep = [timestep for i in range(no_frames)]
            row['timestep_btwn_frame'] = timestep
        arr = np.array(row)
        if paco:
            arr = [_convert_to_list(r) for r in arr]
        arr = list(zip(*arr))
        arr = [np.array(row) for row in arr]
        split_arr.append(arr)
    print('here')
    return np.array(split_arr)



def _soft_assignment(dict_path='../../data/action_db/clusters/merged_centers.npz', data_fpath='../../data/action_db/training/train_data.h5', output_fname="train_soft_assign.npz"):
    ''' Do soft assignment
        Default parameters = for train dataset
    '''

    global_dict = np.load(dict_path, encoding='latin1')
    if data_fpath.endswith('.h5'):
        LMA_train = pd.read_hdf(data_fpath)
    else:
        LMA_train = pd.read_csv(data_fpath).iloc[:, 1:]

    if rmv_emotions != []:
        for emotion in rmv_emotions:
            print('Discarding emotion' + emotion)
            LMA_train = LMA_train[LMA_train['emotion']!=emotion]

    thresh = global_dict['thresh']
    global_dict = global_dict['merged_centers']
    output = {}
    print('Doing soft assignment on dataset: ' + data_fpath)
    for emotion in emotions:
        print('Doing soft assignment for emotion: ' + emotion)
        df = LMA_train.loc[LMA_train['emotion'] == emotion]
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
        train_set = np.load(train_fpath, encoding='latin1')['soft_assign'].item()
    train_X, train_y = get_soft_assignment(train_set)

    if not os.path.isfile(test_fpath) or override_soft_assign:
        print("Soft assignment on test dataset")
        test_set = _soft_assignment(data_fpath=os.path.dirname(test_fpath) + '/test_data.h5', output_fname='test_soft_assign.npz')
    else:
        print("Loading emotion test sets...")
        test_set = np.load(test_fpath, encoding='latin1')['soft_assign'].item()
    test_X, test_y = get_soft_assignment(test_set)
    train_X = [x[5:] for x in train_X]
    return train_X, train_y, test_X, test_y


def _GMM(train_X, train_y):
    ''' E-M algorithm to learn GMM parameters '''
    def _calc_cov(a):
        # compute a * aT
        return [np.multiply(a_i, a) for a_i in a]

    _epsilon = 0
    def _nearPostiiveSemiDefinite(A, epsilon=0):
        n = A.shape[0]
        eigval, eigvec = np.linalg.eig(A)
        val = np.matrix(np.maximum(eigval,epsilon))
        vec = np.matrix(eigvec)
        T = 1/(np.multiply(vec,vec) * val.T)
        T = np.matrix(np.sqrt( np.diag( np.array(T).reshape((n)) ) ))
        B = T * vec * np.diag(np.array(np.sqrt(val)).reshape((n)))
        out = B*B.T
        return out

    dists = [] # in order of emotions
    parameters = {}
    for emotion in emotions:
        print("Computing GMM parameters for emotion " + emotion)
        obs = [np.array(train_X[i]) for i, y in enumerate(train_y) if y==emotion]
        obs = [o_t for seq in obs for o_t in seq]
        mu_j = np.sum(obs, axis=0)/len(obs)
        cov_j = np.sum([_calc_cov(o_t-mu_j) for o_t in obs], axis=0) / len(obs)

        # cov_j =  np.array([_calc_cov(o_t-mu_j) for o_t in obs]) / len(obs)
        pd_cov_j = cov_j
        pos_def = False
        iteration = 1
        while not pos_def:
            print([iteration])
            try:
                L = np.linalg.cholesky(pd_cov_j)
                pos_def = True
                cov_j = pd_cov_j
            except:
                _epsilon = _epsilon + 0.0000001
                pd_cov_j = _nearPostiiveSemiDefinite(cov_j, _epsilon)
                iteration += 1

        # make cov matrix positive definite
        dists.append(pmg.MultivariateGaussianDistribution(mu_j, cov_j))
        parameters[emotion] = [mu_j, cov_j]
    return dists, parameters
    # GMM_model = pmg.GeneralMixtureModel.from_samples(pmg.MultivariateGaussianDistribution, n_components=4, X=train_X)



def train_model(train_X, train_y, hmm_json_fpath):
    ''' cross validate for HMM learning - TODO: after run once while HMM is doing learning, need to consider gender, intensity '''

    gmm_dists, gmm_parameters = _GMM(train_X, train_y)

    # initialise initial and transition probabilities using modified k means clustering
    print('Initialising start and transition probabilities.')
    no_emotions = len(emotions)
    counts = {emotion: 0 for emotion in emotions}
    for y in train_y:
        counts[y] += 1

    trans_mat = np.diag([0.8]*no_emotions)
    trans_mat = [[0.2 / (no_emotions - 1) if i == 0 else i for i in row] for row in trans_mat]
    starts = [1/no_emotions for e in emotions]
    # starts = [counts[emotion]/len(train_y) for emotion in emotions]

    print('Saving initial parameters...')
    parameters_fpath = os.path.dirname(hmm_json_fpath) + "/paco_parameters.npz" if paco else os.path.dirname(hmm_json_fpath) + "/action_db_parameters.npz"
    np.savez_compressed(parameters_fpath, trans_mat=trans_mat, starts=starts,gmm_parameters=gmm_parameters)
    print('Done.')

    print('Training GMM HMM model')

    assert len(train_X) == len(train_y)
    model = pmg.HiddenMarkovModel.from_matrix(transition_probabilities=trans_mat, distributions=gmm_dists, starts=starts, state_names=emotions, verbose=True)
    weights_by_emotion = {emotion: 1/(len(train_y)*counts[emotion]) for emotion in emotions}
    weights = [weights_by_emotion[emotion] for emotion in train_y]
    labels = np.array([[y]*len(train_X[i]) for i,y in enumerate(train_y)])
    # assert labels.shape == train_X.shape
    # model.fit(train_X, labels=labels, algorithm='labeled', verbose=True)
    model.bake()

    # TODO: use log probabilities
    # Persist HMM
    model_json = pmg.HiddenMarkovModel.to_json(model)
    print('Saving HMM model')
    with open(hmm_json_fpath, 'w') as file:
        simplejson.dump(model_json, file)
    return model




def hmm_inference(model, test_X, test_y, model_output_dir='../../data/action_db', plot_recog_rate=True):
    def _top_k_classes(predictions, k):
        ''' Identify top k classes assigned '''
        counter = Counter(predictions)
        print(counter)
        top_k = counter.most_common(k)
        top_k = [most_common[0] for most_common in top_k]
        top_k = [k for k in top_k if k != len(emotions)]  # remove start state state = last state in transition matr
        print(top_k)
        top_k = [emotions[most_common] for most_common in top_k]  # Get in 'ang', etc form

        return top_k

    RR_1 = {emotion: {emotion: 0 for emotion in emotions} for emotion in emotions}
    RR_2= {emotion: {emotion: 0 for emotion in emotions} for emotion in emotions}
    RR_cum = {emotion: {emotion: 0 for emotion in emotions} for emotion in emotions}
    gt_counts = {emotion: 0 for emotion in emotions}
    print(model.node_count())
    model.plot()
    print('Predicting emotion from model...')

    for i, vid in enumerate(test_X):
        gt = test_y[i]
        print("Predicting emotion for gt " + gt)
        gt_counts[gt] += 1
        predictions = model.predict(vid, algorithm='baum-welch')
        print(predictions)
        top_2 = _top_k_classes(predictions, 2)

        RR_1[gt][top_2[0]] += 1
        if len(top_2) > 1:
            RR_2[gt][top_2[1]] += 1
    for emotion1 in RR_1:
        for emotion2 in RR_1[emotion1]:
            RR_1[emotion1][emotion2] = RR_1[emotion1][emotion2] / gt_counts[emotion1]
            RR_2[emotion1][emotion2] = RR_2[emotion1][emotion2] / gt_counts[emotion1]
            # RR_2[emotion1][emotion2] = RR_2[emotion1][emotion2] / gt_counts[emotion1]
            RR_cum[emotion1][emotion2] = RR_1[emotion1][emotion2] + RR_2[emotion1][emotion2]

    # normalise RR_cum
    for emotion1 in RR_1:
        for emotion2 in RR_1[emotion1]:
            RR_cum[emotion1][emotion2] = RR_cum[emotion1][emotion2] / np.sum([RR_cum[emotion1][e] for e in emotions])

    print("Writing recognition results to " + model_output_dir + "/model_output.npz...")
    np.savez_compressed(model_output_dir + '/model_output.npz', RR_1=RR_1, RR_2=RR_2, RR_cum=RR_cum)
    print('Done')
    # plot recog_rate to see if any differences in confusion matr and this
    if plot_recog_rate:
        RR_0 = {emotion: 0 for emotion in emotions}
        RR_1 = {emotion: 0 for emotion in emotions}
        RR_cum = {emotion: 0 for emotion in emotions}
        I_0 = {emotion: 0 for emotion in emotions}
        I_1 = {emotion: 0 for emotion in emotions}
        I = {emotion: 0 for emotion in emotions}

        print('Predicting emotion from model...')
        for i, vid in enumerate(test_X):
            gt = test_y[i]
            predictions = model.predict(vid)
            I[gt] += 1
            if top_2[0] == gt:
                I_0[gt] += 1
            elif len(top_2) > 1:  # might only be 1 prediction across timeseries
                I_1[gt] += 1

        # Calculate RR(0), RR(1) for each emotion
        for emotion in emotions:
            RR_0[emotion] = I_0[emotion] / I[emotion]
            RR_1[emotion] = I_1[emotion] / I[emotion]
            RR_cum[emotion] = RR_0[emotion] + RR_1[emotion]

        with open('../../data/paco/model_output.txt', 'w') as f:
            print("Writing recognition results to data/model_output.txt...")
            f.write('Emotion,RR_0,RR_1,RR_cum \n')
            for emotion in emotions:
                f.write(emotion + ',' + str(RR_0[emotion]) + "," + str(RR_1[emotion]) + "," + str(RR_cum[emotion]) + "\n")
            print('Done')


def hmm(override_model=False, override_soft_assign=False):
    if paco:
        print('HMM for paco dataset')
        train_X, train_y, test_X, test_y = get_train_test_soft_assign(train_fpath='../../data/paco/training/train_soft_assign.npz', test_fpath='../../data/paco/test/test_soft_assign.npz', override_soft_assign=override_soft_assign)
    else:
        print('HMM for action db')
        train_X, train_y, test_X, test_y = get_train_test_soft_assign(override_soft_assign=override_soft_assign)
    # Load model
    hmm_json = '../../models/HMM_paco.json' if paco else '../../models/HMM.json'
    if not os.path.isfile(hmm_json) or override_model:
        model = train_model(train_X, train_y, hmm_json_fpath=hmm_json)
    else:
        print('Loading model from ' + hmm_json)
        with open(hmm_json, 'rb') as f:
            model = simplejson.load(f)
            model = pmg.HiddenMarkovModel.from_json(model)

    # Test model
    model_output_dir = '../../data/paco' if paco else '../../data/action_db'
    hmm_inference(model, test_X, test_y, model_output_dir=model_output_dir)


def plot_recog_rate():
    if paco:
        model_output_fname = '../../data/paco/model_output.txt'
        full_emotion_names = {'ang': 'anger', 'fea': 'fear', 'hap': 'happiness', 'neu': 'neutral', 'sad': 'sadness'}
    else:
        model_output_fname = '../../data/action_db/model_output.txt'
        full_emotion_names = {'ang': 'anger', 'fea': 'fear', 'hap': 'happiness', 'sad': 'sadness',
                              'unt': 'untrustworthiness'}

    d = {'emotion': [], 'Key': [], 'recognition rate (%)': []}

    with open(model_output_fname, 'r') as f:
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
            d['Key'].append('cumulative recognition rates at pos 1,2')
            d['recognition rate (%)'].append(float(recog_rates[3])*100)

    df = pd.DataFrame(data=d)
    sns.factorplot(x='emotion', y='recognition rate (%)', hue='Key', data=df, kind='bar')
    sns.despine(offset=10, trim=True)
    print('Saving plot...')
    if paco:
        plt.savefig('../../plots/paco/recog_rate.png')
    else:
        plt.savefig('../../plots/action_db/recog_rate.png')
    plt.show()


def confusion_mat():
    n_emotions = len(emotions)
    if paco:
        recog_rates = np.load('../../data/paco/model_output.npz', encoding='latin1')
    else:
        recog_rates = np.load('../../data/action_db/model_output.npz', encoding='latin1')
    RR_1 = recog_rates['RR_1'].item()
    RR_cum = recog_rates['RR_cum'].item()
    conf_mat_RR_1 = np.zeros((n_emotions,n_emotions))
    conf_mat_RR_cum = np.zeros((n_emotions,n_emotions))

    for i,emotion1 in enumerate(emotions):
        for j,emotion2 in enumerate(emotions):
            conf_mat_RR_1[i][j] = RR_1[emotion1][emotion2] # flip indices so that gt is on y axis in confusion matrix plot
            conf_mat_RR_cum[i][j] = RR_cum[emotion1][emotion2]

    df_RR_1 = pd.DataFrame(conf_mat_RR_1, index=emotions, columns=emotions)
    df_RR_cum = pd.DataFrame(conf_mat_RR_cum, index=emotions, columns=emotions)

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



if __name__ == '__main__':
    # paco db
    paco=True
    rmv_emotions = []
    emotions = ['ang', 'fea', 'hap', 'neu', 'sad']
    hmm(override_model=True)
    confusion_mat()


    # actions db
    # paco=False
    # rmv_emotions = []
    # emotions = ['ang', 'fea', 'hap', 'sad', 'unt']
    # hmm(override_model=True, override_soft_assign=False)
    # confusion_mat()




