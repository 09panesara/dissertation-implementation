import pandas as pd
import numpy as np
import os
from scipy.spatial.distance import euclidean
from pomegranate import *
import random
from collections import Counter
import simplejson
import pandas as pd
import ast
import pomegranate as pmg
from pomegranate import *
from utils.plot_results import _read_model_results, confusion_mat

class HMM:
    def __init__(self, paco, train, test, train_soft_assign_path, test_soft_assign_path,
                 rmv_emotions=[], model_results_path='../models/paco/model_results.npz', csv=True):
        '''
        :param paco: boolean for whether dataset is paco or not (if not, is action_db)
        :param train: dataframe for train dataset
        :param test: dataframe for test dataset
        :param train_soft_assign_path: path to create soft assignment to merged centers for training data set/
                                       path to file if soft assignment already exists
        :param test_soft_assign_path: path to create soft assignment to merged centers for test data set/
                                       path to file if soft assignment already exists
        :param rmv_emotions: list of emotions to exclude in model training/inference
        :param model_results_path: path to save results from HMM model on test dataset
        '''
        self.paco = paco
        if paco:
            self.emotions = ['ang', 'hap', 'neu', 'sad']
        else:
            self.emotions = ['ang', 'fea', 'hap', 'sad', 'neu', 'unt']
        self.train = train
        self.test = test
        self.rmv_emotions = rmv_emotions
        self.train_soft_assign_path = train_soft_assign_path
        self.test_soft_assign_path = test_soft_assign_path
        self.model_results_path = model_results_path
        self.csv = csv



    def _flatten_by_frame_by_row(self, df):
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
        for i, row in df.iterrows(): # Each row = different video for corresponding emotion
            no_frames = len(row['rise_sink'])
            if 'timestep_btwn_frame' in df.columns.values:
                timestep = row['timestep_btwn_frame']
                timestep = [timestep for i in range(no_frames)]
                row['timestep_btwn_frame'] = timestep
            arr = np.array(row)
            if self.csv:
                arr = [_convert_to_list(r) for r in arr]
            arr = list(zip(*arr))
            arr = [np.array(row) for row in arr]
            split_arr.append(arr)
        return np.array(split_arr)



    def _soft_assignment(self, data, soft_assign_path):
        ''' Do soft assignment
            Default parameters = for train dataset
        '''
        dict_path = os.path.dirname(self.train_soft_assign_path) + '/merged_centers.npz'
        print('Dictionary path for merged centers is: ' + dict_path)
        global_dict = np.load(dict_path, encoding='latin1')
        # if data_fpath.endswith('.h5'):
        #     LMA_train = pd.read_hdf(data_fpath)
        # else:
        #     LMA_train = pd.read_csv(data_fpath).iloc[:, 1:]

        if self.rmv_emotions != []:
            for emotion in self.rmv_emotions:
                print('Discarding emotion' + emotion)
                data = data[data['emotion']!=emotion]
        thresh = global_dict['thresh']
        centers_emotions = global_dict['centers_emotions']
        global_dict = global_dict['merged_centers']
        output = {}
        print('Doing soft assignment on dataset')
        for emotion in self.emotions:
            print('Doing soft assignment for emotion: ' + emotion)
            df = data.loc[data['emotion'] == emotion]
            soft_assign_v = self._flatten_by_frame_by_row(df)
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


        np.savez_compressed(soft_assign_path, soft_assign=output, thresh=thresh)
        print('Saving to ' + soft_assign_path + '.')
        return output


    def process_soft_assignment(self, soft_assign):
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



    def get_train_test_soft_assign(self):
        if self.override_soft_assign or not os.path.isfile(self.train_soft_assign_path):
            print("Soft assignment on train dataset")
            train_set = self._soft_assignment(data=self.train, soft_assign_path=self.train_soft_assign_path)
            test_set = self._soft_assignment(data=self.test, soft_assign_path=self.test_soft_assign_path)
        else:
            print("Loading train soft assignment ...")
            train_set = np.load(self.train_soft_assign_path, encoding='latin1')
            train_set = train_set['soft_assign'].item()
            print("Loading test soft assignment ...")
            test_set = np.load(self.test_soft_assign_path, encoding='latin1')
            test_set = test_set['soft_assign'].item()

        train_X, train_y = self.process_soft_assignment(train_set)
        test_X, test_y = self.process_soft_assignment(test_set)

        return train_X, train_y, test_X, test_y



    def hmm(self, n_components=6, override_model=True, override_soft_assign=False):
        self.override_model = override_model
        self.override_soft_assign = override_soft_assign
        self.n_components = n_components
        train_X, train_y, test_X, test_y = self.get_train_test_soft_assign()
        # Load model
        hmm_json_dir = '../models/paco' if self.paco else '../models/action_db'

        if not os.path.isdir(hmm_json_dir) or self.override_model:
            # hidden state sequence for each observation sequence
            # train_labels = _get_labels(train_X)
            # TODO: check_acc(train_y, train_labels)
            models = self.train_model(train_X, train_y, hmm_json_dir=hmm_json_dir)
        else:
            models = {emotion: None for emotion in self.emotions}
            for emotion in self.emotions:
                print('Loading model for emotion ' + emotion)
                with open(hmm_json_dir + "/model_"  + emotion + ".json", 'rb') as f:
                    model = simplejson.load(f)
                    model = pmg.HiddenMarkovModel.from_json(model)
                    models[emotion] = model

        # Test model
        self.hmm_inference(models, test_X, test_y)

    def _gen_model(self, data, emotion):
        model = pmg.HiddenMarkovModel.from_samples(pmg.NormalDistribution, n_components=self.n_components, X=data, algorithm='baum-welch', max_iterations=1000, verbose=True)

        # model = pmg.HiddenMarkovModel.from_samples(pmg.NormalDistribution, n_components=self.n_components, X=data, algorithm='baum-welch', max_iterations=100, verbose=True)
        print('Training HMM model')
        model.fit(data, algorithm='baum-welch', verbose=True, max_iterations=1000)
        model.bake()
        # Persist HMM
        model_json = pmg.HiddenMarkovModel.to_json(model)
        print('Saving HMM model')
        # with open(hmm_json_dir + "/model_"  + emotion + ".json", 'w') as file:
        #     simplejson.dump(model_json, file)
        # model.plot()
        return model

    def train_model(self, train_X, train_y, hmm_json_dir):
        ''' cross validate for HMM learning - TODO: after run once while HMM is doing learning, need to consider gender, intensity '''


        HMM_models = {}
        for emotion in self.emotions:
            print('Getting train set for emotion: ' + emotion)
            train_emotion = [train_X[i] for i, y in enumerate(train_y) if y==emotion]
            print('Here')
            HMM_models[emotion] = self._gen_model(train_emotion, emotion)

        return HMM_models




    def hmm_inference(self, models, test_X, test_y):
        EMOTIONS = self.emotions
        RR_1 = {emotion: {emotion: 0 for emotion in EMOTIONS} for emotion in EMOTIONS}
        RR_2= {emotion: {emotion: 0 for emotion in EMOTIONS} for emotion in EMOTIONS}
        RR_cum = {emotion: {emotion: 0 for emotion in EMOTIONS} for emotion in EMOTIONS}
        gt_counts = {emotion: 0 for emotion in EMOTIONS}
        pred_pos_1 = {emotion: {emotion: 0 for emotion in EMOTIONS} for emotion in EMOTIONS} # Number predicted at position 1
        pred_pos_2 = {emotion: {emotion: 0 for emotion in EMOTIONS} for emotion in EMOTIONS} # Number predicted at position 2

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
            pred_pos_1[gt][emotion_pred_0] += 1
            pred_pos_2[gt][emotion_pred_1] += 1



        for emotion1 in RR_1:
            for emotion2 in RR_1[emotion1]:
                RR_1[emotion1][emotion2] = RR_1[emotion1][emotion2] / gt_counts[emotion1]
                RR_2[emotion1][emotion2] = RR_2[emotion1][emotion2] / gt_counts[emotion1]
                RR_cum[emotion1][emotion2] = RR_1[emotion1][emotion2] + RR_2[emotion1][emotion2]



        print("Writing recognition results to " + self.model_results_path + "...")
        np.savez_compressed(self.model_results_path, RR_1=RR_1, RR_2=RR_2, RR_cum=RR_cum, pred_pos_1=pred_pos_1, pred_pos_2=pred_pos_2)
        print('Done')



    def _read_results(self):
        ''' returns df_correct_1, df_correct_cum '''
        return _read_model_results(self.model_results_path, self.emotions)


    def confusion_mat(self, show_plot=False):
        df_correct_1, df_correct_cum = self._read_results()
        return confusion_mat(df_correct_1, df_correct_cum, show_plot=show_plot)


    def get_emotion_hidden_state_series(self, X, y, dict_path):
        EMOTIONS = self.emotions
        def _kth_most_common(lst, k):
            data = Counter(lst)
            return data.most_common(k)[k - 1][0]

        def _get_labels(X, merged_centers_fpath):
            ''' get labels of emotion of cluster each frame in obs is closest to '''
            centers_emotions = np.load(merged_centers_fpath, encoding='latin1')['centers_emotions']

            labels = [[centers_emotions[np.argmax(obs)] for obs in x] for x in X]
            return labels

        labels = _get_labels(dict_path)

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

# can call:
#  df_RR_1, df_RR_cum = self._read_Results()
#  utils.plot_results.call confusion_mat(df_RR_1, df_RR_cum)