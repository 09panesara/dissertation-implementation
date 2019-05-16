import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.spatial.distance import euclidean
import random
from collections import Counter
import simplejson
import pandas as pd
import ast
import pomegranate as pmg
from pomegranate import *
from utils.plot_results import _read_model_results, confusion_mat
from sklearn.cluster import KMeans as sklearn_kmeans
import scipy

class StudentTDistribution():
    def __init__(self, mu, std, df=1.0):
        self.mu = mu
        self.std = std
        self.df = df
        self.parameters = (self.mu, self.std)
        self.d = 1
        self.summaries = np.zeros(3)

    def probability(self, X):
        return np.exp(self.log_probability(X))

    def log_probability(self, X):
        return scipy.stats.t.logpdf(X, self.df, self.mu, self.std)

    def summarize(self, X, w=None):
        if w is None:
            w = np.ones(X.shape[0])

        X = X.reshape(X.shape[0])
        self.summaries[0] += w.sum()
        self.summaries[1] += X.dot(w)
        self.summaries[2] += (X ** 2.).dot(w)

    def from_summaries(self, inertia=0.0):
        self.mu = self.summaries[1] / self.summaries[0]
        self.std = self.summaries[2] / self.summaries[0] - self.summaries[1] ** 2 / (self.summaries[0] ** 2)
        self.std = np.sqrt(self.std)
        self.parameters = (self.mu, self.std)
        self.clear_summaries()

    def clear_summaries(self, inertia=0.0):
        self.summaries = np.zeros(3)

    @classmethod
    def from_samples(cls, X, weights=None, df=1):
        d = StudentTDistribution(0, 0, df)
        d.summarize(X, weights)
        d.from_summaries()
        return d

    @classmethod
    def blank(cls):
        return StudentTDistribution(0, 0)

class HMMGMM:
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


    def _GMM(self, train_X, train_y):
        ''' E-M algorithm to learn GMM parameters '''

        # TODO adjust for new labels
        def _calc_cov(a):
            # compute a * aT
            return [np.multiply(a_i, a) for a_i in a]

        _epsilon = 0

        def _nearPostiiveSemiDefinite(A, epsilon=0):
            n = A.shape[0]
            eigval, eigvec = np.linalg.eig(A)
            val = np.matrix(np.maximum(eigval, epsilon))
            vec = np.matrix(eigvec)
            T = 1 / (np.multiply(vec, vec) * val.T)
            T = np.matrix(np.sqrt(np.diag(np.array(T).reshape((n)))))
            B = T * vec * np.diag(np.array(np.sqrt(val)).reshape((n)))
            out = B * B.T
            return out

        dists = []  # in order of emotions
        parameters = {}
        for emotion in self.emotions:
            print("Computing GMM parameters for emotion " + emotion)
            obs = [np.array(train_X[i]) for i, y in enumerate(train_y) if y == emotion]
            obs = [o_t for seq in obs for o_t in seq]
            mu_j = np.sum(obs, axis=0) / len(obs)
            cov_j = np.sum([_calc_cov(o_t - mu_j) for o_t in obs], axis=0) / len(obs)

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

    def _gen_gmm(self, train_X, train_y, no_components=3):
        dists = []
        for emotion in self.emotions:
            print("Computing GMM parameters for emotion " + emotion)
            obs = [np.array(train_X[i]) for i, y in enumerate(train_y) if y == emotion]
            print(len(obs))
            gmm = pmg.GeneralMixtureModel.from_samples(pmg.MultivariateGaussianDistribution, n_components=no_components, X=obs)
            dists.append(gmm)
        return dists


    def train_model(self, train_X, train_y, hmm_json_dir):
        ''' cross validate for HMM learning '''

        # gmm_dists, gmm_parameters = self._GMM(train_X, train_y)
        # gmm_dists = self._gen_gmm(train_X, train_y)
        # print('Initialising start and transition probabilities.')
        # no_emotions = len(self.emotions)
        # counts = {emotion: 0 for emotion in self.emotions}
        # for y in train_y:
        #     counts[y] += 1
        #
        # trans_mat = np.diag([0.8] * no_emotions)
        # trans_mat = [[0.2 / (no_emotions - 1) if i == 0 else i for i in row] for row in trans_mat]
        # starts = [1 / no_emotions for e in self.emotions]
        # # starts = [counts[emotion]/len(train_y) for emotion in emotions]
        # initialise initial and transition probabilities using modified k means clustering
        n_emotions = len(self.emotions)


        distributions = []
        n_mixture_components = 6
        for emotion in self.emotions:
            X_subset = [np.array(train_X[i]) for i, y in enumerate(train_y) if y == emotion]
            X_subset = [o_t for seq in X_subset for o_t in seq]
            distribution = pmg.GeneralMixtureModel.from_samples(StudentTDistribution, n_mixture_components, X_subset)
            distributions.append(distribution)

        trans_mat = np.ones((n_emotions, n_emotions), dtype='float32') / n_emotions
        starts = np.ones(n_emotions, dtype='float32') / n_emotions

        model = pmg.HiddenMarkovModel.from_matrix(trans_mat, distributions, starts, verbose=True)
        model.fit(train_X, verbose=True)


        print('Saving initial parameters...')
        parameters_fpath = os.path.dirname(hmm_json_dir) + "/paco/paco_parameters.npz" if self.paco else os.path.dirname(
            hmm_json_dir) + "/action_db/action_db_parameters.npz"
        np.savez_compressed(parameters_fpath, trans_mat=trans_mat, starts=starts, gmm_dists=distributions)
        print('Done.')

        print('Training GMM HMM model')

        assert len(train_X) == len(train_y)
        # model = pmg.HiddenMarkovModel.from_matrix(transition_probabilities=trans_mat, distributions=distributions,
        #                                           starts=starts, verbose=True, state_names=self.emotions)

        # labels = np.array([[y] * len(train_X[i]) for i, y in enumerate(train_y)])
        # model = pmg.HiddenMarkovModel.from_samples(distribution=pmg.PoissonDistribution, n_components=len(self.emotions), X=train_X,  algorithm='baum-welch', state_names=self.emotions, verbose=True)
        # weights_by_emotion = {emotion: 1 / (len(train_y) * counts[emotion]) for emotion in self.emotions}
        # weights = [weights_by_emotion[emotion] for emotion in train_y]

        # model.fit(train_X, algorithm='baum-welch', verbose=True)
        model.bake()

        return model


    def hmm(self, override_model=True, override_soft_assign=False):
        self.override_model = override_model
        self.override_soft_assign = override_soft_assign
        train_X, train_y, test_X, test_y = self.get_train_test_soft_assign()
        # Load model
        hmm_json_dir = '../models/paco' if self.paco else '../models/action_db'

        if not os.path.isdir(hmm_json_dir) or self.override_model:
            # hidden state sequence for each observation sequence
            # train_labels = _get_labels(train_X)
            # TODO: check_acc(train_y, train_labels)
            model = self.train_model(train_X, train_y, hmm_json_dir=hmm_json_dir)
        else:
            with open(hmm_json_dir + "/model_output.json", 'rb') as f:
                model = simplejson.load(f)
                model = pmg.HiddenMarkovModel.from_json(model)

        # Test model
        self.hmm_inference(model, test_X, test_y)


    def hmm_inference(self, model, test_X, test_y):
        def _kth_most_common(lst, k):
            # remove last index = start
            kth_most_common = Counter(lst[1:]).most_common(k)
            kth_most_common = [item[0] for item in kth_most_common[:k]]
            return kth_most_common

        EMOTIONS = self.emotions
        RR_1 = {emotion: {emotion: 0 for emotion in EMOTIONS} for emotion in EMOTIONS}
        RR_2 = {emotion: {emotion: 0 for emotion in EMOTIONS} for emotion in EMOTIONS}
        RR_cum = {emotion: {emotion: 0 for emotion in EMOTIONS} for emotion in EMOTIONS}
        gt_counts = {emotion: 0 for emotion in EMOTIONS}
        pred_pos_1 = {emotion: {emotion: 0 for emotion in EMOTIONS} for emotion in
                      EMOTIONS}  # Number predicted at position 1
        pred_pos_2 = {emotion: {emotion: 0 for emotion in EMOTIONS} for emotion in
                      EMOTIONS}  # Number predicted at position 2

        print('Predicting emotion from model...')

        for i, vid in enumerate(test_X):
            gt = test_y[i]
            print("Predicting emotion for gt " + gt)
            gt_counts[gt] += 1
            predictions = model.predict(vid, algorithm='viterbi')
            print(predictions)
            top_2 = _kth_most_common(predictions, 2) # ignore start state
            emotion_pred_0 = EMOTIONS[top_2[0]]
            emotion_pred_1 = None
            if len(top_2) > 1:
                emotion_pred_1 = EMOTIONS[top_2[1]]
                print(emotion_pred_1)
            # Deal with indexing
            print("Predicted emotion is: " + emotion_pred_0)
            if emotion_pred_1 != None:
                print("Second predicted emotion is: " + emotion_pred_1)

            RR_1[gt][emotion_pred_0] += 1
            pred_pos_1[gt][emotion_pred_0] += 1
            if emotion_pred_1 != None:
                RR_2[gt][emotion_pred_1] += 1
                pred_pos_2[gt][emotion_pred_1] += 1

        for emotion1 in RR_1:
            for emotion2 in RR_1[emotion1]:
                RR_1[emotion1][emotion2] = RR_1[emotion1][emotion2] / gt_counts[emotion1]
                RR_2[emotion1][emotion2] = RR_2[emotion1][emotion2] / gt_counts[emotion1]
                RR_cum[emotion1][emotion2] = RR_1[emotion1][emotion2] + RR_2[emotion1][emotion2]

        print("Writing recognition results to " + self.model_results_path + "...")
        np.savez_compressed(self.model_results_path, RR_1=RR_1, RR_2=RR_2, RR_cum=RR_cum, pred_pos_1=pred_pos_1,
                            pred_pos_2=pred_pos_2)
        print('Done')


    def _read_results(self):
        ''' returns df_correct_1, df_correct_cum '''
        return _read_model_results(self.model_results_path, self.emotions)


    def confusion_mat(self, show_plot=False):
        df_correct_1, df_correct_cum = self._read_results()
        return confusion_mat(df_correct_1, df_correct_cum, show_plot=show_plot)

    def _get_labels(self, X, merged_centers_fpath):
        ''' get labels of emotion of cluster each frame in obs is closest to '''
        centers_emotions = np.load(merged_centers_fpath, encoding='latin1')['centers_emotions']

        labels = [[centers_emotions[np.argmax(obs)] for obs in x] for x in X]
        return labels


    def get_emotion_hidden_state_series(self, X, y, dict_path):
        EMOTIONS = self.emotions
        def _kth_most_common(lst, k):
            data = Counter(lst)
            return data.most_common(k)[k - 1][0]



        labels = self._get_labels(X, dict_path)

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

    def plot_recog_rate(self):
        if self.paco:
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
                d['recognition rate (%)'].append(float(recog_rates[1]) * 100)
                d['emotion'].append(emotion)
                d['Key'].append('recognition rates at position 2')
                d['recognition rate (%)'].append(float(recog_rates[2]) * 100)
                d['emotion'].append(emotion)
                d['Key'].append('cumulative recognition rates at pos 1,2')
                d['recognition rate (%)'].append(float(recog_rates[3]) * 100)

        df = pd.DataFrame(data=d)
        sns.factorplot(x='emotion', y='recognition rate (%)', hue='Key', data=df, kind='bar')
        sns.despine(offset=10, trim=True)
        print('Saving plot...')
        if self.paco:
            plt.savefig('../../plots/paco/recog_rate.png')
        else:
            plt.savefig('../../plots/action_db/recog_rate.png')
        plt.show()






