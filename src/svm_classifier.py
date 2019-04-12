from sklearn import svm
import pandas as pd
import numpy as np
import os
from utils.data_utils import convert_to_list
from utils.split_train_test import cross_val
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import pickle
from sklearn.externals import joblib
from collections import Counter
from utils.plot_results import _read_model_results, confusion_mat

def prepare_data(LMA_features_path, LMA_svm_path):
    ''' takes every 5 frames of LMA feature data, concatenates 15 frames together'''
    print('Reading LMA features...')
    LMA = pd.read_csv(LMA_features_path).iloc[:,1:]
    cols_to_ignore = ['emotion', 'subject', 'action']
    if 'intensity' in LMA.columns:
        cols_to_ignore.append('intensity')
    columns = cols_to_ignore + ['data']
    svm_df = pd.DataFrame(columns=columns)
    print('Preparing data for SVM...')
    svm_dic = {'subject': [], 'emotion': [], 'data': []}
    if 'intensity' in LMA.columns:
        svm_dic['intensity'] = []

    for i, row in LMA.iterrows():
        print('row ' + str(i))
        arr = np.array(row.drop(cols_to_ignore))
        arr = [convert_to_list(r) for r in arr]
        arr = list(zip(*arr))
        arr = [frame for frame in arr]
        # arr = [x for frame in arr for x in frame] # flatten frames into single array
        arr = [frame for i, frame in enumerate(arr) if i % 5 == 0]  # only take every 5th frame
        svm_dic['subject'].append(row['subject'])
        svm_dic['emotion'].append(row['emotion'])
        if 'intensity' in LMA.columns:
            svm_dic['intensity'].append(row['intensity'])

        svm_dic['data'].append(arr)

    svm_df = pd.DataFrame(svm_dic)
    print('Writing SVM df to ' + LMA_svm_path)
    svm_df.to_hdf(LMA_svm_path, key='df', mode='w')
    return svm_df




def perform_svm_cv(cross_val_dir, window_size=15, window_overlap=14):
    def _split_by_frame(data, labels, keep_by_subject=False):
        # split into 15-frame segments
        data = [[X[i:i + window_size] for i in range(0, len(X) - window_size + 1, window_size - window_overlap)] for X in data]
        data = [[[item for frame in window for item in frame] for window in subject] for subject in data]

        labels = [[labels[i]] * len(X) for i, X in enumerate(data)]
        # flatten by frame
        if keep_by_subject:
            return data, labels
        data = [frame for subject in data for frame in subject]
        labels = [frame_emotion for Y in labels for frame_emotion in Y]
        return data, labels


    if 'paco' in cross_val_dir:
        LMA_features_path = '../data/paco/LMA_features.csv'
        LMA_svm_path = '../data/paco/SVM/LMA_SVM.h5'
        models_dir = '../models/paco/SVM'
        emotions = ['ang', 'hap', 'neu', 'sad']
    else:
        LMA_features_path = '../data/action_db/LMA_features.csv'
        LMA_svm_path = '../data/action_db/SVM/LMA_SVM.h5'
        models_dir = '../models/action_db/SVM'
        emotions = ['ang', 'fea', 'hap', 'neu', 'sad', 'unt']
    LMA_cross_val_path = cross_val_dir + '/LMA_features_cross_val.h5'

    if not os.path.isfile(LMA_cross_val_path):
        if not os.path.isfile(LMA_svm_path):
            svm_df = prepare_data(LMA_features_path, LMA_svm_path)
        else:
            svm_df = pd.read_hdf(LMA_svm_path)
        svm_df = cross_val(svm_df, os.path.dirname(LMA_cross_val_path))
    else:
        svm_df = pd.read_hdf(LMA_cross_val_path)




    cols_to_ignore = ['emotion', 'subject', 'fold']
    if 'intensity' in svm_df.columns:
        cols_to_ignore.append('intensity')

    for f in range(10):
        fold = str(f)

        train = svm_df.loc[svm_df['fold']!=f]
        train_Y = np.array(train['emotion'])
        test = svm_df.loc[svm_df['fold']==f]

        test_Y = np.array(test['emotion'])
        train_X = np.array(train['data'])
        test_X = np.array(test['data'])
        print('Preparing datasets for SVM')
        train_X, train_Y = _split_by_frame(train_X, train_Y)
        test_X, test_Y = _split_by_frame(test_X, test_Y, keep_by_subject=True)


        # scaling = MinMaxScaler(feature_range=(-1, 1)).fit(train_X)
        # train_X = scaling.transform(train_X)
        # test_X = scaling.transform(test_X)

        train_X =preprocessing.scale(train_X)
        # test_X = preprocessing.scale(test_X)


        print('Training SVM classifier')
        clf = svm.SVC(gamma='scale', decision_function_shape='ovr', random_state=0)
        print('Fitting SVM classifier')
        clf.fit(train_X, train_Y)
        print('Saving model...')
        model_filename = models_dir + '/model_fold_' + fold + '.sav'
        joblib.dump(clf, open(model_filename, 'wb'))

        # load the model later from disk
        # loaded_model = joblib.load(filename)

        gt_counts = {emotion: 0 for emotion in emotions}
        pred_pos_1 = {emotion: {emotion: 0 for emotion in emotions} for emotion in
                      emotions}  # Number predicted at position 1
        pred_pos_2 = {emotion: {emotion: 0 for emotion in emotions} for emotion in
                      emotions}  # Number predicted at position 2

        print('Predicting emotion from model...')

        for i, vid in enumerate(test_X):
            scaled = preprocessing.scale(vid)
            gt = test_Y[i][0]
            print("Predicting emotion for gt " + gt)
            gt_counts[gt] += 1
            predictions = clf.predict(scaled)
            counts = Counter(predictions)
            top_2 = counts.most_common(2)
            print(top_2)
            emotion_pred_0 = top_2[0][0]
            print("Top predicted emotion is: " + emotion_pred_0)
            pred_pos_1[gt][emotion_pred_0] += 1

            if len(top_2) > 1:
                emotion_pred_1 = top_2[1][0]
                print("Second predicted emotion is: " + emotion_pred_1)
                pred_pos_2[gt][emotion_pred_1] += 1

        model_results_path = models_dir + '/output/model_output_' + fold + '.npz'
        print("Writing recognition results to " + model_results_path + "...")
        np.savez_compressed(model_results_path, pred_pos_1=pred_pos_1, pred_pos_2=pred_pos_2)
        print('Done')

        # os.system('say "Confusion matrix for fold"')
        # df_correct_1, df_correct_cum = _read_model_results(model_results_path, emotions)
        # confusion_mat(df_1=df_correct_1, df_cum=df_correct_cum, show_plot=True)

    # return df_correct_1_sum, df_correct_cum_sum

def perform_svm_LOSO(LOSO_dir, window_size=15, window_overlap=14):
    def _split_by_frame(data, labels, keep_by_subject=False):
        # split into 15-frame segments
        data = [[X[i:i + window_size] for i in range(0, len(X) - window_size + 1, window_size - window_overlap)] for X in data]
        data = [[[item for frame in window for item in frame] for window in subject] for subject in data]

        labels = [[labels[i]] * len(X) for i, X in enumerate(data)]
        # flatten by frame
        if keep_by_subject:
            return data, labels
        data = [frame for subject in data for frame in subject]
        labels = [frame_emotion for Y in labels for frame_emotion in Y]
        return data, labels


    if 'paco' in LOSO_dir:
        LMA_features_path = '../data/paco/LMA_features.csv'
        LMA_svm_path = '../data/paco/SVM/LMA_SVM.h5'
        models_dir = '../models/paco/SVM'
        emotions = ['ang', 'hap', 'neu', 'sad']
        subjects = ['ale', 'ali', 'alx', 'amc', 'bar', 'boo', 'chr', 'dav', 'din', 'dun', 'ele', 'emm', 'gra', 'ian',
                    'jan', 'jen', 'jua', 'kat', 'lin', 'mac', 'mar', 'mil', 'ndy', 'pet', 'rac', 'ros', 'she', 'shi',
                    'ste', 'vas']
    else:
        LMA_features_path = '../data/action_db/LMA_features.csv'
        LMA_svm_path = '../data/action_db/SVM/LMA_SVM.h5'
        models_dir = '../models/action_db/SVM'
        emotions = ['ang', 'fea', 'hap', 'neu', 'sad', 'unt']
        subjects = ['10f', '11f', '12m', '13f', '14f', '15m', '16f', '17f', '18f', '19m', '1m', '20f', '21m', '22f',
                    '23f', '24f', '25m', '26f', '27m', '28f', '29f', '2f', '3m', '4f', '5m', '6f', '7f', '8m', '9f']

    LMA_path = LOSO_dir + '/LMA_SVM.h5'

    if not os.path.isfile(LMA_path):
        if not os.path.isfile(LMA_svm_path):
            svm_df = prepare_data(LMA_features_path, LMA_svm_path)
        else:
            svm_df = pd.read_hdf(LMA_svm_path)
        svm_df = cross_val(svm_df, os.path.dirname(LMA_path))
    else:
        svm_df = pd.read_hdf(LMA_path)




    cols_to_ignore = ['emotion', 'subject', 'fold']
    if 'intensity' in svm_df.columns:
        cols_to_ignore.append('intensity')

    for subject in subjects:


        train = svm_df.loc[svm_df['subject']!=subject]
        train_Y = np.array(train['emotion'])
        test = svm_df.loc[svm_df['subject']==subject]

        test_Y = np.array(test['emotion'])
        train_X = np.array(train['data'])
        test_X = np.array(test['data'])
        print('Preparing datasets for SVM')
        train_X, train_Y = _split_by_frame(train_X, train_Y)
        test_X, test_Y = _split_by_frame(test_X, test_Y, keep_by_subject=True)


        # scaling = MinMaxScaler(feature_range=(-1, 1)).fit(train_X)
        # train_X = scaling.transform(train_X)
        # test_X = scaling.transform(test_X)

        train_X =preprocessing.scale(train_X)
        # test_X = preprocessing.scale(test_X)


        print('Training SVM classifier')
        clf = svm.SVC(gamma='scale', decision_function_shape='ovr', random_state=0)
        print('Fitting SVM classifier')
        clf.fit(train_X, train_Y)
        print('Saving model...')
        model_filename = models_dir + '/model_fold_' + subject + '.sav'
        joblib.dump(clf, open(model_filename, 'wb'))

        # load the model later from disk
        # loaded_model = joblib.load(filename)

        gt_counts = {emotion: 0 for emotion in emotions}
        pred_pos_1 = {emotion: {emotion: 0 for emotion in emotions} for emotion in
                      emotions}  # Number predicted at position 1
        pred_pos_2 = {emotion: {emotion: 0 for emotion in emotions} for emotion in
                      emotions}  # Number predicted at position 2

        print('Predicting emotion from model...')

        for i, vid in enumerate(test_X):
            scaled = preprocessing.scale(vid)
            gt = test_Y[i][0]
            print("Predicting emotion for gt " + gt)
            gt_counts[gt] += 1
            predictions = clf.predict(scaled)
            counts = Counter(predictions)
            top_2 = counts.most_common(2)
            print(top_2)
            emotion_pred_0 = top_2[0][0]
            print("Top predicted emotion is: " + emotion_pred_0)
            pred_pos_1[gt][emotion_pred_0] += 1

            if len(top_2) > 1:
                emotion_pred_1 = top_2[1][0]
                print("Second predicted emotion is: " + emotion_pred_1)
                pred_pos_2[gt][emotion_pred_1] += 1

        model_results_path = models_dir + '/output/model_output_' + subject + '.npz'
        print("Writing recognition results to " + model_results_path + "...")
        np.savez_compressed(model_results_path, pred_pos_1=pred_pos_1, pred_pos_2=pred_pos_2)
        print('Done')



def plot_results(models_folder):
    if 'paco' in models_folder:
        emotions = ['ang', 'hap', 'neu', 'sad']
    else:
        emotions = ['ang', 'fea', 'hap', 'neu', 'sad', 'unt']
    no_folds = 10
    # get 10 results
    # initialise dataframes for recognition rate at position 1, recognition rate at pos 2
    df_correct_1_sum = pd.DataFrame(np.zeros((len(emotions), len(emotions))), index=emotions, columns=emotions)
    df_correct_cum_sum = pd.DataFrame(np.zeros((len(emotions), len(emotions))), index=emotions, columns=emotions)

    i = 0

    for file in os.listdir(models_folder):
        if file.endswith('.npz'):
            i += 1
            df_correct_1, df_correct_cum = _read_model_results(models_folder + "/" + file, emotions)
            df_correct_1_sum = df_correct_1_sum.add(df_correct_1, fill_value=0)
            df_correct_cum_sum = df_correct_cum_sum.add(df_correct_cum, fill_value=0)

    assert i == no_folds
    # calc average for 10 folds
    # df_correct_1_avg = df_correct_1_sum.divide(no_folds)
    # df_correct_cum_avg = df_correct_cum_sum.divide(no_folds)
    # write results
    print("Writing recognition results to " + models_folder + '/10_fold_cv_results.h5')
    df_correct_1_sum.to_hdf(models_folder + '/10_fold_cv_results.h5', key='df_RR_1_sum', mode='w')
    df_correct_cum_sum.to_hdf(models_folder + '/10_fold_cv_results.h5', key='df_RR_cum_sum', mode='w')
    print('Done')
    # confusion matrix
    confusion_mat(df_1=df_correct_1_sum, df_cum=df_correct_cum_sum, show_plot=True)


if __name__ == '__main__':
    # perform_svm_cv('../data/paco/SVM/10_fold_cross_val', window_size=1, window_overlap=0)
    # plot_results('../models/paco/SVM/output')
    perform_svm_LOSO('../data/paco/SVM/LOSO', window_size=15, window_overlap=5)