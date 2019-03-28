from __future__ import division
import numpy as np
import pandas as pd
import random
from skmultilearn.model_selection import iterative_train_test_split
from sklearn.model_selection import StratifiedKFold



def split(df, train_size=60, val_size=20, test_size=20):
    '''
    Takes LMA file containing feature vectors, splits into training, validation and test set with:
    - roughly equal proportion of each emotion
    - roughly equal number of each actor
    saves train, validation, test set
    '''
    NO_OF_VIDEOS = 563
    # Shuffle order of rows
    df = df.reindex(np.random.permutation(df.index))

    columns = df.columns.values
    train_df = pd.DataFrame(columns=columns)
    validation_df = pd.DataFrame(columns=columns)
    test_df = pd.DataFrame(columns=columns)

    df_len = NO_OF_VIDEOS
    train_val_test_split_total = train_size + val_size + test_size
    train_size = int(train_size / train_val_test_split_total * df_len)
    val_size = int(val_size / train_val_test_split_total * df_len)
    test_size = int(test_size / train_val_test_split_total * df_len)

    train_size += df_len - (train_size + val_size + test_size)


    emotions = set(df['emotion'])
    for emotion in emotions:
        subjects = set(df[df['emotion']==emotion]['subject'])
        for subject in subjects:
            df_emotion_subject = df[df['emotion']==emotion][df['subject']==subject]
            actions = set(df_emotion_subject['action'])
            if len(actions == 1):
                # Only 'Walking1'
                for i, row in df_emotion_subject.iterrows():
                    if i % 3 == 0:
                        # Different action but same intensity, subject, emotion
                        if train_df.loc[(train_df['subject'] == row['subject']) & (train_df['emotion'] == row['emotion']) &
                                        (train_df['intensity'] == row['intensity'])]:
                            if random.randint(0, 1) == 0:
                                validation_df.append(row.copy(), ignore_index=True)
                            else:
                                test_df.append(row.copy(), ignore_index=True)
                        else: # Only Walking1 for subject, emotion at that intensity
                            train_df.append(row.copy(), ignore_index=True)
                    elif i % 3 == 1:
                        if train_df.loc[(train_df['subject'] == row['subject']) & (train_df['emotion'] == row['emotion']) &
                                        (train_df['intensity'] == row['intensity'])]:
                            if random.randint(0, 1) == 0:
                                test_df.append(row.copy(), ignore_index=True)
                            else:
                                train_df.append(row.copy(), ignore_index=True)
                        else: # Only Walking1 for subject, emotion at that intensity
                            validation_df.append(row.copy(), ignore_index=True)
                    else: # i % 3 == 2
                        if train_df.loc[(train_df['subject'] == row['subject']) & (train_df['emotion'] == row['emotion']) &
                                        (train_df['intensity'] == row['intensity'])]:
                            if random.randint(0, 1) == 0:
                                train_df.append(row.copy(), ignore_index=True)
                            else:
                                validation_df.append(row.copy(), ignore_index=True)
                        else: # Only Walking1 for subject, emotion at that intensity
                            test_df.append(row.copy(), ignore_index=True)

    ''' Write datasets to data/ '''
    train_df.to_hdf('../data/training/train_data.h5', key='df', mode='w')
    validation_df.to_hdf('../data/training/validation_data.h5', key='df', mode='w')
    test_df.to_hdf('../data/test/test_data.h5', key='df', mode='w')


def split_train_test(df, train_size=80, test_size=20, folder='../data/action_db'):
    def _convert_index_to_subj_emotion(y):
        y = [subjects[y[0]-1], EMOTIONS[y[1]]]
        return y

    if 'paco' not in folder:
        EMOTIONS = ['ang', 'fea', 'hap', 'neu', 'sad', 'unt']
        subjects = ['1m', '2f', '3m', '4f', '5m', '6f', '7f', '8m', '9f', '10f', '11f', '12m', '13f', '14f', '15m', '16f',
                    '17f', '18f', '19m', '20f', '21m', '22f', '23f', '24f', '25m', '26f', '27m', '28f', '29f']
    else:
        # emotions = ['ang', 'fea', 'hap', 'sad', 'neu']
        EMOTIONS = ['ang', 'hap', 'sad', 'neu']
        subjects = ['ale', 'ali', 'alx', 'amc', 'bar', 'boo', 'chr', 'dav', 'din', 'dun', 'ele', 'emm', 'gra', 'ian', 'jan', 'jen', 'jua', 'kat', 'lin', 'mac', 'mar', 'mil', 'ndy', 'pet', 'rac', 'ros', 'she', 'shi', 'ste', 'vas']

    df = df.reindex(np.random.permutation(df.index))

    emotions_dict = {EMOTIONS[i]: int(i) for i in range(len(EMOTIONS))}
    subjects_dict = {subjects[i]: int(i + 1) for i in range(len(subjects))}
    y_classes = ['subject', 'emotion']
    y = df[y_classes]

    y['subject'] = y['subject'].map(subjects_dict)
    y['emotion'] = y['emotion'].map(emotions_dict)
    y = np.array(y)

    X = df.drop(y_classes, axis=1)
    columns = X.columns.values
    X = np.array(X)
    columns = np.append(columns, y_classes)

    assert len(X) == len(y)

    X_train, y_train, X_test, y_test = iterative_train_test_split(X, y, test_size=test_size/(test_size+train_size))
    y_train = np.apply_along_axis(_convert_index_to_subj_emotion, 1, y_train)
    y_test = np.apply_along_axis(_convert_index_to_subj_emotion, 1, y_test)

    train_df = pd.DataFrame(data=np.hstack((X_train, y_train)), columns=columns)
    test_df = pd.DataFrame(data=np.hstack((X_test, y_test)), columns=columns)

    print('Counts in train set')
    for emotion in EMOTIONS:
        print(emotion + ": " + str(len(train_df[train_df['emotion'] == emotion])))
    print('Counts in test set')
    for emotion in EMOTIONS:
        print(emotion + ": " + str(len(test_df[test_df['emotion'] == emotion])))

    print('Saving training data...')
    train_df.to_csv(folder + '/training/train_data.csv')
    # train_df.to_hdf(folder + '/training/train_data.h5', key='df', mode='w')
    print('Saving test data...')
    test_df.to_hdf(folder + '/test/test_data.h5', key='df', mode='w')
    print('Done.')
    return train_df, test_df


def stratified_split(X, y, folds=5):
    ''' Multiclass stratified split '''
    obs, classes = y.shape
    dist = y.sum(axis=0).astype('float')
    dist /= dist.sum()
    index_list = []
    fold_dist = np.zeros((folds, classes), dtype='float')
    for _ in range(folds):
        index_list.append([])
    for i in range(obs):
        if i < folds:
            target_fold = i
        else:
            normed_folds = fold_dist.T / fold_dist.sum(axis=1)
            how_off = normed_folds.T - dist
            target_fold = np.argmin(np.dot((y[i] - .5).reshape(1, -1), how_off.T))
        fold_dist[target_fold] += y[i]
        index_list[target_fold].append(i)

    print('Folds are')
    folds = [X[index] for index in index_list]
    classes = [y[index] for index in index_list]
    return folds, classes


def cross_val(df, n=10):
    print('Spliting into 10 folds...')
    df = df.reindex(np.random.permutation(df.index))
    drop_classes = ['subject', 'emotion']

    y = np.array(df['emotion'])
    X = df.drop(drop_classes, axis=1)
    X = np.array(X)
    # stratified 10 fold cross val
    skf = StratifiedKFold(n_splits=n)
    # skf.get_n_splits(X, y)
    split = skf.split(X,y)
    df_w_folds = df
    df_w_folds['fold'] = 0
    i = 0
    for train_indices, test_indices in split:
        df_w_folds.iloc[train_indices, df_w_folds.columns.get_loc('fold')] = i
        i += 1

    print(set(df_w_folds['fold']))

    df_w_folds.to_hdf('../data/paco/10_fold_cross_val/LMA_features_crossfold.h5', key='df', mode='w')

    print('Saving...Done.')
    return folds



if __name__ == '__main__':
    X = np.array([[]])

    y = np.array([['sad', '29f'], ['sad', '29f'], ['sad', '28m'], ['ang', '28m'], ['ang', '01m'], ['unt', '01m'], ['fea', '01m'], ['fea', '28m'], ['fea', '23m']])
    emotions = {'ang': 0, 'fea': 1, 'hap': 2, 'neu': 3, 'sad': 4, 'unt': 5}
    y = np.array([np.array([emotions[labels[0]], int(labels[1][:-1])]) for labels in y])
    index_list = stratified_split(y, 2)
    print('Folds are')
    folds = [y[index] for index in index_list]
    print(folds)


