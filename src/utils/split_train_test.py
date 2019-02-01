from __future__ import division
import numpy as np
import pandas as pd
import random

NO_OF_VIDEOS = 563


'''
TODO:
split data into 80:20 training test
Then split training into 80:20 train, test 5 times for cross-validation 
'''

def split(df, train_split=60, val_split=20, test_split=20):
    '''
    Takes keypoints (OR LMA file?) containing feature vectors, splits into training, validation and test set with:
    - roughly equal proportion of each emotion
    - roughly equal number of each actor
    :return: train, validation, test set
    '''
    # Shuffle order of rows
    df = df.reindex(np.random.permutation(df.index))

    columns = df.columns.values
    train_df = pd.DataFrame(columns=columns)
    validation_df = pd.DataFrame(columns=columns)
    test_df = pd.DataFrame(columns=columns)

    df_len = NO_OF_VIDEOS
    train_val_test_split_total = train_split+val_split+test_split
    train_split = int(train_split/train_val_test_split_total * df_len)
    val_split = int(val_split/train_val_test_split_total * df_len)
    test_split = int(test_split/train_val_test_split_total * df_len)

    train_split += df_len - (train_split + val_split + test_split)


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
    train_df.to_hdf('../data/train_data.h5', key='df', mode='w')
    validation_df.to_hdf('../data/validation_data.h5', key='df', mode='w')
    test_df.to_hdf('../data/test_data.h5', key='df', mode='w')



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

if __name__ == '__main__':
    # X = np.array([[]])
    y = np.array([['sad', '29f'], ['sad', '29f'], ['sad', '28m'], ['ang', '28m'], ['ang', '01m'], ['unt', '01m'], ['fea', '01m'], ['fea', '28m'], ['fea', '23m']])
    emotions = {'ang': 0, 'fea': 1, 'hap': 2, 'neu': 3, 'sad': 4, 'unt': 5}
    y = np.array([np.array([emotions[labels[0]], int(labels[1][:-1])]) for labels in y])
    index_list = stratified_split(y, 2)
    print('Folds are')
    folds = [y[index] for index in index_list]
    print(folds)