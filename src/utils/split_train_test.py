import numpy as np
from __future__ import division
import pandas as pd
import random

no_videos = 563



def split(df, train_split=60, val_split=20, test_split=20):
    '''
    Takes keypoints file containing feature vectors, splits into training, validation and test set with:
    - roughly equal proportion of each emotion
    - roughly equal number of each actor
    :param keypoints_3d:
    :return: train, validation, test set
    '''
    columns = df.columns.values
    train_df = pd.DataFrame(columns=columns)
    validation_df = pd.DataFrame(columns=columns)
    test_df = pd.DataFrame(columns=columns)

    df_len = no_videos
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






