from kMedians import kmedians
import numpy as np
from utils import data_utils
import pandas as pd
import os

def flatten_by_frame(df):
    columns = df.columns.values
    for column_name in columns:
        col = df[[column_name]]

    return

def generate_lexicon(emotion, train):
    '''
    Generates lexicon for reference poseset per emotion category
    :return:
    '''

    # obtain action sequences for emotion
    no_frames = 130
    X =[]

    # Carry out k-medians
    k_medians = kmedians(10, no_frames)
    k_medians = k_medians.fit(X)
    predicted_clusters = k_medians._get_clusters(X)
    kmedians._visualise_clusters(predicted_clusters)



emotions = ['ang', 'fea', 'hap', 'neu', 'sad', 'unt']

LMA_train, LMA_test = data_utils.get_train_test_set()
# LMA_train = LMA_train.drop(['subject'], axis=1)
#
#
# for emotion in emotions:
#     df = LMA_train.loc[LMA_train['emotion'] == emotion]
#     df = df.drop(['emotion'], axis=1)
#     generate_lexicon(emotion, np.array(df))

