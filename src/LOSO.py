from utils import data_utils
from merge_posesets import merge_centers
from HMM.hmm import HMM
import os
import pandas as pd
import numpy as np
from utils.plot_results import confusion_mat, _read_model_results

''' Carries out LOSO pipeline (with clustering offloaded onto Google colab notebook) '''
''' Leave one subject out -> Clustering on train for each validation subject (offloaded) -> merge posesets -> hmm -> average results'''


def perform_LOSO(paco, LOSO_dir, models_folder, override_merged_centers=False, merged_centers_thresh=0):
    if paco:
        emotions = ['ang', 'hap', 'neu', 'sad']
        no_subjects = 30
        subjects = ['ale', 'ali', 'alx', 'amc', 'bar', 'boo', 'chr', 'dav', 'din', 'dun', 'ele', 'emm', 'gra', 'ian',
                    'jan', 'jen', 'jua', 'kat', 'lin', 'mac', 'mar', 'mil', 'ndy', 'pet', 'rac', 'ros', 'she', 'shi',
                    'ste', 'vas']

        df = pd.read_csv('../data/paco/LMA_features.csv').iloc[:, 1:]
    else:
        emotions = ['ang', 'fea', 'hap', 'neu', 'sad', 'unt']
        no_subjects = 29
        subjects = ['10f', '11f', '12m', '13f', '14f', '15m', '16f', '17f', '18f', '19m', '1m', '20f', '21m', '22f',
                    '23f', '24f', '25m', '26f', '27m', '28f', '29f', '2f', '3m', '4f', '5m', '6f', '7f', '8m', '9f']
        df = pd.read_hdf('../data/action_db/LMA_features.h5')

    override_soft_assign = override_merged_centers
    # initialise dataframes for recognition rate at position 1, recognition rate at pos 2
    df_correct_1_sum = pd.DataFrame(np.zeros((len(emotions), len(emotions))), index=emotions, columns=emotions)
    df_correct_cum_sum = pd.DataFrame(np.zeros((len(emotions), len(emotions))), index=emotions, columns=emotions)

    for subject in subjects:
        LOSO_subject_dir = LOSO_dir + '/' + subject
        clusters_dir = LOSO_subject_dir + '/clusters'
        if len(os.listdir(clusters_dir)) == 0:
            print('No clusters found for LOSO subject ' + subject + '.')  # Need to obtain clusters from google colab script
            continue
        if not os.path.isfile(LOSO_subject_dir+'/merged_centers.npz') or override_merged_centers:
            merge_centers(emotions, thresh=300000000, dir=clusters_dir)
        model_results_path = models_folder + '/model_output_' + subject + '.npz'
        train = df.loc[df['subject']!=subject]
        test = df.loc[df['subject']==subject]
        hmmModel = HMM(paco=paco, train=train, test=test,
                       train_soft_assign_path=LOSO_subject_dir+'/train_soft_assign.npz',
                       test_soft_assign_path=LOSO_subject_dir+'/test_soft_assign.npz',
                       model_results_path=model_results_path)
        hmmModel.hmm(override_model=True, override_soft_assign=override_soft_assign, n_components=6)
        hmmModel.confusion_mat()
        df_correct_1, df_correct_cum = hmmModel._read_results()
        df_correct_1_sum = df_correct_1_sum.add(df_correct_1, fill_value=0)
        df_correct_cum_sum = df_correct_cum_sum.add(df_correct_cum, fill_value=0)


    return df_correct_1_sum, df_correct_cum_sum


def plot_results(emotions, models_folder, write_recog_results=True):
    # get 30 results

    # initialise dataframes for recognition rate at position 1, recognition rate at pos 2
    df_correct_1_sum = pd.DataFrame(np.zeros((len(emotions), len(emotions))), index=emotions, columns=emotions)
    df_correct_cum_sum = pd.DataFrame(np.zeros((len(emotions), len(emotions))), index=emotions, columns=emotions)

    for file in os.listdir(models_folder):
        if file.endswith('.npz'):
            df_correct_1, df_correct_cum = _read_model_results(models_folder + "/" + file, emotions)
            df_correct_1_sum = df_correct_1_sum.add(df_correct_1, fill_value=0)
            df_correct_cum_sum = df_correct_cum_sum.add(df_correct_cum, fill_value=0)

    # calc average for 30 subjects
    # df_correct_1_avg = df_correct_1_avg.divide(no_subjects)
    # df_correct_cum_avg = df_correct_cum_avg.divide(no_subjects)
    # write results
    if write_recog_results:
        print("Writing recognition results to " + models_folder + '/LOSO_results.h5')
        df_correct_1_sum.to_hdf(models_folder + '/LOSO_results.h5', key='df_RR_1_sum', mode='w')
        df_correct_cum_sum.to_hdf(models_folder + '/LOSO_results.h5', key='df_RR_cum_sum', mode='w')
        print('Done')
    # confusion matrix
    confusion_mat(df_1=df_correct_1_sum, df_cum=df_correct_cum_sum, show_plot=True)





if __name__ == '__main__':
    # df_correct_1_sum, df_correct_cum_sum = _read_model_results('../models/paco/output/LOSO/model_output_ale.npz', ['ang', 'hap', 'neu', 'sad'])
    # confusion_mat(df_1=df_correct_1_sum, df_cum=df_correct_cum_sum, show_plot=True)
    # perform_LOSO(paco=True, LOSO_dir='../data/paco/LOSO', models_folder='../models/paco/output/LOSO',
    #              override_merged_centers=True, merged_centers_thresh=300000000)

    plot_results(['ang', 'hap', 'neu', 'sad'], '../models/paco/output/10_fold_cross_val/v1-thresh=0', write_recog_results=False)
