from utils import data_utils
from merge_posesets import merge_centers_paco
from HMM.hmm import HMM
import os
import pandas as pd
import numpy as np
from utils.plot_results import confusion_mat, _read_model_results

''' Carries out LOSO pipeline (with clustering offloaded onto Google colab notebook) '''
''' Leave one subject out -> Clustering on train for each validation subject (offloaded) -> merge posesets -> hmm -> average results'''

LOSO_dir = '../data/paco/LOSO'


no_subjects = 30
emotions = ['ang', 'hap', 'neu', 'sad']

models_folder = '../models/paco/output/LOSO'


subjects = ['ale', 'ali', 'alx', 'amc', 'bar', 'boo', 'chr', 'dav', 'din', 'dun', 'ele', 'emm', 'gra', 'ian', 'jan', 'jen', 'jua', 'kat', 'lin', 'mac', 'mar', 'mil', 'ndy', 'pet', 'rac', 'ros', 'she', 'shi', 'ste', 'vas']

def perform_LOSO(override_merged_centers=False):
    override_soft_assign = override_merged_centers
    # initialise dataframes for recognition rate at position 1, recognition rate at pos 2
    df_correct_1_avg = pd.DataFrame(np.zeros((len(emotions), len(emotions))), index=emotions, columns=emotions)
    df_correct_cum_avg = pd.DataFrame(np.zeros((len(emotions), len(emotions))), index=emotions, columns=emotions)
    for subject in subjects:
        df = pd.read_csv('../data/paco/LMA_features.csv').iloc[:, 1:]
        LOSO_subject_dir = LOSO_dir + '/' + subject
        clusters_dir = LOSO_subject_dir + '/clusters'
        if len(os.listdir(clusters_dir)) == 0:
            print('No clusters found for LOSO subject ' + subject + '.')  # Need to obtain clusters from google colab script
            continue
        if not os.path.isfile(LOSO_subject_dir+'/merged_centers.npz') or override_merged_centers:
            merge_centers_paco(thresh=0, dir=clusters_dir)
        model_results_path = models_folder + '/model_output_' + subject + '.npz'
        train = df.loc[df['subject']!=subject]
        test = df.loc[df['subject']==subject]
        del df
        hmmModel = HMM(paco=True, train=train, test=test,
                       train_soft_assign_path=LOSO_subject_dir+'/train_soft_assign.npz',
                       test_soft_assign_path=LOSO_subject_dir+'/test_soft_assign.npz',
                       model_results_path=model_results_path)
        hmmModel.hmm(override_model=True, override_soft_assign=override_soft_assign, n_components=6)
        hmmModel.confusion_mat()
        df_correct_1, df_correct_cum = hmmModel._read_results()
        df_correct_1_avg = df_correct_1_avg.add(df_correct_1, fill_value=0)
        df_correct_cum_avg = df_correct_cum_avg.add(df_correct_cum, fill_value=0)

    return df_correct_1_avg, df_correct_cum_avg

# get 30 results

# initialise dataframes for recognition rate at position 1, recognition rate at pos 2
df_correct_1_avg = pd.DataFrame(np.zeros((len(emotions), len(emotions))), index=emotions, columns=emotions)
df_correct_cum_avg = pd.DataFrame(np.zeros((len(emotions), len(emotions))), index=emotions, columns=emotions)

i = 0


for file in os.listdir(models_folder):
    if file.endswith('.npz'):
        i += 1
        df_correct_1, df_correct_cum = _read_model_results(models_folder + "/" + file, emotions)
        df_correct_1_avg = df_correct_1_avg.add(df_correct_1, fill_value=0)
        df_correct_cum_avg = df_correct_cum_avg.add(df_correct_cum, fill_value=0)

assert i == no_subjects
# calc average for 30 subjects
df_correct_1_avg = df_correct_1_avg.divide(no_subjects)
df_correct_cum_avg = df_correct_cum_avg.divide(no_subjects)
# write results
print("Writing recognition results to " + models_folder + '/LOSO_results.h5')
df_correct_1_avg.to_hdf(models_folder + '/LOSO_results.h5', key='df_RR_1_avg', mode='w')
df_correct_cum_avg.to_hdf(models_folder + '/LOSO_results.h5', key='df_RR_cum_avg', mode='w')
print('Done')
# confusion matrix
confusion_mat(df_1=df_correct_1_avg, df_cum=df_correct_cum_avg, show_plot=True)







