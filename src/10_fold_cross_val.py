from utils import data_utils
from merge_posesets import merge_centers
from HMM.hmm import HMM
import os
import pandas as pd
import numpy as np
from utils.plot_results import confusion_mat, _read_model_results
from utils.split_train_test import cross_val


''' Carries out 10fold cross val pipeline (with clustering offloaded onto Google colab notebook) '''
''' Split into 10 folds -> Clustering on train for each validation fold (offloaded) -> merge posesets -> hmm -> average results'''


def perform_cv(paco, cross_val_dir, models_folder, override_merged_centers=False, merged_centers_thresh=0):
    if paco:
        emotions = ['ang', 'hap', 'neu', 'sad']
    else:
        emotions = ['ang', 'fea', 'hap', 'neu', 'sad', 'unt']
    override_soft_assign = override_merged_centers
    # initialise dataframes for recognition rate at position 1, recognition rate at pos 2
    df_correct_1_sum = pd.DataFrame(np.zeros((len(emotions), len(emotions))), index=emotions, columns=emotions)
    df_correct_cum_sum = pd.DataFrame(np.zeros((len(emotions), len(emotions))), index=emotions, columns=emotions)
    df = data_utils.get_cross_val(cross_val_dir='../data/paco/10_fold_cross_val')
    # df = pd.read_hdf(cross_val_dir + '/LMA_features_cross_val.h5')
    for f in range(3,4):
        fold = str(f)
        fold__dir = cross_val_dir + '/fold' + fold
        clusters_dir = fold__dir + '/clusters'
        if len(os.listdir(clusters_dir)) == 0:
            print('No clusters found for fold ' + fold + '.')
            continue
        if not os.path.isfile(fold__dir+'/merged_centers.npz') or override_merged_centers:
            merge_centers(emotions, thresh=merged_centers_thresh, dir=clusters_dir)
        model_results_path = models_folder + '/model_output_' + fold + '.npz'
        train = df.loc[df['fold']!=f]
        test = df.loc[df['fold']==f]
        hmmModel = HMM(paco=paco, train=train, test=test,
                       train_soft_assign_path=fold__dir+'/train_soft_assign.npz',
                       test_soft_assign_path=fold__dir+'/test_soft_assign.npz',
                       model_results_path=model_results_path)
        hmmModel.hmm(override_model=True, override_soft_assign=override_soft_assign, n_components=6)
        hmmModel.confusion_mat()
        df_correct_1, df_correct_cum = hmmModel._read_results()
        df_correct_1_sum = df_correct_1_sum.add(df_correct_1, fill_value=0)
        df_correct_cum_sum = df_correct_cum_sum.add(df_correct_cum, fill_value=0)
        os.system('say "Confusion matrix for fold"')
        confusion_mat(df_1=df_correct_1, df_cum=df_correct_cum, show_plot=True)

    return df_correct_1_sum, df_correct_cum_sum


def plot_results(paco, models_folder):
    if paco:
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
    print(i)
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
    df_correct_1_sum, df_correct_cum_sum = perform_cv(paco=True, cross_val_dir='../data/paco/10_fold_cross_val', models_folder='../models/paco/output/10_fold_cross_val', override_merged_centers=False, merged_centers_thresh=0)
    plot_results(paco=True, models_folder='../models/paco/output/10_fold_cross_val')
    # for i in range(10):
    #     df_correct_1, df_correct_cum = _read_model_results('../models/paco/output/10_fold_cross_val/n_b_size thresh=0_5/model_output_' + str(i) + '.npz', emotions = ['ang', 'hap', 'neu', 'sad'])
    #     confusion_mat(df_1=df_correct_1, df_cum=df_correct_cum, show_plot=True)





