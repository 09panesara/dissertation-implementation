from utils import data_utils
from merge_posesets import merge_centers_paco
from HMM.hmm import HMM
import os
import pandas as pd
import numpy as np
from utils.plot_results import confusion_mat, _read_model_results



''' Carries out 10fold cross val pipeline (with clustering offloaded onto Google colab notebook) '''
''' Split into 10 folds -> Clustering on train for each validation fold (offloaded) -> merge posesets -> hmm -> average results'''


cross_val_dir = '../data/paco/10_fold_cross_val'
no_folds = 10

emotions = ['ang', 'hap', 'neu', 'sad']

models_folder = '../models/paco/output/10_fold_cross_val'



def perform_cv(override_merged_centers=False):
    override_soft_assign = override_merged_centers
    # initialise dataframes for recognition rate at position 1, recognition rate at pos 2
    df_correct_1_avg = pd.DataFrame(np.zeros((len(emotions), len(emotions))), index=emotions, columns=emotions)
    df_correct_cum_avg = pd.DataFrame(np.zeros((len(emotions), len(emotions))), index=emotions, columns=emotions)
    for f in range(1,10):
        fold = str(f)
        df = pd.read_hdf('../data/paco/10_fold_cross_val/LMA_features_cross_val.h5')
        fold__dir = cross_val_dir + '/fold' + fold
        clusters_dir = fold__dir + '/clusters'
        if len(os.listdir(clusters_dir)) == 0:
            print('No clusters found for fold ' + fold + '.')
            continue
        if not os.path.isfile(fold__dir+'/merged_centers.npz') or override_merged_centers:
            merge_centers_paco(thresh=450000000, dir=clusters_dir)
        model_results_path = models_folder + '/model_output_' + fold + '.npz'
        train = df.loc[df['fold']!=f]
        test = df.loc[df['fold']==f]
        del df
        hmmModel = HMM(paco=True, train=train, test=test,
                       train_soft_assign_path=fold__dir+'/train_soft_assign.npz',
                       test_soft_assign_path=fold__dir+'/test_soft_assign.npz',
                       model_results_path=model_results_path)
        hmmModel.hmm(override_model=True, override_soft_assign=override_soft_assign, n_components=5)
        hmmModel.confusion_mat()
        df_correct_1, df_correct_cum = hmmModel._read_results()
        df_correct_1_avg = df_correct_1_avg.add(df_correct_1, fill_value=0)
        df_correct_cum_avg = df_correct_cum_avg.add(df_correct_cum, fill_value=0)
        os.system('say "Confusion matrix for fold"')
        confusion_mat(df_1=df_correct_1, df_cum=df_correct_cum, show_plot=True)

    return df_correct_1_avg, df_correct_cum_avg

def plot_results():
    # get 10 results
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

    assert i == no_folds
    # calc average for 10 folds
    df_correct_1_avg = df_correct_1_avg.divide(no_folds)
    df_correct_cum_avg = df_correct_cum_avg.divide(no_folds)
    # write results
    print("Writing recognition results to " + models_folder + '/10_fold_cv_results.h5')
    df_correct_1_avg.to_hdf(models_folder + '/10_fold_cv_results.h5', key='df_RR_1_avg', mode='w')
    df_correct_cum_avg.to_hdf(models_folder + '/10_fold_cv_results.h5', key='df_RR_cum_avg', mode='w')
    print('Done')
    # confusion matrix
    confusion_mat(df_1=df_correct_1_avg, df_cum=df_correct_cum_avg, show_plot=True)


if __name__ == '__main__':
#     df_correct_1_avg, df_correct_cum_avg = perform_cv(override_merged_centers=False)
    plot_results()









