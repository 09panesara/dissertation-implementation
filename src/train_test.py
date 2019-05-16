from utils import data_utils
from merge_posesets import merge_centers
from HMM.hmm import HMM
from HMM.hmm_gmm import HMMGMM
import os
import pandas as pd
import numpy as np
from utils.plot_results import confusion_mat, _read_model_results

''' Carries out HMM on train-test set '''


def test(paco, models_folder, train_test_dir, override_merged_centers=False, thresh=0):
    if paco:
        emotions = ['ang', 'hap', 'neu', 'sad']
    else:
        emotions = ['ang', 'fea', 'hap', 'neu', 'sad', 'unt']
    override_soft_assign = override_merged_centers
    # initialise dataframes for recognition rate at position 1, recognition rate at pos 2
    train = pd.read_csv(train_test_dir + '/training/train_data.csv').iloc[:, 1:]
    test = pd.read_hdf(train_test_dir + '/test/test_data.h5')
    clusters_dir = train_test_dir + '/clusters'
    if not os.path.isfile(train_test_dir+'/merged_centers.npz') or override_merged_centers:
        merge_centers(emotions, thresh=thresh, dir=clusters_dir)
    model_results_path = models_folder + '/model_output.npz'

    hmmModel = HMM(paco=paco, train=train, test=test,
                   train_soft_assign_path=train_test_dir+'/train_soft_assign.npz',
                   test_soft_assign_path=train_test_dir+'/test_soft_assign.npz',
                   model_results_path=model_results_path)
    hmmModel.hmm(override_model=True, override_soft_assign=override_soft_assign, n_components=6)

    hmmModel.confusion_mat()
    df_correct_1, df_correct_cum = hmmModel._read_results()

    # write results
    # print("Writing recognition results to " + models_folder + '/train_test_results.h5')
    # df_correct_1.to_hdf(models_folder + '/train_test_results.h5', key='df_RR_1', mode='w')
    # df_correct_cum.to_hdf(models_folder + '/train_test_results.h5', key='df_RR_cum', mode='w')
    # print('Done')

    # confusion matrix
    confusion_mat(df_1=df_correct_1, df_cum=df_correct_cum, show_plot=True)

if __name__ =='__main__':
    test(paco=True, models_folder='../models/paco/output/train-test', train_test_dir='../data/paco/train-test', override_merged_centers=False, thresh=300000000)
    # df_correct_1, df_correct_cum = _read_model_results('../models/action_db/output/train-test/model_output.npz', emotions=['ang', 'fea', 'hap', 'neu', 'sad', 'unt'])
    # confusion_mat(df_1=df_correct_1, df_cum=df_correct_cum, show_plot=True)






