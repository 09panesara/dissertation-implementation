import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def _read_model_results(model_results_path, emotions):
    print('Reading results from ' + model_results_path + '...')
    n_emotions = len(emotions)
    results = np.load(model_results_path, encoding='latin1')
    pred_pos_1 = results['pred_pos_1'].item()
    pred_pos_2 = results['pred_pos_2'].item()

    conf_mat_pred_pos_1 = np.zeros((n_emotions, n_emotions))
    conf_mat_pred_pos_cum = np.zeros((n_emotions, n_emotions))

    for i, emotion1 in enumerate(emotions):
        for j, emotion2 in enumerate(emotions):
            conf_mat_pred_pos_1[i][j] = pred_pos_1[emotion1][emotion2]
            conf_mat_pred_pos_cum[i][j] = pred_pos_1[emotion1][emotion2] + pred_pos_2[emotion1][emotion2]

    df_correct_1 = pd.DataFrame(conf_mat_pred_pos_1, index=emotions, columns=emotions)
    df_correct_cum = pd.DataFrame(conf_mat_pred_pos_cum, index=emotions, columns=emotions)
    return df_correct_1, df_correct_cum


def confusion_mat(df_1, df_cum, show_plot=False):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 12))
    ax1.title.set_text('Recognition rate at position 1')
    sns.set(font_scale=1.4)  # for label size
    sns.heatmap(df_1, annot=True, annot_kws={"size": 16}, ax=ax1)  # font size

    ax1.set_ylabel('Ground truth emotion')

    ax2.title.set_text('Cumulative recognition rates for position 1, 2')
    sns.set(font_scale=1.4)  # for label size
    sns.heatmap(df_cum, annot=True, annot_kws={"size": 16}, ax=ax2)  # font size
    ax2.set_ylabel('Ground truth emotion')
    if show_plot:
        plt.show()

