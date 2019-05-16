import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import math
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.patches as patches
from scipy import interp
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib import cm

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
    ax1.title.set_text('Recognition at position 1')
    sns.set(font_scale=1.4)  # for label size
    sns.heatmap(df_1, annot=True, annot_kws={"size": 16}, ax=ax1, cmap="Blues")  # font size
    # ax1.set_xlabel('Predicted Emotion at Position 1')
    # ax1.set_ylabel('Ground truth emotion')

    ax2.title.set_text('Cumulative recognition at positions 1 or 2')
    sns.set(font_scale=1.4)  # for label size
    sns.heatmap(df_cum, annot=True, annot_kws={"size": 16}, ax=ax2, cmap="Blues")  # font size
    # ax2.set_xlabel('Cumulative Predicted Emotion at Positions 1, 2')
    # ax2.set_ylabel('Ground truth emotion')
    if show_plot:
        plt.show()

def confusion_mat_from_arr(pos_1, cum, emotions):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 12))
    ax1.title.set_text('Recognition at position 1')
    sns.set(font_scale=1.4)  # for label size
    sns.heatmap(pos_1, annot=True, annot_kws={"size": 16}, ax=ax1, cmap="Blues", xticklabels=emotions, yticklabels=emotions)  # font size
    # ax1.set_xlabel('Predicted Emotion at Position 1')
    # ax1.set_ylabel('Ground truth emotion')

    ax2.title.set_text('Cumulative recognition at positions 1 or 2')
    sns.set(font_scale=1.4)  # for label size
    sns.heatmap(cum, annot=True, annot_kws={"size": 16}, ax=ax2, cmap="Blues", xticklabels=emotions, yticklabels=emotions)  # font size
    # ax2.set_xlabel('Cumulative Predicted Emotion at Positions 1, 2')
    # ax2.set_ylabel('Ground truth emotion')
    plt.show()


def macro_F1_precision_recall(results):
    ''' takes in results (confusion matrix) as an array and returns average of f1-scores computed label-wise '''
    results = np.array(results)
    n_emotions = len(results)
    tp = np.diag(results)
    fp = [np.sum(results[:,i])-results[i,i] for i in range(n_emotions)]
    fn = [np.sum(results[i,:])-results[i,i] for i in range(n_emotions)]

    precision = [tp[i] / (tp[i]+fp[i]) for i in range(n_emotions)]
    recall = [tp[i] / (tp[i] + fn[i]) for i in range(n_emotions)]
    f1_score = [0 for i in range(n_emotions)]

    for i in range(n_emotions):
        curr_f1_score = 2*precision[i]*recall[i] / (precision[i]+recall[i])
        if math.isnan(curr_f1_score):
            # defined according to https://github.com/dice-group/gerbil/wiki/Precision,-Recall-and-F1-measure
            if tp[i] == 0 and fp[i] == 0 and fn[i] == 0:
                curr_f1_score = 1
                precision[i] = 1
                recall[i] = 1
            else: # fp or fn = 0 and the other one isn't 0
                curr_f1_score = 0
                precision[i] = 0
                recall[i] = 0
        f1_score[i] = curr_f1_score


    f1_score = np.sum(f1_score) / n_emotions
    precision = np.sum(precision) / n_emotions
    recall = np.sum(recall) / n_emotions

    return f1_score, precision, recall


def micro_F1_precision_recall(results):
    ''' takes in results as an array and returns f1 score of global results i.e. ignoring label '''
    results = np.array(results)
    n_emotions = len(results)
    tp = np.sum(np.diag(results))
    fp = np.sum([np.sum(results[:, i]) - results[i,i] for i in range(n_emotions)])
    fn = np.sum([np.sum(results[i, :]) - results[i,i] for i in range(n_emotions)])

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * precision * recall / (precision + recall)

    if math.isnan(f1_score):
        # defined according to https://github.com/dice-group/gerbil/wiki/Precision,-Recall-and-F1-measure
        if tp == 0 and fp == 0 and fn == 0:
            f1_score = 1
            precision = 1
            recall = 1
        else:  # fp or fn = 0 and the other one isn't 0
            f1_score = 0
            precision = 0
            recall = 0

    return f1_score, precision, recall


def f1_scores_table(model_output_folder, output_file_path):
    ''' Calculates f1 scores (micro, macro) per subject on LOSO and outputs to csv to be able to be plotted '''
    if 'paco' in model_output_folder:
        emotions = ['ang', 'hap', 'neu', 'sad']
    else:
        emotions = ['ang', 'fea', 'hap', 'neu', 'sad', 'unt']

    df_scores = pd.DataFrame({'label': [], 'micro_F1': [], 'macro_F1': []})

    for file in os.listdir(model_output_folder):
        if file.endswith('.npz') and file.startswith('model_output_'):
            label = file[13:-4]
            print('Calculating scores for ' + label)
            df_correct_1, _ = _read_model_results(model_output_folder + '/' + file, emotions)
            results = np.array(df_correct_1)
            micro_f1_score, _,_ = micro_F1_precision_recall(results)
            macro_f1_score, _, _ = macro_F1_precision_recall(results)
            n = results.sum()
            z = 1.96
            micro_conf_UB_95 = macro_f1_score + z * math.sqrt(micro_f1_score * (1 - micro_f1_score) / results.sum())
            micro_conf_LB_95 = macro_f1_score - z * math.sqrt(micro_f1_score * (1 - micro_f1_score) / results.sum())
            micro_CI_error = micro_conf_UB_95 - micro_conf_LB_95
            macro_conf_UB_95 = macro_f1_score + z * math.sqrt(macro_f1_score * (1 - macro_f1_score) / results.sum())
            macro_conf_LB_95 = macro_f1_score - z * math.sqrt(macro_f1_score*(1-macro_f1_score) / results.sum())
            macro_CI_error = macro_conf_UB_95 - macro_conf_LB_95
            scores = {'label': label,
                      'micro_F1': micro_f1_score,
                      'macro_F1': macro_f1_score,
                      'micro_conf_UB_95': micro_conf_UB_95,
                      'micro_conf_LB_95': micro_conf_LB_95,
                      'micro_CI_error': micro_CI_error,
                      'macro_conf_UB_95': macro_conf_UB_95,
                      'macro_conf_LB_95': macro_conf_LB_95,
                      'macro_CI_error': macro_CI_error
                      }
            df_scores = df_scores.append(scores, ignore_index=True)
    df_scores = df_scores.sort_values('label')
    df_scores.to_csv(output_file_path)




def plot_ROC_curve(model_output_folder, CV=True):
    ''' TO COME BACK TO '''
    if 'paco' in model_output_folder:
        emotions = ['ang', 'hap', 'neu', 'sad']
    else:
        emotions = ['ang', 'fea', 'hap', 'neu', 'sad', 'unt']

    # plot arrows
    fig1 = plt.figure(figsize=[12, 12])
    ax1 = fig1.add_subplot(111, aspect='equal')
    ax1.add_patch(
        patches.Arrow(0.45, 0.5, -0.25, 0.25, width=0.3, color='green', alpha=0.5)
    )
    ax1.add_patch(
        patches.Arrow(0.5, 0.45, 0.25, -0.25, width=0.3, color='red', alpha=0.5)
    )

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for file in os.listdir(model_output_folder):
        if file.endswith('.npz') and file.startswith('model_output_'):
            label = file[13:-4] # get subject name/fold number
            print('Calculating scores for ' + label)

        gts = []
        preds = []
        df_correct_1, _ = _read_model_results(model_output_folder + '/' + file, emotions)
        results = np.array(df_correct_1)

        n_emotions = len(results)
        tp = np.diag(results)

        fp = np.array([np.sum(results[:, i]) - results[i, i] for i in range(n_emotions)])
        fn = np.array([np.sum(results[i, :]) - results[i, i] for i in range(n_emotions)])
        print(fp)
        print(fn)
        print(tp)
        print(results.sum())
        tn = np.array([results.sum()]*n_emotions - (fp + fn + tp))
        print(tn)
        fpr = fp / (fp + tn)
        tpr = tp / (tp + fn)

        # Compute ROC curve and ROC area for each class
        n_classes = len(emotions)
        roc_auc = dict()
        print(fpr)
        print(tpr)
        # calculate tpr, fpr for multiclass globally
        for i in range(n_classes):
            roc_auc[i] = auc(fpr[i], tpr[i])
            print(roc_auc[i])

        # for i in range(len(results)):
        #     row = results[i,:]
        #     gt = [emotions[i]]*int(np.sum(row))
        #     gt_pred = [emotions[j] for j, count in enumerate(row) for n in range(int(count)) ]
        #     gts += gt
        #     preds += gt_pred

        # Compute micro-average ROC curve and ROC area
        # fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # get one vs all for roc curve multi-class plot

        # Plot of a ROC curve for a specific class
        # plt.figure()
        # plt.plot(fpr[2], tpr[2], label='ROC curve (area = %0.2f)' % roc_auc[2])
        # plt.plot([0, 1], [0, 1], 'k--')
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('Receiver operating characteristic example')
        # plt.legend(loc="lower right")
        # plt.show()

        # Plot ROC curve
        plt.figure()
        # plt.plot(fpr["micro"], tpr["micro"],
        #          label='micro-average ROC curve (area = {0:0.2f})'
        #                ''.format(roc_auc["micro"]))
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                           ''.format(emotions[i], roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right")
        plt.show()

def _calc_confidence_interval(scores):
    '''
    :param: val = micro-F1/whatever accuracy value
    :param: n = no. samples = size of dataset being tested
    Calculates 95% confidence interval and returns LB, UB
    '''
    n = len(scores)
    x = [1.96 * math.sqrt(score*(1-score)/n) for score in scores]
    return x



def barplot_LOSO_f1_scores(csv_paths, titles, colors, paco=False):
    results = [pd.read_csv(csv_path) for csv_path in csv_paths]
    no_subjects = len(results[0])
    subjects = list(results[0].loc[:,'subject'])

    fig, ax = plt.subplots()

    ind = np.arange(no_subjects) # the x locations for the groups
    width = 0.35  # the width of the bars
    db = 'Body Movement Library' if paco else 'Action Database'
    no_results = len(results)

    micro_F1_y = [df['micro_F1'] for df in results]
    micro_F1_errors = [_calc_confidence_interval(f1_scores) for f1_scores in micro_F1_y]
    macro_F1_y = [df['macro_F1'] for df in results]
    macro_F1_errors = [_calc_confidence_interval(f1_scores) for f1_scores in macro_F1_y]
    print([(f1_score, macro_F1_errors[0][i]) for i, f1_score in enumerate(macro_F1_y)])


    # Plot macro-F1 scores
    for i in range(no_results):
        # micro_F1_y = df['micro_F1']
        # micro_F1_errors = _calc_confidence_interval(micro_F1_y)
        # macro_F1_y = df['macro_F1']
        # macro_F1_errors = _calc_confidence_interval(macro_F1_y)
        ax.bar(ind+i*width, micro_F1_y[i], width, color=colors[i], yerr=micro_F1_errors[i])



    ax.set_title('Micro-F1 Scores per subject in LOSO Cross-Validation on ' + db)
    ax.set_xticks(ind + width / no_results)
    ax.set_xticklabels(subjects)
    #
    ax.legend(titles)
    ax.autoscale_view()
    plt.xlabel('Subject')
    plt.ylabel('Micro-F1 Score')
    plt.ylim(bottom=0)
    plt.show()

    fig, ax = plt.subplots()
    # Plot macro-F1 scores
    for i in range(no_results):
        ax.bar(ind+i*width, macro_F1_y[i], width, color=colors[i], yerr=macro_F1_errors[i])

    ax.set_title('Macro-F1 Scores per subject in LOSO Cross-Validation on ' + db)
    ax.set_xticks(ind + width / no_results)
    ax.set_xticklabels(subjects)
    #
    ax.legend(titles)
    ax.autoscale_view()
    plt.xlabel('Subject')
    plt.ylabel('Macro-F1 Score')
    plt.ylim(bottom=0)
    plt.show()


def barplot_10_fold_f1_scores(csv_paths, titles, colors, paco=False):
    results = [pd.read_csv(csv_path) for csv_path in csv_paths]
    no_subjects = len(results[0])
    folds = list(results[0].loc[:,'label'])

    fig, ax = plt.subplots()

    ind = np.arange(no_subjects) # the x locations for the groups
    width = 0.35  # the width of the bars
    db = 'Body Movement Library' if paco else 'Action Database'
    no_results = len(results)

    micro_F1_y = [df['micro_F1'] for df in results]
    micro_F1_errors = [_calc_confidence_interval(f1_scores) for f1_scores in micro_F1_y]
    macro_F1_y = [df['macro_F1'] for df in results]
    macro_F1_errors = [_calc_confidence_interval(f1_scores) for f1_scores in macro_F1_y]
    print([(f1_score, macro_F1_errors[0][i]) for i, f1_score in enumerate(macro_F1_y)])


    # Plot macro-F1 scores
    for i in range(no_results):
        ax.bar(ind+i*width, micro_F1_y[i], width, color=colors[i], yerr=micro_F1_errors[i])


    ax.set_title('Micro-F1 Scores per Fold in 10-Fold Cross-Validation on ' + db)
    ax.set_xticks(ind + width / no_results)
    ax.set_xticklabels(folds)
    #
    ax.legend(titles)
    ax.autoscale_view()
    plt.xlabel('Fold')
    plt.ylabel('Micro-F1 Score')
    plt.ylim(bottom=0)
    plt.show()

    fig, ax = plt.subplots()
    # Plot macro-F1 scores
    for i in range(no_results):
        ax.bar(ind+i*width, macro_F1_y[i], width, color=colors[i], yerr=macro_F1_errors[i])

    ax.set_title('Macro-F1 Scores per Fold in 10-Fold Cross-Validation on ' + db)
    ax.set_xticks(ind + width / no_results)
    ax.set_xticklabels(folds)
    #
    if len(titles) > 1:
        ax.legend(titles)
    ax.autoscale_view()
    plt.xlabel('Fold')
    plt.ylabel('Macro-F1 Score')
    plt.ylim(bottom=0)
    plt.show()



if __name__ == '__main__':
    pos_1 = np.array([[12, 2, 8, 1, 0, 0], [5, 0, 8, 3, 3, 3], [13, 3, 8, 5, 0, 2], [3, 2, 7, 13, 4, 8], [1, 5, 0, 5, 7, 7], [3, 3, 3, 11, 3, 10]])

    cum = np.array([[20, 2, 22, 1, 0, 1], [10, 2, 15, 6, 5, 6], [18, 7, 22, 7, 2, 6], [10, 6, 11, 21, 4, 22], [2, 11, 1, 11, 9, 16], [5, 8, 8, 19, 5, 21]])
    confusion_mat_from_arr(pos_1, cum, emotions=['ang', 'fea', 'hap', 'neu', 'sad', 'unt'])
    # colors =['#4171b5', '#9be7ff']
    colors = ['#2c145e', '#77f1ff']
    # f1_scores_table('../../models/paco/output/10_fold_cross_val/v1-thresh=0', '../../data/paco/10_fold_cross_val/10_fold_cross_val_f1_scores_thresh=0.csv')
    # barplot_10_fold_f1_scores(['../../data/paco/10_fold_cross_val/10_fold_cross_val_f1_scores_thresh=0.csv', '../../data/paco/10_fold_cross_val/10_fold_cross_val_f1_scores_thresh=450000000.csv'],
    #                     titles=['thresh=0', 'thresh=450000000'], colors=colors, paco=True)
