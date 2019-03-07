import pandas as pd
from matplotlib.pylab import plt
import  numpy as np
from utils.data_utils import convert_to_list
import os
''' split into emotions
per feature, visualise over time
'''
PACO = True

if PACO:
    EMOTIONS = ['ang', 'fea', 'hap', 'sad', 'neu']
    colors = {'ang': 'r', 'fea': 'm', 'hap': 'y', 'sad': 'c', 'neu': 'k'}
    subject = "ale"
    plots_dir = '../plots/paco/feature_analysis'

print('Loading data')
LMA_train = pd.read_csv('../data/paco/training/train_data.csv').iloc[:, 1:]
subjects = list(set(LMA_train['subject']))
print(subjects)
for subject in subjects:
    os.mkdir(plots_dir + "/" + subject)
    subject_df = LMA_train.loc[LMA_train['subject'] == subject]
    per_emotion_data = {emotion: subject_df.loc[subject_df['emotion'] == emotion] for emotion in EMOTIONS}
    for emotion in EMOTIONS:
        print(len(per_emotion_data[emotion]))
    features = LMA_train.columns.values
    rmv_features = ['subject', 'emotion', 'timestep_btwn_frame', 'action']
    features = [feature for feature in features if feature not in rmv_features]


    for feature in features:
        print("Feature analysis for feature " + feature)
        data = {emotion: per_emotion_data[emotion][feature] for emotion in EMOTIONS}
        data = {emotion: [convert_to_list(vid) for vid in data[emotion]] for emotion in EMOTIONS}
        data = {emotion: np.array(data[emotion]) for emotion in EMOTIONS}

        for emotion in EMOTIONS:
            print(emotion)
            print("Plotting for emotion " + emotion)
            emotion_data = data[emotion]
            for vid in emotion_data:
                plt.plot(np.arange(len(vid)), vid, label=emotion, color=colors[emotion])
        plt.title(feature + " over time for subject " + subject)
        # plt.legend([colors[emotion] for emotion in EMOTIONS], EMOTIONS)
        plt.legend()
        print('Saving plot...')
        if 'Neck/Nose' in feature:
            feature_name = feature.replace('Neck/Nose', 'Neck')
        else:
            feature_name = feature

        plt.savefig(plots_dir + "/" + subject + "/" + feature_name + '.png')
        plt.close()
        print('Done.')
