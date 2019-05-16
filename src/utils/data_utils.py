import numpy as np
import pandas as pd
import os
import glob
import re
import shutil
from utils.split_train_test import split_train_test, cross_val
from moviepy.editor import VideoFileClip
import ast

# Joints in H3.6M -- data has 32 joints, but only 17 that move; these are the indices.
H36_INDICES_3D_POSE_BASELINE = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]
H36_INDICES_3D_POSE_BASELINE = [[i*3, i*3+1, i*3+2] for i in H36_INDICES_3D_POSE_BASELINE]
H36_NAMES = ['Hip', 'RHip', 'RKnee', 'RFoot', 'LHip', 'LKnee', 'LFoot', 'Spine', 'Thorax', 'Neck/Nose', 'Head', 'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist']

def join_3d_keypoints(keypoints_folder, output_dir='../../data'):
    '''
    Takes keypoints npz files from videopose on each of the videos, and concatenates into one npz file
    :param keypoints_folder: path to videopose keypoints files
    :param output_dir: path to output concatenated npz file
    '''
    keypoints_list = list(glob.iglob(keypoints_folder + '/*.npz'))
    no_keypoints_files = len(keypoints_list)

    keypoints_3D = {}

    for i, file in enumerate(keypoints_list):
        filename = os.path.basename(file)[len('output')+1:-4] # keypoint filenames befin with 'output_'
        subject = re.search('\d+f|\d+m', filename).group()
        subject_end_pos = [m.end() for m in re.finditer(subject, filename)][0]
        action = re.search('[a-zA-Z]+(1|2)', filename[subject_end_pos:]).group()
        action_end_pos = [m.end() for m in re.finditer(action, filename)][0]
        emotion = filename[action_end_pos+1:action_end_pos+4]
        intensity = filename[-3:]

        keypoints = np.load(file, encoding='latin1')['positions_3d']

        print(str(i+1) + "/"+ str(no_keypoints_files) + ": " + filename)
        if subject not in keypoints_3D:
            keypoints_3D[subject] = {}
        if action not in keypoints_3D[subject]:
            keypoints_3D[subject][action] = {}
        if emotion not in keypoints_3D[subject][action]:
            keypoints_3D[subject][action][emotion] = {}
        if intensity not in keypoints_3D[subject][action][emotion]:
            keypoints_3D[subject][action][emotion][intensity] = [None, None, None, None]

        keypoints_3D[subject][action][emotion][intensity] = keypoints.astype('float32')

    print('Saving...')
    np.savez_compressed(output_dir+'/videopose_keypoints', positions_3d=keypoints_3D)
    print('Done.')



def load_action_db_keypoints(keypoints_folder='../data/action_db', kpts_filename = 'hmr_3d_keypoints.npz', normalised=False):
    ''' Loads 3d-pose-baseline keypoints  '''

    assert(os.path.isdir(keypoints_folder))
    if not os.path.isfile(keypoints_folder + '/' + kpts_filename):
        print('3D keypoints file not found')
        return pose_baseline_to_h36m('../data/action_db/3d-pose-baseline')
    else:
        if normalised:
            print('Loading Actions DB hmr 3d normalised keypoints...')
            return np.load(keypoints_folder + '/' + 'normalised_keypoints.npz', encoding='latin1')['positions_3d'].item()
        else:
            print('Loading Actions DB hmr 3d keypoints...')
            return np.load(keypoints_folder + '/' + kpts_filename, encoding='latin1')['positions_3d'].item()



def pose_baseline_to_h36m(path, output_dir='../data/action_db'):
    '''
    Converts 3d-pose-baseline keypoints into H36M keypoints, aggregates into one file <output_dir>/3dpb-keypoints.npz
    :param path:
    :param output_dir:
    :return: NA
    '''
    print('Generating single keypoints file from keypoints ...')
    kpts_list = list(glob.iglob(path + '/*.npz'))
    # TODO Missing left hip -> replace with right hip value
    positions_3d = {}
    for file in kpts_list:
        name = os.path.basename(file)[:-4]
        subject = str(int(name[:3])) + str(name[3])
        emotion = name[5:8]
        intensity = name[9:12]
        if (float(intensity) < 5 and emotion != 'neu') or (emotion == 'neu' and float(intensity) < 3.0):
            continue
        action = 'walking'

        kpts = np.load(file, encoding='latin1')['positions_3d']
        curr_positions = [frame[0] for frame in kpts]
        curr_positions = [[frame[xyz] for xyz in H36_INDICES_3D_POSE_BASELINE] for frame in curr_positions]

        if subject not in positions_3d:
            positions_3d[subject] = {}
        if action not in positions_3d[subject]:
            positions_3d[subject][action] = {}
        if emotion not in positions_3d[subject][action]:
            positions_3d[subject][action][emotion] = {}
        if intensity not in positions_3d[subject][action][emotion]:
            positions_3d[subject][action][emotion][intensity] = []


        positions_3d[subject][action][emotion][intensity].append(curr_positions)

    print('Saving...')
    np.savez_compressed(output_dir + '/3dpb_keypoints.npz', positions_3d=positions_3d)
    print('Done.')
    return positions_3d




def get_timestep(timesteps_path, videos_dir='../VideoPose3D/videos/walking_videos', keypoints_dir='../data/action_db', normalised=True):
    # loads timesteps for actions database videos
    if os.path.isfile(timesteps_path):
        print('Loading timesteps...')
        return np.load(timesteps_path, encoding='latin1')['timesteps'].item()
    else:
        print('Generating timesteps...')
        timesteps = {}
        openpose_dir = '../3d-pose-baseline/walking_openpose/'

        positions_3d = load_action_db_keypoints(keypoints_dir, normalised=normalised)
        for subject in positions_3d:
            for action in positions_3d[subject]:
                for emotion in positions_3d[subject][action]:
                    for intensity in positions_3d[subject][action][emotion]:
                        for i, kpts in enumerate(positions_3d[subject][action][emotion][intensity]):
                            if subject not in timesteps:
                                timesteps[subject] = {}
                            if action not in timesteps[subject]:
                                timesteps[subject][action] = {}
                            if emotion not in timesteps[subject][action]:
                                timesteps[subject][action][emotion] = {}
                            if intensity not in timesteps[subject][action][emotion]:
                                timesteps[subject][action][emotion][intensity] = []
                            try:
                                base_filename = (4 - len(subject)) * '0' + subject + '_' + emotion + '_' + intensity + '_' + 'win_' + str(i+1)
                                op_kpts_folder = base_filename + '/'
                                no_frames = len(glob.glob(openpose_dir + op_kpts_folder + '*.json'))
                                video = videos_dir + '/' + base_filename + '.wmv'
                                # divide by no. frames openpose was able to obtain
                                # since openpose tries to operate at fastest frame rate it can
                                vid_duration = VideoFileClip(video).duration
                                timestep = vid_duration / float(no_frames)
                                timesteps[subject][action][emotion][intensity].append(timestep)
                            except:
                                print('Could not find timestep for %s, %s, %s, %s' %(subject, 'Walking' + str(i+1), emotion, intensity))
                                continue


        np.savez_compressed(timesteps_path, timesteps=timesteps)
        return timesteps


def load_paco_keypoints(keypoints_folder='../data/paco', kpts_filename='paco_keypoints.npz', normalised_by=None):
    if not normalised_by:
        print('Loading unnormalised paco 3d keypoints...')
        return np.load(keypoints_folder + '/' + kpts_filename, encoding='latin1')['positions_3d'].item()
    elif normalised_by =='space':
        print('Loading paco kpts normalised by space...')
        return np.load(keypoints_folder + '/normalised_keypoints.npz', encoding='latin1')['positions_3d'].item()
    elif normalised_by =='size':
        print('Loading paco kpts normalised by size...')
        return np.load(keypoints_folder + '/normalised_by_size_kpts.npz', encoding='latin1')['positions_3d'].item()


def load_LMA(folder):
    if 'paco' in folder:
        print('Loading Paco LMA Features...')
        df = pd.DataFrame()
        paco_emotions = ['ang', 'hap', 'neu', 'sad']
        for emotion in paco_emotions:
            df_emotion = pd.read_hdf(folder+'/LMA_features_' + emotion + '.h5')
            df = df.append(df_emotion)

        print('no rows: ' + str(len(df)))
        return df
    print('Loading Action DB LMA Features...')
    return pd.read_hdf(folder + "/LMA_features.h5")



def get_train_test_set(folder='../data/action_db'):
    if not os.path.isfile(folder+ '/training/train_data.h5') and os.path.isfile(folder+ '/training/train_data.csv') and os.path.isfile(folder + '/test/test_data.h5'):
        print('Getting train from csv, test data')
        train = pd.read_csv(folder + '/training/train_data.csv').iloc[:, 1:]
        test = pd.read_hdf(folder + '/test/test_data.h5')
    elif not os.path.isfile(folder + '/training/train_data.h5') and not os.path.isfile(folder + '/test/test_data.h5'):
        print('Generating train and test datasets')
        LMA = load_LMA(folder)
        train, test = split_train_test(LMA, 80, 20, folder)
    else:
        print('Getting train, test data')
        train = pd.read_hdf(folder + '/training/train_data.h5')
        test = pd.read_hdf(folder + '/test/test_data.h5')
    return train, test


def get_cross_val(cross_val_dir='../data/paco/10_fold_cross_val'):
    if not os.path.isfile(cross_val_dir+'/LMA_features_cross_val.h5'):
        print('Performing 10 cross fold split from LMA features')
        if 'paco' in cross_val_dir:
            LMA_features = load_LMA('../data/paco')
        else:
            LMA_features = load_LMA('../data/action_db')
        df = cross_val(LMA_features, cross_val_dir)
    else:
        print('Getting 10 cross fold split')
        df = pd.read_csv(cross_val_dir+'/LMA_features_cross_val.csv').iloc[:,1:]
    return df


def convert_to_list(s):
    try:
        return ast.literal_eval(s)
    except:
        return [float(item) for item in s]

if __name__ == '__main__':
    join_3d_keypoints('../../VideoPose3D/output/keypoints')