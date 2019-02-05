import numpy as np
import pandas as pd
import os
import glob
import re
import shutil
from utils.split_train_test import split_train_test
from moviepy.editor import VideoFileClip

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


def load_3d_keypoints(keypoints_folder='../data', kpts_filename = '3dpb_keypoints.npz'):
    print('Loading 3d keypoints...')
    assert(os.path.isdir(keypoints_folder))
    if not os.path.isfile(keypoints_folder + '/' + kpts_filename):
        return pose_baseline_to_h36m(keypoints_folder + '/3d-pose-baseline')
    else:
        return np.load(keypoints_folder + '/' + kpts_filename, encoding='latin1')['positions_3d'].item()


def pose_baseline_to_h36m(path, output_dir='../data'):
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
        if float(intensity) < 5:
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




def get_timestep(timesteps_path, videos_dir='../VideoPose3D/videos/walking_videos'):
    # TODO
    if os.path.isfile(timesteps_path):
        print('Loading timesteps...')
        return np.load(timesteps_path, encoding='latin1')['timesteps'].item()
    else:
        print('Generating timesteps...')
        timesteps = {}
        openpose_dir = '../data/filtered_openpose/'

        positions_3d = load_3d_keypoints('../data/')
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

# def clean_LMA(path='../data/LMA_features.h5'):
#     df = pd.read_hdf(path)
#     column = df.columns.values
#     for row in df:
#         if

def load_LMA(path='../data/LMA_features.h5'):
    return pd.read_hdf(path)

def get_train_test_set(folder='../data'):
    if not os.path.isfile(folder + '/' + 'train_data.h5') and not os.path.isfile(folder + '/' + 'test_data.h5'):
        print('Generating train and test datasets')
        LMA = load_LMA()
        train, test = split_train_test(LMA, 80, 20)
    else:
        print('Getting train, test data')
        train = pd.read_hdf(folder + '/train_data.h5')
        test = pd.read_hdf(folder + '/test_data.h5')
    return train, test



if __name__ == '__main__':
    join_3d_keypoints('../../VideoPose3D/output/keypoints')