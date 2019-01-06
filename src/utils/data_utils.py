import numpy as np
import os
import glob
import re

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


def load_3d_keypoints(keypoints_folder='../../data'):
    return np.load(keypoints_folder + '/videopose_keypoints.npz', encoding='latin1')['positions_3d'].item()

def load_data(folder='../../data'):
    return np.load(folder + '/data.npz', encoding='latin1')['features'].item()

def get_timestep(videos_dir):
    return [[]*400]

if __name__ == '__main__':
    join_3d_keypoints('../../VideoPose3D/output/keypoints')