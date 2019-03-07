from utils import data_utils
import math
import numpy as np
import os

''' H36M joint names '''
joints = {
    'Hip': 0,
    'RHip': 1,
    'RKnee': 2,
    'RFoot': 3,
    'LHip': 4,
    'LKnee': 5,
    'LFoot': 6,
    'Spine': 7,
    'Thorax': 8,
    'Neck/Nose': 9,
    'Head': 10,
    'LShoulder': 11,
    'LElbow': 12,
    'LWrist': 13,
    'RShoulder': 14,
    'RElbow': 15,
    'RWrist': 16
}

def median_shift_filter(keypoints, n=3):
    ''' performs median shift filter on keypoints by taking prev n and next n frames and computing median
    :param keypoints: 3D array of keypoint frames dimension n x 17 x 2 where n = no. frames
                      [[[x1,y1,z1],[x2,y2,z2],...], [...], ...]

    '''
    no_features = len(keypoints[0])
    assert (no_features > n)
    no_frames = len(keypoints)

    def _calc_median(prev_n, curr, next_n):
        frames = prev_n + [curr] + next_n
        x = [[xyz[0] for xyz in frame] for frame in frames]
        y = [[xyz[1] for xyz in frame] for frame in frames]
        z = [[xyz[2] for xyz in frame] for frame in frames]
        x = list(zip(*x))
        y = list(zip(*y))
        z = list(zip(*z))
        x = [np.median(list(features)) for features in x]
        y = [np.median(list(features)) for features in y]
        z = [np.median(list(features)) for features in z]
        assert len(x) == no_features
        assert len(y) == no_features
        assert len(z) == no_features
        median = [[x[i], y[i], z[i]] for i in range(no_features)]
        return median

    for frame_no in range(no_frames):
        curr = keypoints[frame_no]
        # stores prev n frames
        prev_n = [keypoints[i] for i in range(frame_no - n, frame_no) if i >= 0]
        # stores future n frames
        next_n = [keypoints[i] for i in range(frame_no + 1, frame_no + n) if i < no_frames]
        median = _calc_median(prev_n, curr, next_n)
        keypoints[frame_no] = median

    return keypoints


def linear_interpolate(keypoints):
    ''' perform linear interpolation to account for equipment failure '''
    # TODO


def normalise_space(keypoints):
    '''
    Transformations put shoulders and hip center in same plane parallel to yOz plane, put both shoulders at same height
    :param keypoints: 3D array of keypoint frames dimension n x 17 x 2 where n = no. frames
                      [[[x1,y1,z1],[x2,y2,z2],...], [...], ...]
    :return: normalised keypoints
    '''

    keypoints = np.array(keypoints)
    # mean center all positions such that pelvis (hip center) is (0,0,0)
    hip_centers = [frame[0] for frame in keypoints]
    normalised_kpts = [
        [[joint[0] - hip_centers[i], joint[1] - hip_centers[i], joint[2] - hip_centers[i]] for joint in frame] for
        i, frame in enumerate(keypoints)]
    # already normalised by position, i.e. z axis = up
    # make orientation invariant by making hips parallel to x axis by rotating around z axis i.e. rotating x, y coord
    thetas = [math.atan(frame[joints['LHip']][1] - frame[joints['RHip']][1]) / (
                frame[joints['LHip']][0] - frame[joints['RHip']][0]) for frame in keypoints]
    rotation_mat = [[[math.cos(theta), -1 * math.sin(theta)], [math.sin(theta), math.cos(theta)]] for theta in thetas]
    normalised_kpts = [
        [[np.dot(rotation_mat[i][1], [joint[1], joint[0]]), np.dot(rotation_mat[i][0], [joint[1], joint[0]]), joint[2]]
         for joint in frame] for i, frame in enumerate(normalised_kpts)]
    return np.array(normalised_kpts)




def smooth_paco_keypoints(redo_normalisation=False):
    ''' Carry out normalisation if not already done'''
    if not os.path.isfile('../data/paco/normalised_keypoints.npz') or redo_normalisation:
        print('Normalising paco keypoints')
        keypoints_3d = data_utils.load_paco_keypoints(normalised=False)
        normalised_keypoints = {}

        for subject in keypoints_3d:
            if subject not in normalised_keypoints:
                normalised_keypoints[subject] = {}
            for action in keypoints_3d[subject]:
                if action not in normalised_keypoints[subject]:
                    normalised_keypoints[subject][action] = {}
                for emotion in keypoints_3d[subject][action]:
                    if emotion not in normalised_keypoints[subject][action]:
                        normalised_keypoints[subject][action][emotion] = []
                    for i, data in enumerate(keypoints_3d[subject][action][emotion]):
                        print('Normalising vid ' + str(i) + ' for subject: ' + subject + ', emotion: ' + emotion)
                        kpts = data['keypoints']
                        kpts = median_shift_filter(kpts)
                        kpts = normalise_space(kpts)
                        timestep = data['timestep']
                        normalised_keypoints[subject][action][emotion].append({'keypoints': kpts, 'timestep': timestep})
        print('Saving...')
        np.savez_compressed('../data/paco/normalised_keypoints.npz', normalised_keypoints=normalised_keypoints)
        print('Done.')

if __name__ == '__main__':
    smooth_paco_keypoints()