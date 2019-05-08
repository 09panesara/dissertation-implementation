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


def normalise_space(keypoints):
    '''
    Transformations put hips in same plane parallel to yOz plane, put both shoulders at same height
    :param keypoints: 3D array of keypoint frames dimension n x 17 x 2 where n = no. frames
                      [[[x1,y1,z1],[x2,y2,z2],...], [...], ...]
    :return: normalised keypoints
    '''

    keypoints = np.array(keypoints)
    # mean center all positions such that pelvis (hip center) is (0,0,0)
    hip_centers = [frame[0] for frame in keypoints]
    normalised_kpts = [
        [[joint[0] - hip_centers[i][0], joint[1] - hip_centers[i][1], joint[2] - hip_centers[i][2]] for joint in frame] for
        i, frame in enumerate(keypoints)]
    # already normalised by position, i.e. z axis = up
    # make orientation invariant by making hips parallel to x axis by rotating around z axis i.e. rotating x, y coord so that subject faces positive y-axis.
    thetas = [math.atan(frame[joints['LHip']][1] - frame[joints['RHip']][1]) /
              (frame[joints['LHip']][0] - frame[joints['RHip']][0]) for frame in keypoints]
    rotation_mat = [[[math.cos(theta), -1 * math.sin(theta)], [math.sin(theta), math.cos(theta)]] for theta in thetas]
    print('5 == 4?')
    print(keypoints[10])
    print(keypoints[10])
    normalised_kpts = [
        [[np.dot(rotation_mat[i][1], [joint[1], joint[0]]), np.dot(rotation_mat[i][0], [joint[1], joint[0]]), joint[2]]
         for joint in frame] for i, frame in enumerate(normalised_kpts)]
    return np.array(normalised_kpts)


no_joints = 17
avg_limb_length = np.zeros((no_joints, no_joints))  # matrix
joint_hierarchy = {0: [4, 7, 1], 4: [5], 7: [8], 1: [2], 5: [6], 2: [3], 8: [9, 11, 14], 9: [10], 11: [12], 12: [13], 14: [15], 15: [16]}
get_parent = {j: i for i in joint_hierarchy for j in joint_hierarchy[i]}
print(get_parent)
order = [0, 4, 7, 1, 5, 8, 2, 6, 3, 9, 11, 14, 10, 12, 13, 15, 16]
no_limbs = 16

def dist_btwn_vectors(a, b):
    ''' Returns distance between two 3D points '''
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a - b)

def calc_avg_limb_lengths(seq_kpts):
    '''
    Calculate avg limb length for each body part
    :seq_kpts = array of kpts (= array of frames)
    '''

    def _calc_limb_lengths(frame):
        limb_lengths = np.zeros((no_joints, no_joints))
        for p in range(no_joints):
            if p in joint_hierarchy:
                for c in joint_hierarchy[p]:
                        dist = dist_btwn_vectors(frame[p], frame[c])
                        limb_lengths[p][c] = dist
                        limb_lengths[c][p] = dist

        return limb_lengths


    l = np.zeros((no_joints, no_joints)) # average limb lengths
    for seq in seq_kpts:
        seq_l = np.zeros((no_joints, no_joints))
        for frame in seq:
            limb_lengths = _calc_limb_lengths(frame)
            seq_l = np.add(seq_l, limb_lengths)
            seq_l = np.divide(seq_l, len(seq))
        l = np.add(l, seq_l)
    l = np.divide(l, len(seq_kpts))

    return l





def normalise_by_size(kpts, avg_limb_lengths):
    '''
    kpts = kpts for one vid sequence
    Assumed normalised by space already
    '''
    normalised_kpts = []
    for frame in kpts:
        # assert position of hips = (0,0,0)?
        new_frame = [[] for i in range(no_joints)]
        new_frame[0] = frame[0]
        for joint_index in order[1:]: # order 0 = hip = doesn't need scaling as center
            parent_joint = frame[get_parent[joint_index]]
            alpha = avg_limb_lengths[get_parent[joint_index]][joint_index]
            alpha = alpha / dist_btwn_vectors(parent_joint, frame[joint_index]) # frame[joint_index] = child
            x = parent_joint[0] + alpha * (frame[joint_index][0] - parent_joint[0])
            if str(x) == 'nan':
                print(joint_index)
                print(get_parent[joint_index])
                return
            y = parent_joint[1] + alpha * (frame[joint_index][1] - parent_joint[1])
            z = parent_joint[2] + alpha * (frame[joint_index][2] - parent_joint[2])
            new_frame[joint_index] = [x,y,z] # ENSURE SAME FORMAT OF NP ARRAYS IS KEPT
        assert len(new_frame) == 17
        normalised_kpts.append(new_frame)
    normalised_kpts = np.array(normalised_kpts)
    return normalised_kpts




def smooth_keypoints(redo_normalisation=False, paco=False):
    ''' Carry out normalisation if not already done'''
    if paco:
        print('Smoothing for paco')
        normalised_keypoints_path = '../data/paco/normalised_keypoints.npz'
        normalised_by_size_path = '../data/paco/normalised_by_size_kpts.npz'
    else:
        print('Smoothing for actions db')
        normalised_keypoints_path = '../data/action_db/normalised_keypoints.npz'
        normalised_by_size_path = '../data/action_db/normalised_by_size_kpts.npz'

    if not os.path.isfile(normalised_keypoints_path) or redo_normalisation:
        print('Normalising keypoints...')
        keypoints_3d = data_utils.load_paco_keypoints(normalised=False) if paco else data_utils.load_action_db_keypoints(normalised=False)
        normalised_keypoints = {}

        for subject in keypoints_3d:
            if subject not in normalised_keypoints:
                normalised_keypoints[subject] = {}
            for action in keypoints_3d[subject]:
                if action not in normalised_keypoints[subject]:
                    normalised_keypoints[subject][action] = {}
                for emotion in keypoints_3d[subject][action]:
                    if paco:
                        if emotion not in normalised_keypoints[subject][action]:
                            normalised_keypoints[subject][action][emotion] = []
                        for i, data in enumerate(keypoints_3d[subject][action][emotion]):
                            print('Normalising vid ' + str(i) + ' for subject: ' + subject + ', emotion: ' + emotion)
                            kpts = data['keypoints']
                            kpts = median_shift_filter(kpts)
                            kpts = normalise_space(kpts)
                            timestep = data['timestep']
                            normalised_keypoints[subject][action][emotion].append({'keypoints': kpts, 'timestep': timestep})
                    else:
                        if emotion not in normalised_keypoints[subject][action]:
                            normalised_keypoints[subject][action][emotion] = {}
                        for intensity in keypoints_3d[subject][action][emotion]:
                            if intensity not in normalised_keypoints[subject][action][emotion]:
                                normalised_keypoints[subject][action][emotion][intensity] = []
                            for i, data in enumerate(keypoints_3d[subject][action][emotion][intensity]):
                                print('Normalising vid ' + str(i) + ' for subject: ' + subject + ', emotion: ' + emotion + ', intensity: ' + intensity)
                                kpts = median_shift_filter(data)
                                kpts = normalise_space(kpts)
                                normalised_keypoints[subject][action][emotion][intensity].append({'keypoints': kpts})

        print('Saving...')
        np.savez_compressed(normalised_keypoints_path, positions_3d=normalised_keypoints)
        print('Done.')
    else:
        normalised_keypoints = data_utils.load_paco_keypoints(normalised=True) if paco else data_utils.load_action_db_keypoints(normalised=True)

    ''' Normalise by size '''
    if not os.path.isfile(normalised_by_size_path) or redo_normalisation:
        normalised_by_size = {}
        all_kpts = []
        # get all keypoints
        for subject in normalised_keypoints:
            if subject not in normalised_by_size:
                normalised_by_size[subject] = {}
            for action in normalised_keypoints[subject]:
                if action not in normalised_by_size[subject]:
                    normalised_by_size[subject][action] = {}
                for emotion in normalised_keypoints[subject][action]:
                    if paco:
                        if emotion not in normalised_by_size[subject][action]:
                            normalised_by_size[subject][action][emotion] = []
                        for i, data in enumerate(normalised_keypoints[subject][action][emotion]):
                            kpts = data['keypoints']
                            all_kpts.append(kpts)

                    else:
                        if emotion not in normalised_by_size[subject][action]:
                            normalised_by_size[subject][action][emotion] = {}
                            for intensity in keypoints_3d[subject][action][emotion]:
                                if intensity not in normalised_keypoints[subject][action][emotion]:
                                    normalised_keypoints[subject][action][emotion][intensity] = []
                                for i, data in enumerate(normalised_keypoints[subject][action][emotion][intensity]):
                                    all_kpts.append(data)

        avg_limb_length = calc_avg_limb_lengths(all_kpts)

        # normalise by size
        for subject in normalised_keypoints:
            if subject not in normalised_by_size:
                normalised_by_size[subject] = {}
            for action in normalised_keypoints[subject]:
                if action not in normalised_by_size[subject]:
                    normalised_by_size[subject][action] = {}
                for emotion in normalised_keypoints[subject][action]:
                    if paco:
                        if emotion not in normalised_by_size[subject][action]:
                            normalised_by_size[subject][action][emotion] = []
                        for i, data in enumerate(normalised_keypoints[subject][action][emotion]):
                            print('Normalising vid ' + str(i) + ' for subject: ' + subject + ', emotion: ' + emotion + ' by size.')
                            kpts = data['keypoints']
                            all_kpts.append(kpts)
                            kpts = normalise_by_size(kpts, avg_limb_length)
                            timestep = data['timestep']
                            normalised_by_size[subject][action][emotion].append({'keypoints': kpts, 'timestep': timestep})
                    else:
                        if emotion not in normalised_by_size[subject][action]:
                            normalised_by_size[subject][action][emotion] = {}
                            for intensity in keypoints_3d[subject][action][emotion]:
                                if intensity not in normalised_keypoints[subject][action][emotion]:
                                    normalised_keypoints[subject][action][emotion][intensity] = []
                                for i, data in enumerate(normalised_keypoints[subject][action][emotion][intensity]):
                                    print('Normalising vid ' + str(i) + ' for subject: ' + subject + ', emotion: ' + emotion + ', intensity: ' + intensity + ' by size.')
                                    kpts = normalise_by_size(data, avg_limb_length)
                                    normalised_by_size[subject][action][emotion][intensity].append({'keypoints': kpts})
        # print('Saving...')
        # np.savez_compressed(normalised_by_size_path, positions_3d=normalised_by_size)
        # print('Done.')





if __name__ == '__main__':
    smooth_keypoints(redo_normalisation=True, paco=True)