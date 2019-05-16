from utils import data_utils
import numpy as np
import pandas as pd
import h5py
import os
import math



def calculate_angle(a, c, b=[0, 0, 0], degrees=False):
    ''' Returns angle between ab and bc '''
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cos_angle)
    if degrees:
        return np.degrees(angle)
    else:
        return angle

def dist_btwn_vectors(a, b):
    ''' Returns distance between two 3D points '''
    return np.linalg.norm(a-b)

def dist_btwn_pt_and_axis(p0, p1, p2):
    return np.linalg.norm(np.cross(p2-p1,p0-p1))/np.linalg.norm(p2-p1)

def _mean(x, y):
    return (x+y)/2;

def generate_LMA_features(keypoints, timestep_between_frame, include_joints=False):
    if include_joints:
        LMA_vector = {joints_by_index[i] +'_x': [frame[i][0] for frame in keypoints] for i in range(17)}
        LMA_vector.update({joints_by_index[i] + '_y': [frame[i][1] for frame in keypoints] for i in range(17)})
        LMA_vector.update({joints_by_index[i] + '_z': [frame[i][2] for frame in keypoints] for i in range(17)})
    else:
        LMA_vector = {}

    # Approximate center of mass as avg of points
    center_of_mass = [[np.mean(frame[:,0]), np.mean(frame[:,1]), np.mean(frame[:,2])] for frame in keypoints]
    ''' 
    Body:
    Spatial dissymmetry between two hands, two knees, two feet respectively. 0.5 = perfectly symmetrical. 
    Distance between left hand and left shoulder
    Distance between right hand and right shoulder
    Distance between left foot and left hip
    Distance between right foot and right hip 
    Angle at right elbow
    Angle at left elbow
    Angle at right knee
    Angle at left knee
   
    '''

    # dist between left/right wrist and axis binding center of hip and center of shoulders/neck
    d_LWrist_center = [dist_btwn_pt_and_axis(frame[joints['LWrist']], frame[joints['Neck/Nose']], frame[joints['Hip']]) for t, frame in enumerate(keypoints)]
    d_RWrist_center = [dist_btwn_pt_and_axis(frame[joints['RWrist']], frame[joints['Neck/Nose']], frame[joints['Hip']]) for t, frame in enumerate(keypoints)]

    # dist between left/right foot and axis binding center of hip and center of feet
    feet_centers = [_mean(frame[joints['LFoot']], frame[joints['RFoot']]) for frame in keypoints]
    d_LFoot_center = [dist_btwn_pt_and_axis(frame[joints['LFoot']], frame[joints['Hip']], feet_centers[t]) for t, frame in enumerate(keypoints)]
    d_RFoot_center = [dist_btwn_pt_and_axis(frame[joints['RFoot']], frame[joints['Hip']], feet_centers[t]) for t, frame in enumerate(keypoints)]

    # dist between left/right knees and axis binding center of knees and center of feet
    knee_centers = [_mean(frame[joints['LKnee']], frame[joints['RKnee']]) for frame in keypoints]
    d_LKnee_center = [dist_btwn_pt_and_axis(frame[joints['LKnee']], knee_centers[t], feet_centers[t]) for t, frame in enumerate(keypoints)]
    d_RKnee_center = [dist_btwn_pt_and_axis(frame[joints['RKnee']], knee_centers[t], feet_centers[t]) for t, frame in enumerate(keypoints)]

    d_LWrist_center = np.array(d_LWrist_center)
    d_RWrist_center = np.array(d_RWrist_center)
    d_LKnee_center = np.array(d_LKnee_center)
    d_RKnee_center = np.array(d_RKnee_center)
    d_LFoot_center = np.array(d_LFoot_center)
    d_RFoot_center = np.array(d_RFoot_center)

    LMA_vector['dissym_hands'] = list(np.divide(d_LWrist_center, d_LWrist_center + d_RWrist_center))
    LMA_vector['dissym_knees'] = list(np.divide(d_LKnee_center, d_LKnee_center + d_RKnee_center))
    LMA_vector['dissym_feet'] =  list(np.divide(d_LFoot_center, d_LFoot_center + d_RFoot_center))

    # distances
    LMA_vector['d_LWrist_LShoulder'] = [dist_btwn_vectors(frame[joints['LWrist']], frame[joints['LShoulder']]) for frame in keypoints]
    LMA_vector['d_RWrist_RShoulder'] = [dist_btwn_vectors(frame[joints['RWrist']], frame[joints['RShoulder']]) for frame in keypoints]
    LMA_vector['d_LFoot_LHip'] = [dist_btwn_vectors(frame[joints['LFoot']], frame[joints['LHip']]) for frame in keypoints]
    LMA_vector['d_RFoot_RHip'] = [dist_btwn_vectors(frame[joints['RFoot']], frame[joints['RHip']]) for frame in keypoints]

    # angle at elbows, knees
    LMA_vector['angle_LElbow'] = [calculate_angle(frame[joints['LShoulder']], frame[joints['LWrist']], frame[joints['LElbow']]) for frame in keypoints]
    LMA_vector['angle_RElbow'] = [calculate_angle(frame[joints['RShoulder']], frame[joints['RWrist']], frame[joints['RElbow']]) for frame in keypoints]
    LMA_vector['angle_LKnee'] = [calculate_angle(frame[joints['LHip']], frame[joints['LFoot']], frame[joints['LKnee']]) for frame in keypoints]
    LMA_vector['angle_LRKnee'] = [calculate_angle(frame[joints['RHip']], frame[joints['RFoot']], frame[joints['RKnee']]) for frame in keypoints]

    '''
    Effort: 
    (Flow) 3rd order derivative of joints (jerk)
    
    (Weight) vertical (y) components of velocity and acceleration sequences associated with joints 
    '''

    velocity = np.array([frame - keypoints[t-1] if t > 0 else np.zeros((17, 3)) for t, frame in enumerate(keypoints)])
    velocity = np.divide(velocity, timestep_between_frame)

    accel = np.array([v-velocity[t-1] if t > 0 else np.zeros((17, 3)) for t, v in enumerate(velocity)])
    accel = np.divide(accel, timestep_between_frame)

    flow = np.array([a - accel[t-1] if t > 0 else np.zeros((17, 3)) for t, a in enumerate(accel)])
    flow = np.divide(flow, timestep_between_frame)

    velocity = [[joint_veloc[1] for joint_veloc in frame] for frame in velocity ]
    accel = [[joint_accel[1] for joint_accel in frame] for frame in accel]

    LMA_vector.update({'veloc_y_' + joints_by_index[i]: [frame_veloc[i] for frame_veloc in velocity] for i in range(17)}) # No velocity/accel at timestep 0
    LMA_vector.update({'accel_y_' + joints_by_index[i]: [frame_accel[i] for frame_accel in accel] for i in range(17)})
    LMA_vector.update({'jerk_x_' + joints_by_index[i]: [frame_flow[i][0] for frame_flow in flow] for i in range(17)})
    LMA_vector.update({'jerk_y_' + joints_by_index[i]: [frame_flow[i][1] for frame_flow in flow] for i in range(17)})
    LMA_vector.update({'jerk_z_' + joints_by_index[i]: [frame_flow[i][2] for frame_flow in flow] for i in range(17)})

    ''' 
    Shape: 
    C(t) = contraction of body 
    (Spreading vs Enclosing) Avg distance between every joint to vertical axis of body extending from Head to Spine.
    (Rise vs Sink) Avg distance between every joints from center of mass
    (Shaping) Amplitudes A_x, A_y, A_z in directions perpendicular to vertical, horizontal and sagittal planes.
    OR ampltiude difference between knees, hips, shoulders, hands, feet respectively
    '''

    C_t = [(np.linalg.norm(frame[joints['Hip']] - frame[joints['LWrist']]) +
            np.linalg.norm(frame[joints['Hip']] - frame[joints['RWrist']])
            ) / 2 for frame in keypoints]

    spread_enclos_measure = [[dist_btwn_pt_and_axis(joint, frame[joints['Head']], frame[joints['Spine']]) for joint in frame] for t, frame in enumerate(keypoints)]
    spread_enclos_measure = [np.mean(frame) for frame in spread_enclos_measure]

    rise_sink_measure = [np.mean([dist_btwn_vectors(joint,center_of_mass[t]) for joint in frame]) for t, frame in enumerate(keypoints)]

    A_x = [np.max(frame[:,0])-np.min(frame[:,0]) for frame in keypoints]
    A_y = [np.max(frame[:,1])-np.min(frame[:,1]) for frame in keypoints]
    A_z = [np.max(frame[:,2])-np.min(frame[:,2]) for frame in keypoints]

    LMA_vector['contraction'] = C_t
    LMA_vector['spread_enclos'] = spread_enclos_measure
    LMA_vector['rise_sink'] = rise_sink_measure
    LMA_vector['Ampl_x'] = A_x
    LMA_vector['Ampl_y'] = A_y
    LMA_vector['Ampl_z'] = A_z


    '''
    Space: elbow forward-backward motion, forward tilt angle 
    '''
    forward_tilt_angle = [calculate_angle(np.array([0, 1, 0]), frame[joints['Hip']]-frame[joints['Head']]) for frame in keypoints]
    LElbow_fb_motion = [dist_btwn_vectors(frame[joints['LElbow']],keypoints[i-1][joints['LElbow']]) if i > 0 else 0 for i, frame in enumerate(keypoints)]
    RElbow_fb_motion = [dist_btwn_vectors(frame[joints['RElbow']],keypoints[i-1][joints['RElbow']]) if i > 0 else 0 for i, frame in enumerate(keypoints)]
    LKnee_fb_motion = [dist_btwn_vectors(frame[joints['LKnee']], keypoints[i - 1][joints['LKnee']]) if i > 0 else 0
                        for i, frame in enumerate(keypoints)]
    RKnee_fb_motion = [dist_btwn_vectors(frame[joints['RKnee']], keypoints[i - 1][joints['RKnee']]) if i > 0 else 0
                        for i, frame in enumerate(keypoints)]


    LMA_vector['forward_tilt_angle'] = forward_tilt_angle
    LMA_vector['LElbow_fb_motion'] = LElbow_fb_motion
    LMA_vector['RElbow_fb_motion'] = RElbow_fb_motion
    LMA_vector['LKnee_fb_motion'] = LKnee_fb_motion
    LMA_vector['RKnee_fb_motion'] = RKnee_fb_motion

    ''' Convert to list '''
    LMA_vector = {key: list(LMA_vector[key]) for key in LMA_vector}
    return LMA_vector




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





joints_by_index = list(joints.keys())


def LMA_action_db(include_joints=False):
    # if not os.path.isfile('../data/action_db/3dpb_keypoints.npz'):
    #     data_utils.pose_baseline_to_h36m(path='../data/3d-pose-baseline', output_dir='../data/3d-pose-baseline')

    keypoints_3d = data_utils.load_action_db_keypoints(keypoints_folder='../data/action_db', normalised=True)

    LMA_df = pd.DataFrame()
    timesteps = data_utils.get_timestep(timesteps_path='../data/action_db/timesteps.npz',
                                        videos_dir='../VideoPose3D/videos/walking_videos', normalised=True)  # float

    for subject in keypoints_3d:
        for action in keypoints_3d[subject]:
            for emotion in keypoints_3d[subject][action]:
                for intensity in keypoints_3d[subject][action][emotion]:
                    for i, data in enumerate(keypoints_3d[subject][action][emotion][intensity]):
                        print('Subject: %s, action: %s, emotion: %s, intensity: %s' %(subject, action, emotion, intensity))
                        kpts = data['keypoints']
                        LMA_features = {'subject': subject, 'action': action, 'emotion': emotion, 'intensity': intensity}
                        kpts = [np.array(frame) for frame in kpts]
                        LMA_features.update(generate_LMA_features(kpts, timestep_between_frame=timesteps[subject][action][emotion][intensity][i], include_joints=include_joints))
                        LMA_df = LMA_df.append(LMA_features, ignore_index=True)

    # Write pandas dataframe to compressed h5.py file
    LMA_df.to_hdf('../data/action_db/LMA_features.h5', key='df', mode='w')


def LMA_paco(include_joints=False, normalised_by='size'):
    print('Computing LMA feature vectors for PACO dataset...')
    paco_emotions = ['ang', 'hap', 'neu', 'sad']
    keypoints_3d = data_utils.load_paco_keypoints(normalised_by=normalised_by)

    LMA_df = pd.DataFrame()

    ''' Laban Movement Analysis Features Extraction '''
    for subject in keypoints_3d:
        for action in keypoints_3d[subject]:
            for emotion in keypoints_3d[subject][action]:
                if emotion == 'fea': # Ignore fear
                    continue
                for i, data in enumerate(keypoints_3d[subject][action][emotion]):
                    kpts = data['keypoints']
                    timestep = data['timestep']
                    print('Subject: %s, action: %s, emotion: %s' % (
                    subject, action, emotion))
                    LMA_features = {'subject': subject, 'action': action, 'emotion': emotion}
                    LMA_vector = generate_LMA_features(kpts, timestep_between_frame=timestep, include_joints=include_joints)
                    LMA_features.update(LMA_vector)
                    LMA_df = LMA_df.append(LMA_features, ignore_index=True)

    for emotion in paco_emotions:
        df = LMA_df.loc[LMA_df['emotion']==emotion]
        print('Saving...')
        df.to_hdf('../data/paco/LMA_features_' + emotion + '.h5', key='df', mode='w')
        print('Done.')



if __name__ == '__main__':
    # LMA_paco(include_joints=False, normalised_by='size')
    data_utils.get_train_test_set(folder='../data/paco')



