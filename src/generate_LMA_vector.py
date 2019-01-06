from utils import data_utils
import numpy as np
import pandas as pd
import h5py


def normalise_keypoints(keypoints):
    '''
    Transformations put shoulders and hip center in same plane parallel to yOz plane, put both shoulders at same height
    :param keypoints: 3D array of keypoint frames dimension n x 17 x 2 where n = no. frames
                      [[[x1,y1,z1],[x2,y2,z2],...], [...], ...]
    :return:
    '''
    keypoints = np.array(keypoints)
    print(keypoints[:5])

    # Translate body to set hip center at origin of landmark
    normalised_kpts = keypoints - np.tile(keypoints[:, :1], [17, 1])
    # Rotate axis z->x, x-> y, y->z
    normalised_kpts = [[[joint[2], joint[0], joint[1]] for joint in frame] for frame in normalised_kpts]
    # Rotate body around y axis to set left and right shoulders in plane // to yOz plane

    # Rotate body around z axis to set shoulder and hip centers in a plane // to yOz plane
    # Rotate body around x acis to set let and right shoulders in plane // to zOx plane
    # Translate body to put hip center at initial position
    normalised_kpts = normalised_kpts + np.tile(keypoints[:, :1], [17, 1])

    return normalised_kpts


def calculate_angle(a, c, b=[0,0,0], degrees=False):
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


# TODO: test
def generate_LMA_features(keypoints, timestep_between_frame):
    keypoints = normalise_keypoints(keypoints)
    LMA_vector = {}

    # Approximate center of mass as avg of points
    center_of_mass = [[np.mean(frame[:,0]), np.mean(frame[:,1]), np.mean(frame[:,2])] for frame in keypoints]
    ''' 
    Body:
    Spatial dyssemtry between two hands, two knees, two feet respectively. 0.5 = perfectly symmetrical. 
    Distance between left hand and left shoulder
    Distance between right hand and right shoulder
    Distance between left foot and left hip
    Distance between right foot and right hip 
    Distance between right knee and centre of gravity
    Distance between left knee and centre of gravity
   
    '''

    # TODO: Check for handedness of actors.
    # TODO: Check neck/nose = center of shoulders
    axis_hip_shoulders = [frame[joints['Neck/Nose']] - frame[joints['Hip']] for frame in keypoints] # axis binding center of hip and center of shoulders/neck
    axis_hip_feet = [frame[joints['Hip']] - np.mean(frame[joints['LFeet']], frame[joints['RFeet']]) for frame in keypoints] # axis binding center of hip and center of feet
    axis_knee_feet = [frame[joints['Hip']] - np.mean(frame[joints['LKnee']], frame[joints['RKnee']]) for frame in keypoints] # axis binding center of knees and center of feet

    d_LHand_center = [dist_btwn_vectors(frame[joints['LHand']], axis_hip_shoulders[t]) for t, frame in enumerate(keypoints)]
    d_RHand_center = [dist_btwn_vectors(frame[joints['RHand']], axis_hip_shoulders[t]) for t, frame in enumerate(keypoints)]

    d_LFoot_center = [dist_btwn_vectors(frame[joints['LFoot']], axis_hip_feet[t]) for t, frame in enumerate(keypoints)]
    d_RFoot_center = [dist_btwn_vectors(frame[joints['RFoot']], axis_hip_feet[t]) for t, frame in enumerate(keypoints)]

    d_LKnee_center = [dist_btwn_vectors(frame[joints['LKnee']], axis_knee_feet[t]) for t, frame in enumerate(keypoints)]
    d_RKnee_center = [dist_btwn_vectors(frame[joints['RKnee']], axis_knee_feet[t]) for t, frame in enumerate(keypoints)]

    LMA_vector['dys_hands'] = np.divide(d_LHand_center, np.sum(d_LHand_center, d_RHand_center)) # TODO: replace with non-dominant hand/knee/foot
    LMA_vector['dys_knees'] = np.divide(d_LKnee_center, np.sum(d_LKnee_center, d_RKnee_center))
    LMA_vector['dys_feet'] =  np.divide(d_LFoot_center, np.sum(d_LFoot_center, d_RFoot_center))


    LMA_vector['d_LHand_LShoulder'] = [dist_btwn_vectors(frame[joints['LHand']], frame[joints['LShoulder']]) for frame in keypoints]
    LMA_vector['d_RHand_RShoulder'] = [dist_btwn_vectors(frame[joints['RHand']], frame[joints['RShoulder']]) for frame in keypoints]
    LMA_vector['d_LFoot_LHip'] = [dist_btwn_vectors(frame[joints['LFoot']], frame[joints['LHip']]) for frame in keypoints]
    LMA_vector['d_RFoot_RHip'] = [dist_btwn_vectors(frame[joints['RFoot']], frame[joints['RHip']]) for frame in keypoints]
    LMA_vector['d_LKnee_center'] = d_LKnee_center
    LMA_vector['d_RKnee_center'] = d_RKnee_center

    '''
    Effort: 
    (Flow) 3rd order derivative of joints (jerk)
    
    (Weight) vertical (y) components of velocity and acceleration sequences associated with joints 
    '''

    velocity = [[0.0]*17]
    velocity[1:] = [frame - keypoints[t-1] for t, frame in enumerate(keypoints[1:])]
    velocity = np.divide(velocity, timestep_between_frame) # TODO: check are floats, should be

    accel = [[0.0]*17]
    accel[1:] = [v[t]-velocity[t-1] for t, v in enumerate(velocity[1:])]
    accel = np.divide(accel, timestep_between_frame)

    flow = [[0.0]*17]
    flow[1:] = [a[t] - accel[t-1] for t, a in enumerate(accel[1:])]
    flow = np.divide(flow, timestep_between_frame)

    velocity = [joint_veloc[1] for frame in velocity for joint_veloc in frame]
    accel = [joint_accel[1] for frame in accel for joint_accel in frame]

    LMA_vector.update({'veloc_y_'+ joints_by_index[i]: velocity[i] for i in range(17)}) # No velocity/accel at timestep 0
    LMA_vector.update({'accel_y_' + joints_by_index[i]: accel[i] for i in range(17)})
    LMA_vector.update({'jerk_' + joints_by_index[i]: flow[i] for i in range(17)})

    ''' 
    Shape: 
    C(t) = contraction of body 
    (Spreading vs Enclosing) Avg distance between every joint to vertical axis of body extending from Head to Spine.
    (Rise vs Sink) Avg distance between every joints from center of mass
    (Shaping) Amplitudes A_x, A_y, A_z in directions perpendicular to vertical, horizontal and saggital planes.
    OR ampltiude difference between knees, hips, shoulders, hands, feet respectively
    '''

    C_t = [(np.absolute(keypoints[frame[joints['Hip']]] - keypoints[frame[joints['LWrist']]]) +
            np.absolute(keypoints[frame[joints['Hip']]] - keypoints[frame[joints['RWrist']]])
            ) / 2 for frame in keypoints]

    vert_axis = [frame[joints['Head']] - frame[joints['Spine']] for frame in keypoints]
    spread_enclos_measure = [dist_btwn_vectors(frame, vert_axis[t]) for t, frame in enumerate(keypoints)] # TODO: come back to and check it works fine with repeating vert_axis for distance
    spread_enclos_measure = [np.mean(frame) for frame in spread_enclos_measure]

    rise_sink_measure = [np.mean(dist_btwn_vectors(frame,center_of_mass[t])) for t, frame in enumerate(keypoints)]

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
    forward_tilt_angle = [calculate_angle([0, 1, 0], frame[joints['Hip']]-frame[joints['Head']]) for frame in keypoints]
    LElbow_fb_motion = [dist_btwn_vectors(frame[joints['LElbow']],keypoints[i-1][joints['LElbow']]) if i > 0 else 0 for i, frame in enumerate(keypoints)]
    RElbow_fb_motion = [dist_btwn_vectors(frame[joints['RElbow']],keypoints[i-1][joints['RElbow']]) if i > 0 else 0 for i, frame in enumerate(keypoints)]

    LMA_vector['forward_tilt_angle'] = forward_tilt_angle
    LMA_vector['LElbow_fb_motion'] = LElbow_fb_motion
    LMA_vector['LElbow_fb_motion'] = RElbow_fb_motion

    return LMA_vector



keypoints_3d = data_utils.load_3d_keypoints(keypoints_folder='../data')
LMA_df = pd.DataFrame(columns=['subject', 'action', 'emotion', 'intensity', '...'])
# TODO: update with LMA feature names

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

timesteps = data_utils.get_timestep('../VideoPose3D/videos/walking_videos')  # float

for subject in keypoints_3d:
    for action in keypoints_3d[subject]:
        for emotion in keypoints_3d[subject][action]:
            for intensity in keypoints_3d[subject][action][emotion]:
                LMA_df.append()
                LMA_features = {'subject': subject, 'action': action, 'emotion': emotion, 'intensity': intensity}\
                    .update(generate_LMA_features(keypoints_3d[subject][action][emotion][intensity], timestep_between_frame=timesteps[subject][action][emotion][intensity]))
                LMA_df.append(LMA_features, ignore_index=True)



# Write pandas dataframe to compressed h5.py file
LMA_df.to_hdf('../data/LMA_features.h5', key='df', mode='w')