''' Script to convert CSM files to np array '''
import os
import glob
import numpy as np


h36m_joints = [
    'Hip',
    'RHip',
    'RKnee',
    'RFoot',
    'LHip',
    'LKnee',
    'LFoot',
    'Spine',
    'Thorax',
    'Neck/Nose',
    'Head',
    'LShoulder',
    'LElbow',
    'LWrist',
    'RShoulder',
    'RElbow',
    'RWrist'
]


joints_x = {h36m_joints[i]: i*3 for i in range(17)}
joints_y = {h36m_joints[i]: i*3+1 for i in range(17)}
joints_z = {h36m_joints[i]: i*3+2 for i in range(17)}


def process_CSM(dir, output_dir='../../data/paco'):
    dirs = os.listdir(dir)
    dirs = [dir + "/" + d for d in dirs]
    dirs = [dir for dir in dirs if os.path.isdir(dir)]

    CSM_markers =   ['LFHD',
                     'RFHD',
                     'LBHD',
                     'RBHD',
                     'C7',
                     'CLAV',
                     'STRN',
                     'LSHO',
                     'LELB',
                     'LWRA',
                     'LWRB',
                     'LFIN',
                     'RSHO',
                     'RELB',
                     'RWRA',
                     'RWRB',
                     'RFIN',
                     'T10',
                     'SACR',
                     'LFWT',
                     'RFWT',
                     'LBWT',
                     'RBWT',
                     'LKNE',
                     'RKNE',
                     'LANK',
                     'RANK',
                     'LHEL',
                     'RHEL',
                     'LMT5',
                     'RMT5',
                     'LTOE',
                     'RTOE']

    #TODO use alternative markers if these are in file.
    CSM_alt_order = ['TopHead',
                     'LFrontHead',
                     'RFrontHead',
                     'LBackHead',
                     'RBackHead',
                     'TopSpine',
                     'Sternum',
                     'Shoulder_Asym',
                     'Chest',
                     'LShoulder',
                     'LOuterElbow',
                     'LWristThumb',
                     'LWristPinky',
                     'LHand',
                     'RShoulder',
                     'ROuterElbow',
                     'RWristThumb',
                     'RWristPinky',
                     'RHand',
                     'MiddleBack',
                     'LowerBack',
                     'LFrontWaist',
                     'RFrontWaist',
                     'LBackWaist',
                     'RBackWaist',
                     'LOuterKnee',
                     'ROuterKnee',
                     'LAnkle',
                     'RAnkle',
                     'LHeel',
                     'RHeel',
                     'LOuterMeta',
                     'ROuterMeta',
                     'LToe',
                     'RToe']

    # markers to use if alternative matche
    CSM_alt_markers = ['TopHead',
                     'LFHD',
                     'RFHD',
                     'LBHD',
                     'RBHD',
                     'C7',
                     'Sternum',
                     'Shoulder_Asym',
                     'Chest',
                     'LSHO',
                     'LELB',
                     'LWristThumb',
                     'LWristPinky',
                     'LFIN',
                     'RSHO',
                     'RELB',
                     'RWristThumb',
                     'RWristPinky',
                     'RFIN',
                     'T10',
                     'LowerBack',
                     'LFWT',
                     'RFWT',
                     'LBWT',
                     'RBWT',
                     'LOuterKnee',
                     'ROuterKnee',
                     'LAnkle',
                     'RAnkle',
                     'LHEL',
                     'RHEL',
                     'LOuterMeta',
                     'ROuterMeta',
                     'LToe',
                     'RToe']
    ''' 
    CSM acronym meanings
        Left Front head                   'LFHD',
        Right Front head                  'RFHD',
        Left Back head                    'LBHD',
        Right Back head                   'RBHD',
        Top of Spine                      'C7',
        Clavicle                          'CLAV',
        Center chest                      'STRN',
        Left Shoulder                     'LSHO',
        Left Elbow                        'LELB',
        Left Wrist Inner near thumb       'LWRA',
        Left Wrist Outer opposite thumb   'LWRB',
        Left Hand                         'LFIN',
        Right Shoulder                    'RSHO',
        Right Elbow                       'RELB',
        Right Wrist Inner near thumb      'RWRA',
        Right Wrist Outer opposite thumb  'RWRB',
        Right Hand                        'RFIN',
        Middle of Back                    'T10',
        Lower Back                        'SACR',
        Left Front Waist                  'LFWT',
        Right Front Waist                 'RFWT',
        Left Back Waist                   'LBWT',
        Right Back Waist                  'RBWT',
        Left Knee                         'LKNE',
        Right Knee                        'RKNE',
        Left Outer Ankle                  'LANK',
        Right Outer Ankle                 'RANK',
        Left Heel                         'LHEL',
        Right Heel                        'RHEL',
        Left Outer Metatarsal             'LMT5',
        Right Outer Metatarsal            'RMT5',
        Left Toe                          'LTOE',
        Right Toe                         'RTOE'
        
    '''
    files_to_manually_process = []
    positions_3d = {}
    emotions = {'an': 'ang', 'af': 'fea', 'sa': 'sad', 'ha': 'hap', 'nu': 'neu'}

    for csm_dir in dirs:
        files = sorted(list(glob.glob(csm_dir + '/*.csm')))

        files = [f for f in files if 'walk' in f]
        for filename in files:
            print(filename)
            if '_e_' in filename:
                print('THIS HAS _e_')
                files_to_manually_process.append(filename)
                continue
            curr_positions = [] # curr_positions = [[x0,y0,z0,x1,y1,z1,...],...] where each sublist = frame
            fname = os.path.basename(filename)
            index = fname.find('_')
            subject = fname[:index]
            fname = fname[index+1:]
            index = fname.find('_')
            action = fname[:index]
            action = 'walking' if action == 'walk' else action
            fname = fname[index+1:]
            index = fname.find('_')
            emotion = emotions[fname[:index]]

            # add to positions_3d
            if subject not in positions_3d:
                positions_3d[subject] = {}
            if action not in positions_3d[subject]:
                positions_3d[subject][action] = {}
            if emotion not in positions_3d[subject][action]:
                positions_3d[subject][action][emotion] = []

            with open(filename, 'r') as f:
                use_alt_markers = False
                no_frames = ''
                while not no_frames.startswith('$NumFrames'):
                    no_frames = f.readline()
                no_frames = int(no_frames[len('$NumFrames'):].strip())
                curr_frame = ''
                while not curr_frame.startswith('$Rate'):
                    curr_frame = f.readline()
                # get timestep between frames
                timestep = 1 / int(curr_frame.split(" ")[-1].strip())

                while not curr_frame.startswith('$Order'):
                    curr_frame = f.readline()
                curr_frame = f.readline().strip()
                file_order = curr_frame.split(" ")
                if file_order != CSM_markers:
                    if file_order == CSM_alt_order:
                        use_alt_markers = True
                        print('Alternative CSM markers')
                    else:
                        print('File ' + filename + ' needs to be manually processed.')
                        files_to_manually_process.append(filename)
                        continue

                while not curr_frame.startswith('$Points'):
                    curr_frame = f.readline()
                # At first frame
                for i in range(no_frames):
                    frame_joints = np.zeros(51)
                    frame = f.readline()
                    while frame == '\n': # skip empty lines to advance to next frame
                        frame = f.readline()

                    frame = frame.split('\t')
                    frame = frame[1:] # ignore frame number
                    if frame[-1] == '\n':
                        frame = frame[:-1]

                    # split into x,y,z

                    x = [pt for i, pt in enumerate(frame) if i%3 == 0]
                    y = [pt for i, pt in enumerate(frame) if i % 3 == 1]
                    z = [pt for i, pt in enumerate(frame) if i % 3 == 2]

                    # 'DROPOUT' = missing data
                    x = [0 if i=='DROPOUT' else float(i) for i in x]
                    y = [0 if i == 'DROPOUT' else float(i) for i in y]
                    z = [0 if i == 'DROPOUT' else float(i) for i in z]

                    no_csm_joints = int(len(frame)/3)

                    if not use_alt_markers:
                        x = {CSM_markers[i]: x[i] for i in range(no_csm_joints)}
                        y = {CSM_markers[i]: y[i] for i in range(no_csm_joints)}
                        z = {CSM_markers[i]: z[i] for i in range(no_csm_joints)}
                    else:
                        x = {CSM_alt_markers[i]: x[i] for i in range(no_csm_joints)}
                        y = {CSM_alt_markers[i]: y[i] for i in range(no_csm_joints)}
                        z = {CSM_alt_markers[i]: z[i] for i in range(no_csm_joints)}

                    # get joints data

                    frame_joints[joints_x['LHip']] = (x['LFWT']+x['LBWT']) / 2
                    frame_joints[joints_y['LHip']] = (y['LFWT'] + y['LBWT']) / 2
                    frame_joints[joints_z['LHip']] = (z['LFWT'] + z['LBWT']) / 2
                    frame_joints[joints_x['RHip']] = (x['RFWT'] + x['RBWT']) / 2
                    frame_joints[joints_y['RHip']] = (y['RFWT'] + y['RBWT']) / 2
                    frame_joints[joints_z['RHip']] = (z['RFWT'] + z['RBWT']) / 2

                    frame_joints[joints_x['Hip']] = (frame_joints[joints_x['LHip']] +  frame_joints[joints_x['RHip']]) / 2
                    frame_joints[joints_y['Hip']] = (frame_joints[joints_y['LHip']] +  frame_joints[joints_y['RHip']]) / 2
                    frame_joints[joints_z['Hip']] = (frame_joints[joints_z['LHip']] +  frame_joints[joints_z['RHip']]) / 2

                    frame_joints[joints_x['LKnee']] = (x['LFWT'] + x['LBWT']) / 2
                    frame_joints[joints_y['LKnee']] = (y['LFWT'] + y['LBWT']) / 2
                    frame_joints[joints_z['LKnee']] = (z['LFWT'] + z['LBWT']) / 2
                    frame_joints[joints_x['RKnee']] = (x['RFWT'] + x['RBWT']) / 2
                    frame_joints[joints_y['RKnee']] = (y['RFWT'] + y['RBWT']) / 2
                    frame_joints[joints_z['RKnee']] = (z['RFWT'] + z['RBWT']) / 2

                    frame_joints[joints_x['LFoot']] = x['LHEL']
                    frame_joints[joints_y['LFoot']] = y['LHEL']
                    frame_joints[joints_z['LFoot']] = z['LHEL']
                    frame_joints[joints_x['RFoot']] = x['RHEL']
                    frame_joints[joints_y['RFoot']] = y['RHEL']
                    frame_joints[joints_z['RFoot']] = z['RHEL']

                    # used for spread-enclose measure so middle of back/spine is what's needed
                    frame_joints[joints_x['Spine']] = x['T10']
                    frame_joints[joints_y['Spine']] = y['T10']
                    frame_joints[joints_z['Spine']] = z['T10']

                    # approximately top of spine - see h36m skeleton, thorax keypoints ~ top of spine
                    frame_joints[joints_x['Thorax']] = x['C7']
                    frame_joints[joints_y['Thorax']] = y['C7']
                    frame_joints[joints_z['Thorax']] = z['C7']


                    # head is center of LFHD RFHD LBHD RBHD
                    frame_joints[joints_x['Head']] = (x['LFHD'] + x['RFHD'] + x['LBHD'] + x['RBHD']) / 4
                    frame_joints[joints_y['Head']] = (y['LFHD'] + y['RFHD'] + y['LBHD'] + y['RBHD']) / 4
                    frame_joints[joints_z['Head']] = (z['LFHD'] + z['RFHD'] + z['LBHD'] + z['RBHD']) / 4

                    if use_alt_markers:
                        # used in LMA vector for center of shoulders - top of spine = best approximation
                        frame_joints[joints_x['Neck/Nose']] = x['C7']
                        frame_joints[joints_y['Neck/Nose']] = y['C7']
                        frame_joints[joints_z['Neck/Nose']] = z['C7']

                        frame_joints[joints_x['LWrist']] = (x['LWristThumb'] + x['LWristPinky']) / 2
                        frame_joints[joints_y['LWrist']] = (y['LWristThumb'] + y['LWristPinky']) / 2
                        frame_joints[joints_z['LWrist']] = (z['LWristThumb'] + z['LWristPinky']) / 2
                        frame_joints[joints_x['RWrist']] = (x['RWristThumb'] + x['RWristPinky']) / 2
                        frame_joints[joints_y['RWrist']] = (y['RWristThumb'] + y['RWristPinky']) / 2
                        frame_joints[joints_z['RWrist']] = (z['RWristThumb'] + z['RWristPinky']) / 2

                    else:
                        frame_joints[joints_x['Neck/Nose']] = (x['C7'] + x['CLAV']) / 2
                        frame_joints[joints_y['Neck/Nose']] = (y['C7'] + y['CLAV']) / 2
                        frame_joints[joints_z['Neck/Nose']] = (z['C7'] + z['CLAV']) / 2

                        frame_joints[joints_x['LWrist']] = (x['LWRA'] + x['LWRB']) / 2
                        frame_joints[joints_y['LWrist']] = (y['LWRA'] + y['LWRB']) / 2
                        frame_joints[joints_z['LWrist']] = (z['LWRA'] + z['LWRB']) / 2
                        frame_joints[joints_x['RWrist']] = (x['RWRA'] + x['RWRB']) / 2
                        frame_joints[joints_y['RWrist']] = (y['RWRA'] + y['RWRB']) / 2
                        frame_joints[joints_z['RWrist']] = (z['RWRA'] + z['RWRB']) / 2

                    frame_joints[joints_x['LShoulder']] = x['LSHO']
                    frame_joints[joints_y['LShoulder']] = y['LSHO']
                    frame_joints[joints_z['LShoulder']] = z['LSHO']
                    frame_joints[joints_x['RShoulder']] = x['RSHO']
                    frame_joints[joints_y['RShoulder']] = y['RSHO']
                    frame_joints[joints_z['RShoulder']] = z['RSHO']

                    frame_joints[joints_x['LElbow']] = x['LELB']
                    frame_joints[joints_y['LElbow']] = y['LELB']
                    frame_joints[joints_z['LElbow']] = z['LELB']
                    frame_joints[joints_x['RElbow']] = x['RELB']
                    frame_joints[joints_y['RElbow']] = y['RELB']
                    frame_joints[joints_z['RElbow']] = z['RELB']




                    assert len(frame_joints) == 51
                    frame_joints = [[frame_joints[i], frame_joints[i+1], frame_joints[i+1]] for i in range(0,51,3)]

                    # append to curr_positions
                    curr_positions.append(frame_joints)

                positions_3d[subject][action][emotion].append({'keypoints': curr_positions, 'timestep': timestep})



    print('Saving...')
    np.savez_compressed(output_dir + '/keypoints.npz', positions_3d=positions_3d)
    print('Done.')
    print("Files that need manual processing: ")
    print(files_to_manually_process)

    return positions_3d





if __name__ == '__main__':
    process_CSM('../../data/paco/csm')

# TODO: need to adjust train, test split to split only on emotion and subject
