import numpy as np
import os
import glob
import simplejson
from run_openpose import VideoPose3D




def _remove_frame(keypoints):
    # TODO: add tracking for intermediate values
    kpts_len = len(keypoints) # 25*3

    n_occluded_joints = 0
    for i in range(kpts_len):
        joint = keypoints[i]
        if len(joint) == 0 or (float(joint[0]) == 0.0 and float(joint[1]) == 0.0 and float(joint[2]) == 0):
            n_occluded_joints += 1

    return n_occluded_joints >= (kpts_len/2)


def get_2D_keypoints():
    keypoints_folder = ('../openpose/output/vid1')

    keypoints_list = list(glob.iglob(keypoints_folder + '/*.json'))
    vid_list = [keypoints_list]
    keypoints_output_dir = '../data'

    indices_to_use = []
    keypoints_2d = []


    openpose_to_coco = [0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11]


    for keypoints_list in vid_list:
        vid_name = ''
        subject = ''
        emotion = ''
        action = ''
        intensity = ''
        action = ''
        for file in keypoints_list:
            vid_name = file[len(os.path.dirname(file))+1:] #TODO check
            vid_name = vid_name[:-len('json')-1]
            subject = str(int(vid_name[:3])) + str(vid_name[3])
            emotion = vid_name[5:8]
            intensity = vid_name[9:12]
            action = 'Walking' + vid_name[17] if 'win' in vid_name else 'unknown_action'

            fp = open(file, 'r')
            json = simplejson.load(fp)
            if json['people'] != []:
                keypoints = json['people'][0]['pose_keypoints_2d']
                keypoints = np.array(keypoints)
                keypoints = [keypoints[i*3:i*3+3] for i in openpose_to_coco]

                if not _remove_frame(keypoints):
                    keypoints_2d.append(keypoints)




        metadata = {'layout_name': u'coco', u'num_joints': 17, u'keypoints_symmetry': [[1, 3, 5, 7, 9, 11, 13, 15], [2, 4, 6, 8, 10, 12, 14, 16]]}
        positions_2d = {'1m': {'Walking1': [keypoints_2d]}}


        print('Saving 2D keypoints...')
        np.savez_compressed(os.path.join(keypoints_output_dir, 'openpose_keypoints.npz'), positions_2d=positions_2d, metadata=metadata)
        print('Done.')


# get_2D_keypoints()

def obtain_3D_keypoints(subject, action, emotion, intensity):
    keypoints_output_dir = '../data'
    args = ['-k', 'detectron_pt_coco', '-arc', '3,3,3,3,3', '-c', 'checkpoint', '--evaluate', 'd-pt-243.bin',
            '--render',
            '--viz-camera', '0', '--viz-size', '3', '--viz-downsample', '1', '--viz-limit', '100']

    os.chdir('../VideoPose3D')
    VideoPose = VideoPose3D(args)
    predictions = VideoPose.main('../VideoPose3D/videos/walking_videos/001m_ang_2.6_win_1.wmv', subject, action,
                                 emotion, intensity)
    print('Saving 3D keypoints...')
    np.savez_compressed(os.path.join(keypoints_output_dir,
                                     'openpose3d_' + subject + '_' + action + '_' + emotion + '_' + intensity + '.npz'),
                        positions_3d=predictions)
    print('Done.')

obtain_3D_keypoints('1m', 'Walking1', 'ang', '2.6')