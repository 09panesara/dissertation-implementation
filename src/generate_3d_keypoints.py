from run import VideoPose3D
import glob
import os
import numpy as np

os.chdir('../VideoPose3D')
vid_folder = 'videos/walking_videos'
vid_list = list(glob.iglob(vid_folder + '/*.wmv'))
keypoints_output_dir = 'output/keypoints'

args = ['-k', 'detectron_pt_coco', '-arc', '3,3,3,3,3', '-c', 'checkpoint', '--evaluate', 'cpn-pt-243.bin', '--render',
        '--viz-camera', '0', '--viz-size', '3', '--viz-downsample', '1', '--viz-limit', '100']

VideoPose = VideoPose3D(args)

for video in vid_list:
    if '001m_ang_2.6_win_1.wmv' in video:
        print(video)
        vid_name = video[len(os.path.dirname(video))+1:]
        vid_name = vid_name[:-len('wmv')-1]
        subject = str(int(vid_name[:3])) + str(vid_name[3])
        emotion = vid_name[5:8]
        intensity = vid_name[9:12]
        action = 'Walking' + vid_name[-1] if 'win' in vid_name else 'unknown_action'

        predictions = VideoPose.main(video, subject, action, emotion, intensity)
        print('Saving...')
        np.savez_compressed(os.path.join(keypoints_output_dir, 'output_'+subject+'_'+action+'_'+emotion+'_'+intensity+'.npz'), positions_3d=predictions)
        print('Done.')


