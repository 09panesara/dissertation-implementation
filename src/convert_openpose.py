import numpy as np
import os
import glob
import simplejson

keypoints_folder = ('../openpose/output/vid1')

keypoints_list = list(glob.iglob(keypoints_folder + '/*.json'))
keypoints_output_dir = '../data'




for file in keypoints_list:
    fp = open(file, 'r')
    json = simplejson.load(fp)
    if json['people'] != []:
        keypoints = json['people']
    print('Saving...')
    np.savez_compressed(os.path.join(keypoints_output_dir, 'output_'+subject+'_'+action+'_'+emotion+'_'+intensity+'.npz'), positions_3d=predictions)
    print('Done.')
