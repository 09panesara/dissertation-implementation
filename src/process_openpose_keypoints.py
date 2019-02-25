import simplejson as json
from collections import Counter
import numpy as np
import glob
import os
import shutil
import re
import random
from filterpy.kalman import KalmanFilter

class ProcessOpenPose2DData():
    def __init__(self, kpts_dir='../data/openpose_output'):
        self.keypoints_dir = kpts_dir

        self.keypoint_mapping = [
            "Nose",
            "Neck",
            "RShoulder",
            "RElbow",
            "RWrist",
            "LShoulder",
            "LElbow",
            "LWrist",
            "MidHip",
            "RHip",
            "RKnee",
            "RAnkle",
            "LHip",
            "LKnee",
            "LAnkle",
            "REye",
            "LEye",
            "REar",
            "LEar",
            "LBigToe",
            "LSmallToe",
            "LHeel",
            "RBigToe",
            "RSmallToe",
            "RHeel"]

        self.no_videos = 563


    def remove_empty_openpose_kpts(self, remove_occluded_at_end=True):
        print('Cleaning openpose keypoints')

        directories = sorted([self.keypoints_dir + "/" + name for name in os.listdir(self.keypoints_dir) if os.path.isdir(self.keypoints_dir + "/" + name)])
        json_data = []
        for directory in directories:
            print('Processing ' + directory)
            kpts_list = sorted(list(glob.iglob(directory + '/*.json')))
            # Filter out json files that are empty, providing all following are empty
            non_empty_jsons = []

            # Are there any keypoints collected for json files with < 25 joints c
            without_25 = []
            occluded = []

            for kpts_json in kpts_list:
                with open(kpts_json, 'r') as f:
                    data = json.load(f)
                if data['people'] != []:
                    non_empty_jsons.append(kpts_json)
                    if len(data['people'][0]['pose_keypoints_2d']) != 75:
                        without_25.append(kpts_json, len(data['people'][0]['pose_keypoints_2d']))
                        f.close()
                    if self._count_occluded_kpts(data['people'][0]['pose_keypoints_2d']) > 12: # If more than half of joints were occluded
                        occluded.append(kpts_json)
                else:
                    f.close()
                    print("Removing " + kpts_json + ' ...')
                    os.remove(kpts_json)



            json_data.append(non_empty_jsons)
            print('Occluded')
            print(occluded)

            occluded_kpt_numbers = [int(frame[-19:-15]) for frame in occluded]
            # Remove those with consecutive frames in a row where openpose fails that are near the end
            if remove_occluded_at_end and len(occluded) > 0:
                ''' TODO test'''
                kpts_file_base = occluded[0][:-18]
                self._remove_end_occluded_frames(kpts_file_base, occluded_kpt_numbers, last_kpt_no=int(non_empty_jsons[-1][-19:-15]))

            # Interpolate rest



            # # Remove frames where <= 2 frames in a row that have over half of joints occluded
            # occluded = [occluded[i] for i, frame_no in enumerate(occluded_kpt_numbers) if not((frame_no-1) in occluded_kpt_numbers or (frame_no+1) in occluded_kpt_numbers)]

            # Any keypoints with wrong number of keypoints - NO
            if len(without_25) > 0:
                print('No. keypoint sets without 25 keypoints collected ' + str(len(without_25)))


    ''' TODO: test'''
    def _remove_end_occluded_frames(self, file_base, occluded_kpt_numbers, last_kpt_no):
        remove = [kpt_no for kpt_no in occluded_kpt_numbers if kpt_no in range(last_kpt_no-10, last_kpt_no)]
        if remove != []:
            remove_from = remove[0]
            kpts_to_remove = [file_base + (3-len(str(i)))*'0' + str(i) + '_keypoints.json' for i in range(remove_from, last_kpt_no+1)]
            for kpts_json in kpts_to_remove:
                print('Removing ' + kpts_json)
                try:
                    os.remove(kpts_json)
                except:
                    print('No such file - possibly already removed')



    def _count_occluded_kpts(self, kpts):
        ''' Count number of keypoints not collected due to occlusion/noise/etc
        :kpts: takes loaded keypoints json data
        :return: number of occluded joints
        '''
        counts = Counter(kpts)
        no_zeros = int(counts[0] / 3)
        return no_zeros

    def _interpolate(self):
        '''
        For missing frames i.e. 3 consecutive 0's, get previous frames that were missing
        :return:
        '''
        directories = [self.keypoints_dir + "/" + name for name in os.listdir(self.keypoints_dir) if
                       os.path.isdir(self.keypoints_dir + "/" + name)]
        for directory in directories:
            kpts_list = sorted(list(glob.iglob(directory + '/*.json')))

    def _check_keypoints(self, filename):
        f = open(filename, 'r')
        data = json.load(f)
        if data['people'] != []:
            pose_keypoints = data['people'][0]['pose_keypoints_2d']
        else:
            pose_keypoints = []
        return pose_keypoints

    def generate_empty_keypoints_dict(self, to_examine, to_examine_files):
        empty_keypoints = {}
        for i, fileset in enumerate(to_examine_files):
            curr_empty_keypoints = []
            for file_no, file in enumerate(fileset):
                with open(file, 'r') as f:
                    data = json.load(f)
                if data['people'] == []:
                    curr_empty_keypoints.append(file)
                f.close()
            empty_keypoints[to_examine[i]] = curr_empty_keypoints
        return empty_keypoints


    def _check_keypoints_in_range(self, file_prefix, start_no, end_no):
        for i in range(start_no, end_no + 1):
            filename = self.keypoints_dir + '/' + file_prefix + '/' + file_prefix + (12-len(start_no))*'0' + str(i) + '_keypoints.json'
            self._check_keypoints(filename)

    def _check_all_no_keypoints(self, file_prefix, start_no, end_no):
        for i in range(start_no, end_no):
            filename = file_prefix + str(start_no) + '_keypoints.json'
            f = open(filename, 'r')
            data = json.load(f)
            if data['people'] != []:
                print("false")
                return


    def _write_back(filename, data):
        with open(filename, 'w') as f:
            f.write(str(data))





    # def apply_kf_to_missing(self):
    #     # Treat remaining empty keypoints as full NAs we need to replace using kalman filter
    #     # keypoints = self.keypoints
    #     # keypoints = [[[item if item!=0 or isinstance(item, float) else np.ma for item in video_keypoints] for video_keypoints in video] for video in keypoints]
    #     # # anything with 0 =noise, 0.0 = actual angle
    #     # # self.write_back('keypoints.txt', keypoints)
    #     # self.keypoints = keypoints
    #     keypoints = self.keypoints
    #     keypoints = np.ma.asarray(keypoints)
    #     # Replace nans with ma.masked
    #     keypoints = [[[item if item != 0 or isinstance(item, float) else ma.masked for item in video_keypoints] for video_keypoints in video] for video in keypoints]
    #     kf = KalmanFilter.em(X=keypoints, n_iter=5)
    #     (filtered_state_means, filtered_state_covariances) = kf.filter(measurements)
    #     (smoothed_state_means, smoothed_state_covariances) = kf.smooth(measurements)
    #     return


    def _filter_by_highest_intensity(self):
        directories = sorted([self.keypoints_dir + "/" + name for name in os.listdir(self.keypoints_dir) if
                              os.path.isdir(self.keypoints_dir + "/" + name)])
        filtered_dir_path = self.keypoints_dir + '/../' + 'filtered_openpose'
        if not os.path.isdir(filtered_dir_path):
            os.mkdir(filtered_dir_path)
        filtered_directories = []
        emotions = ['ang', 'fea', 'hap', 'neu', 'sad', 'unt']
        print('Filtering keypoints...')
        for i in range(30):
            for emotion in emotions:
                dirs = [directory for directory in directories if emotion in directory and (str(i) + 'm' in directory or str(i) + 'f' in directory)]
                intensity = [re.findall('\d\.\d', os.path.basename(dir))[0] for dir in dirs]
                dirs = [dir for i, dir in enumerate(dirs) if float(intensity[i]) >= 5]
                filtered_directories += dirs
        print('Copying keypoints...')
        for filtered_dir in filtered_directories:
            shutil.copytree(filtered_dir, filtered_dir_path + "/" + os.path.basename(filtered_dir))



    def _rts_smoother(self, dir):
        print('Applying rts smoother to openpose keypoints')
        directories = sorted([dir + "/" + name for name in os.listdir(dir) if
                              os.path.isdir(dir + "/" + name)])
        kpts_list = [sorted(list(glob.iglob(directory + '/*.json'))) for directory in directories]
        ''' process keypoints in array format '''

        # filter data with Kalman filter, than run smoother on it
        kpts = kpts_list[0]
        kpts = [self._check_keypoints(kpts_path) for kpts_path in kpts]
        kpts = [np.array(vid_kpts) for vid_kpts in kpts]
        kpts = np.array(kpts)
        # kpts = [kpt.reshape(75,1) for kpt in kpts]
        zs = kpts[0]
        print(zs.shape)
        zs.reshape(75)
        print(zs.shape)
        random.seed(123)
        fk = KalmanFilter(dim_x=75, dim_z=75)
        fk.x = zs[0] # initial state
        fk.F = np.ones(75)
        fk.H = np.eye(75)
        fk.Q = 0.1 # process uncertainty
        fk.R = 11 # state uncertainty, TODO iterate with different values

        mu, cov, _, _ = fk.batch_filter(zs)
        M, P, C, _ = fk.rts_smoother(mu, cov)


processOpenPose = ProcessOpenPose2DData()
# processOpenPose.remove_empty_openpose_kpts()
# processOpenPose._filter_by_highest_intensity()
processOpenPose._rts_smoother(dir='../data/filtered_openpose')

